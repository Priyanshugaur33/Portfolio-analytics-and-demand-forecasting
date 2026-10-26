"""Portfolio Analytics & Demand Forecasting (E-commerce) - end-to-end pipeline.

Steps: generate messy data (15,000+ rows, 5+ categories, 4 regions) -> ETL/cleaning (~12% missing,
duplicates, outliers) -> EDA -> regression forecasting model -> KPI exports for Power BI.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(7)

# ---------------- 1. Generate a messy dataset ----------------
N = 15_300
cats = ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Books"]
regions = ["North", "South", "East", "West"]
cat_base = dict(zip(cats, [60, 45, 35, 30, 25, 15]))
reg_mult = dict(zip(regions, [1.1, 1.0, 0.9, 1.2]))
date = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, N), unit="D")
cat = rng.choice(cats, N); reg = rng.choice(regions, N)
price = np.round(np.array([cat_base[c] for c in cat]) * rng.uniform(0.8, 1.6, N), 2)
discount = rng.choice([0, 5, 10, 15, 20, 30], N, p=[.3, .2, .2, .15, .1, .05])
promo = rng.choice([0, 1], N, p=[.75, .25])
month = date.month
season = 1 + 0.35 * np.isin(month, [10, 11, 12]) + 0.1 * np.isin(month, [5, 6])
demand = (cat_base_arr := np.array([cat_base[c] for c in cat])) * np.array([reg_mult[r] for r in reg]) * season \
    * (1 + discount / 100 * 1.5) * (1 + 0.2 * promo) * (60 / price * 0.6 + 0.6)
units = np.maximum(1, np.round(demand * rng.normal(1, 0.08, N))).astype(float)

raw = pd.DataFrame({"order_date": date, "category": cat, "region": reg, "unit_price": price,
                    "discount_pct": discount.astype(float), "promo_flag": promo.astype(float),
                    "units_sold": units})
raw["revenue"] = (raw.units_sold * raw.unit_price * (1 - raw.discount_pct / 100)).round(2)
raw = pd.concat([raw, raw.sample(300, random_state=1)], ignore_index=True)  # duplicates
for col, frac in [("unit_price", .04), ("discount_pct", .05), ("promo_flag", .03), ("region", .04), ("units_sold", .02)]:
    raw.loc[raw.sample(frac=frac, random_state=hash(col) % 1000).index, col] = np.nan  # ~ messy
out_idx = raw.sample(150, random_state=3).index
raw.loc[out_idx, "units_sold"] *= 12  # outliers
raw.to_csv("data/raw_sales.csv", index=False)
print("Raw rows:", len(raw), "| missing cells % of rows with any NaN:", round(100 * raw.isna().any(axis=1).mean(), 1))

# ---------------- 2. ETL / cleaning ----------------
df = pd.read_csv("data/raw_sales.csv", parse_dates=["order_date"])
before = len(df)
df = df.drop_duplicates()
dups_removed = before - len(df)
miss_before = df.isna().sum().sum()
df["region"] = df.region.fillna(df.region.mode()[0])
df["promo_flag"] = df.promo_flag.fillna(0)
df["discount_pct"] = df.groupby("category").discount_pct.transform(lambda s: s.fillna(s.median()))
df["unit_price"] = df.groupby("category").unit_price.transform(lambda s: s.fillna(s.median()))
# Outliers: IQR cap within each category (winsorise)
def cap(s):
    q1, q3 = s.quantile([.25, .75]); hi = q3 + 3 * (q3 - q1)
    return s.clip(upper=hi)
n_out = int((df.units_sold > df.groupby("category").units_sold.transform(lambda s: s.quantile(.75) + 3 * (s.quantile(.75) - s.quantile(.25)))).sum())
df["units_sold"] = df.groupby("category").units_sold.transform(cap)
df["units_sold"] = df.units_sold.fillna(df.groupby("category").units_sold.transform("median"))
df["revenue"] = (df.units_sold * df.unit_price * (1 - df.discount_pct / 100)).round(2)
print(f"Cleaning: {dups_removed} duplicates removed, {miss_before} missing cells imputed, {n_out} outliers capped")

# ---------------- 3. EDA outputs ----------------
df["month"] = df.order_date.dt.month
df["year"] = df.order_date.dt.year
df["ym"] = df.order_date.dt.to_period("M").astype(str)
df.groupby("category").agg(revenue=("revenue", "sum"), units=("units_sold", "sum")).sort_values("revenue", ascending=False).to_csv("outputs/revenue_by_category.csv")
df.groupby("region").agg(revenue=("revenue", "sum"), units=("units_sold", "sum")).to_csv("outputs/revenue_by_region.csv")
monthly = df.groupby("ym").agg(revenue=("revenue", "sum"), units=("units_sold", "sum")).reset_index()
monthly.to_csv("outputs/monthly_trend.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 3.5)); ax.plot(monthly.ym, monthly.revenue, marker="o"); ax.set_title("Monthly revenue")
plt.xticks(rotation=90, fontsize=7); plt.tight_layout(); plt.savefig("outputs/monthly_revenue.png", dpi=120)

# ---------------- 4. Forecasting model ----------------
feat = pd.get_dummies(df[["category", "region"]], drop_first=True)
X = pd.concat([feat, df[["unit_price", "discount_pct", "promo_flag", "month"]],
               pd.get_dummies(df.month, prefix="m", drop_first=True)], axis=1).astype(float)
X = X.loc[:, ~X.columns.duplicated()]
y = df.units_sold
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
lin = LinearRegression().fit(Xtr, ytr)
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1).fit(Xtr, ytr)
res = {}
for name, m in [("LinearRegression", lin), ("RandomForest", rf)]:
    p = m.predict(Xte)
    res[name] = {"R2": round(r2_score(yte, p), 3), "MAE": round(mean_absolute_error(yte, p), 2),
                 "MAPE_%": round(100 * mean_absolute_percentage_error(yte, p), 1),
                 "accuracy_%(100-MAPE)": round(100 - 100 * mean_absolute_percentage_error(yte, p), 1)}
print(pd.DataFrame(res).T)
imp = pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False).head(8)
print("\nTop RF features:\n", imp.round(3))
best = rf if res["RandomForest"]["R2"] >= res["LinearRegression"]["R2"] else lin
df["predicted_units"] = best.predict(X).round(1)
df.to_csv("outputs/clean_with_forecast.csv", index=False)
pd.DataFrame(res).T.to_csv("outputs/model_metrics.csv")

# ---------------- 5. KPIs ----------------
kpis = {
    "total_revenue": round(df.revenue.sum(), 0), "total_units": int(df.units_sold.sum()),
    "avg_order_value": round(df.revenue.mean(), 2), "avg_discount_pct": round(df.discount_pct.mean(), 2),
    "promo_share_pct": round(100 * df.promo_flag.mean(), 1),
    "top_category": df.groupby("category").revenue.sum().idxmax(), "top_region": df.groupby("region").revenue.sum().idxmax(),
    "rows": len(df), "categories": df.category.nunique(), "regions": df.region.nunique(),
}
pd.Series(kpis).to_csv("outputs/kpis.csv", header=["value"])
print("\nKPIs:", kpis)

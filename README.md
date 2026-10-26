# Portfolio Analytics & Demand Forecasting (E-commerce)

An end-to-end analytics pipeline for e-commerce sales: data cleaning (ETL), exploratory analysis, a regression model that forecasts units sold, and KPI exports for a Power BI dashboard.

## The problem

Sales data arrives messy, and planning needs a view of what drives demand and how many units to expect. This project cleans the data, finds the drivers (category, region, price, discount, promotion, season) and forecasts units sold.

## Pipeline

```
Raw sales data (15,600 rows, messy)
  -> ETL / cleaning: duplicates, missing values, outliers
  -> EDA: revenue by category and region, monthly trend
  -> Model: Linear Regression vs Random Forest (80/20 train-test split)
  -> Evaluation: R2, MAE, MAPE
  -> KPI exports for Power BI
```

## Cleaning

| Problem | Fix | Why |
|---|---|---|
| Duplicate rows | `drop_duplicates()` | Avoid double-counting revenue |
| Missing region | Most frequent value | Preserves the distribution |
| Missing price / discount | Median within category | Robust to outliers, category-aware |
| Missing promo flag | 0 (assumed no promo) | Stated assumption |
| Outliers in units sold | Capped at Q3 + 3 x IQR per category | Keeps rows, limits distortion |

On the synthetic data: 208 duplicates removed, 2,804 missing cells imputed, 158 outliers capped, leaving 15,392 clean rows across 6 categories and 4 regions.

## Model

- **Target:** units sold
- **Features:** category, region, unit price, discount %, promo flag, month (one-hot encoded)
- **Models:** Linear Regression (baseline) and Random Forest (200 trees)

| Model | R2 | MAE (units) | MAPE |
|---|---|---|---|
| Linear Regression | 0.738 | 7.45 | 10.1% |
| Random Forest | 0.747 | 7.21 | 9.6% |

**"Accuracy" here means 100 minus MAPE**, so about 90% (90.4% for Random Forest) on the held-out test set. Accuracy is not a standard regression metric, so R2, MAE and MAPE are reported as well.

**Top drivers (Random Forest):** month (seasonality), Electronics category, unit price, discount %, promo flag.

## Findings (synthetic data)

- Electronics is the top revenue category and North is the top region.
- Q4 (October to December) shows a clear seasonal uplift.
- Higher discounts and promotions raise units sold.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py      # generates raw data, cleans, models, writes outputs/
```

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `clean_with_forecast.csv` | Cleaned data with predicted units |
| `model_metrics.csv` | R2, MAE, MAPE per model |
| `revenue_by_category.csv`, `revenue_by_region.csv` | Revenue and units breakdowns |
| `monthly_trend.csv`, `monthly_revenue.png` | Monthly revenue trend |
| `kpis.csv` | Headline KPIs |

Load the CSVs into Power BI for a multi-page dashboard: executive summary, category and region performance, discount and promo analysis, and forecast vs actual.

## Limitations

- **Random train-test split.** For forecasting, a time-based split avoids leaking future information into training. This is the first thing to change.
- **Synthetic data.** The relationships were generated, so performance on real data would differ.
- **Simple imputation.** Median and mode fills ignore relationships between columns.

## Next steps

Time-based validation, lag and rolling-average features, holiday effects, gradient boosting, SARIMA or Prophet, and prediction intervals.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Power BI.

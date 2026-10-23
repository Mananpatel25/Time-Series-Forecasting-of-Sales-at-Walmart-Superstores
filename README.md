# Forecasting Weekly Sales at Walmart Stores

Two approaches to the same question — *how much will each store sell next week?* — compared on the Kaggle Walmart Store Sales dataset (45 stores, 143 weeks, with holiday flags, temperature, fuel price, CPI and unemployment).

## Results

| Notebook | Approach | Best model | Result |
|---|---|---|---|
| `walmart-sales-forecasting.ipynb` | Supervised regression with engineered calendar + holiday + lag features | Random forest | WMAE **5,850 → 1,801** after feature selection |
| `walmart-sales-prediction.ipynb` | Model comparison on scaled target | Random forest / XGBoost | R² 0.979 / 0.973 (linear 0.923, KNN 0.922) |
| `walmart-sales-forecasting.ipynb` | Classical time series | `auto_arima` → ARIMA(3,0,2) | AIC 1782.9 |

WMAE (weighted mean absolute error, holiday weeks weighted 5x) is the Kaggle competition metric — it penalises missing a Thanksgiving week far more than a normal one.

## What mattered

- **Holiday-week features** carried most of the signal; macro variables (CPI, unemployment) added almost nothing week to week.
- **Lag features** (sales 1, 2, 52 weeks back) are what made the tree models competitive with ARIMA on seasonality.
- **Feature selection** was the single biggest win — dropping noisy columns cut WMAE by two-thirds.

## Run

```bash
pip install pandas numpy scikit-learn xgboost statsmodels pmdarima matplotlib seaborn
```
Download `train.csv`, `features.csv`, `stores.csv` from the [Kaggle competition](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting), place next to the notebooks, run.

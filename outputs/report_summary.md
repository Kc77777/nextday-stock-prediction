# Prediction Report — N225

*Generated: 2025-09-09 13:01:07*

## Executive Summary
- **Data**: `Market.csv` — ticker with max rows selected (here: **N225**), Adjusted Close preferred, business-day resampling, interpolation & ffill/bfill.
- **Task**: Predict **next-day log return** (regression) + **direction** (classification from sign).
- **Models compared (CV)**: Ridge (with scaling), RandomForest, GradientBoosting
- **Best model** (by CV RMSE): **Ridge**
- **Holdout (~1Y) regression**: MAE = 0.00798, RMSE = 0.01055, MAPE = 114.26%, R² = -0.0343
- **Headline ‘prediction accuracy’** (for PPT, from 100 − MAPE): **0.0%** *(not a standard metric; use MAPE/MAE/RMSE/R² in the report)*
- **Direction (Up/Down) on holdout**: Accuracy = 0.456, Precision = 0.477, Recall = 0.402, F1 = 0.436

## Data & Preprocessing
- Range used: **1965-10-11 → 2021-06-02**
- Features: lags of returns, rolling means/stds, SMA distances, Bollinger %B, RSI, MACD, and cyclical calendar encodings.
- Targets: `y_logret_next` (regression) and `y_up_next` (classification label).

## Model Choice — What & Why
- **Ridge**: linear baseline with regularization; works well on high-dimensional technicals when relationships are mostly linear.
- **RandomForest / GradientBoosting**: capture **nonlinear** interactions between technical features; robust to outliers.

## Results
**Cross-Validation (mean over folds):**
| Model            |      MAE |     RMSE |        MAPE |        R2 |
|:-----------------|---------:|---------:|------------:|----------:|
| Ridge            | 0.008601 | 0.012378 | 9.02625e+08 | -0.041591 |
| RandomForest     | 0.011618 | 0.015783 | 5.21569e+09 | -2.51336  |
| GradientBoosting | 0.014646 | 0.019585 | 7.55399e+09 | -8.07082  |

**Holdout (~1Y) — Best Model:**
|       |     MAE |     RMSE |    MAPE |        R2 |
|:------|--------:|---------:|--------:|----------:|
| Ridge | 0.00798 | 0.010549 | 114.255 | -0.034264 |

**Confusion Matrix (Holdout, Direction from Sign)**
Rows = True [Down, Up]; Columns = Predicted [Down, Up]
|           |   Pred:Down |   Pred:Up |
|:----------|------------:|----------:|
| True:Down |          62 |        58 |
| True:Up   |          79 |        53 |

**Top Features** (by importance / |coef|):
| feature   |   importance |
|:----------|-------------:|
| ret       |     0.033539 |
| logret    |     0.031103 |
| sma_50    |     0.013977 |
| sma_10    |     0.013125 |
| sma_100   |     0.009024 |
| Close     |     0.006598 |
| sma_5     |     0.005839 |
| sma_20    |     0.004328 |
| sma_200   |     0.003367 |
| prc_sma_5 |     0.00287  |

## How to Use
- Retrain daily (expanding window) and **predict next day**.
- Convert regression to signal: **long if predicted return > 0**, else cash (or hedge).
- Monitor both regression metrics (MAPE/MAE/RMSE/R²) and direction metrics (Accuracy/F1).

## Why No Confusion Matrix for Regression?
- Confusion matrix applies to **classification**. We derive direction by thresholding the regression prediction’s sign; that’s what the matrix shows.

## Limitations & Next Steps
- Univariate price-only features limit signal; add **exogenous** features (sector ETFs, VIX, macro).
- Use **walk-forward** evaluation and hyperparameter tuning for robustness.
- Consider **ensembles** and add **regularization**/feature selection to reduce noise.

## Figures (saved)
- Holdout forecast plot: `outputs/holdout_forecast.png`
- Confusion matrix: `outputs/confusion_matrix.png`
- Equity curve: `outputs/equity_curve.png`
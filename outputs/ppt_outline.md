# Slide 1 — Problem & Data
- Ticker: **N225**; Range: 1965-10-11 → 2021-06-02
- Target: next-day log return; Direction = sign(target)

# Slide 2 — Features & Models
- Technicals (lags, rolling stats, SMA, RSI, MACD, calendar)
- Models: Ridge, RandomForest, GradientBoosting

# Slide 3 — CV & Best Model
- Best by CV RMSE: **Ridge**
- (Insert CV table image or numbers)

# Slide 4 — Holdout Performance
- MAE=0.00798, RMSE=0.01055, MAPE=114.26%, R²=-0.0343
- Headline (100 − MAPE): **0.0%** (informal)

# Slide 5 — Direction & Confusion Matrix
- Acc=0.456, Prec=0.477, Rec=0.402, F1=0.436
- Use `outputs/confusion_matrix.png` image

# Slide 6 — Equity Curve & Takeaways
- Include `outputs/equity_curve.png`
- Short horizon signals can be useful; add exogenous data for lift
- Deployment: batch retrain, dashboard for analysts
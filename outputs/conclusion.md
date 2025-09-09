# Conclusion — N225

**Data window:** 1965-10-11 → 2021-06-02

Best regression model: **Ridge** on ~1Y holdout — MAE=0.00798, RMSE=0.01055, MAPE=114.26%, R²=-0.0343 (headline “prediction accuracy” ≈ **0.0%**, derived as 100−MAPE). Tuned classifier threshold: **τ=0.195** (chosen on CV).

**Direction winner:** Classifier (Logistic) @ τ=0.195 — best F1=0.688 (Acc=0.524).
Final equity (holdout): Buy&Hold=1.29×, RegSign=1.04×, Cls@0.5=1.11×, Cls@τ*=1.29×. Highest: **Buy&Hold** (1.29×).

**Practical takeaway:** For this dataset, the regression model’s raw sign signal and the classifier produce modest directional skill. Short-horizon predictive power is limited by noise; to improve, add exogenous features (sector/FX/volatility/macro), tune hyperparameters with walk-forward validation, and consider ensembling (e.g., regression + classifier). Use the tuned decision threshold for any directional strategy.

**Files for slides:**
- `outputs/holdout_forecast.png` — prediction vs actual
- `outputs/confusion_matrix.png` — direction (from regression sign)
- `outputs/classifier_confusion_matrix.png` and `outputs/classifier_confusion_matrix_tuned.png` — classifier
- `outputs/holdout_equity_comparison.png` — equity curves
- `outputs/report_summary.md` — full report
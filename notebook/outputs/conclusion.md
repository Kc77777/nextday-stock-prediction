# Conclusion — N225

**Data window:** N/A

Regression holdout metrics not available.

**Direction winner:** N/A (no comparison table).
Equity comparison not available.

**Practical takeaway:** For this dataset, the regression model’s raw sign signal and the classifier produce modest directional skill. Short-horizon predictive power is limited by noise; to improve, add exogenous features (sector/FX/volatility/macro), tune hyperparameters with walk-forward validation, and consider ensembling (e.g., regression + classifier). Use the tuned decision threshold for any directional strategy.

**Files for slides:**
- `outputs/holdout_forecast.png` — prediction vs actual
- `outputs/confusion_matrix.png` — direction (from regression sign)
- `outputs/classifier_confusion_matrix.png` and `outputs/classifier_confusion_matrix_tuned.png` — classifier
- `outputs/holdout_equity_comparison.png` — equity curves
- `outputs/report_summary.md` — full report
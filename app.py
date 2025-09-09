# app.py — Stock NEXT-DAY PREDICTION (Regression + Classifier) with CV, tuning & plots
# Works with Market.csv (Index=Ticker, Date, Adj Close/Close)
import os, io, math, copy, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 140

# ====== PAGE CONFIG / THEME ======
st.set_page_config(page_title="Stock Prediction — Market.csv", layout="wide")
st.markdown("""
<style>
/* subtle polish */
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
.small { color:#6b7280; font-size:0.9rem; }
.metric-row div[data-testid="metric-container"] { background:#f8fafc; border-radius:12px; padding:8px 12px; }
hr { border: none; height: 1px; background: #e5e7eb; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Next-Day Prediction (Regression + Classifier)")
st.caption("Input: **Market.csv** · Picks ticker with most data · Business-day resample · Interpolate/ffill/bfill · Predict next-day log return + direction")

# ====== OPTIONAL: XGBoost availability ======
try:
    from xgboost import XGBRegressor, XGBClassifier
    _XGB_OK = True
except Exception:
    _XGB_OK = False

# ====== sklearn imports (estimators only; metrics coded manually) ======
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier

# ====== Safe TimeSeriesSplit (use sklearn if available; else shim) ======
try:
    from sklearn.model_selection import TimeSeriesSplit
except Exception:
    class TimeSeriesSplit:
        def __init__(self, n_splits=5): self.n_splits = int(n_splits)
        def split(self, X, y=None, groups=None):
            n = len(X)
            if n < (self.n_splits + 1):
                t = max(5, n // 5); s = max(1, n - t)
                yield np.arange(0, s), np.arange(s, n); return
            t = n // (self.n_splits + 1)
            for i in range(self.n_splits):
                s = (i+1)*t; e = min((i+2)*t, n)
                if s >= e: break
                yield np.arange(0, s), np.arange(s, e)

# ====== HELPERS: metrics (version-agnostic) ======
def reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    diff = y_true - y_pred
    mae  = float(np.mean(np.abs(diff)))
    mse  = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    denom = np.where(np.abs(y_true) < 1e-12, 1e-12, np.abs(y_true))
    mape = float(np.mean(np.abs(diff) / denom) * 100.0)
    var  = float(np.var(y_true))
    r2   = float(1.0 - (mse / var)) if var > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def confusion_matrix_simple(y_true, y_pred):
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    cm = np.zeros((2,2), dtype=int)
    for t,p in zip(y_true, y_pred): cm[t, p] += 1
    return cm

def cls_metrics_from_cm(cm):
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    acc  = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))
    f1   = (2*prec*rec) / max(1e-12, (prec + rec))
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

# ====== DATA LOADING / CLEANING ======
@st.cache_data(show_spinner=False)
def load_market_df(file_bytes: bytes | None):
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif os.path.exists("Market.csv"):
        df = pd.read_csv("Market.csv")
    else:
        raise FileNotFoundError("Upload Market.csv or place it next to app.py")
    df = df.rename(columns={"Index":"Ticker", "Adj Close":"Adj_Close"})
    if "Date" not in df.columns:
        raise RuntimeError("No 'Date' column in CSV.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    price_col = "Adj_Close" if "Adj_Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        raise RuntimeError("Need 'Adj Close' or 'Close' in CSV.")
    if "Ticker" not in df.columns:
        df["Ticker"] = "UNKNOWN"
    df = df[["Date","Ticker",price_col]].drop_duplicates(["Date","Ticker"], keep="last")
    counts = df.groupby("Ticker")["Date"].count().sort_values(ascending=False)
    chosen = counts.index[0]
    sub = df[df["Ticker"]==chosen].copy().sort_values("Date").set_index("Date")
    s = sub[price_col].astype(float).resample("B").mean().interpolate("time").ffill().bfill()
    s.name = "Close"
    return s, chosen

def add_features(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"Close": s})
    df["ret"] = df["Close"].pct_change()
    df["logret"] = np.log(df["Close"]).diff()
    for L in [1,2,3,5,10,20]:
        df[f"logret_lag{L}"] = df["logret"].shift(L)
    for win in [5,10,20,50]:
        df[f"roll_mean_{win}"] = df["logret"].rolling(win).mean()
        df[f"roll_std_{win}"]  = df["logret"].rolling(win).std()
    for win in [5,10,20,50,100,200]:
        ma = df["Close"].rolling(win).mean()
        df[f"sma_{win}"] = ma
        df[f"prc_sma_{win}"] = df["Close"] / ma - 1
    mid20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    upper20 = mid20 + 2*std20
    lower20 = mid20 - 2*std20
    df["pctB_20"] = (df["Close"] - lower20) / (upper20 - lower20)
    delta = df["Close"].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-12)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd; df["macd_signal"] = signal; df["macd_hist"] = macd - signal
    df["dow"] = df.index.dayofweek; df["month"] = df.index.month
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/5); df["dow_cos"] = np.cos(2*np.pi*df["dow"]/5)
    df["m_sin"] = np.sin(2*np.pi*df["month"]/12); df["m_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["y_logret_next"] = df["logret"].shift(-1)
    df["y_up_next"] = (df["y_logret_next"] > 0).astype(int)
    return df.dropna().copy()

# ====== SIDEBAR ======
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload Market.csv (optional)", type=["csv"])
    holdout_days = st.number_input("Holdout size (business days)", min_value=60, max_value=504, value=252, step=6)
    n_splits = st.slider("CV splits", 2, 8, 5)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.subheader("Regression models")
    reg_opts = ["Ridge", "RandomForest", "GradientBoosting"] + (["XGBRegressor"] if _XGB_OK else [])
    pick_reg = st.multiselect("Use models", reg_opts, default=["Ridge","RandomForest"] + (["XGBRegressor"] if _XGB_OK else []))
    optimize_metric = st.selectbox("Select best by", ["RMSE","MAE","MAPE","R2"])

    st.subheader("Classifier (direction)")
    use_classifier = st.checkbox("Train classifier (Logistic/RF/GB/XGB)", value=True)
    tune_threshold = st.checkbox("Tune decision threshold (maximize F1)", value=True)

    run_btn = st.button("🚀 Run")

# ====== LOAD / PREP ======
s, ticker = load_market_df(uploaded.read() if uploaded else None)
st.write(f"**Ticker:** `{ticker}`  ·  **Range:** {s.index.min().date()} → {s.index.max().date()}  ·  **N:** {len(s):,}")
with st.expander("Preview cleaned series"):
    st.line_chart(pd.DataFrame({"Close": s}))

fe = add_features(s)
feature_cols = [c for c in fe.columns if c not in ["y_logret_next","y_up_next"]]

H = min(holdout_days, max(60, len(fe)//20))
train_df = fe.iloc[:-H].copy()
test_df  = fe.iloc[-H:].copy()
X_train = train_df[feature_cols].values
y_train_reg = train_df["y_logret_next"].values
X_test  = test_df[feature_cols].values
y_test_reg = test_df["y_logret_next"].values
y_train_cls = train_df["y_up_next"].astype(int).values
y_test_cls  = test_df["y_up_next"].astype(int).values

# ====== BUILD MODELS ======
def build_reg_models(selected, seed=42):
    models = {}
    if "Ridge" in selected:
        models["Ridge"] = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=seed))])
    if "RandomForest" in selected:
        models["RandomForest"] = RandomForestRegressor(n_estimators=400, max_depth=None, n_jobs=-1, random_state=seed)
    if "GradientBoosting" in selected:
        models["GradientBoosting"] = GradientBoostingRegressor(random_state=seed)
    if "XGBRegressor" in selected and _XGB_OK:
        models["XGBRegressor"] = XGBRegressor(
            random_state=seed, n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist"
        )
    return models

def build_cls_models(seed=42):
    models = {
        "Logistic": Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))]),
        "RFClassifier": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed),
        "GBClassifier": GradientBoostingClassifier(random_state=seed),
    }
    if _XGB_OK:
        models["XGBClassifier"] = XGBClassifier(
            random_state=seed, n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
        )
    return models

def get_probs_or_scores(model, X):
    try:
        return model.predict_proba(X)[:,1]
    except Exception:
        try:
            s = model.decision_function(X)
            smin, smax = float(np.min(s)), float(np.max(s))
            return (s - smin) / (smax - smin + 1e-12)
        except Exception:
            return model.predict(X).astype(float)

# ====== RUN PIPELINE ======
tab_data, tab_reg, tab_cls, tab_cmp = st.tabs(["📄 Data", "📈 Regression", "🧮 Classifier", "🧪 Compare"])

with tab_data:
    st.subheader("Data & Features")
    st.write(f"Feature rows: **{len(fe):,}**  ·  Train: **{len(train_df):,}**  ·  Holdout: **{len(test_df):,}**  ·  #Features: **{len(feature_cols)}**")
    st.dataframe(fe[feature_cols + ["y_logret_next","y_up_next"]].head(10))
    st.caption("Targets: `y_logret_next` (regression), `y_up_next` (classification)")

if run_btn:
    # ===== Regression CV =====
    with tab_reg:
        st.subheader("Cross-Validation (Regression)")
        reg_models = build_reg_models(pick_reg, seed=seed)
        if not reg_models:
            st.warning("Select at least one regression model in the sidebar.")
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            rows = []
            for name, model in reg_models.items():
                for fold, (tr, va) in enumerate(tscv.split(X_train)):
                    X_tr, X_va = X_train[tr], X_train[va]
                    y_tr, y_va = y_train_reg[tr], y_train_reg[va]
                    model.fit(X_tr, y_tr)
                    y_hat = model.predict(X_va)
                    m = reg_metrics(y_va, y_hat)
                    m.update({"Model": name, "Fold": fold})
                    rows.append(m)
            cv_df = pd.DataFrame(rows)
            cv_summary = cv_df.groupby("Model")[["MAE","RMSE","MAPE","R2"]].mean().sort_values(optimize_metric, ascending=(optimize_metric != "R2"))
            st.dataframe(cv_summary.style.format("{:.6f}"))
            best_name = cv_summary.index[0]
            st.success(f"Best by **{optimize_metric}**: **{best_name}**")

            # Fit best on full train, evaluate holdout
            best_model = reg_models[best_name]
            best_model.fit(X_train, y_train_reg)
            y_pred = best_model.predict(X_test)
            hold = reg_metrics(y_test_reg, y_pred)

            c1, c2, c3, c4 = st.columns(4, gap="small")
            with c1: st.metric("MAE", f"{hold['MAE']:.6f}")
            with c2: st.metric("RMSE", f"{hold['RMSE']:.6f}")
            with c3: st.metric("MAPE", f"{hold['MAPE']:.2f}%")
            with c4: st.metric("R²", f"{hold['R2']:.4f}")

            fig1, ax = plt.subplots(figsize=(9,3.5))
            ax.plot(test_df.index, y_test_reg, label="True next-day logret")
            ax.plot(test_df.index, y_pred, label=f"Predicted ({best_name})")
            ax.set_title(f"{ticker} — Holdout (last {H}B)"); ax.legend(); ax.grid(False)
            st.pyplot(fig1)

            # Equity curve from regression-sign strategy
            ret_next_simple = np.expm1(y_test_reg)
            sig_reg = (y_pred > 0).astype(int)
            eq_bh = (1 + ret_next_simple).cumprod()
            eq_reg = (1 + sig_reg * ret_next_simple).cumprod()
            fig2, ax2 = plt.subplots(figsize=(9,3.5))
            ax2.plot(test_df.index, eq_bh, label="Buy & Hold")
            ax2.plot(test_df.index, eq_reg, label="RegSign strategy")
            ax2.set_title(f"{ticker} — Holdout Equity"); ax2.legend(); ax2.grid(False)
            st.pyplot(fig2)

            # Downloads
            pred_df = pd.DataFrame({
                "date": test_df.index, "y_true_logret_next": y_test_reg,
                "y_pred_logret_next": y_pred, "y_true_up": y_test_cls,
                "y_pred_up_regsign": sig_reg
            }).set_index("date")
            st.download_button("⬇️ Download holdout predictions (CSV)",
                               pred_df.to_csv(index=True).encode(),
                               file_name="holdout_predictions.csv",
                               mime="text/csv")

    # ===== Classifier track =====
    if use_classifier:
        with tab_cls:
            st.subheader("Direction Classifier")
            cls_models = build_cls_models(seed=seed)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            rows = []
            for name, model in cls_models.items():
                for fold, (tr, va) in enumerate(tscv.split(X_train)):
                    X_tr, X_va = X_train[tr], X_train[va]
                    y_tr, y_va = y_train_cls[tr], y_train_cls[va]
                    model.fit(X_tr, y_tr)
                    p = get_probs_or_scores(model, X_va)
                    y_hat = (p >= 0.5).astype(int)
                    cm = confusion_matrix_simple(y_va, y_hat)
                    m = cls_metrics_from_cm(cm)
                    m.update({"Model":name, "Fold":fold})
                    rows.append(m)
            cv_cls = pd.DataFrame(rows)
            cv_cls_summary = cv_cls.groupby("Model")[["Accuracy","Precision","Recall","F1"]].mean().sort_values("F1", ascending=False)
            st.dataframe(cv_cls_summary.style.format("{:.3f}"))

            best_cls_name = cv_cls_summary.index[0]
            best_cls_model = cls_models[best_cls_name].fit(X_train, y_train_cls)
            p_test = get_probs_or_scores(best_cls_model, X_test)
            y_pred_cls_default = (p_test >= 0.5).astype(int)
            cm_def = confusion_matrix_simple(y_test_cls, y_pred_cls_default)
            m_def = cls_metrics_from_cm(cm_def)

            c1, c2, c3, c4 = st.columns(4, gap="small")
            with c1: st.metric("Acc (τ=0.5)", f"{m_def['Accuracy']:.3f}")
            with c2: st.metric("Precision", f"{m_def['Precision']:.3f}")
            with c3: st.metric("Recall", f"{m_def['Recall']:.3f}")
            with c4: st.metric("F1", f"{m_def['F1']:.3f}")

            # Confusion matrix plot
            fig_cm, ax = plt.subplots(figsize=(3.6,3.6))
            im = ax.imshow(cm_def, cmap="Blues")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Down","Up"]); ax.set_yticklabels(["Down","Up"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm_def[i, j], ha="center", va="center", color="black")
            ax.set_title(f"Confusion (τ=0.5) — {best_cls_name}"); ax.set_xlabel("Pred"); ax.set_ylabel("True")
            fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig_cm)

            # Optional threshold tuning
            best_tau = 0.5
            if tune_threshold:
                st.markdown("**Threshold tuning (maximize F1 via CV)**")
                tscv = TimeSeriesSplit(n_splits=n_splits)
                y_val_all, p_val_all = [], []
                for tr, va in tscv.split(X_train):
                    model = copy.deepcopy(cls_models[best_cls_name]).fit(X_train[tr], y_train_cls[tr])
                    p = get_probs_or_scores(model, X_train[va])
                    y_val_all.append(y_train_cls[va]); p_val_all.append(p)
                yv = np.concatenate(y_val_all); pv = np.concatenate(p_val_all)
                thrs = np.linspace(0.05, 0.95, 181)
                rows = []
                for tau in thrs:
                    y_hat = (pv >= tau).astype(int)
                    cm = confusion_matrix_simple(yv, y_hat)
                    m = cls_metrics_from_cm(cm)
                    rows.append({"threshold":tau, **m})
                thr_df = pd.DataFrame(rows).sort_values(["F1","Accuracy"], ascending=[False, False])
                best_tau = float(thr_df.iloc[0]["threshold"])
                st.write(f"Best τ (CV F1): **{best_tau:.3f}**")

                fig_thr, axthr = plt.subplots(figsize=(8.5,3))
                axthr.plot(thr_df["threshold"], thr_df["F1"], label="F1")
                axthr.plot(thr_df["threshold"], thr_df["Accuracy"], label="Accuracy")
                axthr.axvline(best_tau, ls="--", label=f"Best τ={best_tau:.3f}")
                axthr.set_title("Threshold tuning (CV)"); axthr.set_xlabel("τ"); axthr.set_ylabel("Score")
                axthr.legend(); st.pyplot(fig_thr)

                # Apply tuned τ on holdout
                y_pred_cls_tuned = (p_test >= best_tau).astype(int)
                cm_tuned = confusion_matrix_simple(y_test_cls, y_pred_cls_tuned)
                m_tuned = cls_metrics_from_cm(cm_tuned)
                st.markdown("**Holdout (tuned τ)**")
                c1, c2, c3, c4 = st.columns(4, gap="small")
                with c1: st.metric("Acc (τ*)", f"{m_tuned['Accuracy']:.3f}")
                with c2: st.metric("Precision", f"{m_tuned['Precision']:.3f}")
                with c3: st.metric("Recall", f"{m_tuned['Recall']:.3f}")
                with c4: st.metric("F1", f"{m_tuned['F1']:.3f}")

                fig_cmt, axt = plt.subplots(figsize=(3.6,3.6))
                imt = axt.imshow(cm_tuned, cmap="Purples")
                axt.set_xticks([0,1]); axt.set_yticks([0,1])
                axt.set_xticklabels(["Down","Up"]); axt.set_yticklabels(["Down","Up"])
                for i in range(2):
                    for j in range(2):
                        axt.text(j, i, cm_tuned[i, j], ha="center", va="center", color="black")
                axt.set_title(f"Confusion (τ*) — {best_cls_name}"); axt.set_xlabel("Pred"); axt.set_ylabel("True")
                fig_cmt.colorbar(imt, ax=axt, fraction=0.046, pad=0.04)
                st.pyplot(fig_cmt)

            # Downloads
            out = pd.DataFrame({"y_true": y_test_cls, "p_test": p_test})
            st.download_button("⬇️ Download classifier probabilities (CSV)",
                               out.to_csv(index=False).encode(),
                               file_name="classifier_holdout_probs.csv",
                               mime="text/csv")

    # ===== Comparison tab =====
    with tab_cmp:
        st.subheader("Compare Strategies on Holdout")
        # regression sign available if regression ran
        if "y_pred" in locals():
            sig_reg = (y_pred > 0).astype(int)
        else:
            # quick baseline: use lag1 sign if regression wasn't run
            lag1_idx = feature_cols.index("logret_lag1") if "logret_lag1" in feature_cols else None
            sig_reg = (X_test[:, lag1_idx] > 0).astype(int) if lag1_idx is not None else np.zeros_like(y_test_cls)

        # classifier signals (if trained)
        if use_classifier:
            p_test = locals().get("p_test", None)
            if p_test is None:
                st.info("Run the Classifier tab to compute probabilities for comparison.")
                p_test = np.zeros_like(y_test_cls, dtype=float)
            sig_cls_05 = (p_test >= 0.5).astype(int)
            best_tau_val = locals().get("best_tau", 0.5)
            sig_cls_tuned = (p_test >= best_tau_val).astype(int)
        else:
            sig_cls_05 = np.zeros_like(y_test_cls); sig_cls_tuned = np.zeros_like(y_test_cls); best_tau_val = 0.5

        def pack_metrics(name, sig):
            cm = confusion_matrix_simple(y_test_cls, sig)
            m = cls_metrics_from_cm(cm)
            return {"Strategy": name, **m, "CM": cm}

        rows = [
            pack_metrics("RegSign (from Regression)", sig_reg),
            pack_metrics("Classifier @0.5", sig_cls_05),
            pack_metrics(f"Classifier @τ={best_tau_val:.3f}", sig_cls_tuned),
        ]
        comp = pd.DataFrame([{k:v for k,v in r.items() if k!="CM"} for r in rows])
        st.dataframe(comp.style.format({"Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}))

        # Equity curves
        ret_next_simple = np.expm1(y_test_reg)
        eq_bh   = (1 + ret_next_simple).cumprod()
        eq_reg  = (1 + sig_reg      * ret_next_simple).cumprod()
        eq_05   = (1 + sig_cls_05   * ret_next_simple).cumprod()
        eq_tune = (1 + sig_cls_tuned* ret_next_simple).cumprod()
        fig, ax = plt.subplots(figsize=(9,3.5))
        ax.plot(test_df.index, eq_bh, label="Buy & Hold")
        ax.plot(test_df.index, eq_reg, label="RegSign")
        ax.plot(test_df.index, eq_05,  label="Cls @0.5")
        ax.plot(test_df.index, eq_tune,label=f"Cls @τ={best_tau_val:.3f}")
        ax.set_title(f"{ticker} — Equity Curves (Holdout)"); ax.legend(); ax.grid(False)
        st.pyplot(fig)

        # Download metrics
        st.download_button("⬇️ Download comparison metrics (CSV)",
                           comp.to_csv(index=False).encode(),
                           file_name="holdout_direction_comparison.csv",
                           mime="text/csv")

else:
    with tab_reg:
        st.info("Set options in the sidebar and click **Run**.")
    with tab_cls:
        st.info("Enable **Train classifier** in the sidebar and click **Run**.")
    with tab_cmp:
        st.info("Run Regression and/or Classifier to compare strategies.")

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("💡 Tip: For stronger performance, add exogenous features (sector ETFs, VIX, macro), do walk-forward tuning, and consider ensembling.")

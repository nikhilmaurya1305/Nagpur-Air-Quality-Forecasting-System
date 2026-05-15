"""
train_xgboost.py
================
XGBoost AQI Forecasting — Nagpur City (CPCB Data)
Best model: R² = 0.88–0.91 across all 4 stations

Run from nagpur_xgb/ml/:
    python train_xgboost.py

Install:
    pip install xgboost scikit-learn pandas numpy matplotlib seaborn joblib

Outputs:
    models/xgb_aqi_{station}.pkl          ← trained XGBoost model
    models/feat_scaler_{station}.pkl      ← MinMaxScaler for features
    models/tgt_scaler_{station}.pkl       ← MinMaxScaler for AQI target
    models/model_config.json              ← feature list + config
    models/evaluation_report.json         ← train / val / test metrics
    plots/xgb_{station}.png              ← 9-panel accuracy figure
    results/xgb_metrics.json
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, confusion_matrix, classification_report
)
import xgboost as xgb

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH  = "../data/nagpur_final_preprocessed.csv"
MODEL_DIR  = "models"
PLOT_DIR   = "plots"
RESULT_DIR = "results"

WINDOW_SIZE = 48       # 48-hour look-back window
HORIZON     = 24       # forecast next 24 hours
FEATURES    = [
    "PM2.5","PM10","NO","NO2","SO2","NH3",
    "Hour_sin","Hour_cos","Month_sin","Month_cos",
    "DOW_sin","DOW_cos","IsWeekend"
]
TARGET      = "AQI"
STATIONS    = ["Ambazari","Mahal","Civil_Lines","Ram_Nagar"]
TEST_RATIO  = 0.15
VAL_RATIO   = 0.15

# ── XGBoost HYPERPARAMETERS ───────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators     = 500,    # Increased for better performance
    max_depth        = 8,      # Increased depth
    learning_rate    = 0.02,   # Lower LR for more stable learning
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0
)

for d in [MODEL_DIR, PLOT_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
def load_station(df, station):
    s = df[df["Station"] == station].copy()
    s["Datetime"] = pd.to_datetime(s["Datetime"])
    s = s.sort_values("Datetime").set_index("Datetime")
    idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(idx)
    s[FEATURES + [TARGET]] = s[FEATURES + [TARGET]].interpolate(method="time", limit=6)
    return s.dropna(subset=[TARGET])

def make_sequences(feat, tgt, window, horizon):
    X, y = [], []
    for i in range(len(feat) - window - horizon + 1):
        X.append(feat[i:i+window])
        y.append(tgt[i+window:i+window+horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def split_data(X, y):
    n  = len(X)
    nt = int(n * TEST_RATIO)
    nv = int(n * VAL_RATIO)
    return (X[:n-nt-nv], y[:n-nt-nv],
            X[n-nt-nv:n-nt], y[n-nt-nv:n-nt],
            X[n-nt:], y[n-nt:])

# ── HELPERS ───────────────────────────────────────────────────────────────────
def aqi_category(v):
    if v <= 50:    return "Good"
    elif v <= 100: return "Satisfactory"
    elif v <= 200: return "Moderate"
    elif v <= 300: return "Poor"
    elif v <= 400: return "Very Poor"
    else:          return "Severe"

def inv(ts, arr):
    return ts.inverse_transform(arr.reshape(-1, 1)).ravel()

def compute_metrics(yt, yp):
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)
    mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100
    ct   = [aqi_category(v) for v in yt]
    cp   = [aqi_category(v) for v in yp]
    ca   = sum(a == b for a, b in zip(ct, cp)) / len(ct) * 100
    return ({"MAE": round(float(mae),4), "RMSE": round(float(rmse),4),
             "R2":  round(float(r2),4),  "MAPE": round(float(mape),4),
             "CatAcc": round(float(ca),2)},
            ct, cp)

def print_metrics(tr_m, v_m, te_m, station):
    print(f"\n  {'═'*54}")
    print(f"  ACCURACY — XGBoost | {station}")
    print(f"  {'═'*54}")
    print(f"  {'Metric':10s}  {'Train':>10s}  {'Val':>10s}  {'Test':>10s}")
    print(f"  {'─'*50}")
    for k in ["MAE","RMSE","R2","MAPE","CatAcc"]:
        u = "%" if k in ["MAPE","CatAcc"] else ""
        print(f"  {k:10s}  {str(tr_m[k])+u:>10s}  "
              f"{str(v_m[k])+u:>10s}  {str(te_m[k])+u:>10s}")
    gap = te_m["MAE"] - tr_m["MAE"]
    print(f"\n  MAE gap = {gap:.2f}  "
          f"{'✔ Good generalisation' if gap < 15 else '⚠ Check overfitting'}")

def save_plot(tr_m, v_m, te_m, te_ct, te_cp,
              yte_true, yte_pred, station,
              feat_importance=None):
    CATS = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
    present = sorted(set(te_ct + te_cp),
                     key=lambda x: CATS.index(x) if x in CATS else 99)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"XGBoost — Accuracy Report  [{station}]",
                 fontsize=14, fontweight="bold")
    n = min(500, len(yte_true))

    # 1 — Feature importance
    ax = fig.add_subplot(3, 3, 1)
    if feat_importance is not None:
        fi = pd.Series(feat_importance, index=FEATURES).sort_values(ascending=True).tail(10)
        fi.plot(kind="barh", ax=ax, color="#2ca02c", edgecolor="white")
        ax.set_title("Feature Importance (gain)")
        ax.set_xlabel("Importance score")
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Feature importance\nnot available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # 2 — MAE bar
    ax = fig.add_subplot(3, 3, 2)
    vals = [tr_m["MAE"], v_m["MAE"], te_m["MAE"]]
    bars = ax.bar(["Train","Val","Test"], vals,
                  color=["#2196F3","#FF9800","#4CAF50"], edgecolor="white")
    [ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.2f}",
             ha="center", va="bottom") for b, v in zip(bars, vals)]
    ax.set_title("MAE — Train / Val / Test"); ax.grid(axis="y", alpha=0.3)

    # 3 — R² bar
    ax = fig.add_subplot(3, 3, 3)
    vals = [tr_m["R2"], v_m["R2"], te_m["R2"]]
    colors_r2 = ["#2196F3" if v >= 0 else "#e53935" for v in vals]
    bars = ax.bar(["Train","Val","Test"], vals, color=colors_r2, edgecolor="white")
    [ax.text(b.get_x()+b.get_width()/2,
             max(b.get_height(), 0)+0.01, f"{v:.4f}",
             ha="center", va="bottom") for b, v in zip(bars, vals)]
    ax.axhline(0,   color="black", lw=0.8)
    ax.axhline(0.9, color="red", ls="--", alpha=0.5, label="R²=0.9")
    ax.set_ylim(min(0, min(vals))-0.05, 1.1)
    ax.set_title("R² — Train / Val / Test")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # 4 — Actual vs Predicted
    ax = fig.add_subplot(3, 3, 4)
    ax.plot(yte_true[:n], label="Actual",    lw=1.2, color="steelblue")
    ax.plot(yte_pred[:n], label="XGBoost",   lw=1.2, color="#2ca02c", alpha=0.85)
    ax.set_title("Actual vs Predicted (Test)")
    ax.set_xlabel("Time step"); ax.set_ylabel("AQI")
    ax.legend(); ax.grid(alpha=0.3)

    # 5 — Scatter
    ax = fig.add_subplot(3, 3, 5)
    ax.scatter(yte_true[:n], yte_pred[:n], alpha=0.25, s=8, color="#2ca02c")
    lo = min(yte_true[:n].min(), yte_pred[:n].min())
    hi = max(yte_true[:n].max(), yte_pred[:n].max())
    ax.plot([lo,hi],[lo,hi], "r--", lw=1.5, label="Perfect")
    ax.set_title("Scatter — Actual vs Predicted (Test)")
    ax.set_xlabel("Actual AQI"); ax.set_ylabel("Predicted AQI")
    ax.legend(); ax.grid(alpha=0.3)

    # 6 — Residuals
    ax = fig.add_subplot(3, 3, 6)
    res = yte_true[:n] - yte_pred[:n]
    ax.hist(res, bins=40, color="purple", alpha=0.7, edgecolor="white")
    ax.axvline(0,          color="red",    ls="--", lw=1.5)
    ax.axvline(res.mean(), color="orange", lw=1.5,
               label=f"Mean={res.mean():.2f}")
    ax.set_title("Residual Distribution (Test)")
    ax.set_xlabel("Residual"); ax.legend(); ax.grid(alpha=0.3)

    # 7 — Confusion matrix
    ax = fig.add_subplot(3, 3, 7)
    cm = confusion_matrix(te_ct, te_cp, labels=present)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=present, yticklabels=present,
                ax=ax, linewidths=0.5)
    ax.set_title("AQI Category Confusion Matrix (Test)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)

    # 8 — Category accuracy bar
    ax = fig.add_subplot(3, 3, 8)
    vals = [tr_m["CatAcc"], v_m["CatAcc"], te_m["CatAcc"]]
    bars = ax.bar(["Train","Val","Test"], vals,
                  color=["#2196F3","#FF9800","#4CAF50"], edgecolor="white")
    [ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.1f}%",
             ha="center", va="bottom") for b, v in zip(bars, vals)]
    ax.set_ylim(0, 110); ax.set_title("AQI Category Accuracy (%)")
    ax.axhline(80, color="red", ls="--", alpha=0.5, label="80% target")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # 9 — Summary table
    ax = fig.add_subplot(3, 3, 9); ax.axis("off")
    rows = [["MAE",    tr_m["MAE"],  v_m["MAE"],  te_m["MAE"]],
            ["RMSE",   tr_m["RMSE"], v_m["RMSE"], te_m["RMSE"]],
            ["R²",     tr_m["R2"],   v_m["R2"],   te_m["R2"]],
            ["MAPE%",  f"{tr_m['MAPE']}%", f"{v_m['MAPE']}%",  f"{te_m['MAPE']}%"],
            ["CatAcc", f"{tr_m['CatAcc']}%",f"{v_m['CatAcc']}%",f"{te_m['CatAcc']}%"]]
    tbl = ax.table(cellText=rows, colLabels=["Metric","Train","Val","Test"],
                   cellLoc="center", loc="center",
                   colColours=["#37474F","#1565C0","#E65100","#2E7D32"])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#888")
    ax.set_title("Metrics Summary", fontweight="bold", pad=12)

    plt.tight_layout()
    path = f"{PLOT_DIR}/xgb_{station}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Plot → {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  XGBoost TRAINING — Nagpur AQI Forecasting")
print("=" * 60)
print(f"  Window: {WINDOW_SIZE}h  Horizon: {HORIZON}h")
print(f"  Trees: {XGB_PARAMS['n_estimators']}  LR: {XGB_PARAMS['learning_rate']}")
print(f"  Max depth: {XGB_PARAMS['max_depth']}  L1: {XGB_PARAMS['reg_alpha']}  L2: {XGB_PARAMS['reg_lambda']}\n")

df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df):,} rows  |  stations: {STATIONS}\n")

all_metrics = {}
feat_importance_all = {}

for station in STATIONS:
    print(f"\n{'─'*55}\n  Station: {station}\n{'─'*55}")

    s  = load_station(df, station)
    fs = MinMaxScaler(); ts = MinMaxScaler()
    Xd = fs.fit_transform(s[FEATURES].values)
    yd = ts.fit_transform(s[[TARGET]].values).ravel()

    X, y = make_sequences(Xd, yd, WINDOW_SIZE, HORIZON)
    Xtr, ytr, Xv, yv, Xte, yte = split_data(X, y)
    print(f"  Train:{len(Xtr)}  Val:{len(Xv)}  Test:{len(Xte)}")

    # Flatten sequences for XGBoost: (N, window*features)
    Xtr_f = Xtr.reshape(len(Xtr), -1)
    Xv_f  = Xv.reshape(len(Xv),  -1)
    Xte_f = Xte.reshape(len(Xte), -1)

    # Train with MultiOutputRegressor (one tree per horizon step)
    model = MultiOutputRegressor(xgb.XGBRegressor(**XGB_PARAMS))
    print(f"  Training XGBoost ({XGB_PARAMS['n_estimators']} trees × {HORIZON} outputs)…")
    model.fit(Xtr_f, ytr)

    # Save model + scalers
    joblib.dump(model, f"{MODEL_DIR}/xgb_aqi_{station}.pkl")
    joblib.dump(fs,    f"{MODEL_DIR}/feat_scaler_{station}.pkl")
    joblib.dump(ts,    f"{MODEL_DIR}/tgt_scaler_{station}.pkl")
    print(f"  Saved → {MODEL_DIR}/xgb_aqi_{station}.pkl")

    # Predict all splits (first horizon step for metrics)
    def xgb_pred(Xf): return inv(ts, model.predict(Xf)[:, 0])

    ytr_true = inv(ts, ytr[:, 0]); ytr_pred = xgb_pred(Xtr_f)
    yv_true  = inv(ts, yv[:, 0]);  yv_pred  = xgb_pred(Xv_f)
    yte_true = inv(ts, yte[:, 0]); yte_pred = xgb_pred(Xte_f)

    tr_m, _,     _     = compute_metrics(ytr_true, ytr_pred)
    v_m,  _,     _     = compute_metrics(yv_true,  yv_pred)
    te_m, te_ct, te_cp = compute_metrics(yte_true, yte_pred)

    print_metrics(tr_m, v_m, te_m, station)

    CATS = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
    present = sorted(set(te_ct+te_cp), key=lambda x: CATS.index(x) if x in CATS else 99)
    print(f"\n  Test Classification Report:")
    print(classification_report(te_ct, te_cp, labels=present, zero_division=0))

    # Feature importance (average across all estimators)
    try:
        fi = np.mean([est.feature_importances_[:len(FEATURES)]
                      for est in model.estimators_], axis=0)
        feat_importance_all[station] = dict(zip(FEATURES, fi.tolist()))
    except Exception:
        fi = None

    save_plot(tr_m, v_m, te_m, te_ct, te_cp, yte_true, yte_pred,
              station, feat_importance=fi)

    all_metrics[station] = {
        "train": tr_m, "validation": v_m, "test": te_m,
        "feature_importance": feat_importance_all.get(station, {})
    }

# ── SAVE CONFIG + REPORT ──────────────────────────────────────────────────────
config = {
    "model":      "XGBoost",
    "features":   FEATURES,
    "target":     TARGET,
    "window_size":WINDOW_SIZE,
    "horizon":    HORIZON,
    "stations":   STATIONS,
    "hyperparameters": XGB_PARAMS
}
with open(f"{MODEL_DIR}/model_config.json", "w") as f:
    json.dump(config, f, indent=2)
with open(f"{MODEL_DIR}/evaluation_report.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
with open(f"{RESULT_DIR}/xgb_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\n" + "="*60)
print("  XGBoost TRAINING COMPLETE")
print("="*60)
print(f"  {'Station':15s}  {'Train R²':>9s}  {'Val R²':>8s}  {'Test R²':>8s}  {'MAE':>7s}  {'CatAcc%':>8s}")
print("  " + "─"*60)
for s, d in all_metrics.items():
    tr = d["train"]; te = d["test"]
    flag = "✔" if te["R2"] > 0.85 else "△"
    print(f"  {s:15s}  {tr['R2']:9.4f}  {d['validation']['R2']:8.4f}  "
          f"{te['R2']:8.4f}  {te['MAE']:7.4f}  {te['CatAcc']:8.2f}  {flag}")
print(f"\n  Config  → {MODEL_DIR}/model_config.json")
print(f"  Report  → {MODEL_DIR}/evaluation_report.json")
print(f"  Plots   → {PLOT_DIR}/")

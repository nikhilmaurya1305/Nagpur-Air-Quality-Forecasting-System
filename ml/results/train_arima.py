"""
train_arima.py
==============
SARIMA (Seasonal ARIMA) – AQI Forecasting for Nagpur City
Research baseline model vs deep learning approaches

Strategy:
    - Per-station, per-pollutant AQI time series
    - Auto-selects best (p,d,q)(P,D,Q,s) order via AIC grid search
      OR uses auto_arima from pmdarima (faster, recommended)
    - One-step ahead rolling forecast on test set (standard ARIMA eval)
    - Also provides 24-step ahead static forecast from last training point
    - Seasonal period s=24 (daily cycle in hourly AQI data)

Install:
    pip install pmdarima statsmodels

Output files:
    models/arima_{station}.pkl
    results/arima_metrics.json
    plots/arima_{station}_pred.png
"""

import os, json, warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import pmdarima as pm
    USE_AUTOARIMA = True
    print("OK: pmdarima found - using auto_arima for order selection")
except ImportError:
    USE_AUTOARIMA = False
    print("WARNING: pmdarima not found - using manual SARIMA(2,1,2)(1,1,1,24)")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

# ── CONFIG ────────────────────────────────────────────────────────────────
DATA_PATH   = "data/nagpur_final_preprocessed.csv"
MODEL_DIR   = "ml/models"
PLOT_DIR    = "ml/plots"
RESULT_DIR  = "ml/results"
TARGET      = "AQI"
STATIONS    = ["Ambazari","Mahal","Civil_Lines","Ram_Nagar"]
SEASONAL_S  = 24       # 24 hours = daily seasonality
TEST_RATIO  = 0.005
HORIZON     = 24       # static forecast steps ahead

# For manual SARIMA (used if pmdarima not installed)
MANUAL_ORDER         = (2, 1, 2)
MANUAL_SEASONAL_ORDER= (1, 1, 1, SEASONAL_S)

for d in [MODEL_DIR, PLOT_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── HELPERS ───────────────────────────────────────────────────────────────
def load_station_aqi(df, station):
    """Return clean hourly AQI series for a station."""
    s = df[df["Station"] == station][["Datetime", TARGET]].copy()
    s["Datetime"] = pd.to_datetime(s["Datetime"])
    s = s.sort_values("Datetime").set_index("Datetime")
    idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(idx)
    # Interpolate short gaps; drop remaining NaN
    s[TARGET] = s[TARGET].interpolate(method="time", limit=6)
    s = s.dropna()
    return s[TARGET]

def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {"MAE": round(mae,4), "RMSE": round(rmse,4),
            "R2":  round(r2,4),  "MAPE": round(mape,4)}

def rolling_forecast_arima(train_vals, test_vals, order, seasonal_order=None):
    """
    Walk-forward (rolling) one-step-ahead forecast.
    Standard evaluation protocol for ARIMA in time-series research.
    Each step: refit on history, predict 1 step, add actual to history.
    Uses a faster approximate refit (append + update) for speed.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    history = list(train_vals)
    preds   = []
    n_test  = len(test_vals)

    print(f"  Rolling forecast: {n_test} steps", end="", flush=True)
    dot_every = max(n_test // 20, 1)

    for i, actual in enumerate(test_vals):
        try:
            if seasonal_order:
                model = SARIMAX(history,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            else:
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(history, order=order)

            res   = model.fit(disp=False)
            yhat  = res.forecast(steps=1)[0]
        except Exception:
            yhat = history[-1]    # fallback: persist last value

        preds.append(yhat)
        history.append(actual)

        if i % dot_every == 0:
            print(".", end="", flush=True)

    print(" done")
    return np.array(preds)

def fit_auto_arima(train_vals):
    """Use pmdarima auto_arima to select best SARIMA order."""
    model = pm.auto_arima(
        train_vals,
        start_p=1, start_q=1,
        max_p=4,   max_q=4,
        d=None,            # auto-select differencing
        seasonal=True,
        m=SEASONAL_S,
        start_P=0, start_Q=0,
        max_P=2,   max_Q=2,
        D=1,
        information_criterion="aic",
        stepwise=True,     # fast stepwise search
        suppress_warnings=True,
        error_action="ignore",
        n_jobs=1
    )
    print(f"  Best order: ARIMA{model.order} x SARIMA{model.seasonal_order}")
    return model

def plot_results(train, test, pred_roll, pred_static,
                 station, horizon):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ── Plot 1: rolling one-step forecast on test set
    n = min(len(pred_roll), 500)
    axes[0].plot(test.values[:n],   label="Actual AQI",  lw=1.5)
    axes[0].plot(pred_roll[:n],     label="SARIMA 1-step forecast",
                 lw=1.5, alpha=0.85, color="tomato")
    axes[0].set_title(f"SARIMA — Rolling 1-Step Forecast  [{station}]")
    axes[0].set_xlabel("Time step"); axes[0].set_ylabel("AQI")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # ── Plot 2: 24-step ahead static forecast
    tail  = 72   # show last 72 h of train + forecast
    hist_tail = train.values[-tail:]
    x_hist = range(tail)
    x_fc   = range(tail, tail + horizon)

    axes[1].plot(x_hist, hist_tail, label="Historical AQI", lw=1.5)
    axes[1].plot(x_fc,  pred_static[:horizon], label=f"{horizon}-h Forecast",
                 lw=2, color="tomato", marker="o", markersize=3)
    axes[1].axvline(tail, color="grey", linestyle="--", alpha=0.5)
    axes[1].set_title(f"SARIMA — {horizon}-Hour Static Forecast  [{station}]")
    axes[1].set_xlabel("Hour offset"); axes[1].set_ylabel("AQI")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/arima_{station}_pred.png", dpi=120)
    plt.close()
    print(f"  Plot saved -> {PLOT_DIR}/arima_{station}_pred.png")

# ── MAIN ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  SARIMA TRAINING — Nagpur AQI Forecasting")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
all_metrics = {}

for station in STATIONS:
    print(f"\n{'-'*55}")
    print(f"  Station: {station}")
    print(f"{'-'*55}")

    series = load_station_aqi(df, station)
    print(f"  Series length: {len(series):,}")

    n_test = int(len(series) * TEST_RATIO)
    train  = series.iloc[-1000-n_test:-n_test]
    test   = series.iloc[-n_test:]
    print(f"  Train: {len(train):,}  Test: {len(test):,}")

    # ── Fit model ──────────────────────────────────────────────────
    if USE_AUTOARIMA:
        # Fit on training data
        fitted = fit_auto_arima(train.values)
        best_order    = fitted.order
        best_seasonal = fitted.seasonal_order

        # Rolling forecast using statsmodels (pmdarima predict is slow for rolling)
        pred_rolling = rolling_forecast_arima(
            train.values, test.values,
            best_order, best_seasonal
        )

        # Static 24-h ahead forecast
        pred_static = fitted.predict(n_periods=HORIZON)

        # Save model
        with open(f"{MODEL_DIR}/arima_{station}.pkl", "wb") as f:
            pickle.dump(fitted, f)

    else:
        # Manual SARIMA without pmdarima
        best_order    = MANUAL_ORDER
        best_seasonal = MANUAL_SEASONAL_ORDER
        print(f"  Using manual order: ARIMA{best_order} × {best_seasonal}")

        pred_rolling = rolling_forecast_arima(
            train.values, test.values,
            best_order, best_seasonal
        )

        # Fit on full train for static forecast
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        final = SARIMAX(train.values,
                        order=best_order,
                        seasonal_order=best_seasonal,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
        pred_static = final.forecast(steps=HORIZON)

        with open(f"{MODEL_DIR}/arima_{station}.pkl", "wb") as f:
            pickle.dump(final, f)

    # Clip negative forecasts (AQI ≥ 0)
    pred_rolling = np.clip(pred_rolling, 0, 600)
    pred_static  = np.clip(pred_static,  0, 600)

    m = compute_metrics(test.values, pred_rolling)
    all_metrics[station] = {**m,
        "order": str(best_order),
        "seasonal_order": str(best_seasonal)}

    print(f"\n  OK MAE={m['MAE']}  RMSE={m['RMSE']}"
          f"  R2={m['R2']}  MAPE={m['MAPE']}%")

    plot_results(train, test, pred_rolling, pred_static,
                 station, HORIZON)

with open(f"{RESULT_DIR}/arima_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\n" + "="*60)
print("  SARIMA TRAINING COMPLETE")
print("="*60)
for s, m in all_metrics.items():
    print(f"  {s:15s}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}"
          f"  R²={m['R2']:.4f}  MAPE={m['MAPE']:.2f}%")
print(f"\n  Metrics -> {RESULT_DIR}/arima_metrics.json")
print("\nNote: SARIMA rolling forecast is slow (~1–3 hrs per station).")
print("For faster results: pip install pmdarima  and use stepwise=True.")

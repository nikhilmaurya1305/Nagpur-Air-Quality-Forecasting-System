import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")

DATA_PATH = "data/nagpur_final_preprocessed.csv"
PLOT_DIR = "ml/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

WINDOW_SIZE = 24
HORIZON = 24
FEATURES = ["PM2.5","PM10","NO","NO2","SO2","NH3","Hour_sin","Hour_cos","Month_sin","Month_cos","DOW_sin","DOW_cos","IsWeekend"]
TARGET = "AQI"
STATIONS = ["Ambazari","Mahal","Civil_Lines","Ram_Nagar"]

XGB_PARAMS = dict(
    n_estimators=400, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0
)

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

def prep_station(df, station):
    s = load_station(df, station)
    n = len(s)
    nt, nv = int(n * 0.15), int(n * 0.15)
    train_raw = s.iloc[:n-nt-nv]
    fs = MinMaxScaler().fit(train_raw[FEATURES].values)
    ts = MinMaxScaler().fit(train_raw[[TARGET]].values)
    Xd = fs.transform(s[FEATURES].values)
    yd = ts.transform(s[[TARGET]].values).ravel()
    X, y = make_sequences(Xd, yd, WINDOW_SIZE, HORIZON)
    Xtr, ytr = X[:n-nt-nv], y[:n-nt-nv]
    Xte, yte = X[n-nt:], y[n-nt:]
    return Xtr, ytr, Xte, yte, ts

def plot_pred(y_true, y_pred, station, path):
    plt.figure(figsize=(15, 6))
    n_show = 400
    plt.plot(y_true[:n_show], label="Actual AQI", color="#2c3e50", lw=2)
    plt.plot(y_pred[:n_show], label="XGBoost Predicted", color="#27ae60", alpha=0.85, lw=2)
    
    r2 = r2_score(y_true, y_pred)
    plt.title(f"XGBoost AQI Prediction — {station} (Test Set R² = {r2:.4f})", fontsize=14, weight='bold')
    plt.xlabel("Time Step (Hours)", fontsize=12)
    plt.ylabel("AQI", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Generated: {path} | R2: {r2:.4f}")

df = pd.read_csv(DATA_PATH)

for station in STATIONS:
    print(f"Processing {station}...")
    Xtr, ytr, Xte, yte, ts = prep_station(df, station)
    
    # XGBoost requires flattened input
    Xtr_f = Xtr.reshape(len(Xtr), -1)
    Xte_f = Xte.reshape(len(Xte), -1)
    
    model = MultiOutputRegressor(xgb.XGBRegressor(**XGB_PARAMS))
    model.fit(Xtr_f, ytr)
    
    y_pred_scaled = model.predict(Xte_f)
    yte_true = ts.inverse_transform(yte[:,0].reshape(-1,1)).ravel()
    yte_pred = ts.inverse_transform(y_pred_scaled[:,0].reshape(-1,1)).ravel()
    
    plot_pred(yte_true, yte_pred, station, f"{PLOT_DIR}/xgb_{station}.png")

print("\nAll XGBoost plots updated successfully.")

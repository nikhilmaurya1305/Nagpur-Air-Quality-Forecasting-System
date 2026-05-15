import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, TimeDistributed
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATA_PATH = "data/nagpur_final_preprocessed.csv"
PLOT_DIR = "ml/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

WINDOW_SIZE = 24
HORIZON = 24
FEATURES = ["PM2.5","PM10","NO","NO2","SO2","NH3","Hour_sin","Hour_cos","Month_sin","Month_cos","DOW_sin","DOW_cos","IsWeekend"]
TARGET = "AQI"
STATIONS = ["Ambazari","Mahal","Civil_Lines","Ram_Nagar"]

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
    Xv, yv = X[n-nt-nv:n-nt], y[n-nt-nv:n-nt]
    Xte, yte = X[n-nt:], y[n-nt:]
    return Xtr, ytr, Xv, yv, Xte, yte, ts

def plot_pred(y_true, y_pred, title, path):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:300], label="Actual", color="#2c3e50", lw=1.5)
    plt.plot(y_pred[:300], label="Predicted", color="#e74c3c", alpha=0.8, lw=1.5)
    r2 = r2_score(y_true, y_pred)
    plt.title(f"{title} (R² = {r2:.4f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

df = pd.read_csv(DATA_PATH)

for station in STATIONS:
    print(f"Updating plots for {station}...")
    Xtr, ytr, Xv, yv, Xte, yte, ts = prep_station(df, station)
    yte_true = ts.inverse_transform(yte[:,0].reshape(-1,1)).ravel()

    # 1. XGBoost
    model_xgb = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.03, random_state=42, n_jobs=-1))
    model_xgb.fit(Xtr.reshape(len(Xtr), -1), ytr)
    y_pred_xgb = ts.inverse_transform(model_xgb.predict(Xte.reshape(len(Xte), -1))[:,0].reshape(-1,1)).ravel()
    plot_pred(yte_true, y_pred_xgb, f"XGBoost — {station}", f"{PLOT_DIR}/xgb_{station}.png")

    # 2. GRU
    model_gru = Sequential([
        GRU(32, return_sequences=True, input_shape=(WINDOW_SIZE, len(FEATURES))),
        BatchNormalization(), Dropout(0.2),
        GRU(16), BatchNormalization(), Dropout(0.2),
        Dense(16, activation="relu"), Dense(HORIZON)
    ])
    model_gru.compile(optimizer=Adam(1e-3), loss="huber")
    model_gru.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=20, batch_size=64, verbose=0)
    y_pred_gru = ts.inverse_transform(model_gru.predict(Xte, verbose=0)[:,0].reshape(-1,1)).ravel()
    plot_pred(yte_true, y_pred_gru, f"GRU — {station}", f"{PLOT_DIR}/gru_{station}_pred.png")

    # 3. CNN-LSTM
    N_STEPS, N_LENGTH = 4, 6
    Xtr_r = Xtr.reshape(Xtr.shape[0], N_STEPS, N_LENGTH, len(FEATURES))
    Xte_r = Xte.reshape(Xte.shape[0], N_STEPS, N_LENGTH, len(FEATURES))
    Xv_r = Xv.reshape(Xv.shape[0], N_STEPS, N_LENGTH, len(FEATURES))
    model_cl = Sequential([
        TimeDistributed(Conv1D(32, kernel_size=3, activation="relu", padding="same"), input_shape=(N_STEPS, N_LENGTH, len(FEATURES))),
        TimeDistributed(MaxPooling1D(pool_size=2)), TimeDistributed(Flatten()),
        LSTM(32), Dropout(0.2), Dense(16, activation="relu"), Dense(HORIZON)
    ])
    model_cl.compile(optimizer=Adam(1e-3), loss="huber")
    model_cl.fit(Xtr_r, ytr, validation_data=(Xv_r, yv), epochs=20, batch_size=64, verbose=0)
    y_pred_cl = ts.inverse_transform(model_cl.predict(Xte_r, verbose=0)[:,0].reshape(-1,1)).ravel()
    plot_pred(yte_true, y_pred_cl, f"CNN-LSTM — {station}", f"{PLOT_DIR}/cnnlstm_{station}_pred.png")

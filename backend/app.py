"""
app.py — Flask Backend for Nagpur AQI Dashboard (XGBoost)
==========================================================
XGBoost is the primary forecasting model (R² = 0.88–0.91).

Run:
    pip install flask flask-cors xgboost scikit-learn pandas joblib numpy
    python app.py

API Endpoints:
    GET  /api/stations                → list of 4 stations
    GET  /api/current                 → latest AQI + pollutants per station
    GET  /api/forecast/<station>      → 24-h XGBoost AQI forecast
    GET  /api/history/<station>       → last 7 days hourly AQI
    GET  /api/aqi_trend               → 30-day city-wide daily AQI
    GET  /api/pollutants/<station>    → latest pollutant breakdown vs NAAQS
    GET  /api/health_advisory         → group-specific health advisories
    GET  /api/model_metrics           → train/val/test metrics per station
    GET  /api/feature_importance/<station> → XGBoost feature importance
    POST /api/predict                 → custom AQI prediction from user input
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

from flask import Flask, jsonify, request
from flask_cors import CORS

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "../data/nagpur_final_preprocessed.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "../ml/models")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# ── SERVE FRONTEND ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")

# ── LOAD CONFIG ───────────────────────────────────────────────────────────────
print("Loading model config …")
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

FEATURES    = CFG["features"]
TARGET      = CFG["target"]
WINDOW_SIZE = CFG["window_size"]
HORIZON     = CFG["horizon"]
STATIONS    = CFG["stations"]

# ── LOAD DATASET ──────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

# ── LOAD XGBoost MODELS ───────────────────────────────────────────────────────
print("Loading XGBoost models …")
MODELS       = {}
FEAT_SCALERS = {}
TGT_SCALERS  = {}

for station in STATIONS:
    mpath = os.path.join(MODEL_DIR, f"xgb_aqi_{station}.pkl")
    fpath = os.path.join(MODEL_DIR, f"feat_scaler_{station}.pkl")
    tpath = os.path.join(MODEL_DIR, f"tgt_scaler_{station}.pkl")
    if os.path.exists(mpath):
        MODELS[station]       = joblib.load(mpath)
        FEAT_SCALERS[station] = joblib.load(fpath)
        TGT_SCALERS[station]  = joblib.load(tpath)
        print(f"  OK {station}")
    else:
        print(f"  FAIL {station} — run train_xgboost.py first")

# ── AQI HELPERS ───────────────────────────────────────────────────────────────
AQI_BANDS = [
    (0,   50,  "Good",         "#00e400",
     "Air quality is satisfactory. Outdoor activities are safe."),
    (51,  100, "Satisfactory", "#d4d700",
     "Minor discomfort for sensitive people. Most can be outdoors."),
    (101, 200, "Moderate",     "#ff7e00",
     "People with respiratory issues may experience discomfort."),
    (201, 300, "Poor",         "#ff0000",
     "Everyone may experience health effects. Limit outdoor activity."),
    (301, 400, "Very Poor",    "#99004c",
     "Health alert. Everyone may experience more serious effects."),
    (401, 500, "Severe",       "#7e0023",
     "Health emergency conditions. Stay indoors."),
]

def aqi_info(aqi):
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return {"category":"Unknown","color":"#888","advisory":"Data unavailable"}
    aqi = float(aqi)
    for lo, hi, cat, color, adv in AQI_BANDS:
        if lo <= aqi <= hi:
            return {"category":cat, "color":color, "advisory":adv}
    return {"category":"Severe","color":"#7e0023",
            "advisory":"Hazardous. Avoid all outdoor activity."}

def get_station_df(station):
    return df[df["Station"] == station].copy().sort_values("Datetime")

# ── XGBoost FORECAST ──────────────────────────────────────────────────────────
def predict_xgb(station):
    """
    Build the latest WINDOW_SIZE-hour feature window,
    flatten for XGBoost, and return 24 predicted AQI values.
    """
    if station not in MODELS:
        return None

    sdf = get_station_df(station).dropna(subset=FEATURES).tail(WINDOW_SIZE)
    if len(sdf) < WINDOW_SIZE:
        return None

    # Scale features → flatten → predict
    X_scaled = FEAT_SCALERS[station].transform(sdf[FEATURES].values)
    X_flat   = X_scaled.reshape(1, -1)                         # (1, window*features)
    y_scaled = MODELS[station].predict(X_flat)                 # (1, horizon)
    y_aqi    = TGT_SCALERS[station].inverse_transform(
                   y_scaled.reshape(-1, 1)).ravel()            # (horizon,)
    return np.clip(y_aqi, 0, 500).tolist()

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/api/stations")
def get_stations():
    return jsonify(STATIONS)


@app.route("/api/current")
def current_aqi():
    result = {}
    for station in STATIONS:
        sdf = get_station_df(station).dropna(subset=[TARGET])
        if sdf.empty:
            continue
        row  = sdf.iloc[-1]
        aqi  = float(row[TARGET])
        info = aqi_info(aqi)
        result[station] = {
            "aqi":      round(aqi, 1),
            "category": info["category"],
            "color":    info["color"],
            "advisory": info["advisory"],
            "datetime": str(row["Datetime"]),
            "PM2.5":    round(float(row.get("PM2.5", 0) or 0), 2),
            "PM10":     round(float(row.get("PM10",  0) or 0), 2),
            "NO2":      round(float(row.get("NO2",   0) or 0), 2),
            "SO2":      round(float(row.get("SO2",   0) or 0), 2),
            "NH3":      round(float(row.get("NH3",   0) or 0), 2),
        }
    return jsonify(result)


@app.route("/api/forecast/<station>")
def forecast(station):
    if station not in STATIONS:
        return jsonify({"error": "Unknown station"}), 404

    preds = predict_xgb(station)
    if preds is None:
        return jsonify({"error": "Model unavailable — run train_xgboost.py"}), 503

    now   = datetime.now()
    times = [(now + timedelta(hours=i+1)).strftime("%Y-%m-%d %H:00")
             for i in range(HORIZON)]

    data = [{"datetime": times[i],
             "aqi":      round(preds[i], 1),
             **aqi_info(preds[i])}
            for i in range(HORIZON)]

    # Peak forecast info
    peak_aqi  = max(preds)
    peak_hour = preds.index(max(preds)) + 1

    return jsonify({
        "station":    station,
        "model":      "XGBoost",
        "forecast":   data,
        "peak_aqi":   round(peak_aqi, 1),
        "peak_hour":  peak_hour,
        "peak_info":  aqi_info(peak_aqi)
    })


@app.route("/api/history/<station>")
def history(station):
    if station not in STATIONS:
        return jsonify({"error": "Unknown station"}), 404

    sdf    = get_station_df(station).dropna(subset=[TARGET])
    cutoff = sdf["Datetime"].max() - pd.Timedelta(days=7)
    sdf    = sdf[sdf["Datetime"] >= cutoff]

    data = [{"datetime": str(row["Datetime"]),
             "aqi":      round(float(row[TARGET]), 1),
             **aqi_info(float(row[TARGET]))}
            for _, row in sdf.iterrows()]
    return jsonify({"station": station, "history": data})


@app.route("/api/aqi_trend")
def aqi_trend():
    cutoff = df["Datetime"].max() - pd.Timedelta(days=30)
    recent = df[df["Datetime"] >= cutoff].copy()
    daily  = recent.groupby(recent["Datetime"].dt.date)[TARGET].mean().dropna()
    data   = [{"date": str(d), "aqi": round(float(v), 1), **aqi_info(float(v))}
              for d, v in daily.items()]
    return jsonify(data)


@app.route("/api/pollutants/<station>")
def pollutants(station):
    if station not in STATIONS:
        return jsonify({"error": "Unknown station"}), 404

    sdf = get_station_df(station).dropna(subset=["PM2.5","PM10","NO2","SO2","NH3"])
    if sdf.empty:
        return jsonify({})
    row = sdf.iloc[-1]

    # NAAQS 24-h limits (µg/m³)
    poll = {
        "PM2.5": {"value": round(float(row["PM2.5"]),2), "unit":"µg/m³", "limit":60,  "safe":60},
        "PM10":  {"value": round(float(row["PM10"]), 2), "unit":"µg/m³", "limit":100, "safe":100},
        "NO2":   {"value": round(float(row["NO2"]),  2), "unit":"µg/m³", "limit":80,  "safe":80},
        "SO2":   {"value": round(float(row["SO2"]),  2), "unit":"µg/m³", "limit":80,  "safe":80},
        "NH3":   {"value": round(float(row["NH3"]),  2), "unit":"µg/m³", "limit":400, "safe":200},
    }
    return jsonify({"station":station, "datetime":str(row["Datetime"]),
                    "pollutants":poll})


@app.route("/api/health_advisory")
def health_advisory():
    advisories = {}
    for station in STATIONS:
        sdf = get_station_df(station).dropna(subset=[TARGET])
        if sdf.empty:
            continue
        aqi  = float(sdf.iloc[-1][TARGET])
        info = aqi_info(aqi)

        if aqi <= 50:
            groups = {
                "General":  "No restrictions. Air quality is satisfactory.",
                "Children": "Safe for outdoor play and sports.",
                "Elderly":  "No special precautions needed.",
                "Athletes": "Exercise freely outdoors."
            }
        elif aqi <= 100:
            groups = {
                "General":  "Sensitive individuals should limit prolonged outdoor exertion.",
                "Children": "Reduce prolonged outdoor play if feeling discomfort.",
                "Elderly":  "Monitor for symptoms such as coughing or shortness of breath.",
                "Athletes": "Sensitive people should reduce intense prolonged exercise."
            }
        elif aqi <= 200:
            groups = {
                "General":  "Reduce prolonged or heavy outdoor exertion. Take more breaks.",
                "Children": "Avoid prolonged outdoor activities. Keep windows closed.",
                "Elderly":  "Stay indoors if possible. Avoid busy roads.",
                "Athletes": "Reduce intensity and duration. Move workouts indoors."
            }
        else:
            groups = {
                "General":  "Avoid all outdoor activity. Stay indoors with windows closed.",
                "Children": "No outdoor activities. Keep children indoors.",
                "Elderly":  "Remain indoors. Seek medical attention if symptoms occur.",
                "Athletes": "Cancel outdoor training. Indoor exercise only."
            }

        advisories[station] = {
            "aqi":      round(aqi, 1),
            "category": info["category"],
            "color":    info["color"],
            "general":  info["advisory"],
            "groups":   groups
        }
    return jsonify(advisories)


@app.route("/api/model_metrics")
def model_metrics():
    rpath = os.path.join(MODEL_DIR, "evaluation_report.json")
    if not os.path.exists(rpath):
        return jsonify({"error": "Run train_xgboost.py first"}), 404
    with open(rpath) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/feature_importance/<station>")
def feature_importance(station):
    """Return XGBoost feature importance for a station."""
    if station not in STATIONS:
        return jsonify({"error": "Unknown station"}), 404

    rpath = os.path.join(MODEL_DIR, "evaluation_report.json")
    if not os.path.exists(rpath):
        return jsonify({"error": "No report found"}), 404

    with open(rpath) as f:
        report = json.load(f)

    fi = report.get(station, {}).get("feature_importance", {})
    if not fi:
        return jsonify({"error": "Feature importance not available"}), 404

    # Sort descending
    sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    return jsonify({"station": station, "importance": sorted_fi})


@app.route("/api/predict", methods=["POST"])
def predict_custom():
    """
    Custom AQI prediction from user-supplied pollutant values.

    POST body (JSON):
    {
        "station": "Ambazari",
        "PM2.5": 85, "PM10": 130,
        "NO": 12, "NO2": 38, "SO2": 15, "NH3": 22,
        "hour": 14, "month": 6, "dow": 2
    }
    Returns predicted AQI for t+1 through t+24.
    """
    body    = request.get_json(force=True)
    station = body.get("station", STATIONS[0])

    if station not in MODELS:
        return jsonify({"error": f"Model for {station} not found"}), 404

    hour  = int(body.get("hour",  datetime.now().hour))
    month = int(body.get("month", datetime.now().month))
    dow   = int(body.get("dow",   datetime.now().weekday()))

    single_row = {
        "PM2.5":     float(body.get("PM2.5", 60)),
        "PM10":      float(body.get("PM10",  90)),
        "NO":        float(body.get("NO",    10)),
        "NO2":       float(body.get("NO2",   30)),
        "SO2":       float(body.get("SO2",   12)),
        "NH3":       float(body.get("NH3",   20)),
        "Hour_sin":  np.sin(2*np.pi*hour/24),
        "Hour_cos":  np.cos(2*np.pi*hour/24),
        "Month_sin": np.sin(2*np.pi*(month-1)/12),
        "Month_cos": np.cos(2*np.pi*(month-1)/12),
        "DOW_sin":   np.sin(2*np.pi*dow/7),
        "DOW_cos":   np.cos(2*np.pi*dow/7),
        "IsWeekend": 1 if dow >= 5 else 0
    }

    # Replicate into WINDOW_SIZE rows to fill the look-back window
    feat_matrix = np.tile([single_row[f] for f in FEATURES], (WINDOW_SIZE, 1))
    X_scaled    = FEAT_SCALERS[station].transform(feat_matrix)
    X_flat      = X_scaled.reshape(1, -1)

    y_scaled = MODELS[station].predict(X_flat)
    y_aqi    = TGT_SCALERS[station].inverse_transform(
                   y_scaled.reshape(-1, 1)).ravel()
    y_aqi    = np.clip(y_aqi, 0, 500)

    # Return t+1 prediction as primary + full 24-h forecast
    aqi_t1   = float(y_aqi[0])
    info     = aqi_info(aqi_t1)
    now      = datetime.now()
    forecast = [{"hour": i+1,
                 "datetime": (now+timedelta(hours=i+1)).strftime("%H:00"),
                 "aqi": round(float(y_aqi[i]),1),
                 **aqi_info(float(y_aqi[i]))}
                for i in range(HORIZON)]

    return jsonify({
        "station":       station,
        "model":         "XGBoost",
        "predicted_aqi": round(aqi_t1, 1),
        "category":      info["category"],
        "color":         info["color"],
        "advisory":      info["advisory"],
        "forecast_24h":  forecast
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

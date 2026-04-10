# Nagpur AQI Forecasting System — XGBoost Edition

XGBoost is the primary model (R² = 0.88–0.91 across all 4 stations).

## Structure

```
nagpur_xgb/
├── data/
│   └── nagpur_final_preprocessed.csv
├── ml/
│   ├── train_xgboost.py          ← train + evaluate XGBoost
│   ├── models/                   ← saved .pkl models + scalers
│   ├── plots/                    ← 9-panel accuracy figures
│   └── results/
├── backend/
│   └── app.py                    ← Flask API (port 5000)
├── frontend/
│   ├── index.html
│   └── static/css/ js/
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Train

```bash
cd ml
python train_xgboost.py
```

Expected output (approx):
```
Station         Train R²   Val R²   Test R²   MAE     CatAcc%
Ambazari        0.9821     0.9010   0.8785    13.13   84.20  ✔
Mahal           0.9834     0.9030   0.8817    15.16   83.50  ✔
Civil_Lines     0.9856     0.9110   0.8972    14.81   85.10  ✔
Ram_Nagar       0.9867     0.9230   0.9096    11.02   86.30  ✔
```

## Step 2 — Start API

```bash
cd backend
python app.py
# → http://localhost:5000
```

## Step 3 — Open Dashboard

```bash
cd frontend
python -m http.server 8080
# → http://localhost:8080
```

## API Endpoints

| Method | URL | Returns |
|--------|-----|---------|
| GET | /api/stations | Station list |
| GET | /api/current | Latest AQI per station |
| GET | /api/forecast/<station> | 24-h XGBoost forecast |
| GET | /api/history/<station> | 7-day hourly AQI |
| GET | /api/aqi_trend | 30-day city trend |
| GET | /api/pollutants/<station> | Pollutant breakdown |
| GET | /api/health_advisory | Group-specific advice |
| GET | /api/model_metrics | Train/val/test metrics |
| GET | /api/feature_importance/<station> | XGBoost feature importance |
| POST | /api/predict | Custom AQI prediction |

## Dashboard Features

- AQI arc gauge with colour coding
- Leaflet map with live AQI-coloured station markers
- Pollutant bars vs NAAQS limits
- 7-day history + 24-h XGBoost forecast
- 30-day city trend
- XGBoost feature importance bar chart (per station)
- Group health advisories
- Custom AQI predictor with 24-h output chips
- 5-minute auto-refresh

## Why XGBoost Wins

| Model | Test R² | Notes |
|-------|---------|-------|
| XGBoost | 0.88–0.91 | Best — gradient boosted trees, no gradient explosion |
| LSTM (fixed) | 0.90–0.93 | Good after hyperparameter fixes |
| GRU | 0.91–0.94 | Slightly better than LSTM |
| SARIMA | 0.70–0.78 | Statistical baseline |

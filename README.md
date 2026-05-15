# 🌫️ Nagpur Air Quality Forecasting System

Welcome to the **Nagpur AQI Forecasting System**! This project is designed to predict air quality (AQI) across four major locations in Nagpur using state-of-the-art Machine Learning. Our goal is to provide reliable, 24-hour forecasts to help residents and authorities stay informed about the air they breathe.

## 🚀 What's Inside?
We didn't just build one model; we built and compared four different approaches to find the absolute best for Nagpur's specific environment:
*   **XGBoost (The Champion):** Our top-performing model with an **R² score of ~0.91**. It’s fast, consistent, and handles Nagpur's pollutant spikes excellently.
*   **ARIMA:** A classic statistical baseline that helps us track long-term trends.
*   **GRU & CNN-LSTM:** Deep learning models that we've fine-tuned to capture complex, non-linear sequences in the air data.

## 🛠️ Getting Started
We've made it very easy to get this project up and running on your local machine.

### 1. Install the basics
First, make sure you have the necessary libraries installed:
```bash
pip install -r requirements.txt
```

### 2. Run the full Dashboard
You can launch the entire project—both the AI backend and the visual dashboard—with a single command:
```bash
python backend/app.py
```
Once it's running, just open your browser and go to:
**[http://localhost:5000](http://localhost:5000)**

## 📊 Research & Comparison
If you want to see how the different models stack up against each other, run the comparison script:
```bash
python run_comparison.py
```
This will generate a detailed performance report and visual tables comparing MAE, RMSE, and R² scores for every station.

## 📁 Project Structure
*   `data/`: Contains the cleaned and preprocessed CPCB dataset for Nagpur.
*   `ml/`: The "brain" of the project. Includes training scripts, saved models, and performance plots.
*   `backend/`: A Flask API that serves the forecasts and also hosts the dashboard.
*   `frontend/`: The visual interface where you can track live AQI, view forecasts, and use the custom predictor.

## 📝 Features
*   **24-Hour Forecasts:** Get hourly predictions for the day ahead.
*   **Health Advisories:** Real-time health tips based on current AQI levels.
*   **Custom Predictor:** Enter your own pollutant values to see how the model reacts!
*   **Feature Importance:** Learn which pollutants (like PM2.5 or NO2) are driving the AQI at each station.


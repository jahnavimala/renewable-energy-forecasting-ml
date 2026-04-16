
# Renewable Energy Forecasting using ML & DL (Research-Oriented)

## Overview
Comparative study of ARIMA, Random Forest (tuned), XGBoost, and LSTM for time-series forecasting of renewable energy with exogenous weather features.

## Dataset
Synthetic dataset with realistic seasonality + weather features (temperature, humidity, wind_speed, irradiance). Replace with real datasets for experiments.

## Features
- Time features: hour, dayofweek, month
- Rolling stats: mean/std (3, 6)
- Lag features: 1..6
- Models: ARIMA, RF (GridSearchCV), XGBoost, LSTM
- Outputs: metrics.csv, experiments.csv, plots

## How to Run
pip install -r requirements.txt
python main.py

## Outputs
- results/metrics.csv
- results/experiments.csv
- results/plots/comparison.png
- results/plots/feature_importance.png

## Research Notes
- RF/XGB leverage exogenous + lag features
- LSTM models temporal dependencies
- Extend with real weather data and transformers

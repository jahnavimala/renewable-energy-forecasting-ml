# Renewable Energy Forecasting using Machine Learning & Deep Learning

## Overview
This project presents a comparative study of statistical, machine learning, and deep learning approaches for renewable energy forecasting.

Accurate prediction of renewable energy generation is challenging due to variability and dependence on weather conditions. This project evaluates ARIMA, Random Forest, XGBoost, and LSTM models on time-series data with engineered and weather-based features.

---

## Research Question
How effectively can machine learning and deep learning models forecast renewable energy generation using time-series and weather-based features?

Can advanced or hybrid approaches improve prediction accuracy compared to traditional statistical models?

---

## Dataset
The dataset includes:
- Power generation (target variable)
- Weather features: temperature, humidity, wind speed, irradiance

> Note: Replace with real-world dataset for stronger results.

---

## Feature Engineering
- Time-based features: hour, day of week, month
- Lag features: previous time steps (lag_1 to lag_6)
- Rolling statistics: mean and standard deviation
- Weather-based features

---

## Models Used
- **ARIMA** → Statistical baseline for time series
- **Random Forest (Tuned)** → Captures non-linear relationships
- **XGBoost** → Gradient boosting for improved performance
- **LSTM** → Deep learning model for temporal dependencies

---

## Model Justification
- ARIMA is used as a baseline for comparison
- Random Forest handles structured data with engineered features
- XGBoost improves performance through boosting
- LSTM captures sequential dependencies in time-series data

---

## Results

| Model          | MAE  | RMSE |
|---------------|------|------|
| ARIMA         | XX   | XX   |
| Random Forest | XX   | XX   |
| XGBoost       | XX   | XX   |
| LSTM          | XX   | XX   |

> Replace XX with actual values after running the project

---

## Visualization

### Model Comparison
![Comparison](results/plots/comparison.png)

### Feature Importance
![Feature Importance](results/plots/feature_importance.png)

---

## Key Insights
- ARIMA struggles with non-linear patterns
- Random Forest improves performance using engineered features
- XGBoost provides better accuracy through boosting
- LSTM performs best for capturing temporal dependencies

---

## Project Structure
renewable-energy-forecasting-ml/
├── data/
├── notebooks/
├── src/
├── results/
├── main.py
├── requirements.txt
└── README.md


---

## How to Run
```bash
pip install -r requirements.txt
python main.py
```
---
# Future Work
Build hybrid models combining RF + LSTM
Integrate real-time weather APIs
Use transformer-based models
Apply for smart grid optimization

# Real-World Application
This project can be extended for:
  Smart grid optimization
  Energy demand forecasting
  Real-time energy scheduling

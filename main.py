
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

from src.preprocessing import load_data, create_features, create_lag_features
from src.models import train_arima, train_rf, tune_rf, train_xgb
from src.lstm_model import prepare_lstm_data, train_lstm
from src.evaluation import evaluate

# Load and feature engineer
df = load_data("data/energy.csv")
df = create_features(df)

# Train-test split (time-based)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# ---------------- ARIMA (baseline on univariate) ----------------
arima_model = train_arima(train['power'])
arima_pred = arima_model.forecast(steps=len(test))

# ---------------- RF (with lag + exogenous features) ----------------
df_lag = create_lag_features(df, lag=6)
train_lag = df_lag.iloc[:train_size]
test_lag = df_lag.iloc[train_size:]

X_train = train_lag.drop('power', axis=1)
y_train = train_lag['power']
X_test = test_lag.drop('power', axis=1)
y_test = test_lag['power']

rf_model = tune_rf(X_train, y_train)  # hyperparameter tuning
rf_pred = rf_model.predict(X_test)

# ---------------- XGBoost (if available) ----------------
xgb_pred = None
try:
    xgb_model = train_xgb(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
except Exception as e:
    print("XGBoost not available, skipping:", e)

# ---------------- LSTM ----------------
X_lstm, y_lstm, scaler = prepare_lstm_data(df[['power']])
split = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

lstm_model = train_lstm(X_train_lstm, y_train_lstm)
lstm_pred = lstm_model.predict(X_test_lstm)

# inverse scale
lstm_pred = scaler.inverse_transform(lstm_pred)
y_test_lstm = scaler.inverse_transform(y_test_lstm)

# ---------------- Evaluation ----------------
mae_arima, rmse_arima = evaluate(test['power'], arima_pred)
mae_rf, rmse_rf = evaluate(y_test, rf_pred)
mae_lstm, rmse_lstm = evaluate(y_test_lstm, lstm_pred)

rows = [
    {"Model": "ARIMA", "MAE": float(mae_arima), "RMSE": float(rmse_arima)},
    {"Model": "RandomForest(tuned)", "MAE": float(mae_rf), "RMSE": float(rmse_rf)},
    {"Model": "LSTM", "MAE": float(mae_lstm), "RMSE": float(rmse_lstm)},
]

if xgb_pred is not None:
    mae_xgb, rmse_xgb = evaluate(y_test, xgb_pred)
    rows.append({"Model": "XGBoost", "MAE": float(mae_xgb), "RMSE": float(rmse_xgb)})

results = pd.DataFrame(rows)
results.to_csv("results/metrics.csv", index=False)

# experiments log
exp_path = "results/experiments.csv"
try:
    prev = pd.read_csv(exp_path)
except:
    prev = pd.DataFrame()

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_rows = []
for r in rows:
    exp_rows.append({
        "run_id": run_id,
        "model": r["Model"],
        "MAE": r["MAE"],
        "RMSE": r["RMSE"]
    })
exp_df = pd.DataFrame(exp_rows)
pd.concat([prev, exp_df], ignore_index=True).to_csv(exp_path, index=False)

# feature importance (RF)
importances = rf_model.feature_importances_
plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("results/plots/feature_importance.png")
plt.close()

# plot comparison (aligned to test index)
plt.figure(figsize=(10,5))
plt.plot(test.index, test['power'], label="Actual")
plt.plot(test.index, arima_pred, label="ARIMA")
plt.plot(test_lag.index, rf_pred, label="RF")
if xgb_pred is not None:
    plt.plot(test_lag.index, xgb_pred, label="XGB")
plt.legend()
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig("results/plots/comparison.png")
plt.close()

print("Saved results to results/metrics.csv and plots/")

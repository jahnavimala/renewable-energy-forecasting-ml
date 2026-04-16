
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train_arima(series):
    model = ARIMA(series, order=(5,1,0))
    return model.fit()

def train_rf(X, y):
    model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
    model.fit(X, y)
    return model

def tune_rf(X, y):
    params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None]
    }
    grid = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=3, n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_

def train_xgb(X, y):
    try:
        from xgboost import XGBRegressor
    except Exception:
        raise RuntimeError("xgboost not installed")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    return model


import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.sort_index()
    df = df.fillna(method='ffill')
    return df

def create_features(df):
    data = df.copy()
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month

    data['rolling_mean_3'] = data['power'].rolling(3).mean()
    data['rolling_std_3'] = data['power'].rolling(3).std()
    data['rolling_mean_6'] = data['power'].rolling(6).mean()
    data['rolling_std_6'] = data['power'].rolling(6).std()

    return data.dropna()

def create_lag_features(df, lag=3):
    data = df.copy()
    for i in range(1, lag+1):
        data[f'lag_{i}'] = data['power'].shift(i)
    return data.dropna()

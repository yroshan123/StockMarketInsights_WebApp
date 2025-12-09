import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch, numpy as np

def time_split_index(n_rows: int, test_ratio: float):
    cut = int(round(n_rows * (1 - test_ratio)))
    return max(cut, 1)

def make_sequences(df: pd.DataFrame, feature_cols, target_col, seq_len: int):
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(df[feature_cols].iloc[i:i+seq_len].values)
        y.append(df[target_col].iloc[i+seq_len])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y

def build_features_for_ticker(stock_df: pd.DataFrame, sent_df: pd.DataFrame, seq_len=20, test_ratio=0.2, horizon=7):
    s = stock_df.copy()
    s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
    s = s.dropna(subset=["Date"]).sort_values("Date")

    t = sent_df.copy()
    t["Date"] = pd.to_datetime(t["Date"], errors="coerce")
    t = t.dropna(subset=["Date"]).sort_values("Date")

    # Daily mean sentiment per calendar day
    daily_sent = t.groupby(t["Date"].dt.date)["Sentiment_Score"].mean().reset_index()
    daily_sent["Date"] = pd.to_datetime(daily_sent["Date"])

    # Align to trading dates (left join on price)
    df = pd.merge(s, daily_sent, on="Date", how="left")
    df["Sentiment_Score"] = df["Sentiment_Score"].ffill().bfill()

    # Price-derived features
    df["ma_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["z_score"] = (df["Close"] - df["ma_20"]) / df["Close"].rolling(window=7, min_periods=1).std()

    # Future returns for multiple horizons
    for lag in [1, 3, 7, 14]:
        df[f"future_returns_lag_{lag}"] = df["Close"].shift(-lag) / df["Close"] - 1.0

    # Sentiment dynamics
    df["sentiment_volatility"] = df["Sentiment_Score"].rolling(window=3, min_periods=1).std()
    df["lagged_sentiment"] = df["Sentiment_Score"].shift(1)

    # Drop rows where critical features or target will be NaN
    target_col = f"future_returns_lag_{horizon}"
    df = df.dropna(subset=[target_col]).dropna().reset_index(drop=True)

    feature_cols = [
        "Open","High","Low","Close","Volume",
        "ma_20","z_score","future_returns_lag_1","future_returns_lag_3",
        "sentiment_volatility"
    ]

    # Time-ordered split
    cut = time_split_index(len(df), test_ratio)
    train_df = df.iloc[:cut].copy()
    test_df  = df.iloc[cut:].copy()

    # Scale on TRAIN only
    feat_scaler = StandardScaler().fit(train_df[feature_cols])
    targ_scaler = StandardScaler().fit(train_df[[target_col]])

    df_scaled = df.copy()
    df_scaled[feature_cols] = feat_scaler.transform(df[feature_cols])
    df_scaled[target_col]   = targ_scaler.transform(df[[target_col]])

    train_scaled = df_scaled.iloc[:cut].copy()
    test_scaled  = df_scaled.iloc[cut:].copy()

    X_train, y_train = make_sequences(train_scaled, feature_cols, target_col, seq_len)
    X_test,  y_test  = make_sequences(test_scaled,  feature_cols, target_col, seq_len)

    return df, feature_cols, target_col, X_train, y_train, X_test, y_test, feat_scaler, targ_scaler
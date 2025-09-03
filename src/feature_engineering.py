def add_lag_features(df, lag=5):
    for i in range(1, lag+1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    return df.dropna()

def split_data(X, y, train_size=0.8):
    train_len = int(len(X) * train_size)
    return X[:train_len], y[:train_len], X[train_len:], y[train_len:]

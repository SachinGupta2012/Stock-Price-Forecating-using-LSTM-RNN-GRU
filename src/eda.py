import matplotlib.pyplot as plt
import seaborn as sns

def plot_close(df):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Close"])
    plt.title("Stock Closing Price Trend")
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

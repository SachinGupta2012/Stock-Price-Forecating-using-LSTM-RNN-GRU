import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="📊 Stock Price Forecast Dashboard", layout="wide")
st.title("🚀 Stock Price Prediction Dashboard")

# -------------------------------
# Load Scaler + Models
# -------------------------------
scaler = joblib.load("models/scaler.pkl")
models = {
    "LSTM": "models/lstm_model.h5",
    "RNN": "models/rnn_model.h5",
    "GRU": "models/gru_model.h5"
}

# Sidebar
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio("Select Model", list(models.keys()))

# Load chosen model
model = load_model(models[model_choice], compile=False)

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/processed_stock_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# -------------------------------
# Prepare Data
# -------------------------------
seq_length = 60
X, y = [], []
data_scaled = scaler.transform(df["Close"].values.reshape(-1, 1))

for i in range(seq_length, len(data_scaled)):
    X.append(data_scaled[i - seq_length:i])
    y.append(data_scaled[i])

X, y = np.array(X), np.array(y)

# -------------------------------
# Predictions
# -------------------------------
y_pred_scaled = model.predict(X, verbose=0)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y)

# -------------------------------
# Metrics
# -------------------------------
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# -------------------------------
# Line Chart (Full Series)
# -------------------------------
st.markdown("### 📈 Actual vs Predicted (Full Series)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["Date"][seq_length:], y_true, label="Actual", color="blue")
ax.plot(df["Date"][seq_length:], y_pred, label="Predicted", color="red", alpha=0.7)
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.set_title(f"{model_choice} - Actual vs Predicted")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------------------
# Metrics Section
# -------------------------------
st.markdown("### 📊 Model Performance Metrics (Full Series)")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("MAPE", f"{mape:.2f}%")

# -------------------------------
# Extra Visualizations
# -------------------------------
st.markdown("### 🔥 Error Analysis Visualizations")

# Residuals
residuals = y_true - y_pred

colA, colB = st.columns(2)

# Residual Plot
with colA:
    st.markdown("#### 📍 Residual Plot")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, residuals, alpha=0.5, color="purple")
    ax.axhline(0, linestyle="--", color="black")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    st.pyplot(fig)

# Error Distribution
with colB:
    st.markdown("#### 📊 Error Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals.flatten(), bins=30, kde=True, color="orange", ax=ax)
    ax.set_title("Distribution of Prediction Errors")
    st.pyplot(fig)

# -------------------------------
# DataFrame
# -------------------------------
st.markdown("### 📋 Comparison Data (Last 30 Rows)")
comparison_df = pd.DataFrame({
    "Date": df["Date"][seq_length:],
    "Actual_Close": y_true.flatten(),
    "Predicted_Close": y_pred.flatten(),
    "Residuals": residuals.flatten()
})
styled_df = comparison_df.tail(30).style.background_gradient(cmap="Blues").format(
    {"Actual_Close": "{:.2f}", "Predicted_Close": "{:.2f}", "Residuals": "{:.2f}"}
)
st.dataframe(styled_df, use_container_width=True)

# -------------------------------
# Summary Bar Chart
# -------------------------------
st.markdown("### 📊 Error Summary")
error_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "MAPE"],
    "Value": [rmse, mae, mape]
})

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x="Metric", y="Value", data=error_df, palette="coolwarm", ax=ax)
ax.set_title(f"{model_choice} Performance Metrics")
st.pyplot(fig)

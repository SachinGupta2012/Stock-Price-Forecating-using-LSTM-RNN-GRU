# 📈 Stock Price Prediction (RNN • LSTM • GRU)

An end‑to‑end deep learning project for **time‑series stock price forecasting** built with **TensorFlow/Keras** and deployed with **Streamlit**. The app lets you **switch between RNN, LSTM, and GRU models**, view **full‑series Actual vs Predicted** lines, and explore **modern visualizations** (bar charts, error heatmaps, styled tables) with **evaluation metrics** (RMSE, MAE, MAPE).

---

## 🔗 Table of Contents

* [Overview](#-overview)
* [Features](#-features)
* [Project Structure](#-project-structure)
* [Setup](#-setup)
* [Training](#-training)
* [Saved Artifacts](#-saved-artifacts)
* [Run the App](#-run-the-app)
* [How the App Works](#-how-the-app-works)
* [Metrics & Evaluation](#-metrics--evaluation)
* [Configuration](#-configuration)
* [Troubleshooting](#-troubleshooting)
* [Screenshots](#-screenshots)
* [Roadmap](#-roadmap)
* [License](#-license)

---

## 🧠 Overview

This project trains three recurrent architectures (**SimpleRNN, LSTM, GRU**) to forecast stock closing prices using a sliding window of the previous `seq_length` days. Models are trained on scaled data and persisted to disk. An interactive **Streamlit dashboard** loads any of the saved models, computes **backtests** on the hold‑out set or full series, and renders **resume‑ready** charts and tables.

> **Data expectation:** CSV with at least two columns: `Date` and `Close`.
>
> Path used by default: `data/processed_stock_data.csv`.

---

## ✅ Features

* Multiple models: **RNN, LSTM, GRU** (individually trainable & comparable)
* **50 epochs** default training with validation split and early stopping (optional)
* Clean **model persistence** (`.h5`) + **scaler** (`joblib`)
* **Streamlit UI** with:

  * Model selector (RNN / LSTM / GRU)
  * **Full‑series Actual vs Predicted** line chart
  * **Bar charts** for residuals & segment errors
  * **Styled DataFrame** for last *N* days and/or full backtest
  * **Metrics panel** (RMSE, MAE, MAPE)
* Modular codebase: data loading, preprocessing, sequence creation, modeling, evaluation, visualization

---

## 📁 Project Structure

```
stock/
├─ app.py                      # Streamlit dashboard
├─ main.py                     # Training entry point (trains & saves 3 models)
├─ src/
│  ├─ data_loader.py
│  ├─ preprocessing.py
│  ├─ sequence_preparation.py
│  ├─ split_data.py
│  ├─ modeling.py              # build_lstm / build_rnn / build_gru
│  ├─ evaluation.py
│  └─ visualization.py         # (optional helpers)
├─ data/
│  └─ processed_stock_data.csv
├─ models/                     # created automatically if missing
│  ├─ lstm_model.h5
│  ├─ rnn_model.h5
│  ├─ gru_model.h5
│  └─ scaler.pkl
├─ requirements.txt
└─ README.md
```

---

## 🧰 Setup

### 1) Clone & create venv

```bash
git clone <YOUR_REPO_URL> stock
cd stock
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Data

Place your CSV at `data/processed_stock_data.csv` with columns:

```
Date,Close
2025-01-02,25149.85
...
```

---

## 🏋️ Training

Run training (trains **LSTM**, **RNN**, **GRU** and saves all artifacts):

```bash
python main.py
```

What happens:

* Loads `data/processed_stock_data.csv`
* Scales Close with **MinMaxScaler**
* Creates sequences with `seq_length=60`
* Splits into train/test
* Trains each model for **50 epochs** (adjustable)
* Saves models to `models/*.h5` and scaler to `models/scaler.pkl`

> The script will **create `models/` automatically** if it doesn't exist.

---

## 💾 Saved Artifacts

* `models/lstm_model.h5`
* `models/rnn_model.h5`
* `models/gru_model.h5`
* `models/scaler.pkl`

> **Note:** We deliberately load models in the app with `compile=False` to avoid legacy loss/metric deserialization issues for `.h5` files.

---

## 🖥️ Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

**App Sidebar Controls**

* **Model**: choose **RNN / LSTM / GRU**
* **Chart mode**: Full series or last *N* days
* **Backtest window**: full series or trailing window for metrics
* **Display options**: show/hide residual bars, error heatmap, styled table

---

## 🔍 How the App Works

1. Loads your CSV and sorts by date.
2. Rebuilds sequences using the same `seq_length` used in training.
3. Loads the selected model (`compile=False`) and the saved scaler.
4. Computes predictions for the entire test span (or full series) by rolling window.
5. **Visualizations**:

   * **Actual vs Predicted line chart** (full series or last *N* days)
   * **Residual bar chart** (|Actual − Predicted| by date)
   * **Error heatmap** (aggregated by week or month)
   * **Styled DataFrame** of predictions (sortable, filterable)
6. **Metrics**: RMSE / MAE / MAPE for the chosen window.

---

## 📏 Metrics & Evaluation

* **RMSE** (Root Mean Squared Error): sensitive to large errors.
* **MAE** (Mean Absolute Error): average absolute error.
* **MAPE** (Mean Absolute Percentage Error): relative error (ignore if Close≈0).

**Example (illustrative only):**

```
RMSE : 171.97
MAE  : 123.62
MAPE : 1.19%
```

> For reproducible numbers, keep the same `seq_length`, train/test split, epochs, batch size, and seeds.

---

## ⚙️ Configuration

Most settings live in `main.py` / `src/modeling.py`:

* `seq_length = 60`
* `epochs = 50`
* `batch_size = 32`
* Model units/dropout/layers in `build_lstm`, `build_rnn`, `build_gru`

You can expose these as **Streamlit controls** if desired.

---

## 🛠️ Troubleshooting

**1) `ValueError: Could not deserialize 'keras.metrics.mse' ...`**

* Caused by loading legacy **.h5** with compiled objects.
* ✅ **Fix used here:** `load_model(path, compile=False)` in `app.py`.

**2) `TypeError: DatetimeArray._generate_range() ... 'closed'`**

* Newer pandas uses `inclusive=` instead of `closed=`.
* ✅ Use: `pd.date_range(..., inclusive="right")`.

**3) No GPU / CUDA warnings**

* Informational only. Training runs on CPU if GPU drivers aren’t present.

**4) Models folder missing**

* Training script creates `models/` automatically. Or make it manually.

---

## 🖼️ Screenshots

> Add your own images under `docs/images/` and reference them here.

* App Home (Model selector, metrics)
* Full‑series Actual vs Predicted line chart
* Residuals bar chart & error heatmap
* Styled predictions table

---

### 🙌 Acknowledgements

* TensorFlow/Keras, NumPy, Pandas, Matplotlib, Scikit‑learn
* Streamlit for the rapid UI framework

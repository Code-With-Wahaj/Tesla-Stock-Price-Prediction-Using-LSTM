# Tesla Stock Price Prediction Using LSTM

> A deep learning project that predicts Tesla (TSLA) stock closing prices using a Stacked LSTM neural network, trained on Kaggle and deployed through a Streamlit web application.

---

## 👨‍💻 About the Author

Hi, I am **Wahaj**, a faculty member at Aptech. This project was developed as part of a faculty skill enhancement program to demonstrate the practical implementation of deep learning for time-series forecasting.

| | |
|---|---|
| 🔗 LinkedIn | [muhammad-wahaj-bin-aamir](https://www.linkedin.com/in/muhammad-wahaj-bin-aamir) |
| 🐙 GitHub | [Code-With-Wahaj](https://github.com/Code-With-Wahaj/Predicting-Tesla-Stock-Price-Using-LSTM) |
| 📧 Email | wahajaamir2@gmail.com |
| 🎥 Video Demo | Coming Soon |

---

## 📌 Project Overview

This project predicts the daily closing price of Tesla, Inc. (TSLA) using a Stacked LSTM neural network. The workflow is split into two phases:

**Phase 1 — Model Development (Kaggle)**
- Exploratory Data Analysis
- ADF Stationarity Test
- Data preprocessing and MinMax scaling
- 60-day sliding window sequence generation
- Stacked LSTM model training with Early Stopping
- Model evaluation using RMSE, MAE, and MAPE

**Phase 2 — Deployment (Streamlit Web Application)**
- Trained model exported from Kaggle as `.h5`, `.pkl` files
- Loaded into a local Streamlit app
- Users upload a stock CSV and get next-day predictions and full backtest results
- The model uses the **last 60 trading days of Close prices** to predict the next day's closing price — no other columns are used

---

## 📊 Dataset

**Tesla Stock Data (Updated till 28 Jun 2021)** by varpit94 on Kaggle.

| Property | Value |
|---|---|
| Rows | 2,416 |
| Date Range | June 2010 – June 2021 |
| Target Variable | `Close` price |

**Columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

> Only the `Close` column is used for training and prediction. All other columns are ignored.

📎 [Kaggle Dataset Link](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021)

The dataset is included in this repository as `Tesla.csv`. A 2024 test file is included as `TESLA_test_2024.csv`.

---

## 🧠 Model Architecture

A Stacked LSTM built with TensorFlow and Keras.

```
Input: (60 timesteps, 1 feature)
    │
    ▼
LSTM — 128 units (return_sequences=True)
    │
Dropout — 20%
    │
    ▼
LSTM — 64 units (return_sequences=False)
    │
Dropout — 20%
    │
    ▼
Dense — 1 unit → Predicted Close Price
```

| Config | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Early Stopping | Patience = 15 epochs |
| Max Epochs | 100 |
| Batch Size | 32 |

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| TensorFlow & Keras | LSTM model |
| Streamlit | Web application UI |
| Pandas & NumPy | Data processing |
| Scikit-Learn | MinMaxScaler, error metrics |
| Plotly | Interactive charts |
| Statsmodels | ADF Stationarity Test |
| uv | Package manager |

---

## 📂 Project Structure

```
Tesla Stock Price Prediction Using LSTM/
│
├── README.md
├── Tesla.csv
├── TESLA_test_2024.csv
├── tesla-stock-price-prediction-using-lstm.ipynb
│
└── tesla_stock_app/
    ├── app.py
    ├── lstm_model.h5
    ├── scaler.pkl
    ├── config.pkl
    └── requirements.txt
```

| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `lstm_model.h5` | Trained LSTM model (from Kaggle) |
| `scaler.pkl` | Fitted MinMaxScaler (from Kaggle) |
| `config.pkl` | Model metadata and benchmark metrics |
| `requirements.txt` | Python dependencies |
| `tesla-stock-price-prediction-using-lstm.ipynb` | Kaggle training notebook |
| `Tesla.csv` | Original training dataset |
| `TESLA_test_2024.csv` | 2024 data for testing the deployed app |

---

## 🚀 How to Run This Project

### Prerequisites

- Python 3.13
- `uv` package manager

```bash
pip install uv
```

### Step 1 — Navigate to the App Folder

```bash
cd "Tesla Stock Price Prediction Using LSTM/tesla_stock_app"
```

### Step 2 — Create and Activate a Virtual Environment

```bash
uv venv --python 3.13
```

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
uv pip install -r requirements.txt
```

### Step 4 — Verify Required Model Files

Ensure these three files are present inside `tesla_stock_app/`:

- `lstm_model.h5`
- `scaler.pkl`
- `config.pkl`

These are already included in the repository. If you retrain the model on Kaggle, replace these with the newly downloaded versions.

### Step 5 — Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Step 6 — Using the Application

Upload `Tesla_test_2024.csv` or any stock CSV file that contains a `Date` column and a `Close` column. All other columns (`Open`, `High`, `Low`, `Volume`) are automatically ignored — the model was trained on Close prices only and requires no other input.

The app has three tabs:

**🔮 Tab 1 — Next-Day Forecast**
- Uses the last 60 Close prices from your file to predict the next trading session's closing price
- Displays the predicted price with the dollar and percentage change vs the last known close
- Shows a chart of the exact 60-day input window the model consumed, with the predicted price marked as a green star

**📊 Tab 2 — Backtest & Metrics**
- Slides the 60-day window across your entire uploaded dataset and evaluates the model at every step
- Displays accuracy, MAPE, MAE, MAE as a percentage of average price, and RMSE
- Shows an Actual vs Predicted Close price chart across all trading days
- Shows a colour-coded Prediction Error chart (green = overestimate, red = underestimate) with ±MAE reference lines
- Includes a plain-English explanation of why the two charts have different Y-axis scales

**📈 Tab 3 — Price Chart**
- Displays the historical Close price as an area chart
- Overlays a 7-day (fast) and 30-day (slow) moving average
- Shows a summary panel with all-time high, all-time low, latest close, total return, and average daily price change

---

## 📊 App Features

- CSV and Excel file upload support
- Automatic `Date` and `Close` column detection — no manual selection required
- Benchmark metrics strip (Accuracy, MAPE, MAE, RMSE) always visible at the top
- Plain-English explanation of why dollar MAE and percentage accuracy appear different
- Next-day price prediction with full 60-day input window visualisation and green star marker
- Full dataset backtest with Actual vs Predicted chart
- Colour-coded Prediction Error chart with ±MAE reference lines and zoom-scale explanation
- Interactive historical price chart with dual moving averages
- Dataset summary statistics panel

---

## 📊 Results

| Metric | Value |
|---|---|
| Accuracy | ~96% on test set |
| Formula | Accuracy = 100 − MAPE |

Full RMSE, MAE, and MAPE values are printed at the end of the Kaggle training notebook and are also displayed live inside the Streamlit app after running a backtest.

---

## 🔁 How to Retrain the Model

If you want to retrain the model from scratch instead of using the included files:

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Add the [Tesla Stock Dataset](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021)
3. Upload or paste the code from `tesla-stock-price-prediction-using-lstm.ipynb`
4. Run all cells and wait for training to complete
5. Download the three output files from the **Output** tab:
   - `lstm_model.h5`
   - `scaler.pkl`
   - `config.pkl`
6. Replace the existing files inside `tesla_stock_app/` with the new ones

📓 The full notebook is also publicly available on Kaggle:
[tesla-stock-price-preditiction-using-lstm](https://www.kaggle.com/code/wahajaamir/tesla-stock-price-preditiction-using-lstm)

---

## 📜 License

This project is intended for educational and academic purposes only.
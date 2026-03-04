## Tesla Stock Price Prediction using LSTM

### 👨‍💻 About the Author

Hi, I am Wahaj, a faculty member at Aptech.
This project was developed as part of a faculty skill enhancement program to demonstrate the practical implementation of deep learning for time-series forecasting.

LinkedIn: www.linkedin.com/in/muhammad-wahaj-bin-aamir

GitHub: https://github.com/Code-With-Wahaj/Predicting-Tesla-Stock-Price-Using-LSTM

Email: wahajaamir2@gmail.com

Video Demo: Coming Soon

### 📌 Project Overview

This project predicts stock closing prices of Tesla, Inc. (TSLA) using a Stacked LSTM neural network.

The workflow is divided into two main phases:

### 🔹 1. Model Development (Kaggle)

- Data preprocessing

- Exploratory Data Analysis

- ADF Stationarity Test

- Sequence generation using 60-day window

- LSTM model training

- Model evaluation

### 🔹 2. Deployment (Streamlit Web Application)

- Trained model exported from Kaggle

- Loaded into a local Streamlit application

- Users upload stock CSV files

- App generates predictions and interactive visualizations

-  The model uses the last 60 trading days of closing prices to predict the next trading day’s closing price.

### 📊 Dataset

Tesla Stock Data (Updated till 28 Jun 2021) by varpit94 on Kaggle.

- Rows: 2,416

- Date Range: June 2010 – June 2021

- Target Variable: Close price

#### Dataset Columns

- Date

- Open

- High

- Low

- Close

- Adj Close

- Volume

#### Dataset Link:
https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021

The dataset is included in this repository as:

TESLA.csv

### 🧠 Model Architecture

A Stacked LSTM built using TensorFlow and Keras.

#### Architecture Details

- Input Shape: (60 timesteps, 1 feature)

- LSTM Layer 1: 128 units, return_sequences=True

- Dropout: 20%

- LSTM Layer 2: 64 units, return_sequences=False

- Dropout: 20%

- Dense Layer: 1 unit (Predicted Close Price)

#### Training Configuration

- Optimizer: Adam

- Loss Function: Mean Squared Error (MSE)

- Early Stopping: Patience = 15 epochs

### 🛠 Tech Stack

- Python 3.13

- TensorFlow & Keras

- Streamlit

- Pandas

- NumPy

- Scikit-Learn

- Plotly

- Statsmodels (ADF Test)

- uv (Package Manager)

### 📂 Project Structure
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
    ├── requirements.txt

### 🚀 How to Run This Project
#### 🔹 Prerequisites

- Python 3.13

- uv package manager

If uv is not installed:

pip install uv

#### 🔹 Step 1 – Navigate to Streamlit Folder
cd Tesla Stock Price Prediction Using LSTM/tesla_stock_app

#### 🔹 Step 2 – Create Virtual Environment
uv venv --python 3.13

#### Activate Environment

#### Windows

.venv\Scripts\activate

#### Mac/Linux

source .venv/bin/activate

#### 🔹 Step 3 – Install Dependencies
uv pip install -r requirements.txt

#### 🔹 Step 4 – Verify Required Files

Ensure these files exist inside streamlit_app:

- lstm_model.h5

- scaler.pkl

- config.pkl

These are generated from the Kaggle notebook and are already included.

#### 🔹 Step 5 – Run the Streamlit App

streamlit run app.py

The app will open at:

http://localhost:8501

### 📈 Using the Application

Upload TELSA_test_2024.csv or any stock CSV file

Select:

- Date column

- Close column

- Choose prediction mode

#### 🔹 Prediction Modes
#### Predict Next Day

- Uses last 60 days

- Predicts next trading day's closing price

- Displays price change amount and percentage

#### Test Full Dataset

Evaluates model across entire dataset

Displays:

- Actual vs Predicted chart

- Zoomed visualization

- Error metrics (RMSE, MAE, MAPE)

### 📊 App Features

- CSV and Excel file uploads

- Manual column selection

- ADF stationarity test

- Interactive price chart

- Next-day price prediction

- Full dataset evaluation

- Actual vs Predicted visualization

- Error over time plot

- Error distribution plot

### 🔁 How to Retrain the Model

1. Go to Kaggle

2. Create a new notebook

3. Add the Tesla dataset

4. Upload or paste code from:

tesla-stock-price-prediction-using-lstm.ipynb

5. Run all cells

6. Download the generated files from the Output tab:

lstm_model.h5

scaler.pkl

config.pkl

7. Replace the existing files inside the streamlit_app folder.

### 📓 Kaggle Training Notebook

The full training notebook is available on Kaggle:

https://www.kaggle.com/code/wahajaamir/tesla-stock-price-preditiction-using-lstm


### 📊 Results

~96% accuracy on the test set

Metrics displayed:

- RMSE

- MAE

- MAPE

Metrics are shown during notebook training and inside the Streamlit app.

### 📜 License


This project is intended for educational and academic purposes.

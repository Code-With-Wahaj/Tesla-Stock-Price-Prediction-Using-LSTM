# Tesla Stock Price Prediction using LSTM

## About Me

Hi, I am Wahaj, a faculty member at Aptech. This project was built as part of the faculty 
skill enhancement program to demonstrate practical implementation of deep learning for
time-series forecasting.

- LinkedIn: www.linkedin.com/in/muhammad-wahaj-bin-aamir
- GitHub: https://github.com/Code-With-Wahaj/Predicting-Tesla-Stock-Price-Using-LSTM
- Email: wahajaamir2@gmail.com
- Video Demo: 

## Project Overview

A deep learning project that predicts Tesla (TSLA) stock closing prices using a Stacked LSTM 
neural network. The model is trained on Kaggle and deployed locally through a Streamlit 
web application. Data analysis, preprocessing, and model training are done on Kaggle. 
The trained model is then used in a local Streamlit web app where users can upload stock 
data and get predictions. The model takes the last 60 trading days of closing prices as 
input and predicts the next day's closing price.

## Dataset

The dataset used is Tesla Stock Data (Updated till 28 Jun 2021) by varpit94 on Kaggle. 
It contains 2,416 rows of daily stock data from June 2010 to June 2021. The columns include 
Date, Open, High, Low, Close, Adj Close, and Volume. The target variable we predict is the 
Close price. The dataset is included in the zip file as Tesla.csv.

Dataset link: https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021

## Project Structure

The project folder contains the following files. app.py is the Streamlit web application. 
lstm_model.h5 is the trained LSTM model downloaded from Kaggle. scaler.pkl is the fitted 
MinMaxScaler downloaded from Kaggle. config.pkl is the model configuration downloaded 
from Kaggle. requirements.txt lists all Python dependencies. tesla-stock-price-prediction-using-lstm.ipynb is the Kaggle training notebook. Tesla.csv is the original dataset. TESLA_test_2024 is tesla 2024 data for testing. README.md is this file.

## Tech Stack

The project uses Python 3.13, TensorFlow and Keras for deep learning, 
Streamlit for the web UI, Pandas and NumPy and Scikit-Learn for data processing, 
Plotly for interactive visualizations, Statsmodels for the ADF stationarity test, 
and uv as the package manager.

## Model Architecture

The model is a Stacked LSTM with the following layers. The input shape is 60 timesteps
and 1 feature. The first LSTM layer has 128 units with return_sequences set to True. 
This is followed by a Dropout layer at 20 percent. The second LSTM layer has 64 units 
with return_sequences set to False. This is followed by another Dropout layer at 20 percent. 
The final Dense layer has 1 unit which outputs the predicted price. The optimizer is Adam, 
the loss function is Mean Squared Error, and Early Stopping with a patience of 15 epochs 
is used during training.

## How to Run This Project

### Prerequisites

You need Python 3.13 and the uv package manager. If you do not have uv installed, 
run pip install uv in your terminal.

### Step 1

Extract the zip file and open a terminal inside the project folder by running cd tesla_stock.

### Step 2

Create a virtual environment by running uv venv --python 3.13. Then activate it. 
On Windows run .venv\Scripts\activate and on Mac or Linux run source .venv/bin/activate.

### Step 3

Install all dependencies by running uv pip install -r requirements.txt.

### Step 4

Make sure these three files are in the project folder: lstm_model.h5, scaler.pkl, and config.pkl.
These are generated from the Kaggle notebook and are already included in the zip.

### Step 5

Run the Streamlit app by running streamlit run app.py. The app will open in your 
browser at localhost.

### Step 6

Upload Tesla.csv which is included in the zip or any other stock CSV file. Select the 
Date and Close columns from the dropdowns. Choose a mode. Predict Next Day predicts the 
next trading day's close price based on the last 60 days. Test Full Dataset tests accuracy 
across the entire uploaded file and shows actual vs predicted charts. Click the button and 
view the results.

## How to Retrain the Model

If you want to train the model yourself instead of using the included files, go to kaggle.com 
and create a new notebook. Add the Tesla Stock Dataset from the link above. Upload or paste 
the code from tesla-stock-price-prediction-using-lstm.ipynb. Run all cells and wait for training to finish. 
Download the three output files from the Output tab which are lstm_model.h5, scaler.pkl, 
and config.pkl. Replace the existing files in your project folder with these new ones.

## Results

The model achieves approximately 96 percent accuracy on the test set. RMSE, MAE, 
and MAPE values are displayed after training in the notebook and also shown in the 
Streamlit app after running predictions.

## App Features

The app supports CSV and Excel file uploads. Users can manually select the date and price
columns. The app shows an ADF stationarity test result, an interactive price chart, 
next day price prediction with change amount and percentage, full dataset testing with 
actual vs predicted charts and zoomed view, and error analysis with error over time and 
error distribution plots.

## Author

Muhammad Wahaj Bin Aamir - AI and Flutter Developer

## License

This project is for educational and academic purposes.



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

-MAPE

Metrics are shown during notebook training and inside the Streamlit app.

### 📜 License

This project is intended for educational and academic purposes.
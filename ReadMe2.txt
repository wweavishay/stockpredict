Stock Price Prediction using XGBoost -stockvisual.py


This Python script demonstrates how to predict stock prices using the XGBoost regression model. It fetches historical stock data from Yahoo Finance, preprocesses it, trains an XGBoost model, and predicts future stock prices along with their trend directions.

Requirements
numpy
pandas
matplotlib
yfinance
scikit-learn
xgboost


Installation

Install the required libraries using pip:
pip install -r requirements.txt
OR
pip install numpy pandas matplotlib yfinance scikit-learn xgboost


Copy and run code
python stockvisual.py

Running the Script
it will prompt you to input a stock symbol (e.g., AAPL for Apple Inc.)
and predict its stock price for the next day.


Script Overview

get_stock_data: Fetches historical stock data using Yahoo Finance.
create_dataset: Creates training and test datasets for the model.
train_and_predict_model: Trains an XGBoost model and makes predictions.
predict_tomorrow_stock_price: Predicts the stock price for the next day.
predict_tomorrow_price_direction: Predicts the direction of stock price movement for the next day (Up/Down/Flat).
predict_price_trend: Predicts the trend of stock price movement for a specified number of days.
Various plotting functions to visualize actual vs. predicted prices.
Usage Notes
Input: The script prompts for a stock symbol and fetches data from July 26, 2020, to the current date by default.
Model Training: Utilizes an XGBoost model with a sliding window of 30 days for prediction.
Visualization: Generates plots to visualize actual vs. predicted prices.
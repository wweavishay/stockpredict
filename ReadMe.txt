Stock Market Trading Analysis Tool - main.py
This Python tool utilizes various libraries such as numpy, pandas, matplotlib, yfinance, and scikit-learn to perform stock market analysis, prediction, and visualization.
It allows users to fetch historical stock data, generate buy/sell signals based on moving averages, train a linear regression model for price prediction, and visualize the stock data along with predictions.


Running the Code in PyCharm
Setup:
Install the required libraries using pip:
pip install -r requirements.txt


Copy and run code
python main.py


Script Overview

fetch_stock_data: Retrieves historical stock data from Yahoo Finance and prepares it for analysis.
calculate_moving_averages: Computes short and long moving averages for the stock data.
generate_signals: Produces buy/sell signals based on moving average crossovers.
train_model: Trains a linear regression model using the provided training data.
predict_stock_price: Predicts today's and tomorrow's stock prices using the trained model.
plot_stock_data: Displays a visual representation of the stock data along with moving averages and buy/sell signals.
plot_actual_vs_predicted: Plots the actual and predicted stock prices.
plot_feature_importance: Illustrates feature importance based on the coefficients of the model.
main_menu: Provides a user-friendly interface for interacting with the tool's functionalities.
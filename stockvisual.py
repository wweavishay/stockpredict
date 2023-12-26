import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import datetime as dt

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def get_stock_data(symbol, start_date, end_date, interval):
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return data['Close'].values.reshape(-1, 1), data.index.date

def train_and_predict_model(X_train, y_train, X_test, window_size):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train)
    train_predictions = model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1]))
    test_predictions = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1]))
    return train_predictions, test_predictions

def inverse_transform_data(data, scaler):
    return scaler.inverse_transform(data.reshape(-1, 1))

def calculate_mean_squared_error(actual, predicted):
    return mean_squared_error(actual, predicted)

def plot_results(dates, window_size, train_size, y_train_original, y_test_original, test_predictions_original, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(dates[window_size:train_size], y_train_original, label='Actual Train Price', color='blue')
    plt.plot(dates[train_size:], y_test_original, label='Actual Test Price', color='green')
    plt.plot(dates[train_size:], test_predictions_original, label='Predicted Test Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.title(f'Stock Price Prediction for {symbol}')
    plt.xticks(rotation=45)
    plt.show()

def plot_zoomed_results(dates, y_test_original, test_predictions_original, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(dates[-len(y_test_original):], y_test_original, label='Actual Test Price', color='green')
    plt.plot(dates[-len(y_test_original):], test_predictions_original, label='Predicted Test Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.title(f'Stock Price Prediction for {symbol} (Zoomed-In)')
    plt.xticks(rotation=45)
    plt.show()

def predict_tomorrow_stock_price(model, data, scaler, window_size):
    X, y = create_dataset(data, window_size)
    X = X[-1].reshape(1, window_size)  # Take the last window_size data for prediction
    prediction = model.predict(X)
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
    return prediction[0][0]

def predict_tomorrow_price_direction(model, data, scaler, window_size):
    today_price = data[-1][0]
    tomorrow_prediction = predict_tomorrow_stock_price(model, data, scaler, window_size)
    if tomorrow_prediction > today_price:
        return "Up"
    elif tomorrow_prediction < today_price:
        return "Down"
    else:
        return "Flat"



def predict_price_trend(model, data, scaler, window_size, num_days):
    prediction_trend = []

    # Make predictions for each day in the future
    for i in range(num_days):
        prediction = predict_tomorrow_price_direction(model, data, scaler, window_size)
        tomorrow_date = (end_date + timedelta(days=i+1)).strftime("%Y-%m-%d")
        prediction_trend.append((tomorrow_date, prediction))

        # Update data to include the latest prediction for the next iteration
        next_day_prediction = predict_tomorrow_stock_price(model, data, scaler, window_size)
        data = np.append(data, next_day_prediction)

    return prediction_trend



if __name__ == "__main__":
    # Input parameters
    symbol = input("Enter the stock symbol (e.g., AAPL for Apple Inc.): ")
    start_date_str = "2020-07-26"
    end_date_str = dt.datetime.today().strftime('%Y-%m-%d')
    interval = "1d"  # 15-minute interval

    # Convert the input start_date and end_date strings to datetime objects
    start_date = dt.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = dt.datetime.strptime(end_date_str, '%Y-%m-%d')


    window_size = 30  # Sliding window size for the XGBoost model
    # Fetch stock data and dates within the specified range
    stock_data, dates = get_stock_data(symbol, start_date, end_date, interval)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(stock_data)

    # Split the data into training and test sets
    train_size = int(len(data_normalized) * 0.8)
    train_data = data_normalized[:train_size]
    test_data = data_normalized[train_size - window_size:]

    # Create training dataset
    X_train, y_train = create_dataset(train_data, window_size)

    # Create test dataset
    X_test, y_test = create_dataset(test_data, window_size)

    # Check if the test dataset is large enough for reshaping
    if len(X_test) == 0:
        print("Test dataset is too small for the given window size.")
        exit()

    # Train the model and make predictions
    train_predictions, test_predictions = train_and_predict_model(X_train, y_train, X_test, window_size)

    # Inverse transform the predictions to get the original scale
    y_train_original = inverse_transform_data(y_train, scaler)
    y_test_original = inverse_transform_data(y_test, scaler)
    train_predictions_original = inverse_transform_data(train_predictions, scaler)
    test_predictions_original = inverse_transform_data(test_predictions, scaler)

    # Calculate the Mean Squared Error
    train_mse = calculate_mean_squared_error(y_train_original, train_predictions_original)
    test_mse = calculate_mean_squared_error(y_test_original, test_predictions_original)
    print(f"Train Mean Squared Error: {train_mse}")
    print(f"Test Mean Squared Error: {test_mse}")

    # Plot the results
    plot_results(dates, window_size, train_size, y_train_original, y_test_original, test_predictions_original, symbol)

    # Plot the zoomed-in results
    plot_zoomed_results(dates, y_test_original, test_predictions_original, symbol)

    # Train the model and make predictions
    train_predictions, test_predictions = train_and_predict_model(X_train, y_train, X_test, window_size)

    # Create the XGBoost model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train)

    # Predict stock price for tomorrow
    tomorrow_prediction = predict_tomorrow_stock_price(model, data_normalized, scaler, window_size)
    tomorrow_date = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Predicted stock price for {symbol} on {tomorrow_date}: ${tomorrow_prediction:.2f}")

    # Predict stock price direction for tomorrow
    tomorrow_direction = predict_tomorrow_price_direction(model, data_normalized, scaler, window_size)
    print(f"Predicted stock price direction for {symbol} on {tomorrow_date}: {tomorrow_direction}")


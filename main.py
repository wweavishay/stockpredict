import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime as dt


def save_data_to_csv(data, filename):
    """Save data to a CSV file."""
    data.to_csv(filename, index=False)


def fetch_stock_data(company, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance and prepare it for analysis."""
    data = yf.download(company, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data available for the specified company and date range.")
    data["Target"] = data["Close"].shift(-1)
    data.dropna(inplace=True)
    return data


def calculate_moving_averages(data, short_window=50, long_window=200):
    """Calculate short and long moving averages."""
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    return data


def generate_signals(data):
    """Generate buy/sell signals based on crossovers of moving averages."""
    data['Signal'] = np.where(data['Short_MA'] > data['Long_MA'], 1, -1)
    return data


def train_model(X_train, y_train):
    """Train a linear regression model on the training data."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict_stock_price(model, data):
    """Predict today's and tomorrow's stock prices using the trained model."""
    # Check if there are at least two rows in the data to make predictions
    if len(data) < 2:
        raise ValueError("Insufficient data to make predictions.")

    # Get the most recent two rows of the data DataFrame
    last_two_rows = data.iloc[-2:]

    # Extract the features for today's and tomorrow's stock prices
    today_features = last_two_rows.iloc[0][["Open", "High", "Low", "Close", "Volume"]].values.reshape(1, -1)
    tomorrow_features = last_two_rows.iloc[1][["Open", "High", "Low", "Close", "Volume"]].values.reshape(1, -1)

    # Predict today's and tomorrow's stock prices
    today_price = model.predict(today_features)[0]
    tomorrow_price = model.predict(tomorrow_features)[0]

    return today_price, tomorrow_price


def print_price_prediction(today_price, tomorrow_price, today_close):
    """Print the stock price prediction for tomorrow and compare it with today's closing price."""
    print(f"Predicted Stock Price for Tomorrow: {tomorrow_price:.4f}")

    if tomorrow_price > today_close:
        print("The stock price is expected to increase tomorrow.")
    elif tomorrow_price < today_close:
        print("The stock price is expected to decrease tomorrow.")
    else:
        print("The stock price is expected to remain unchanged tomorrow.")


def plot_actual_vs_predicted_table(y_test, y_pred, company):
    """Plot the actual vs. predicted stock prices and show differences."""
    data = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': y_pred})
    data['Difference'] = data['Predicted Prices'] - data['Actual Prices']

    avg_difference = data['Difference'].mean()
    std_difference = data['Difference'].std()

    print(f"\nSummary Statistics for {company}:")
    print(f"Average Difference: {avg_difference:.4f}")
    print(f"Standard Deviation: {std_difference:.4f}")

    print("\nPredicted vs. Actual Stock Prices with Differences:")
    print(data)


def plot_stock_data(data, company):
    """Plot the stock data with moving averages and buy/sell signals."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label="Stock Price", color='black')
    plt.plot(data.index, data['Short_MA'], label="Short MA (50-day)", color='blue')
    plt.plot(data.index, data['Long_MA'], label="Long MA (200-day)", color='red')

    plt.plot(data[data['Signal'] == 1].index, data['Short_MA'][data['Signal'] == 1], '^', markersize=10, color='g',
             label='Buy Signal')
    plt.plot(data[data['Signal'] == -1].index, data['Short_MA'][data['Signal'] == -1], 'v', markersize=10, color='r',
             label='Sell Signal')

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Price and Moving Averages for {company}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_actual_vs_predicted(data, company, y_test, y_pred, rmse):
    """Plot the actual and predicted stock prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(y_test):], y_test, label="Actual Prices", marker='o', color='blue')
    plt.plot(data.index[-len(y_test):], y_pred, label="Predicted Prices", marker='o', markerfacecolor='red',
             markersize=8, color='green')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Price Prediction for {company}")
    plt.legend()
    plt.grid(True)
    plt.annotate(f"RMSE: {rmse:.4f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='red')
    plt.show()


def plot_feature_importance(model, features):
    """Plot feature importance based on the model's coefficients."""
    feature_importance = pd.Series(model.coef_, index=features.columns)
    feature_importance.plot(kind='bar', color='blue', alpha=0.7, figsize=(10, 6))
    plt.xlabel("Features")
    plt.ylabel("Coefficient (Weight)")
    plt.title("Feature Importance")
    plt.grid(True)
    plt.show()


def main_menu():
    print("Welcome to the Stock Price Prediction Tool!")

    # User inputs for stock symbol and date range
    company = input("Enter the stock symbol (e.g., GOOG): ").upper()
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = dt.datetime.today().strftime('%Y-%m-%d')

    try:
        data = fetch_stock_data(company, start_date, end_date)
        X = data[["Open", "High", "Low", "Close", "Volume"]]
        y = data["Target"]

        data = calculate_moving_averages(data)
        data = generate_signals(data)

        # Sort the data by date in ascending order
        data = data.sort_index()

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        save_data_to_csv(pd.concat([X_train, y_train], axis=1), f"{company}_train_data.csv")
        save_data_to_csv(pd.concat([X_test, y_test], axis=1), f"{company}_test_data.csv")

        while True:
            print("\nMenu:")
            print("1. Plot Stock Data with Moving Averages and Buy/Sell Signals")
            print("2. Predict Tomorrow's Stock Price")
            print("3. Show Summary Statistics for Stock Price Prediction")
            print("4. Plot Feature Importance")
            print("5. Plot Actual vs. Predicted Stock Prices")
            print("6. Exit")

            choice = input("Enter the number corresponding to the function you want to use: ")

            if choice == "1":
                plot_stock_data(data, company)
            elif choice == "2":
                try:
                    today_price, tomorrow_price = predict_stock_price(model, data)
                    today_close = data.iloc[-1]['Close']
                    print_price_prediction(today_price, tomorrow_price, today_close)
                except ValueError as e:
                    print(e)
            elif choice == "3":
                plot_actual_vs_predicted_table(y_test, y_pred, company)
            elif choice == "4":
                plot_feature_importance(model, X)
            elif choice == "5":
                plot_actual_vs_predicted(data, company, y_test, y_pred, rmse)
            elif choice == "6":
                print("Exiting the Stock Price Prediction Tool. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a valid number.")

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main_menu()

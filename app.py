import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates

# Define custom objects dictionary
custom_objects = {
    'Orthogonal': Orthogonal
}

# Load the model with custom objects
model = load_model(r"C:\Users\91869\Downloads\Stock Predictions Model0.keras", custom_objects=custom_objects)

st.header('Stock Analysis And Prediction - Web App')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start_date = st.date_input("Select start date", pd.to_datetime('2012-01-01'), key='start_date')
end_date = st.date_input("Select end date", pd.to_datetime('2022-12-31'), key='end_date')

if stock and start_date and end_date:
    data = yf.download(stock, start=start_date, end=end_date)

    st.subheader('Stock Data')
    st.write(data)

    def plot_closing_price(data, start_date, end_date):
        subset_data = data.loc[start_date:end_date]
        closing_price = subset_data['Close']
        fig = plt.figure(figsize=(8, 6))
        plt.plot(subset_data.index, closing_price, 'g')
        plt.xlabel('Date')
        plt.ylabel('Closing Price ($)')
        plt.title('Closing Price vs Date')
        plt.xticks(rotation=45)  # Rotate x-axis labels
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.show()
        st.pyplot(fig)

    def plot_moving_averages(data, start_date, end_date):
        subset_data = data.loc[start_date:end_date]
        closing_price = subset_data['Close']
        ma_100 = closing_price.rolling(window=100).mean()
        ma_200 = closing_price.rolling(window=200).mean()

        fig = plt.figure(figsize=(8, 6))
        plt.plot(subset_data.index, closing_price, 'g', label='Closing Price ($)')
        plt.plot(subset_data.index, ma_100, 'r', label='100-day Moving Average ($)')
        plt.plot(subset_data.index, ma_200, 'b', label='200-day Moving Average ($)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title("Closing Price vs Date with 100-day & 200-day Moving Averages")
        plt.xticks(rotation=45)  # Rotate x-axis labels
        plt.legend()
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.show()
        st.pyplot(fig)

    def plot_bollinger_bands(data, start_date, end_date):
        subset_data = data.loc[start_date:end_date].copy()
        subset_data['100 Day STD'] = subset_data['Close'].rolling(window=100).std()
        subset_data['ma_100_days'] = subset_data['Close'].rolling(window=100).mean()
        subset_data['Upper Band'] = subset_data['ma_100_days'] + (subset_data['100 Day STD'] * 2)
        subset_data['Lower Band'] = subset_data['ma_100_days'] - (subset_data['100 Day STD'] * 2)

        fig = plt.figure(figsize=(14, 7))
        plt.plot(subset_data['Close'], label='Close Price')
        plt.plot(subset_data['Upper Band'], label='Upper Band', linestyle='--', color='red')
        plt.plot(subset_data['Lower Band'], label='Lower Band', linestyle='--', color='blue')
        plt.fill_between(subset_data.index, subset_data['Lower Band'], subset_data['Upper Band'], color='grey', alpha=0.1)
        plt.title(f'{stock} Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Adjust the interval as needed
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)  # Rotate x-axis labels
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.show()
        st.pyplot(fig)

    # Plot based on user input
    plot_type = st.radio("Select plot type", ("Closing Price vs Date", "Moving Averages", "Bollinger Bands"))
    if plot_type == "Closing Price vs Date":
        plot_closing_price(data, start_date, end_date)
    elif plot_type == "Moving Averages":
        plot_moving_averages(data, start_date, end_date)
    else:
        plot_bollinger_bands(data, start_date, end_date)

    # Using and training the ML model
    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

    scaler = MinMaxScaler(feature_range=(0, 1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1 / scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8, 6))
    plt.plot(data_test[-len(predict):].index, y, 'r', label='Original Price ($)')
    plt.plot(data_test[-len(predict):].index, predict, 'g', label='Predicted Price ($)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title('Original Price vs Predicted Price')
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()
    st.pyplot(fig4)

    # Future Prediction
    st.subheader('Future Stock Price Prediction')

    future_days = 30  # Number of days to predict
    last_100_days = data_test_scale[-100:]  # Last 100 days from the scaled test data
    future_predictions = []

    current_input = last_100_days

    for _ in range(future_days):
        current_input = current_input.reshape((1, current_input.shape[0], 1))
        future_pred = model.predict(current_input)
        future_predictions.append(future_pred[0, 0])

        current_input = np.append(current_input[0, 1:], future_pred[0, 0])
        current_input = current_input.reshape((100, 1))

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = future_predictions * scale  # Inverse transform the predictions

    # Generate dates for future predictions
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=future_days + 1, freq='D')[1:]

    # Plot future predictions
    fig5 = plt.figure(figsize=(8, 6))
    plt.plot(future_dates, future_predictions, 'r', label='Future Predictions ($)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()
    st.pyplot(fig5)

    # Displaying next 7 days predictions in a table
    st.subheader('Predicted Stock Prices for the Next 7 Days')
    next_7_days_predictions = future_predictions[:7]
    next_7_days_dates = future_dates[:7]
    next_7_days_predictions_inr = next_7_days_predictions * 83.06  # Assuming exchange_rate is the conversion factor
    prediction_df = pd.DataFrame({'Date': next_7_days_dates,
                                  'Predicted Price (USD)': next_7_days_predictions.flatten(),
                                  'Predicted Price (INR)': next_7_days_predictions_inr.flatten()})
    prediction_df['Predicted Price (USD)'] = '$' + prediction_df['Predicted Price (USD)'].astype(str)  # Add $ sign
    st.table(prediction_df)
else:
    st.write("Please enter a stock symbol and select a date range.")

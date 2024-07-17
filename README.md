# Stock Price Prediction

This repository contains a web application for stock price analysis and prediction. The application allows users to visualize historical stock data, calculate moving averages, plot Bollinger Bands, and predict future stock prices using a machine learning model.

## Features

- **Stock Data Visualization**: View historical stock data for a selected stock symbol within a specified date range.
- **Closing Price Plot**: Visualize the closing prices of the stock over time.
- **Moving Averages Plot**: Plot the 100-day and 200-day moving averages of the stock.
- **Bollinger Bands Plot**: Display Bollinger Bands to analyze stock volatility.
- **Price Prediction**: Predict future stock prices using a pre-trained machine learning model.
- **Future Stock Price Prediction**: Predict stock prices for the next 30 days.

## File Descriptions

- `app.py`: The main file that contains the Streamlit web application code.
- `model_code.ipynb`: Jupyter notebook containing the code used for training the machine learning model.

## How It Works

### Data Loading

The application uses the `yfinance` library to download historical stock data from Yahoo Finance. The user can input the stock symbol and specify the date range for the data.

### Data Visualization

Three types of plots are available for visualization:

1. **Closing Price vs Date**: Plots the closing prices of the stock over the selected date range.
2. **Moving Averages**: Plots the 100-day and 200-day moving averages along with the closing prices.
3. **Bollinger Bands**: Plots the Bollinger Bands along with the closing prices to analyze volatility.

### Machine Learning Model

The application uses a pre-trained LSTM model to predict future stock prices. The model is loaded using TensorFlow's `load_model` function with custom objects.

The model is trained on the closing prices of the stock, and it uses the last 100 days of data to predict the next day's price.

### Future Price Prediction

The application predicts stock prices for the next 30 days based on the last 100 days of data. The predictions are displayed in a plot and in a table format for the next 7 days.

## Screenshots

### App Homepage
![App Homepage](https://github.com/abhaygupt-a/STOCK-ANALYSIS-AND-PREDICTION---WEB-APP/blob/main/extra/Screenshot%202024-05-27%20105912.png)

### Stock Data Visualization
![Stock Data Visualization](https://github.com/abhaygupt-a/STOCK-ANALYSIS-AND-PREDICTION---WEB-APP/blob/main/extra/Screenshot%202024-05-27%20105930.png)

### Moving Averages Plot
![Moving Averages Plot](https://github.com/abhaygupt-a/STOCK-ANALYSIS-AND-PREDICTION---WEB-APP/blob/main/extra/Screenshot%202024-05-27%20105948.png)

### Original Vs Predicted Price
![Original Vs Predicted Price](https://github.com/abhaygupt-a/STOCK-ANALYSIS-AND-PREDICTION---WEB-APP/blob/main/extra/Screenshot%202024-05-27%20110024.png)

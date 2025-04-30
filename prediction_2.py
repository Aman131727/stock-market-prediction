import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

# Fetch stock data
stock = yf.Ticker("RELIANCE.NS")
df = stock.history(period="5y")
df = df[['Close']]  # We only need closing prices

# Normalize the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Create training and testing datasets
train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60  # 60 days of past data for prediction
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))  # Increased units
model.add(Dropout(0.2))  # Dropout to prevent overfitting
model.add(LSTM(100, return_sequences=False))
model.add(Dense(50, activation='relu'))  # Added activation
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, batch_size=16, epochs=20)  # Increased batch size & epochs

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train.reshape(-1, time_step), Y_train)
rf_predictions = rf_model.predict(X_test.reshape(-1, time_step))

# ARIMA Model
arima_model = ARIMA(df['Close'], order=(5, 1, 0))
arima_fitted = arima_model.fit()
arima_predictions = arima_fitted.forecast(steps=len(X_test))

# LSTM Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Convert back to original scale

# Align rf_predictions and arima_predictions with the length of predictions
rf_predictions = np.array(rf_predictions[:len(predictions)]).reshape(-1, 1)
arima_predictions = np.array(arima_predictions[:len(predictions)]).reshape(-1, 1)

# Ensure test_data is correctly sliced for plotting
actual_prices = scaler.inverse_transform(test_data[time_step + 1:len(predictions) + time_step + 1])

# Future Prediction for 30 Days (1 Month)
future_steps = 30

# LSTM Future Predictions
last_60_days = test_data[-time_step:]  # Get the last 60 days from the test data
future_predictions = []
for _ in range(future_steps):
    last_60_days_reshaped = last_60_days.reshape(1, time_step, 1)
    next_prediction = model.predict(last_60_days_reshaped)
    future_predictions.append(next_prediction[0, 0])
    last_60_days = np.append(last_60_days[1:], next_prediction, axis=0)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# ARIMA Future Predictions
arima_future_predictions = arima_fitted.forecast(steps=future_steps)

# Plot LSTM predictions
plt.figure(figsize=(10, 5))
plt.plot(df.index[-len(predictions):], actual_prices, label="Actual Prices")
plt.plot(df.index[-len(predictions):], predictions, label="Predicted Prices", color='red')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Reliance Industries Stock Price Prediction using LSTM")
plt.legend()
plt.show()

# Plot LSTM vs Random Forest vs ARIMA predictions
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(predictions):], actual_prices, label="Actual Prices")
plt.plot(df.index[-len(predictions):], predictions, label="LSTM Predictions", color='red')
plt.plot(df.index[-len(predictions):], rf_predictions, label="Random Forest Predictions", color='blue')
plt.plot(df.index[-len(predictions):], arima_predictions, label="ARIMA Predictions", color='green')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Predictions: LSTM vs Random Forest vs ARIMA")
plt.legend()
plt.show()

# Plot Future Predictions
future_dates = pd.date_range(start=df.index[-1], periods=future_steps + 1, freq='B')[1:]  # Business days
plt.figure(figsize=(10, 5))
plt.plot(future_dates, future_predictions, label="LSTM Future Predictions", color='red')
plt.plot(future_dates, arima_future_predictions, label="ARIMA Future Predictions", color='green')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Future Stock Price Predictions (1 Month)")
plt.legend()
plt.show()
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd


stock = yf.Ticker("RELIANCE.NS")
df = stock.history(period="1y")

# Display first few rows
print(df.head())

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Close'], label="Reliance Stock Price")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("Reliance Industries Stock Price Over Time")
plt.legend()
plt.show()

df['50_MA'] = df['Close'].rolling(window=50).mean()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Close'], label="Closing Price")
plt.plot(df.index, df['50_MA'], label="50-Day Moving Average", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Price and Moving Average")
plt.legend()
plt.show()




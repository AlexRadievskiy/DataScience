import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fetch_data():
    ticker = "TON11419-USD"
    data = yf.download(ticker, start="2021-01-01", end="2023-01-01", interval="1mo")
    return data

# 1. Результати в файлик
def save_to_csv(df):
    df.to_csv('parsed_data.csv')

# 2. Оцінка тренду реальних данних
def plot_trend(df):
    df['Close'].plot(figsize=(10, 6), title='Dynamics of TON Coin', grid=True, color='blue', linewidth=2)
    plt.show()

# 3. Статистична характеристика
def get_statistics(df):
    return df.describe()

# 4. Нормальні та аномальні
def calculate_errors(df):
    # Середнє та відхилення
    mean = df['Close'].mean()
    std = df['Close'].std()

    # Нормальна
    df['Normal_Error'] = std / np.sqrt(len(df))

    # Аномалка
    df['Anomaly_Error'] = df['Close'] - mean

    return df

# 5. Візуал
def plot_errors(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Actual Data', color='blue', linewidth=2)
    plt.fill_between(df.index, df['Close'] - df['Normal_Error'], df['Close'] + df['Normal_Error'], color='gray', alpha=0.5, label='Normal Error')
    plt.fill_between(df.index, df['Close'] - df['Anomaly_Error'], df['Close'] + df['Anomaly_Error'], color='red', alpha=0.2, label='Anomaly Error')
    plt.legend()
    plt.title('TON Coin with Errors')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    data = fetch_data()
    save_to_csv(data)
    plot_trend(data)
    print(get_statistics(data))
    data_with_errors = calculate_errors(data)
    plot_errors(data_with_errors)

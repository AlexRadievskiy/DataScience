import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Завантаження даних
data = pd.read_csv('parsed_data.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Вибір випадкової дати для додавання аномалій
random_date = random.choice(data.index[:-3])

# Додавання аномалій протягом 3 днів підряд
for i in range(3):
    data.at[random_date + pd.Timedelta(days=i), 'Close'] = data['Close'].mean() * 3

# Візуалізація даних з аномаліями
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], 'bo-', label='With Anomalies')
plt.title('Data with Anomalies')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)

# Підсвічування аномалій
plt.scatter(data.index, data['Close'], color='red', s=50, lw=0, label='Anomaly', zorder=5)
plt.scatter(random_date, data.at[random_date, 'Close'], color='lime', s=50, lw=0, label='Start of Anomaly', zorder=5)

plt.legend()
plt.show()

# Видалення аномалій за допомогою методу IQR
Q1 = data['Close'].quantile(0.25)
Q3 = data['Close'].quantile(0.75)
IQR = Q3 - Q1

# Фільтрація аномалій
filtered_data = data[~((data['Close'] < (Q1 - 1.5 * IQR)) | (data['Close'] > (Q3 + 1.5 * IQR)))]

# Візуалізація очищених даних
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['Close'], 'go-', label='Without Anomalies')
plt.title('Data without Anomalies')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# Візуалізація порівняння
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], 'bo-', label='With Anomalies')
plt.plot(filtered_data['Close'], 'go-', label='Without Anomalies')
plt.title('Comparison of Data with and without Anomalies')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# 4. Визначення показників якості та оптимізація моделі
# 5. Статистичне навчання поліноміальної моделі за методом найменших квадратів (МНК – LSM)

# Подготовка данных
X = np.array((filtered_data.index - filtered_data.index.min()).days).reshape(-1, 1)
y = filtered_data['Close'].values

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Построение полиномиальной регрессии
degree = 3  # Выберите степень полинома
polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
polyreg.fit(X_train, y_train)

# Предсказание
y_pred = polyreg.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(filtered_data.index, filtered_data['Close'], 'go-', label='Actual')
plt.plot(filtered_data.index, polyreg.predict(X), 'r-', label='Predicted')
plt.title('Polynomial Regression')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# 6. Прогнозування (екстраполяцію) параметрів досліджуваного процесу
forecast_days = int(0.1 * len(X))
forecast_index = pd.date_range(filtered_data.index[-1], periods=forecast_days+1)[1:]  # Убран аргумент 'closed'

forecast_X = np.array((forecast_index - filtered_data.index.min()).days).reshape(-1, 1)

forecast_y = polyreg.predict(forecast_X)

# Визуализация прогноза
plt.figure(figsize=(10, 6))
plt.plot(filtered_data.index, filtered_data['Close'], 'go-', label='Actual')
plt.plot(filtered_data.index, polyreg.predict(X), 'r-', label='Predicted')
plt.plot(forecast_index, forecast_y, 'b-', label='Forecasted')
plt.title('Forecasting with Polynomial Regression')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

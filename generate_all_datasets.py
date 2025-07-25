# generate_all_datasets.py
import pandas as pd
import numpy as np
import os

os.makedirs("datasets", exist_ok=True)

# 1. sales_data.csv
np.random.seed(42)
months = pd.date_range(start='2023-01-01', periods=12, freq='ME')
products = ['Laptop', 'Tablet', 'Smartphone', 'Monitor']
sales_data = []
for month in months:
    for product in products:
        sales = np.random.randint(50, 200)
        revenue = sales * np.random.randint(300, 1500)
        sales_data.append([month.strftime('%Y-%m'), product, sales, revenue])
df_sales = pd.DataFrame(sales_data, columns=['Month', 'Product', 'Units Sold', 'Revenue'])
df_sales.to_csv("datasets/sales_data.csv", index=False)

# 2. covid_cases.csv
np.random.seed(0)
days = pd.date_range(start='2020-03-01', periods=100)
states = ['California', 'Texas', 'New York', 'Florida']
covid_data = []
for state in states:
    base = np.random.randint(50, 100)
    cases = base + np.random.poisson(lam=100, size=100).cumsum()
    for i, date in enumerate(days):
        covid_data.append([date.strftime('%Y-%m-%d'), state, cases[i]])
df_covid = pd.DataFrame(covid_data, columns=['Date', 'State', 'Cases'])
df_covid.to_csv("datasets/covid_cases.csv", index=False)

# 3. stock_prices.csv
np.random.seed(1)
dates = pd.date_range('2024-01-01', periods=60)
stocks = ['AAPL', 'GOOG', 'TSLA']
stock_data = []
for stock in stocks:
    price = np.random.uniform(100, 500)
    for date in dates:
        open_price = price + np.random.uniform(-5, 5)
        close_price = open_price + np.random.uniform(-5, 5)
        high = max(open_price, close_price) + np.random.uniform(0, 5)
        low = min(open_price, close_price) - np.random.uniform(0, 5)
        volume = np.random.randint(1000000, 5000000)
        stock_data.append([date.strftime('%Y-%m-%d'), stock, round(open_price, 2), round(close_price, 2),
                           round(high, 2), round(low, 2), volume])
        price = close_price
df_stock = pd.DataFrame(stock_data, columns=['Date', 'Stock', 'Open', 'Close', 'High', 'Low', 'Volume'])
df_stock.to_csv("datasets/stock_prices.csv", index=False)

# 4. weather_data.csv
np.random.seed(10)
cities = ['New York', 'San Francisco', 'Austin', 'Chicago']
dates = pd.date_range(start='2023-06-01', periods=30)
weather_data = []
for city in cities:
    temp = np.random.uniform(15, 35, size=30)
    humidity = np.random.uniform(40, 90, size=30)
    for i, date in enumerate(dates):
        weather_data.append([date.strftime('%Y-%m-%d'), city, round(temp[i], 1), round(humidity[i], 1)])
df_weather = pd.DataFrame(weather_data, columns=['Date', 'City', 'Temperature (C)', 'Humidity (%)'])
df_weather.to_csv("datasets/weather_data.csv", index=False)

print("✅ All datasets generated in 'datasets/' folder.")

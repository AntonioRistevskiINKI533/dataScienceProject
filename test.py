import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# https://www.kaggle.com/datasets/jyotiprasadpal/historical-sales-data-on-daily-basis-of-a-company

store_sales = pd.read_csv("historical_data.csv")

# store_sales = store_sales.drop(store_sales[store_sales.Article_ID != 1132].index)
store_sales = store_sales.drop(['Country_Code','Article_ID'], axis=1)

store_sales['Date'] = pd.to_datetime(store_sales['Date'], format='%Y%m%d')

# store_sales['Date'] = pd.to_datetime(store_sales['Date'])
print(store_sales.head(100))

store_sales['Date'] = store_sales['Date'].dt.to_period("M")
monthly_sales = store_sales.groupby('Date').sum().reset_index()

monthly_sales['Date'] = monthly_sales['Date'].dt.to_timestamp()

# Visualization

plt.figure(figsize=(15,5))
plt.plot(monthly_sales['Date'], monthly_sales['Sold_Units']) ### The date = x axis, sales = y axis
plt.xlabel('Date')
plt.ylabel('Sold_Units')
plt.title('Monthly customer sales')
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

store_sales = pd.read_csv("historical_data.csv")
articles = store_sales.drop(['Date','Country_Code','Sold_Units'], axis=1)
articles = articles.groupby('Article_ID').sum().reset_index()
# fruits = ["apple", "banana", "cherry"]

for ind in articles.index:
  print(articles['Article_ID'][ind])
# store_sales.info()

store_sales332 = store_sales.drop(store_sales[store_sales.Article_ID != 332].index)
store_sales362 = store_sales.drop(store_sales[store_sales.Article_ID != 362].index)
store_sales332 = store_sales332.drop(['Country_Code','Article_ID'], axis=1)
store_sales362 = store_sales362.drop(['Country_Code','Article_ID'], axis=1)
# store_sales.info()

store_sales332['Date'] = pd.to_datetime(store_sales332['Date'], format='%Y%m%d')
store_sales362['Date'] = pd.to_datetime(store_sales362['Date'], format='%Y%m%d')
# store_sales.info()

store_sales332['Date'] = store_sales332['Date'].dt.to_period("M")
store_sales362['Date'] = store_sales362['Date'].dt.to_period("M")
# print(store_sales.head(10))
monthly_sales332 = store_sales332.groupby('Date').sum().reset_index()
monthly_sales362 = store_sales362.groupby('Date').sum().reset_index()
# print(monthly_sales.head(10))
# monthly_sales.info()

monthly_sales332['Date'] = monthly_sales332['Date'].dt.to_timestamp()
monthly_sales362['Date'] = monthly_sales362['Date'].dt.to_timestamp()
# print(monthly_sales.head(10))
# monthly_sales.info()

plt.figure(figsize=(15,5))
line, = plt.plot(monthly_sales332['Date'], monthly_sales332['Sold_Units'], 'r-o', label="Article_ID 332") # 'g--'
line.set_color("yellow")
line2, = plt.plot(monthly_sales362['Date'], monthly_sales362['Sold_Units'], 'r-o', label="Article_ID 362")
plt.xlabel('Date')
plt.ylabel('Sold_Units')
plt.title('Monthly customer sales')
plt.legend()
plt.show()
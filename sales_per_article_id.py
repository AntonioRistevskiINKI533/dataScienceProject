import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

articles_store_sales = pd.read_csv("historical_data.csv")

articles = articles_store_sales.drop(['Date','Country_Code','Sold_Units'], axis=1)
articles = articles.groupby('Article_ID').sum().reset_index()

colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'brown', 'pink', 'teal', 'orange', 'black', 'purple', 'olive', 'gray', 'violet']

plt.figure(figsize=(15, 10))

for ind in articles.index:
  print(articles['Article_ID'][ind])
  # store_sales.info()

  store_sales = pd.read_csv("historical_data.csv")

  store_sales = store_sales.drop(store_sales[store_sales.Article_ID != articles['Article_ID'][ind]].index)
  store_sales = store_sales.drop(['Country_Code','Article_ID'], axis=1)

  store_sales['Date'] = pd.to_datetime(store_sales['Date'], format='%Y%m%d')

  store_sales['Date'] = store_sales['Date'].dt.to_period("M")
  monthly_sales = store_sales.groupby('Date').sum().reset_index()

  monthly_sales['Date'] = monthly_sales['Date'].dt.to_timestamp()

  label_string = "Article_ID "+str(articles['Article_ID'][ind])
  line, = plt.plot(monthly_sales['Date'], monthly_sales['Sold_Units'], 'r-o', label=label_string) # 'g--'
  print(colors[ind])
  line.set_color(colors[ind])

plt.xlabel('Date')
plt.ylabel('Sold_Units')
plt.title('Monthly sales per article')
plt.legend()
plt.show()
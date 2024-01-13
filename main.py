import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

store_sales = pd.read_csv("data.csv")
# print(store_sales.head(10))
# store_sales.info()

store_sales = store_sales.drop(['store','item'], axis=1)
# store_sales.info()

store_sales['date'] = pd.to_datetime(store_sales['date'])
store_sales.info()
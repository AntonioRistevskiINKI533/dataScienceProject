import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series

store_sales = pd.read_csv("data.csv")
# print(store_sales.head(10))
# store_sales.info()

store_sales = store_sales.drop(['store','item'], axis=1)
# store_sales.info()

store_sales['date'] = pd.to_datetime(store_sales['date'])
# store_sales.info()

store_sales['date'] = store_sales['date'].dt.to_period("M")
# print(store_sales.head(10))
monthly_sales = store_sales.groupby('date').sum().reset_index()
# print(monthly_sales.head(10))
# monthly_sales.info()

monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
# print(monthly_sales.head(10))
# monthly_sales.info()

plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Monthly customer sales')
plt.show()

y = pd.DataFrame(monthly_sales)
y.plot()
plt.show()

monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna() # Само за првиот запис нема да може да се пресмета sales_diff, кај неко sales_diff ќе биде NaN, и затоа го отсрануваме со dropna().
# print(monthly_sales.head(10))

plt.figure(figsize=(15,5))
plt.bar(monthly_sales['date'], monthly_sales['sales_diff'], width=12) ### width е ширина на столбовите.
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Monthly customer sales difference")
plt.show()

z = pd.DataFrame(monthly_sales)
z.plot()
plt.show()

y_train, y_test = temporal_train_test_split(y, train_size=0.8) # Можи тука да се најди кој метод не е деприкејтет (0.8 = 80% од y)
fh = ForecastingHorizon(y_test.index, is_relative=False)

#print(fh)
# ModuleNotFoundError: ThetaForecaster requires package 'statsmodels' to be present in the python environment, but 'statsmodels' was not found. 'statsmodels' is a soft dependency and not included in the base sktime installation. Please run: `pip install statsmodels` to install the statsmodels package. To install all soft dependencies, run: `pip install sktime[all_extras]`
# Во терминал со патеката од проектот треба да се направи pip install statsmodels


forecaster = ThetaForecaster(sp=12) #sp = seasonal period (1 = yearly, 4 = quarterly, 12 = monthly)
# y_train.set_index('Date')
print(y_train.head(10))
y_train.info()
forecaster.fit(y_train)

y_pred = forecaster.predict(fh)

# print(y_pred)

map = mean_absolute_percentage_error(y_test, y_pred)
# print(map)

plot_series(y, y_pred)

plt.show()
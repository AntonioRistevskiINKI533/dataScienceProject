import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series

store_sales = pd.read_csv("historical_data.csv")
# print(store_sales.head(10))

# Ја користиме info() функцијата за да провериме дали има null вредности во dataset-от
# Во count важно е да пишува non-null, ако така пишува тогаш е вод ред.

# store_sales.info()

# Ги отфрламе колоните Article_ID и Country_Code

store_sales = store_sales.drop(['Country_Code','Article_ID'], axis=1)
# store_sales.info()

# конвертираме колоната Date од тим integer во тип datetime

store_sales['Date'] = pd.to_datetime(store_sales['Date'], format='%Y%m%d')
# store_sales.info()

# Конвертираме Date во месечен период, и потоа ги сумираме бројот на продадени единици за секој месец

store_sales['Date'] = store_sales['Date'].dt.to_period("M")
# print(store_sales.head(10))
monthly_sales = store_sales.groupby('Date').sum().reset_index()
# print(monthly_sales.head(10))
# monthly_sales.info()

# Резултатната Date колона ја конвертираме во timespan datatype

monthly_sales['Date'] = monthly_sales['Date'].dt.to_timestamp()
# print(monthly_sales.head(10))
# monthly_sales.info()

# Прикажување во форма на граф

plt.figure(figsize=(15,5))
plt.plot(monthly_sales['Date'], monthly_sales['Sold_Units'])
plt.xlabel('Date')
plt.ylabel('Sold_Units')
plt.title('Monthly customer sales')
plt.show()

y = pd.DataFrame(monthly_sales)
y.plot()
plt.show()

# Пресметување разлика во продажби за секој месец

monthly_sales['Sales_Diff'] = monthly_sales['Sold_Units'].diff()
monthly_sales = monthly_sales.dropna() # Само за првиот запис нема да може да се пресмета sales_diff, кај неко sales_diff ќе биде NaN, и затоа го отсрануваме со dropna().
# print(monthly_sales.head(10))

plt.figure(figsize=(15,5))
plt.bar(monthly_sales['Date'], monthly_sales['Sales_Diff'], width=12) ### width е ширина на столбовите.
plt.xlabel("Date")
plt.ylabel("Sold_Units")
plt.title("Monthly customer sales difference")
plt.show()

z = pd.DataFrame(monthly_sales)
z.plot()
plt.show()

y_train, y_test = temporal_train_test_split(y, train_size=0.94) # Можи тука да се најди кој метод не е деприкејтет (0.8 = 80% од y)
# Ова за според davebellaar mora 0.94 a ne 0.8 inaku se javuva: ValueError: x must have 2 complete cycles requires 24 observations. x only has 23 observation(s)
fh = ForecastingHorizon(y_test.index, is_relative=False)

#print(fh)
# ModuleNotFoundError: ThetaForecaster requires package 'statsmodels' to be present in the python environment, but 'statsmodels' was not found. 'statsmodels' is a soft dependency and not included in the base sktime installation. Please run: `pip install statsmodels` to install the statsmodels package. To install all soft dependencies, run: `pip install sktime[all_extras]`
# Во терминал со патеката од проектот треба да се направи pip install statsmodels


forecaster = ThetaForecaster(sp=12) #sp = seasonal period (1 = yearly, 4 = quarterly, 12 = monthly)
# y_train.set_index('Date')
# y_train['Date'] = y_train['Date'].strftime("%d/%m/%Y")
print(y_train.head(1000))
y_train.info()
forecaster.fit(y_train)

y_pred = forecaster.predict(fh)

# print(y_pred)

map = mean_absolute_percentage_error(y_test, y_pred)
# print(map)

plot_series(y, y_pred)

plt.show()
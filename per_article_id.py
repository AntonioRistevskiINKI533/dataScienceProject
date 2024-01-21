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
# print(store_sales.head(10))

# Ја користиме info() функцијата за да провериме дали има null вредности во dataset-от
# Во count важно е да пишува non-null, ако така пишува тогаш е вод ред.

# store_sales.info()

# Ги отфрламе колоните Article_ID и Country_Code

store_sales = store_sales.drop(store_sales[store_sales.Article_ID != 3417].index)
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

print('monthly_sales.head(1000)')
print(monthly_sales.head(1000))
print('monthly_sales.head(1000)')
supervised_data = monthly_sales.drop(['Date', 'Sold_Units'], axis=1)
### print(supervised_data.head(10))

# Подготовка на supervised data
### These datasets are designed to train or “supervise” algorithms into classifying data or predicting outcomes accurately. Using labeled inputs and outputs, the model can measure its accuracy and learn over time.

for i in range(1,13):
    col_name = 'month_' + str(i)
    supervised_data[col_name] = supervised_data['Sales_Diff'].shift(i) # The `DataFrame.shift()` function in Pandas is a method that shifts the values of a DataFrame along a specified axis.
### print(supervised_data.head(10))
supervised_data = supervised_data.dropna().reset_index(drop=True) ### replaces the previous DataFrame index with the new index provided by . reset_index() , otherwise it sets the new index in front of the old index.
## print(supervised_data.head()) ############NEVADI KO SO SE KAJNEGO, POINAKU SE SVRTENI NEKAKO

# Делење на податоците во train и test

print('supervised_data.head(1000)')
print(supervised_data.head(1000))
print('supervised_data.head(1000)')
train_data = supervised_data[:-8] ### This is for the previous 12 months (сите освем последните 12)
test_data = supervised_data[-8:] ### This is for the comming 12 months (само последните 12)
## print("Train data shape", train_data.shape) ### The shape of an array is the number of elements in each dimension.
### print(train_data.head(100)) ### Ги содржи сите редови од supervised_data - индекс 0 до 34 (вкупно 36, сите без последните 12)
## print("Test data shape", test_data.shape)
### print(test_data.head(100)) ### Ги содржи сите редови од supervised_data - индекс 35 до 46 (вкупно 12)

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

X_train, y_train = train_data[:,1:], train_data[:,0:1] ### In the supervised dataframe the first column allways coresponds to the output and the remaining columns act as the input features
X_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()
## print("X_train Shape: ", X_train.shape) ### значи 23 редови а 12 колони
## print("y_train Shape: ", y_train.shape)
## print("X_test Shape: ", X_test.shape)
## print("y_test Shape: ", y_test.shape)

### X е матрица, y е само една линијам една димензија
### But we generally use 'X' instead of 'x' bcz of mathematical representation of matrix using uppercase letter. And here X represents the features matrix. The features matrix is assumed to be two-dimensional, with shape [n_samples, n_features]. And y is represented as target array i.e assumed to be one-dimensional.

# Make prediction data frame to merge the predicted sales prices of all trainer algoritams

sales_dates = monthly_sales['Date'][-8:].reset_index(drop=True) # Само последните 12 месеци.
predict_df = pd.DataFrame(sales_dates)

act_sales = monthly_sales['Sold_Units'][-9:].to_list() # Само последните 13 месеци.
## print(act_sales)

# За да се креира моделот на линеарна регресија и предиктираниот output

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pre = lr_model.predict(X_test)
### print(lr_pre)
### print(X_test)

lr_pre = lr_pre.reshape(-1,1)
# This is a set matrix - contains the input features of the test data, and also the predicted output
lr_pre_test_set = np.concatenate([lr_pre, X_test], axis=1)
### print(lr_pre_test_set)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set) ### Undo the scaling of X according to feature_range.

result_list =[]
for index in range(0, len(lr_pre_test_set)):
    result_list.append(lr_pre_test_set[index][0] + act_sales[index]) ### index - iterator,  [0] - the first index of the iterator
lr_pre_series = pd.Series(result_list, name="Linear Prediction")
predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True)

# print(predict_df)
lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'], monthly_sales['Sold_Units'][-8:])) ### sqrt = square root.
lr_mae = mean_absolute_error(predict_df['Linear Prediction'], monthly_sales['Sold_Units'][-8:]) # -12: - Само последните 12 месеци.
lr_r2 = r2_score(predict_df['Linear Prediction'], monthly_sales['Sold_Units'][-8:])
print("Liner Regression MSE: ", lr_mse)
print("Liner Regression MAE: ", lr_mae)
print("Liner Regression R2: ", lr_r2)

# Визуелизација на предикцијата спрема вистинската продажба

plt.figure(figsize=(15,5))
# Actual sales
plt.plot(monthly_sales['Date'], monthly_sales['Sold_Units'])
# Predicted sales
plt.plot(predict_df['Date'], predict_df['Linear Prediction'])
plt.title("Customer sales forecast using LR model")
plt.xlabel("Date")
plt.ylabel("Sold_Units")
plt.legend(['Actual Sales', 'Predicted sales'])
plt.show()
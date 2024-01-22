import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dateutil.relativedelta import relativedelta

prediction_months = 0
while int(prediction_months) == False: # Проверка дали е int внесениот број.
    prediction_months = input("Внесете број на месеци кој ќе се предвидуваат\n")
    if (int(prediction_months) == False):
        print('Невалиден внес, обидетесе повторно')

prediction_months = int(prediction_months)

predict_in_future = ''
while predict_in_future != 'F' and predict_in_future != 'P':
    predict_in_future = input("За предикција во иднина внесете F, инаку внесете: P\n")
    if predict_in_future != 'F' and predict_in_future != 'P':
        print('Невалиден внес, обидетесе повторно')

countries_store_sales = pd.read_csv("historical_data.csv")

countries = countries_store_sales.drop(['Date','Article_ID','Sold_Units'], axis=1)
countries = countries.groupby('Country_Code').sum().reset_index()

colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'brown', 'pink', 'teal', 'orange', 'black', 'purple', 'olive', 'gray', 'violet']

plt.figure(figsize=(15, 10))

for ind in countries.index:
  print(countries['Country_Code'][ind])
  # store_sales.info()

  store_sales = pd.read_csv("historical_data.csv")

  store_sales = store_sales.drop(store_sales[store_sales.Country_Code != countries['Country_Code'][ind]].index)
  store_sales = store_sales.drop(['Country_Code','Article_ID'], axis=1)

  store_sales['Date'] = pd.to_datetime(store_sales['Date'], format='%Y%m%d')

  store_sales['Date'] = store_sales['Date'].dt.to_period("M")
  monthly_sales = store_sales.groupby('Date').sum().reset_index()

  monthly_sales['Date'] = monthly_sales['Date'].dt.to_timestamp()

  label_string = "Country_Code "+str(countries['Country_Code'][ind])
  line, = plt.plot(monthly_sales['Date'], monthly_sales['Sold_Units'], 'r-o', label=label_string) # 'g--'
  print(colors[ind])
  line.set_color(colors[ind])

  # PREDICTION

  monthly_sales['Sales_Diff'] = monthly_sales['Sold_Units'].diff()
  monthly_sales = monthly_sales.dropna()

  supervised_data = monthly_sales.drop(['Date', 'Sold_Units'], axis=1)

  for i in range(1, 13):
    col_name = 'month_' + str(i)
    supervised_data[col_name] = supervised_data['Sales_Diff'].shift(i)
  supervised_data = supervised_data.dropna().reset_index(drop=True)

  if (predict_in_future == 'F'):
    train_data = supervised_data  ### This is for the previous 12 months (сите освем последните 12)
  elif (predict_in_future == 'P'):
    train_data = supervised_data[:-prediction_months] ### This is for the previous 12 months (сите освем последните 12)

  test_data = supervised_data[-prediction_months:] ### This is for the comming 12 months (само последните 12)

  scaler = MinMaxScaler(feature_range=(-1, 1))
  scaler.fit(train_data)
  train_data = scaler.transform(train_data)
  test_data = scaler.transform(test_data)

  X_train, y_train = train_data[:,1:], train_data[:,0:1]
  X_test, y_test = test_data[:,1:], test_data[:,0:1]
  y_train = y_train.ravel()
  y_test = y_test.ravel()

  sales_dates = monthly_sales['Date'][-prediction_months:].reset_index(drop=True)  # Само последните 12 месеци.
  predict_df = pd.DataFrame(sales_dates)

  if (predict_in_future == 'F'):
    for i in range(0, predict_df.shape[0]):
      predict_df['Date'][i] = predict_df['Date'][i] + relativedelta(months=predict_df.shape[0])

  act_sales = monthly_sales['Sold_Units'][-(prediction_months):].to_list()  # Само последните 13 месеци.

  lr_model = LinearRegression()
  lr_model.fit(X_train, y_train)
  lr_pre = lr_model.predict(X_test)

  lr_pre = lr_pre.reshape(-1, 1)
  lr_pre_test_set = np.concatenate([lr_pre, X_test], axis=1)
  lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

  result_list = []
  for index in range(0, len(lr_pre_test_set)):
    result_list.append(
      lr_pre_test_set[index][0] + act_sales[index])
  lr_pre_series = pd.Series(result_list, name="Linear Prediction")
  predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True)

  lr_mse = np.sqrt(
    mean_squared_error(predict_df['Linear Prediction'], monthly_sales['Sold_Units'][-prediction_months:]))
  lr_mae = mean_absolute_error(predict_df['Linear Prediction'],
                               monthly_sales['Sold_Units'][-prediction_months:])
  lr_r2 = r2_score(predict_df['Linear Prediction'], monthly_sales['Sold_Units'][-prediction_months:])
  print("Liner Regression MSE: ", lr_mse)
  print("Liner Regression MAE: ", lr_mae)
  print("Liner Regression R2: ", lr_r2)

  # plt.figure(figsize=(15, 5))
  # Вистински продажби
  # plt.plot(monthly_sales['Date'], monthly_sales['Sold_Units'])

  # Додавање нова на почетокот за да се спојат Вистинската продажба и предвидената

  Date = (predict_df['Date'][0] - relativedelta(months=1))
  Sold_Units = monthly_sales.loc[monthly_sales['Date'] == Date].iloc[0]['Sold_Units']

  starting_point_row = pd.DataFrame(columns=('Date', 'Linear Prediction'))
  starting_point_row.loc[len(starting_point_row.index)] = [Date, Sold_Units]

  predict_df = pd.concat([starting_point_row, predict_df.loc[:]]).reset_index(drop=True)

  # Предвидени продажби
  label_string = "Country_Code " + str(countries['Country_Code'][ind] + "(Prediction)")
  line, = plt.plot(predict_df['Date'], predict_df['Linear Prediction'], 'g--', label=label_string)  # 'g--'
  print(colors[ind])
  line.set_color(colors[ind])
  # plt.title("Customer sales forecast using LR model")
  # plt.xlabel("Date")
  # plt.ylabel("Sold_Units")
  # plt.legend(['Actual Sales', 'Predicted sales'])
  # plt.show()

plt.title("Customer sales forecast using LR model")
plt.xlabel('Date')
plt.ylabel('Sold_Units')
plt.legend(['Actual Sales', 'Predicted sales'])
plt.show()
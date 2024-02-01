import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dateutil.relativedelta import relativedelta

input_for_column = ''
while input_for_column != 'A' and input_for_column != 'C':
    input_for_column = input("За предвидување за сите производи (Article_ID) внесете A, додека пак за држави (Country_Code) внесете: C\n")
    if input_for_column != 'A' and input_for_column != 'C':
        print('Невалиден внес, обидетесе повторно')

column = ''
other_column = ''
if (input_for_column == 'A'):
  column = 'Article_ID'
  other_column = 'Country_Code'
elif (input_for_column == 'C'):
  column = 'Country_Code'
  other_column = 'Article_ID'

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

elements_store_sales = pd.read_csv("historical_data.csv")

elements = elements_store_sales.drop(['Date',other_column,'Sold_Units'], axis=1)
elements = elements.groupby(column).sum().reset_index()

colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'brown', 'pink', 'teal', 'orange', 'black', 'purple', 'olive', 'gray', 'violet']

plt.figure(figsize=(15, 10))

for ind in elements.index:
  # store_sales.info()

  store_sales = pd.read_csv("historical_data.csv")

  if (input_for_column == 'C'):
    store_sales = store_sales.drop(store_sales[store_sales.Country_Code != elements['Country_Code'][ind]].index)
  elif (input_for_column == 'A'):
    store_sales = store_sales.drop(store_sales[store_sales.Article_ID != elements['Article_ID'][ind]].index)

  store_sales = store_sales.drop(['Country_Code','Article_ID'], axis=1)

  store_sales['Date'] = pd.to_datetime(store_sales['Date'], format='%Y%m%d')

  store_sales['Date'] = store_sales['Date'].dt.to_period("M")
  monthly_sales = store_sales.groupby('Date').sum().reset_index()

  monthly_sales['Date'] = monthly_sales['Date'].dt.to_timestamp()

  label_string = column+" "+str(elements[column][ind])
  line, = plt.plot(monthly_sales['Date'], monthly_sales['Sold_Units'], 'r-o', label=label_string) # 'g--'
  line.set_color(colors[ind])

  # PREDICTION

  monthly_sales['Sales_Diff'] = monthly_sales['Sold_Units'].diff()
  monthly_sales = monthly_sales.dropna()

  supervised_data = monthly_sales.drop(['Date', 'Sold_Units'], axis=1)

  for i in range(1, (prediction_months+1)):
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

  act_sales = monthly_sales['Sold_Units'][-(prediction_months+1):].to_list()  # Само последните 13 месеци.

  lr_model = LinearRegression()
  lr_model.fit(X_train, y_train)
  lr_pre = lr_model.predict(X_test)

  lr_pre = lr_pre.reshape(-1, 1)
  lr_pre_test_set = np.concatenate([lr_pre, X_test], axis=1)
  lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

  result_list = []
  for index in range(0, len(lr_pre_test_set)):
    result_list.append(lr_pre_test_set[index][0] + act_sales[index])
  lr_pre_series = pd.Series(result_list, name="Linear Prediction")
  predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True)

  if (predict_in_future == 'P'):
    lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'], monthly_sales['Sold_Units'][-prediction_months:]))
    lr_mae = mean_absolute_error(predict_df['Linear Prediction'], monthly_sales['Sold_Units'][-prediction_months:])
    lr_r2 = r2_score(predict_df['Linear Prediction'], monthly_sales['Sold_Units'][-prediction_months:])
    print("MSE (Mean squared error): ", lr_mse)
    print("MAE (Mean absolute error): ", lr_mae)
    print("R2 (R square): ", lr_r2)

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
  label_string = label_string + " (предвидена продажба)"
  line, = plt.plot(predict_df['Date'], predict_df['Linear Prediction'], 'g--', label=label_string)  # 'g--'
  line.set_color(colors[ind])
  # plt.title("Customer sales forecast using LR model")
  # plt.xlabel("Date")
  # plt.ylabel("Sold_Units")
  # plt.legend(['Actual Sales', 'Predicted sales'])
  # plt.show()

plt.title("Предвидување на продажби со модел на линеарна регресија")
plt.xlabel('Date')
plt.ylabel('Sold_Units')
plt.legend()
plt.show()
import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

input_for_column = ''
while input_for_column != 'S' and input_for_column != 'I':
    input_for_column = input(
        "За предвидување за сите продавници (store) внесете S, додека пак за производ (item) внесете I\n")
    if input_for_column != 'S' and input_for_column != 'I':
        print('Невалиден внес, обидетесе повторно')

column = ''
other_column = ''
if (input_for_column == 'S'):
    plt.figure(figsize=(30, 15))
    column = 'store'
    other_column = 'item'
elif (input_for_column == 'I'):
    plt.figure(figsize=(60, 40))
    column = 'item'
    other_column = 'store'

prediction_months = 0
while int(prediction_months) == False:  # Проверка дали е int внесениот број.
    prediction_months = input("Внесете број на месеци кој ќе се предвидуваат\n")
    if (int(prediction_months) == False):
        print('Невалиден внес, обидетесе повторно')

prediction_months = int(prediction_months)

predict_in_future = ''
while predict_in_future != 'F' and predict_in_future != 'P':
    predict_in_future = input("За предикција во иднина внесете F, инаку внесете: P\n")
    if predict_in_future != 'F' and predict_in_future != 'P':
        print('Невалиден внес, обидетесе повторно')

data = pd.read_csv("data.csv")

elements = data.drop(['date', other_column, 'sales'], axis=1)
elements = elements.groupby(column).sum().reset_index()

colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'brown', 'pink', 'teal', 'orange', 'black', 'purple','olive', 'gray', 'violet',
          'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'brown', 'pink', 'teal', 'orange', 'black', 'purple','olive', 'gray', 'violet',
          'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'brown', 'pink', 'teal', 'orange', 'black', 'purple','olive', 'gray', 'violet',
          'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'brown', 'pink', 'teal', 'orange', 'black', 'purple','olive', 'gray', 'violet']

for ind in elements.index:

    data = pd.read_csv("data.csv")

    if (input_for_column == 'S'):
        data = data.drop(data[data.store != elements['store'][ind]].index)
    elif (input_for_column == 'I'):
        data = data.drop(data[data.item != elements['item'][ind]].index)

    data['date'] = pd.to_datetime(data['date'])  # , format='%Y%m%d'
    data['date'] = data['date'].dt.to_period("M")
    data = data.groupby('date').sum().reset_index()
    data['date'] = data['date'].dt.to_timestamp()
    data = data.drop(['store', 'item'], axis=1)
    data.columns = ['ds', 'y']

    if (predict_in_future == 'P'):
        scaler = StandardScaler()
        scaled_data = data.copy()
        scaled_data['y'] = scaler.fit_transform(data[['y']])

    model = Prophet()
    last_date = data['ds'].max()

    label_string = column + " " + str(elements[column][ind])
    label_string2 = label_string + " (предвидена продажба)"

    if (predict_in_future == 'P'):
        model.fit(scaled_data)
        future = model.make_future_dataframe(periods=0, freq='M')
        future = future[future['ds'] > last_date - pd.DateOffset(months=prediction_months)]
        forecast = model.predict(future)

        actual_values = scaled_data[scaled_data['ds'] > last_date - pd.DateOffset(months=prediction_months)]['y'].values
        predicted_values = forecast[forecast['ds'] > last_date - pd.DateOffset(months=prediction_months)]['yhat'].values

        mse = mean_squared_error(actual_values, predicted_values)
        print("Mean Squared Error за последните " + str(prediction_months) + " месеци:", mse)
        mae = mean_absolute_error(actual_values, predicted_values)
        print("Mean Absolute Error за последните " + str(prediction_months) + " месеци:", mae)
        r2 = r2_score(actual_values, predicted_values)
        print("R-squared за последните " + str(prediction_months) + " месеци:", r2, "\n---")

        actual_values = scaler.inverse_transform(actual_values.reshape(-1, 1)).flatten()
        predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()

        line1, = plt.plot(data['ds'], data['y'], 'r-o', label=label_string)
        line2, = plt.plot(forecast['ds'], predicted_values, 'g--', label=label_string2)
        line1.set_color(colors[ind])
        line2.set_color(colors[ind])

        # Plot actual vs. predicted values for the last 12 months
        # plt.plot(data[data['ds'] > last_date - pd.DateOffset(months=prediction_months)]['ds'], actual_values) # ,label='Реални продажби'
        # plt.plot(forecast[forecast['ds'] > last_date - pd.DateOffset(months=prediction_months)]['ds'], predicted_values) # ,label='Предвидени продажби'

    elif (predict_in_future == 'F'):
        model.fit(data)
        # Make future dataframe for predictions limited to the last 12 months
        future = model.make_future_dataframe(periods=prediction_months, freq='M')  # Predicting for 365 days into future
        # Generate predictions
        forecast = model.predict(future)

        # Filter data between two dates
        filtered_forecast = forecast.loc[(forecast['ds'] >= last_date)]# 1e6 значи еден милион *

        # Plot actual vs. predicted values for the last 12 months
        line1, = plt.plot(data['ds'], data['y'], 'r-o', label=label_string)
        line2, = plt.plot(filtered_forecast['ds'], filtered_forecast['yhat'], 'g--', label=label_string2)
        line1.set_color(colors[ind])
        line2.set_color(colors[ind])

plt.title("Предвидување на продажби")
plt.xlabel('Датум')
plt.ylabel('Продажби')
plt.legend()
plt.show()
# pip install fbprophet
# pip install Cython
# pip install pystan
# pip install prophet --no-binary :all:
# pip install plotly

import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

# Load your data
data = pd.read_csv('data.csv')

filter_by_column = ''
while filter_by_column != 'S' and filter_by_column != 'I' and filter_by_column != 'N':
    filter_by_column = input("За филтрирање по store внесете S, додека пак за по item внесете I, ако сакате да нема филтрирање внесете N\n")
    if filter_by_column != 'S' and filter_by_column != 'I' and filter_by_column != 'N':
        print('Невалиден внес, обидетесе повторно')

if filter_by_column == "S":
    filter_by_row = input("Внесете Id на store:\n")
    filter_by_row = np.int64(filter_by_row)
    data = data.drop(data[data.store != filter_by_row].index)
elif filter_by_column == "I":
    filter_by_row = input("Внесете Id на item:\n")
    filter_by_row = np.int64(filter_by_row)
    data = data.drop(data[data.item != filter_by_row].index)

prediction_months = 0
while int(prediction_months) == False or int(prediction_months) < 1: # Проверка дали е позитивен int внесениот број.
    prediction_months = input("Внесете број на месеци кој ќе се предвидуваат\n")
    if (int(prediction_months) == False or int(prediction_months) < 1):
        print('Невалиден внес, обидетесе повторно')

prediction_months = int(prediction_months)

predict_in_future = ''
while predict_in_future != 'F' and predict_in_future != 'P':
    predict_in_future = input("За предикција во иднина внесете F, инаку внесете: P\n")
    if predict_in_future != 'F' and predict_in_future != 'P':
        print('Невалиден внес, обидетесе повторно')



data['date'] = np.array(pd.to_datetime(data['date'])) # format='%Y%m%d'
data['date'] = data['date'].dt.to_period("M")
data = data.groupby('date').sum().reset_index()
data['date'] = data['date'].dt.to_timestamp()
# frame = df[df['store'] == 1].copy()
# data = data.copy()
data.drop(['store', 'item'], axis=1, inplace=True)
data.columns = ['ds', 'y']

if (predict_in_future == 'P'):
    # Scale the 'y' values
    scaler = StandardScaler()
    scaled_data = data.copy()
    scaled_data['y'] = scaler.fit_transform(data[['y']])

# Instantiate Prophet model
model = Prophet() # interval_width=0.4 0.95
# Determine the last date in the existing data
last_date = data['ds'].max()

# Fit the model
if (predict_in_future == 'P'):
    model.fit(scaled_data)
    # Make future dataframe for predictions limited to the last 12 months
    future = model.make_future_dataframe(periods=0, freq='M')
    future = future[future['ds'] > last_date - pd.DateOffset(months=prediction_months)]
    forecast = model.predict(future)

    # Extract actual values for the last 12 months
    actual_values = scaled_data[scaled_data['ds'] > last_date - pd.DateOffset(months=prediction_months)]['y'].values
    # Extract predicted values for the last 12 months
    predicted_values = forecast[forecast['ds'] > last_date - pd.DateOffset(months=prediction_months)]['yhat'].values

    #actual_values = scaler.fit_transform(actual_values.reshape(-1, 1))
    #predicted_values = scaler.fit_transform(predicted_values.reshape(-1, 1))
    # Calculate mean squared error for the last 12 months
    mse = mean_squared_error(actual_values, predicted_values)
    print("Mean Squared Error за последните "+str(prediction_months)+" месеци:", mse)
    # Calculate mean absolute error for the last 12 months
    mae = mean_absolute_error(actual_values, predicted_values)
    print("Mean Absolute Error за последните "+str(prediction_months)+" месеци:", mae)
    # Calculate R-squared score for the last 12 months
    r2 = r2_score(actual_values, predicted_values)
    print("R-squared за последните "+str(prediction_months)+" месеци:", r2)

    # Inverse transform predicted values to original scale
    actual_values = scaler.inverse_transform(actual_values.reshape(-1, 1)).flatten()
    predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()

    plt.plot(data['ds'], data['y'], label='Реални продажби')
    plt.plot(forecast['ds'], predicted_values, label='Предвидени продажби')
    plt.xlabel('Датум')
    plt.ylabel('Продажби')
    plt.legend()
    plt.show()

    # Plot actual vs. predicted values for the last 12 months
    plt.figure(figsize=(10, 5))
    plt.plot(data[data['ds'] > last_date - pd.DateOffset(months=prediction_months)]['ds'], actual_values, label='Реални продажби')
    plt.plot(forecast[forecast['ds'] > last_date - pd.DateOffset(months=prediction_months)]['ds'], predicted_values, label='Предвидени продажби')
    plt.xlabel('Датум')
    plt.ylabel('Продажби')
    plt.legend()
    plt.show()

    plot1 = model.plot(forecast)
    plot1.show()
    plot2 = model.plot_components(forecast)
    plot2.show()

elif (predict_in_future == 'F'):
    model.fit(data)
    # Make future dataframe for predictions limited to the last 12 months
    future = model.make_future_dataframe(periods=prediction_months, freq='M')  # Predicting for 365 days into future
    # Generate predictions
    forecast = model.predict(future)

    # Filter data between two dates
    filtered_forecast = forecast.loc[(forecast['ds'] >= last_date)] # 1e6 значи еден милион *
    # Plot actual vs. predicted values for the last 12 months
    plt.plot(data['ds'], data['y'], label='Реални продажби')
    plt.plot(filtered_forecast['ds'], filtered_forecast['yhat'], label='Предвидени продажби')
    plt.xlabel('Датум')
    plt.ylabel('Продажби')
    plt.legend()
    plt.show()

    plot1 = model.plot(forecast)
    plot1.show()
    plot2 = model.plot_components(forecast)
    plot2.show()
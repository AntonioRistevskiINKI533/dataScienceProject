import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import r2_score
from sktime.performance_metrics.forecasting import mean_squared_error, mean_absolute_error

df = pd.read_csv("data.csv")

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

df['Date'] = np.array(pd.to_datetime(df['Date'])) # format='%Y%m%d'

###
df['Date'] = df['Date'].dt.to_period("M")
df = df.groupby('Date').sum().reset_index()
df['Date'] = df['Date'].dt.to_timestamp()
###

# frame = df[df['Article_ID'] == 1].copy()
frame = df.copy()

frame.drop(['Article_ID', 'Country_Code'], axis=1, inplace=True)
frame.columns = ['ds', 'y']

m = Prophet() # interval_width=0.4 0.95
training_run = m.fit(frame)

# Determine the last date in the existing data
last_date = frame['ds'].max()
#print(frame.head(100))
#print(last_date)

future = m.make_future_dataframe(periods=0, freq='M')
#print(future.head(100))
future = future[future['ds'] > last_date - pd.DateOffset(months=12)]
#print(future.head(100))

# future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)
forecast.head()

import matplotlib.pyplot as plt

plt.plot(frame['ds'], frame['y'], label='Actual Data')
plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Data')

plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()

# Show plot
plt.show()

# plot1 = m.plot(forecast)
# plot1.show()

# Extract actual values for the last 12 months
actual_values_last_12_months = frame[frame['ds'] > last_date - pd.DateOffset(months=12)]['y'].values
#print("actual_values_last_12_months")
#print(actual_values_last_12_months)

# Extract predicted values for the last 12 months
predicted_values_last_12_months = forecast[forecast['ds'] > last_date - pd.DateOffset(months=12)]['yhat'].values
#print("predicted_values_last_12_months")
#print(predicted_values_last_12_months)




sampleArray = np.array(actual_values_last_12_months)
print("Our initial array: ", str(actual_values_last_12_months))
print("Original type: " + str(type(actual_values_last_12_months[0])))

# Note usage of astype() function
# np.float can be changed to represent differing types
convertedArray = sampleArray.astype(np.int_)

print("Our final array: ", str(convertedArray))
print("Final type: " + str(type(convertedArray[0])))

sampleArray2 = np.array(predicted_values_last_12_months)
print("Our initial array: ", str(predicted_values_last_12_months))
print("Original type: " + str(type(predicted_values_last_12_months[0])))

# Note usage of astype() function
# np.float can be changed to represent differing types
convertedArray2 = sampleArray2.astype(np.int_)

print("Our final array: ", str(convertedArray2))
print("Final type: " + str(type(convertedArray2[0])))


print(convertedArray)
print(convertedArray2)

# Calculate mean squared error for the last 12 months
mse_last_12_months = mean_squared_error(convertedArray2, convertedArray)
print("Mean Squared Error for the last 12 months:", mse_last_12_months)

# Calculate mean absolute error for the last 12 months
mae_last_12_months = mean_absolute_error(convertedArray2, convertedArray)
print("Mean Absolute Error for the last 12 months:", mae_last_12_months)

# Calculate R-squared score for the last 12 months
r2_last_12_months = r2_score(convertedArray, convertedArray2)
print("R-squared score for the last 12 months:", r2_last_12_months)





plt.plot(frame[frame['ds'] > last_date - pd.DateOffset(months=12)]['ds'], actual_values_last_12_months, label='Actual')
plt.plot(forecast[forecast['ds'] > last_date - pd.DateOffset(months=12)]['ds'], predicted_values_last_12_months, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.plot(convertedArray)
plt.plot(convertedArray2)
plt.show()
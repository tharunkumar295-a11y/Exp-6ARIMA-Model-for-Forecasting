# Experiment 6: ARIMA Model for Forecasting

## Aim

To evaluate predictive models using performance metrics.

## Algorithm

1.  **START**
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  **STOP**



## Program
# Experiment 6: ARIMA Model for Forecasting

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Sample time-series data
data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119,
        104, 118, 115, 126, 141, 135, 125, 149, 170, 170,
        158, 133, 114, 140]

# Create Series
series = pd.Series(data)

# Split into train and test sets
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Fit ARIMA Model (p=2, d=1, q=2)
model = ARIMA(train, order=(2, 1, 2))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Evaluate Forecast
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)

print("Forecasted Values:")
print(forecast)

print("\nRMSE:", rmse)

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Test Data')
plt.plot(test.index, forecast, label='Forecasted Data', color='red')
plt.title('ARIMA Forecasting')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


## Output
<img width="965" height="692" alt="image" src="https://github.com/user-attachments/assets/7de7d4fb-6c0c-4e3f-bd11-bf5348c81bf6" />



## Result
The ARIMA model was successfully implemented for time-series forecasting. Future values were predicted using historical observations, and the forecasting performance was evaluated using Root Mean Squared Error (RMSE). The forecasted values and visualization demonstrated the effectiveness of ARIMA in modeling time-series data.

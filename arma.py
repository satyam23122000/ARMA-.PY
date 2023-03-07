import pandas as pd

data = pd.read_csv(r'C:\Users\satya\OneDrive\Desktop\nifty50.csv')
data.set_index('Date', inplace=True)

import matplotlib.pyplot as plt

plt.plot(data['Close'])
plt.show()

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    # Determine rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # Plot rolling statistics
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend()
    plt.show()
    
    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    df_test = adfuller(timeseries, autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '# Lags Used', 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value
    print(df_output)
    
test_stationarity(data['Close'])

# Apply first-order differencing
data_diff = data.diff().dropna()

test_stationarity(data_diff['Close'])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data_diff, lags=20)
plot_pacf(data_diff, lags=20)
plt.show()

from statsmodels.tsa.arima_model import ARMA

model = ARMA(data_diff['Close'], order=(1, 1))
results = model.fit()
predictions = results.predict(start=len(data_diff), end=len(data_diff))
print(predictions)

train_data = data_diff.iloc[:-30]
test_data = data_diff.iloc[-30:]
model = ARMA(train_data['Close'], order=(1, 1))
results = model.fit()
predictions = results.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(test_data['Close'], predictions)
print('MSE:', mse)






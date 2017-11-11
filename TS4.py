# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:22:54 2017

@author: Andrew
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    plt.close('all')
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

rcParams['figure.figsize'] = 15, 6
parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

DIRECTORY = os.path.dirname('C:/Users/Andrew/Downloads/skills_assessment/')#ran out of time to do relative directory!
training_ts4 = pd.read_csv(os.path.join(DIRECTORY, 'ts4.csv'), parse_dates=[0], index_col='date', date_parser=parser)
test_ts4 = pd.read_csv(os.path.join(DIRECTORY, 'ts4_test.csv'), parse_dates=[0], index_col='date', date_parser=parser)

training_ts4_pd = training_ts4['target']
test_ts4_pd = test_ts4['target']

test_stationarity(training_ts4_pd)

lag_acf = acf(training_ts4_pd, nlags=20)
lag_pacf = pacf(training_ts4_pd, nlags=20, method='ols')

#acf plot: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(training_ts4_pd)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(training_ts4_pd)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.show()
plt.close('all')

#pacf plot:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(training_ts4_pd)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(training_ts4_pd)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()
plt.close('all')



### model fits on training
model = ARIMA(training_ts4_pd, order=(2, 0, 7))
results_AR = model.fit(disp=-1)
print(results_AR.summary())  
plt.plot(training_ts4_pd)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-training_ts4_pd)**2))
plt.show()
plt.close('all')

### model fits on testing
history = [x for x in training_ts4_pd]
predictions = list()
for t in range(len(test_ts4_pd)):
	model = ARIMA(history, order=(2, 0, 7))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_ts4_pd[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test_ts4_pd, predictions)
print('Test MSE: %.3f' % error)


pred_df = pd.DataFrame(np.array(predictions).reshape(len(predictions), 1))
test_ts4_orig = pd.read_csv(os.path.join(DIRECTORY, 'ts4_test.csv'))
test_ts4_orig['target'] = pred_df[0]

#test_ts4_orig.to_csv('ts4_pred.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


import math

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras import callbacks

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

import itertools
from itertools import product
from itertools import combinations

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def split_data(data, sequence):
    train_set = data[:len(data)-90]
    # validation_set = data[355 - sequence:369]
    test_set = data[-90:]

    return train_set, test_set


def timeseries_gen(seq_size, n_features, train, val, test):
    # Train Set
    train_generator = TimeseriesGenerator(train, train.to_numpy(), length=seq_size, batch_size=1)
    print("Total number of samples in the original training data = ", len(train))
    print("Total number of samples in the generated training data = ", len(train_generator))

    # Validation Set
    val_generator = TimeseriesGenerator(val, val.to_numpy(), length=seq_size, batch_size=1)
    print("Total number of samples in the original validation data = ", len(val))
    print("Total number of samples in the validation data = ", len(val_generator))

    # Test Set
    test_generator = TimeseriesGenerator(test, test.to_numpy(), length=seq_size, batch_size=1)
    print("Total number of samples in the original test data = ", len(test))
    print("Total number of samples in the generated test data = ", len(test_generator))
    return train_generator, val_generator, test_generator

def plotprediction(ypredict , col,name="" , pname="" , predtype=''):
    plt.figure(figsize=[12,10] , dpi=140 )
    plt.plot(ypredict.index, ypredict.iloc[:, col], 'y', label='Prediction ')
    plt.plot(ypredict.index, ypredict.iloc[:, 1], 'r', label='Actual ')
    plt.title('Predicted vs  Actual '  + pname + '  in Greece for ' +str(len(ypredict)) + ' days')
    plt.suptitle(predtype)
    plt.xlabel('Date')
    plt.ylabel('deaths')
    plt.legend()
    plt.savefig("Plots\pred" + name +"_"+ predtype+ ".jpeg"  )
    plt.show()

# loc="owid_dataset_fixed.csv"
# Greece_total = pd.read_csv(loc)
# Greece_total['date'] = pd.to_datetime(Greece_total['date']) # convert date column to DateTime
# Greece_total.set_index('date', inplace=True)
#
# analysis = Greece_total[['total_cases']].copy()
# decompose_result_mult = seasonal_decompose(analysis, model="multiplicative")
#
# trend = decompose_result_mult.trend
# seasonal = decompose_result_mult.seasonal
# residual = decompose_result_mult.resid
#
# figure(figsize=(14, 10))
# decompose_result_mult.plot()
# plt.show()
#
#
# ########################################################################################################################
#
# plt.rcParams.update({'figure.figsize':(18,10)})
# from numpy import log
#
#
# analysis = analysis.reset_index(drop=[True])
# result = adfuller(analysis.total_cases.dropna())
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
#
# # Original Series
# fig, axes = plt.subplots(4, 1, sharex=True)
# #
# # axes[0,0].plot(analysis.total_cases);
# # axes[0, 0].set_title('Original Series')
# # plot_acf(analysis.total_cases, ax=axes[0, 1])
# plot_acf(analysis.total_cases, ax=axes[0])
#
# # 1st Differencing
# # axes[1, 0].plot(analysis.total_cases.diff());
# # axes[1, 0].set_title('1st Order Differencing')
# # plot_acf(analysis.total_cases.diff().dropna(), ax=axes[1, 1])
# plot_acf(analysis.total_cases.diff().dropna(), ax=axes[1])
#
# # 2nd Differencing
# # axes[2, 0].plot(analysis.total_cases.diff().diff());
# # axes[2, 0].set_title('2nd Order Differencing')
# # plot_acf(analysis.total_cases.diff().diff().dropna(), ax=axes[2, 1])
# plot_acf(analysis.total_cases.diff().diff().dropna(), ax=axes[2])
#
#
# plot_acf(analysis.total_cases.diff().diff().diff().dropna(), ax=axes[3])
#
# plt.show()

# plot_acf(analysis.total_cases.dropna())
# plot_acf(analysis.total_cases.diff().dropna())
# plot_acf(analysis.total_cases.diff().diff().dropna())
# plot_acf(analysis.total_cases.diff().diff().diff().dropna())
# plot_acf(analysis.total_cases.diff().diff().diff().diff().dropna())
# plot_acf(analysis.total_cases.diff().diff().diff().diff().diff().dropna())
# plot_acf(analysis.total_cases.diff().diff().diff().diff().diff().diff().dropna())
# plt.show()


### AUTO ARIMA ###
from pmdarima.arima import auto_arima
loc="owid_dataset_fixed.csv"
Greece_total = pd.read_csv(loc).set_index('date')
total_cases=Greece_total['total_cases']
train  , test = split_data(total_cases,0)

plt.plot(train)
plt.plot(test)
plt.show()

arima_model =  auto_arima(train,start_p=0, d=1, start_q=0,
                          max_p=6, max_d=6, max_q=6, start_P=0,
                          D=1, start_Q=0, max_P=6, max_D=6,
                          max_Q=6, m=12, seasonal=False,
                          error_action='warn',trace = True,
                          supress_warnings=True,stepwise = True,
                          random_state=20,n_fits = 70 )

arima_model.summary()
prediction = pd.DataFrame(arima_model.predict(n_periods = 90),index=test.index)
prediction.columns = ['predicted_cases']

plt.figure(figsize=(8,5))
plt.plot(train,label="Training")
plt.plot(test,label="Test")
plt.plot(prediction,label="Predicted")
plt.legend(loc = 'lower left')
plt.show()
totalpred=pd.DataFrame()
totalpred=pd.concat([test, prediction], ignore_index=True ,axis=1)

lour1=mean_absolute_percentage_error(test[:7],prediction[:7])
lour2=mean_absolute_percentage_error(test[:14],prediction[:14])
lour3=mean_absolute_percentage_error(test[:30],prediction[:30])
lour4=mean_absolute_percentage_error(test[:60],prediction[:60])
lour5=mean_absolute_percentage_error(test[:90],prediction[:90])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pmdarima.arima import auto_arima

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def adftest(df):
    result = adfuller(df.dropna())
    # ADF Test
    print(f'ADF Statistic: {result[0]}')
    print(f'n_lags: {result[1]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')
    return result

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def split_data(data, sequence):
    # train_set = data[:-9]
    train_set = data[:369]
    # train_set = data[355 - sequence:369]
    # test_set = data[-9:-1]
    test_set = data[369:]
    return train_set, test_set

def final_results(testdf, preddf):
    day_7 = mean_absolute_percentage_error(testdf[:1], preddf[:1])
    day_14 = mean_absolute_percentage_error(testdf[:2], preddf[:2])
    day_30 = mean_absolute_percentage_error(testdf[:4], preddf[:4])
    day_60 = mean_absolute_percentage_error(testdf[:6], preddf[:6])
    day_90 = mean_absolute_percentage_error(testdf[:8], preddf[:8])

    results=pd.DataFrame({day_7,  day_14,  day_30 ,day_60, day_90} )


    return results

def plotres (trainingset , testset ,pred, pname,traindate, testdate):
    plt.figure(figsize=(15,12))
    trainingset=trainingset.set_index(traindate)
    testset=testset.set_index(testdate)
    plt.plot(trainingset,label="Training")
    plt.plot(testset,label="Test")
    plt.plot(pred,label="Predicted")
    plt.legend(loc = 'upper left')
    plt.xlabel('Date')
    plt.ylabel('Total '+ pname)
    plt.show()

def decomposition(df):
    decompose_result_mult = seasonal_decompose(df, model="multiapplicative")
    fig = decompose_result_mult.plot()
    fig.set_size_inches((16, 9))
    fig.tight_layout()
    plt.show()
    return decompose_result_mult

from statsmodels.tsa.stattools import kpss

from statsmodels.tsa.stattools import kpss

def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

loc="owid_dataset_fixed.csv"
# loc="owid_dataset_weekly.csv"

Greece_total = pd.read_csv(loc,parse_dates=True,index_col="date")
train,test = split_data(Greece_total,0)

### Train Set - Test Set Split ###
train = train[['total_deaths']].copy()
train=train.dropna()
train_dates=train.index
train=train.reset_index(drop=True)

test = test[['total_deaths']].copy()
test=test.dropna()
test_dates=test.index
test=test.reset_index(drop=True)

# analysis = Greece_total[['total_deaths']].copy()
## Analyse Data Set
# analysis =Greece_total[["total_deaths"]].dropna()
analysis =train
analysis['lognorm'] = np.log(analysis['total_deaths'])
# decomp_res=decomposition(train) #Seasonal Decomposition


adftestres = adftest(analysis['lognorm'].diff().diff().dropna()) # Use d= 2 for my model
kpss_test(analysis['total_deaths'].diff().diff().dropna())




############## ACF & PACF Plots ################
#Calculate p order PACF

fig=plot_pacf(analysis['lognorm'].diff().diff().dropna())
fig.set_size_inches((16, 9))
plt.show()
#
fig=plot_acf(analysis['lognorm'].diff().diff().dropna())
fig.set_size_inches((16, 9))
plt.show()

plt.figure(figsize=(18,5));
plt.xlabel('date');
plt.ylabel('Total Deaths ');
plt.title('Differenced Time Series of Total Deaths ');
plt.plot(analysis['lognorm'].diff().diff());
plt.show()
##################################################
arima_model = auto_arima(train['total_deaths'], start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=15, max_q=15, # maximum p and q
                      m=30,              # frequency of series
                      # d=1,           # let model determine 'd'
                      # D=1,
                      seasonal=True,   # Seasonality
                      trace=True,

                      # max_order=10,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
#
arima_model.summary()
arima_model.plot_diagnostics(figsize=(18,10))
plt.show()
# #
# #
# # # train=train.reset_index(drop=True)
# # # test=test.reset_index(drop=True)
# from statsmodels.tsa.statespace.sarimax import SARIMAX
#
# # # train=train.reset_index(drop=True)
# model = SARIMAX(train, order=(1, 2, 1), seasonal_order=(2, 0, 1, 7))
# results=model.fit(maxiter=300)
# model.plot_diagnostics(figsize=(18,10))
# plt.show()
# results.summary()
#
# #
# # # Actual vs Fitted
# # model.plot_predict(dynamic=False)
# # plt.show()
# #
# #
## Make Prediction ###
predname='deaths'
# start=len(train)
# end=len(train)+len(test)-1
prediction = pd.DataFrame(arima_model.predict(len(test),dynamic='false'))
prediction.columns = ['predicted_'+ predname]
prediction=prediction.set_index(test_dates)

    ### Results ####
test=test.set_index(test_dates)
plotres(pd.DataFrame(train['total_deaths']) , test, prediction , predname, train_dates,test_dates)
totalpred=pd.concat([prediction, test], ignore_index=True ,axis=1)
totalpred=totalpred.rename(columns={0:'Predicted Deaths', 1:'Actual Deaths'})
finalresults = final_results(test,prediction)

totalpred.plot(figsize=(14,10))
plt.show()


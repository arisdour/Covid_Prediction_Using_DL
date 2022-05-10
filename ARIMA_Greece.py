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
    train_set = data[:369]
    # validation_set = data[355 - sequence:369]
    test_set = data[369:]
    return train_set, test_set

def final_results(testdf, preddf):
    day_7 = mean_absolute_percentage_error(testdf[:7], preddf[:7])
    day_14 = mean_absolute_percentage_error(testdf[:14], preddf[:14])
    day_30 = mean_absolute_percentage_error(testdf[:30], preddf[:30])
    day_60 = mean_absolute_percentage_error(testdf[:60], preddf[:60])
    day_90 = mean_absolute_percentage_error(testdf[:90], preddf[:90])

    results=pd.DataFrame({day_7,  day_14,  day_30 ,day_60, day_90} )


    return results

def plotres (trainingset , testset ,pred, pname):
    plt.figure(figsize=(15,12))
    plt.plot(trainingset,label="Training")
    plt.plot(testset,label="Test")
    plt.plot(pred,label="Predicted")
    plt.legend(loc = 'upper left')
    plt.xlabel('Date')
    plt.ylabel('Total '+ pname)
    plt.show()


loc="owid_dataset_fixed.csv"
Greece_total = pd.read_csv(loc,parse_dates=True,index_col="date")

analysis = Greece_total[['total_cases']].copy()
decompose_result_mult = seasonal_decompose(analysis, model="multiplicative")

# trend = decompose_result_mult.trend
# seasonal = decompose_result_mult.seasonal
# residual = decompose_result_mult.resid

fig = decompose_result_mult.plot()
fig.set_size_inches((16, 9))
# Tight layout to realign things
fig.tight_layout()
plt.show()

adttestres = adftest(analysis.diff().diff())

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

def mainf():

    ### AUTO ARIMA ###
    loc="owid_dataset_fixed.csv"
    Greece_total = pd.read_csv(loc).set_index('date')
    total_cases=Greece_total['total_cases']
    predname='cases'

    #Split Data
    train, test = split_data(total_cases,0)

    plt.plot(train)
    plt.plot(test)
    plt.show()

    ### Make Model ###
    # arima_model =  auto_arima(train,test='adf',start_p=0, d=1, start_q=0,
    #                           max_p=6, max_d=6, max_q=6, start_P=0,
    #                           D=1, start_Q=0, max_P=6, max_D=6,
    #                           max_Q=6, m=12, seasonal=False,
    #                           error_action='warn',trace = True,
    #                           supress_warnings=True,stepwise = True,
    #                           random_state=20,n_fits = 70 )

    arima_model = auto_arima(train, start_p=1, start_q=1,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=6, max_q=6, # maximum p and q
                          m=12,              # frequency of series
                              # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True, )


    arima_model.summary()
    arima_model.plot_diagnostics(figsize=(10,8))
    plt.show()
    ### Make Prediction ###
    prediction = pd.DataFrame(arima_model.predict(n_periods = len(test)),index=test.index)
    prediction.columns = ['predicted_'+ predname]

    ### Results ####
    plotres(train , test, prediction , predname)
    totalpred=pd.concat([test, prediction], ignore_index=True ,axis=1)
    finalresults = final_results(test,prediction)





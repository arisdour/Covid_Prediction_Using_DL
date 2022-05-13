import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pmdarima.arima import auto_arima

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


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



def final_results(dataframe):

    Days_7 = []
    Days_14 = []
    Days_30 = []
    Days_60 = []
    Days_90 = []

    ###############################################################################

    mae = mean_absolute_error(dataframe['Actual'], dataframe['Prediction'])
    mae = float("{:.3f}".format(mae))
    Days_90.append(mae)

    mae_7days = mean_absolute_error(dataframe['Actual'][:7], dataframe['Prediction'][:7])
    mae_7days = float("{:.3f}".format(mae_7days))
    Days_7.append(mae_7days)

    mae_14days = mean_absolute_error(dataframe['Actual'][:14], dataframe['Prediction'][:14])
    mae_14days = float("{:.3f}".format(mae_14days))
    Days_14.append(mae_14days)

    mae_30days = mean_absolute_error(dataframe['Actual'][:30], dataframe['Prediction'][:30])
    mae_30days = float("{:.3f}".format(mae_30days))
    Days_30.append(mae_30days)

    mae_60days = mean_absolute_error(dataframe['Actual'][:60], dataframe['Prediction'][:60])
    mae_60days = float("{:.3f}".format(mae_60days))
    Days_60.append(mae_60days)

    ###############################################################################

    ###############################################################################

    mape = mean_absolute_percentage_error(dataframe['Actual'], dataframe['Prediction'])
    mape = float("{:.3f}".format(mape))
    Days_90.append(mape)

    mape_7days = mean_absolute_percentage_error(dataframe['Actual'][:7], dataframe['Prediction'][:7])
    mape_7days = float("{:.3f}".format(mape_7days))
    Days_7.append(mape_7days)

    mape_14days = mean_absolute_percentage_error(dataframe['Actual'][:14], dataframe['Prediction'][:14])
    mape_14days = float("{:.3f}".format(mape_14days))
    Days_14.append(mape_14days)

    mape_30days = mean_absolute_percentage_error(dataframe['Actual'][:30], dataframe['Prediction'][:30])
    mape_30days = float("{:.3f}".format(mape_30days))
    Days_30.append(mape_30days)

    mape_60days = mean_absolute_percentage_error(dataframe['Actual'][:60], dataframe['Prediction'][:60])
    mape_60days = float("{:.3f}".format(mape_60days))
    Days_60.append(mape_60days)

    ###############################################################################
    ###############################################################################

    mape_n = mean_absolute_percentage_error(dataframe['Actual'], dataframe['Forecast'])
    mape_n = float("{:.3f}".format(mape_n))
    Days_90.append(mape_n)

    mape_7days_n = mean_absolute_percentage_error(dataframe['Actual'][:7], dataframe['Forecast'][:7])
    mape_7days_n = float("{:.3f}".format(mape_7days_n))
    Days_7.append(mape_7days_n)

    mape_14days_n = mean_absolute_percentage_error(dataframe['Actual'][:14], dataframe['Forecast'][:14])
    mape_14days_n = float("{:.3f}".format(mape_14days_n))
    Days_14.append(mape_14days_n)

    mape_30days_n = mean_absolute_percentage_error(dataframe['Actual'][:30], dataframe['Forecast'][:30])
    mape_30days_n = float("{:.3f}".format(mape_30days_n))
    Days_30.append(mape_30days_n)

    mape_60days_n = mean_absolute_percentage_error(dataframe['Actual'][:60], dataframe['Forecast'][:60])
    mape_60days_n = float("{:.3f}".format(mape_60days_n))
    Days_60.append(mape_60days_n)

    ###############################################################################
    ###############################################################################

    ###############################################################################

    mse = mean_squared_error(dataframe['Actual'], dataframe['Prediction'])
    mse = float("{:.3f}".format(mse))
    Days_90.append(mse)

    mse_7days = mean_squared_error(dataframe['Actual'][:7], dataframe['Prediction'][:7])
    mse_7days = float("{:.3f}".format(mse_7days))
    Days_7.append(mse_7days)

    mse_14days = mean_squared_error(dataframe['Actual'][:14], dataframe['Prediction'][:14])
    mse_14days = float("{:.3f}".format(mse_14days))
    Days_14.append(mse_14days)

    mse_30days = mean_squared_error(dataframe['Actual'][:30], dataframe['Prediction'][:30])
    mse_30days = float("{:.3f}".format(mse_30days))
    Days_30.append(mse_30days)

    mse_60days = mean_squared_error(dataframe['Actual'][:60], dataframe['Prediction'][:60])
    mse_60days = float("{:.3f}".format(mse_60days))
    Days_60.append(mse_60days)

    ###############################################################################

    ##############################################################################

    rmse = mean_squared_error(dataframe['Actual'], dataframe['Prediction'], squared=False)
    rmse = float("{:.3f}".format(rmse))
    Days_90.append(rmse)

    rmse_7days = mean_squared_error(dataframe['Actual'][:7], dataframe['Prediction'][:7], squared=False)
    rmse_7days = float("{:.3f}".format(rmse_7days))
    Days_7.append(rmse_7days)

    rmse_14days = mean_squared_error(dataframe['Actual'][:14], dataframe['Prediction'][:14], squared=False)
    rmse_14days = float("{:.3f}".format(rmse_14days))
    Days_14.append(rmse_14days)

    rmse_30days = mean_squared_error(dataframe['Actual'][:30], dataframe['Prediction'][:30], squared=False)
    rmse_30days = float("{:.3f}".format(rmse_30days))
    Days_30.append(rmse_30days)

    rmse_60days = mean_squared_error(dataframe['Actual'][:60], dataframe['Prediction'][:60], squared=False)
    rmse_60days = float("{:.3f}".format(rmse_60days))
    Days_60.append(rmse_60days)
    ####################################################################################################################
    Comp1 = []
    Comp2 = []

    mape_9_Days = mean_absolute_percentage_error(dataframe['Actual'][:9], dataframe['Prediction'][:9])
    mape_9_Days = float("{:.3f}".format(mape_9_Days))
    Comp1.append(mape_9_Days)

    mape_9_Days = mean_absolute_percentage_error(dataframe['Actual'][:9], dataframe['Forecast'][:9])
    mape_9_Days = float("{:.3f}".format(mape_9_Days))
    Comp1.append(mape_9_Days)

    mape_40_Days = mean_absolute_percentage_error(dataframe['Actual'][:40], dataframe['Prediction'][:40])
    mape_40_Days = float("{:.3f}".format(mape_40_Days))
    Comp2.append(mape_40_Days)

    mape_40_Days = mean_absolute_percentage_error(dataframe['Actual'][:40], dataframe['Forecast'][:40])
    mape_40_Days = float("{:.3f}".format(mape_40_Days))
    Comp2.append(mape_40_Days)

    Comparison = pd.DataFrame({" 9 Days": Comp1, " 40 Days": Comp2})
    Comparison.to_csv("Results\Comparison" + ".csv", float_format="%.3f", index=True,
                      header=True)

    ###############################################################################

    Names = ['MAE', 'MAPE','MAPE_Forecast', 'MSE', 'RMSE']
    finalresults = pd.DataFrame(
        {" 7 Days": Days_7, " 14 Days": Days_14, " 30 Days": Days_30, " 60 Days": Days_60, " 90 Days": Days_90,
         'NAMES': Names})
    finalresults = finalresults.set_index(['NAMES'])
    return finalresults

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
                      d=2,           # let model determine 'd'
                      # D=1,
                      seasonal=False,   # Seasonality
                      trace=True,

                      # max_order=10,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
#
arima_model.summary()
arima_model.plot_diagnostics(figsize=(18,10))
plt.show()


## Make Prediction ###
predname='deaths'
# # start=len(train)
# # end=len(train)+len(test)-1
prediction = pd.DataFrame(arima_model.predict(len(test),dynamic='true'))
prediction.columns = ['predicted_'+ predname]
prediction=prediction.set_index(test_dates)

    ### Results ####
test=test.set_index(test_dates)
# plotres(pd.DataFrame(train['total_deaths']) , test, prediction , predname, train_dates,test_dates)
totalpred=pd.concat([prediction, test], ignore_index=True ,axis=1)
totalpred=totalpred.rename(columns={0:'Prediction', 1:'Actual'})



history = [x for x in train['total_deaths']]
predictions4 = list()
testl=test.total_deaths.to_list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(8,2,6))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions4.append(yhat)
	obs = testl[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

predictions4=pd.DataFrame(predictions4).set_index(test_dates)
predictions4=predictions4.rename(columns={'Prediction4':'Forecast'})

totalpred=pd.concat([totalpred, predictions4], ignore_index=True ,axis=1).rename(columns={0 :'Prediction' ,1:'Actual' , 2:'Forecast' })
finalresults = final_results(totalpred)
totalpred.plot(figsize=(14,10))
plt.show()



# train=train.reset_index(drop=True)
# test=test.reset_index(drop=True)
# model = ARIMA(train['total_deaths'], order=(8,2,6))
# model = model.fit()
# print(len(test) )
# lour = pd.DataFrame(model.forecast(len(test)))

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
    # train_set = data[355 - sequence:369]
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

def decomposition(df):
    decompose_result_mult = seasonal_decompose(df, model="additive")
    fig = decompose_result_mult.plot()
    fig.set_size_inches((16, 9))
    fig.tight_layout()
    plt.show()
    return decompose_result_mult


loc="owid_dataset_fixed.csv"

Greece_total = pd.read_csv(loc,parse_dates=True,index_col="date")
train,test = split_data(Greece_total,0)

### Train Set - Test Set Split ###
train = train[['total_cases']].copy()
train=train.dropna()

test = test[['total_cases']].copy()
test=test.dropna()

# analysis = Greece_total[['total_cases']].copy()
## Analyse Data Set
analysis =train
# decomp_res=decomposition(train) #Seasonal Decomposition
adttestres = adftest(analysis.diff().diff()) # Use d= 2 for my model
# adttestres = adftest(analysis.shift(7)) # Use d= 2 for my model



############## ACF & PACF Plots ################
#Calculate p order PACF

fig=plot_pacf(analysis.diff().diff().dropna())
fig.set_size_inches((16, 9))
plt.show()
#
fig=plot_acf(analysis.diff().diff().dropna())
fig.set_size_inches((16, 9))
plt.show()

analysis=analysis.reset_index(drop=True)
##################################################
arima_model = auto_arima(analysis, start_p=0, start_q=0,
                      # test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=7,              # frequency of series
                      # d=1,           # let model determine 'd'
                      # D=1,
                      seasonal=True,   # Seasonality
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

arima_model.summary()
arima_model.plot_diagnostics(figsize=(18,18))
plt.show()

train=train.reset_index(drop=True)
test=test.reset_index(drop=True)
from statsmodels.tsa.statespace.sarimax import SARIMAX
arima_model = SARIMAX(train,order=(5,2,0),seasonal_order=(1,0,1,7))
result = arima_model.fit()
print(result.summary())

result.plot_diagnostics(figsize=(18,18))


### Make Prediction ###
predname='cases'
start=len(train)
end=len(train)+len(test)-1
prediction = pd.DataFrame(arima_model.predict(len(test),dynamic='true' , index=test.index))
prediction.columns = ['predicted_'+ predname]
prediction=prediction.set_index(test.index)

    ### Results ####
plotres(train , test, prediction , predname)
totalpred=pd.concat([test, prediction], ignore_index=False ,axis=1)
finalresults = final_results(test,prediction)

totalpred.plot(figsize=(14,10))
plt.show()
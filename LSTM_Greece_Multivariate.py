import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import tensorflow
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
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
from itertools  import product
from itertools  import combinations 


########################## Functions   ###################################
import LSTM_Greece_Multivariate


def createdata(dataset, features):
    Greece = dataset[features]
    Greece["date"] = Greece_total['date']
    Greece = Greece.dropna(axis=0)

    dates = pd.DataFrame(Greece['date']).reset_index(drop=True)
    Greece = Greece[(features)].reset_index(drop=True)

    return dates, Greece

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def split_data(data, sequence):
    if (len(data)<400):
        
        length = len(data)
        a = round(0.89*(length-60))
                  
        train_set = data[:a].reset_index(drop=True)
        validation_set = data[(a - sequence):(length-60)].reset_index(drop=True)
        test_set = data[length-60:].reset_index(drop=True) 
        
    else:
        train_set = data[:355].reset_index(drop=True)
        validation_set = data[355 - sequence:369].reset_index(drop=True)
        test_set = data[369 - sequence:].reset_index(drop=True) 
    
    return train_set, validation_set, test_set

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

def plotloss(mod, name=""):
    plt.figure(figsize=[12,10] , dpi=140 )
    loss = mod.history['loss']
    val_loss = mod.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("Plots/loss_model" + name +".jpeg"  )
    plt.show()

def plotprediction(ypredict , col,name="" , pname="" , predtype=''):
    plt.figure(figsize=[12,10] , dpi=140 )
    plt.plot(ypredict.index, ypredict.iloc[:, col], 'y', label='Prediction ')
    plt.plot(ypredict.index, ypredict.iloc[:, 1], 'r', label='Actual ')
    plt.title('Predicted vs  Actual '  + pname + '  in Greece for ' +str(len(ypredict)) + ' days')
    plt.suptitle(predtype)
    plt.xlabel('Date')
    plt.ylabel('cases')
    plt.legend()
    plt.savefig("Plots\pred" + name +"_"+ predtype+ ".jpeg"  )
    plt.show()

def inversesets(sequence,feature_list, sc, trainset, validationset, testset, ogdata, dates):
    
    drange =dates.loc[0]
    drange=pd.to_datetime(drange["date"])
    date_index = pd.date_range(drange , periods=len(dates), freq='D')

    
    
    set1 = pd.DataFrame(sc.inverse_transform(trainset),index=date_index[0:len(trainset)])

    set1=set1.set_axis(feature_list, axis=1, inplace=False)
    
    set2 = pd.DataFrame(sc.inverse_transform(validationset),index=date_index[len(trainset) - sequence:len(trainset) + len(validationset) - sequence])
    set2=set2.set_axis(feature_list, axis=1, inplace=False)

    set3 = pd.DataFrame(sc.inverse_transform(testset),index=date_index[-len(testset):])
    set3=set3.set_axis(feature_list, axis=1, inplace=False)
    return set1, set2, set3

def model_create( seq_size , features):
    model = Sequential()
    # model.add(LSTM(44, activation='relu', return_sequences=False, input_shape=(seq_size, features)))
    model.add(LSTM(30, activation='relu', return_sequences=False, input_shape=(seq_size, features)))  #Total cases

    model.add(Dense(n_features))
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.summary()
    return model

def stacked_model_create(seq_size , features):
    model = Sequential()
    model.add(LSTM(20, activation='relu', return_sequences=True, input_shape=(seq_size, features)))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(20, return_sequences=False))

    model.add(Dense(n_features))
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.summary()
    return model

def model_train(i, model, traingenerator, valgenerator, ep):
    history = model.fit(traingenerator, validation_data=valgenerator, epochs=ep, verbose=1)
    model.save('Models/model_' + str(i) + '.h5', overwrite=True)
    plotloss(history,str(i))
    return model

def model_train_earlystop(i, model, traingenerator, valgenerator, ep):
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 5, restore_best_weights = True)


    history = model.fit(traingenerator, validation_data=valgenerator, epochs=ep ,verbose=1,callbacks =[earlystopping])
    model.save('Models\model_' + str(i) + '.h5', overwrite=True)
    # plotloss(history,str(i))
    # avep.append( len(history.history['loss']))
    
    
    return model

def predict(model, sc, valgenerator, validation_set, inverseval, trainset):
    # Forecast   Predict using a for loop
    index = inverseval.index
    predictiondata = pd.DataFrame(inverseval[:seq_size])  # Empty list to populate later with predictions
    predictiondata = pd.DataFrame(trainset[-seq_size:]).reset_index(drop=True)
    # print(predictiondata)
    current_batch = trainset[-seq_size:]
    forecast = pd.DataFrame()

    # Prediction using Validation Generator
    predict1 = model.predict(valgenerator)

    # Predict future, beyond test dates
    future = len(validation_set) - seq_size  # Days
    for i in range(future):
        current_batch = predictiondata[i:seq_size + i]  # Create input for LSTM (Based on sequence size )

        current_batch = current_batch.to_numpy()  # Input to array

        current_batch = current_batch.reshape(1, seq_size, n_features)  # Reshape

        ### Prediction ##

        current_pred = model.predict(current_batch)  # Make a prediction
        # print(current_pred[0])
        current_pred = current_pred[0]  # Convert Prediction to integer
        predictiondata.loc[len(predictiondata.index)] = current_pred

    forecast = predictiondata[-(future):]  # Save results in a dataframe
    forecast = sc.inverse_transform(forecast)  # Inverse Transform to get the actual cases
    forecast = pd.DataFrame(forecast.round())  # Round results
    forecast = forecast.set_index(index[seq_size:], 'Date').rename(columns={0: 'Prediction'})


    forecast = pd.concat([forecast['Prediction'], inverseval['total_cases'][seq_size:]], axis=1,
                         ignore_index=True)  # Concate the two dfs

    forecast = forecast.set_axis(['Prediction', 'Actual'], axis=1, inplace=False)

    predictN4 = sc.inverse_transform(predict1)  # Inverse Transform to get the actual cases
    predictN4 = pd.DataFrame(predictN4.round()).rename(columns={0: 'Prediction N4'})  # Round results
    # print(predictN4)
    predictN4 = predictN4.set_index(index[seq_size:], 'Date')

    total_forecast = pd.concat([forecast, predictN4], axis=1)  # , igonre_index=True)
    # print(total_forecast)

    return total_forecast

def Hyper(parameter1 , parameter2 , parameter3 , repetitions):
    hp1 = list(product(parameter1 , parameter2 ))
    Hyperparameters = list (product(hp1 , parameter3))
    Hyperparameters= pd.DataFrame(Hyperparameters).rename(columns={0: "A", 1: "Nodes"})
    
    Hyperparameters[['Learning Rate' , 'Epochs']]= pd.DataFrame(Hyperparameters['A'].tolist(), index=Hyperparameters.index)
    
    Hyperparameters =Hyperparameters.drop(['A'], axis=1)
    Hyperparameters=Hyperparameters.sort_values(by=['Nodes', 'Learning Rate' ,'Epochs' ])
    Hyperparameters=pd.concat([Hyperparameters]*times)
    
    
    Hyperparameters= list(Hyperparameters.itertuples(index=False, name=None))
    
    
    return Hyperparameters 

def experiments(i, nodes, scaler, seq_size, epochs, n_features, train_generator, val_generator, validation_set,
                train_set, inv_val, inv_test, dates ,lrate):
    
    experimentmodel = model_create( seq_size ,n_features)
    # experimentmodel = stacked_model_create( seq_size ,n_features) #stacked


    experimentmodel = model_train_earlystop(i, experimentmodel, train_generator, val_generator, epochs)  # Train Model

    forecast = predict(experimentmodel, scaler, val_generator, validation_set, inv_val, train_set)
    # plotprediction(forecast ,0,str(i) , pname , 'For Loop Prediction')
    # plotprediction(forecast ,2, str(i), pname, 'Normal Prediction')
    
    
    ##################### Metrics ######################

    mae_4 = mean_absolute_error(forecast['Actual'], forecast['Prediction'])
    MAE_4.append(mae_4)

    mape_4 = mean_absolute_percentage_error(forecast['Actual'], forecast['Prediction'])
    MAPE_4.append(mape_4)

    mse_4 = mean_squared_error(forecast['Actual'], forecast['Prediction'])
    MSE_4.append(mse_4)

    rmse_4 = mean_squared_error(forecast['Actual'], forecast['Prediction'], squared=False)
    RMSE_4.append(rmse_4)

    node.append(nodes)


    mape_4_next_day = mean_absolute_percentage_error(forecast['Actual'][:1], forecast['Prediction'][:1])
    MAPE_4_Next_day.append(mape_4_next_day)
 
    mape_3days = mean_absolute_percentage_error(forecast['Actual'][:3], forecast['Prediction'][:3])
    MAPE_4_3days.append(mape_3days)
    
    mape_7days = mean_absolute_percentage_error(forecast['Actual'][:7], forecast['Prediction'][:7])
    MAPE_4_7days.append(mape_7days)
    
    Epochs.append(epochs)
    Features.append(feature_list)



    # Normal Prediction Metrics

    mape_next_day_n = mean_absolute_percentage_error(forecast['Actual'][:1], forecast['Prediction N4'][:1])
    MAPE_Next_day.append(mape_next_day_n)

    mape_3days_n = mean_absolute_percentage_error(forecast['Actual'][:3], forecast['Prediction N4'][:3])
    MAPE_3days.append(mape_3days_n)

    mape_7days_n = mean_absolute_percentage_error(forecast['Actual'][:7], forecast['Prediction N4'][:7])
    MAPE_7days.append(mape_7days_n)

    mape_4_n = mean_absolute_percentage_error(forecast['Actual'], forecast['Prediction N4'])
    MAPE.append(mape_4_n)
        

    return 

def find_best_model(mape):
    mape = pd.DataFrame(mape)
    min = mape.idxmin()
    j = min[0]
    best_model = tensorflow.keras.models.load_model(r"Models/model_" + str(j) + ".h5")
    print("Best Model is :model_" + str(j) + ".h5")
    return best_model

def final_results(dataframe):
    
    plotprediction(dataframe[:7], 0, "iction_7_day_prediction", pname, 'For Loop Prediction')
    plotprediction(dataframe[:14], 0, "iction_14_day_prediction", pname, 'For Loop Prediction')
    plotprediction(dataframe[:30], 0, "iction_30_day_prediction", pname, 'For Loop Prediction')
    plotprediction(dataframe[:60], 0, "iction_60_day_prediction", pname, 'For Loop Prediction')
    plotprediction(dataframe[:90], 0, "iction_90_day_prediction", pname, 'For Loop Prediction')

    plotprediction(dataframe[:7], 2, "iction_7_day_prediction", pname, 'Normal Prediction')
    plotprediction(dataframe[:14], 2, "iction_14_day_prediction", pname, 'Normal Prediction')
    plotprediction(dataframe[:30], 2, "iction_30_day_prediction", pname, 'Normal Prediction')
    plotprediction(dataframe[:60], 2, "iction_60_day_prediction", pname, 'Normal Prediction')
    plotprediction(dataframe[:90], 2, "iction_90_day_prediction", pname, 'Normal Prediction')

    Days_7= []
    Days_14= []
    Days_30= []
    Days_60= []
    Days_90= []


    ###############################################################################

    mae = mean_absolute_error(dataframe['Actual'], dataframe['Prediction'])
    mae= float("{:.3f}".format(mae))
    Days_90.append(mae)

    mae_7days = mean_absolute_error(dataframe['Actual'][:7], dataframe['Prediction'][:7])
    mae_7days= float("{:.3f}".format(mae_7days))
    Days_7.append(mae_7days)


    mae_14days = mean_absolute_error(dataframe['Actual'][:14], dataframe['Prediction'][:14])
    mae_14days= float("{:.3f}".format(mae_14days))
    Days_14.append(mae_14days)

    mae_30days = mean_absolute_error(dataframe['Actual'][:30], dataframe['Prediction'][:30])
    mae_30days= float("{:.3f}".format(mae_30days))
    Days_30.append(mae_30days)

    mae_60days = mean_absolute_error(dataframe['Actual'][:60], dataframe['Prediction'][:60])
    mae_60days= float("{:.3f}".format(mae_60days))
    Days_60.append(mae_60days)

    ###############################################################################



    ###############################################################################

    mape = mean_absolute_percentage_error(dataframe['Actual'], dataframe['Prediction'])
    mape= float("{:.3f}".format(mape))
    Days_90.append(mape)


    mape_7days = mean_absolute_percentage_error(dataframe['Actual'][:7], dataframe['Prediction'][:7])
    mape_7days= float("{:.3f}".format(mape_7days))
    Days_7.append(mape_7days)

                    
    mape_14days = mean_absolute_percentage_error(dataframe['Actual'][:14], dataframe['Prediction'][:14])
    mape_14days= float("{:.3f}".format(mape_14days))
    Days_14.append(mape_14days)


    mape_30days = mean_absolute_percentage_error(dataframe['Actual'][:30], dataframe['Prediction'][:30])
    mape_30days= float("{:.3f}".format(mape_30days))
    Days_30.append(mape_30days)


    mape_60days = mean_absolute_percentage_error(dataframe['Actual'][:60], dataframe['Prediction'][:60])
    mape_60days= float("{:.3f}".format(mape_60days))
    Days_60.append(mape_60days)


    ###############################################################################
    ###############################################################################

    mape_n = mean_absolute_percentage_error(dataframe['Actual'], dataframe['Prediction N4'])
    mape_n = float("{:.3f}".format(mape_n))
    Days_90.append(mape_n)

    mape_7days_n = mean_absolute_percentage_error(dataframe['Actual'][:7], dataframe['Prediction N4'][:7])
    mape_7days_n = float("{:.3f}".format(mape_7days_n))
    Days_7.append(mape_7days_n)

    mape_14days_n = mean_absolute_percentage_error(dataframe['Actual'][:14], dataframe['Prediction N4'][:14])
    mape_14days_n = float("{:.3f}".format(mape_14days_n))
    Days_14.append(mape_14days_n)

    mape_30days_n = mean_absolute_percentage_error(dataframe['Actual'][:30], dataframe['Prediction N4'][:30])
    mape_30days_n = float("{:.3f}".format(mape_30days_n))
    Days_30.append(mape_30days_n)

    mape_60days_n = mean_absolute_percentage_error(dataframe['Actual'][:60], dataframe['Prediction N4'][:60])
    mape_60days_n = float("{:.3f}".format(mape_60days_n))
    Days_60.append(mape_60days_n)

    ###############################################################################

    ###############################################################################

    mse = mean_squared_error(dataframe['Actual'], dataframe['Prediction'])
    mse= float("{:.3f}".format(mse))
    Days_90.append(mse)


    mse_7days = mean_squared_error(dataframe['Actual'][:7], dataframe['Prediction'][:7])
    mse_7days= float("{:.3f}".format(mse_7days))
    Days_7.append(mse_7days)

    mse_14days = mean_squared_error(dataframe['Actual'][:14], dataframe['Prediction'][:14])
    mse_14days= float("{:.3f}".format(mse_14days))
    Days_14.append(mse_14days)


    mse_30days = mean_squared_error(dataframe['Actual'][:30], dataframe['Prediction'][:30])
    mse_30days= float("{:.3f}".format(mse_30days))
    Days_30.append(mse_30days)


    mse_60days = mean_squared_error(dataframe['Actual'][:60], dataframe['Prediction'][:60])
    mse_60days= float("{:.3f}".format(mse_60days))
    Days_60.append(mse_60days)

    ###############################################################################


    ##############################################################################

    rmse = mean_squared_error(dataframe['Actual'], dataframe['Prediction'], squared=False)
    rmse= float("{:.3f}".format(rmse))
    Days_90.append(rmse)


    rmse_7days = mean_squared_error(dataframe['Actual'][:7], dataframe['Prediction'][:7] , squared=False)
    rmse_7days= float("{:.3f}".format(rmse_7days))
    Days_7.append(rmse_7days)


    rmse_14days = mean_squared_error(dataframe['Actual'][:14], dataframe['Prediction'][:14] , squared=False)
    rmse_14days= float("{:.3f}".format(rmse_14days))
    Days_14.append(rmse_14days)


    rmse_30days = mean_squared_error(dataframe['Actual'][:30], dataframe['Prediction'][:30] , squared=False)
    rmse_30days= float("{:.3f}".format(rmse_30days))
    Days_30.append(rmse_30days)


    rmse_60days = mean_squared_error(dataframe['Actual'][:60], dataframe['Prediction'][:60] , squared=False)
    rmse_60days= float("{:.3f}".format(rmse_60days))
    Days_60.append(rmse_60days)


    ###############################################################################

    Comp1 = []
    Comp2 = []

    mape_9_Days = mean_absolute_percentage_error(dataframe['Actual'][:9], dataframe['Prediction'][:9])
    mape_9_Days = float("{:.3f}".format(mape_9_Days))
    Comp1.append(mape_9_Days)

    mape_9_Days = mean_absolute_percentage_error(dataframe['Actual'][:9], dataframe['Prediction N4'][:9])
    mape_9_Days = float("{:.3f}".format(mape_9_Days))
    Comp1.append(mape_9_Days)

    mape_40_Days = mean_absolute_percentage_error(dataframe['Actual'][:40], dataframe['Prediction'][:40])
    mape_40_Days = float("{:.3f}".format(mape_40_Days))
    Comp2.append(mape_40_Days)

    mape_40_Days = mean_absolute_percentage_error(dataframe['Actual'][:40], dataframe['Prediction N4'][:40])
    mape_40_Days = float("{:.3f}".format(mape_40_Days))
    Comp2.append(mape_40_Days)

    Comparison = pd.DataFrame({" 9 Days": Comp1, " 40 Days": Comp2})
    Comparison.to_csv("Results\Comparison_for_" + str(feature_list) + ".csv", float_format="%.3f", index=True,
                      header=True)

    Comparison = pd.DataFrame({" 9 Days": Comp1, " 40 Days": Comp2})
    Comparison.to_csv("Results\Comparison_for_" + str(feature_list) + ".csv", float_format="%.3f", index=True,
                      header=True)
    ###############################################################################

    Names = ['MAE' , 'MAPE', 'MAPE NORMAL' , 'MSE'  , 'RMSE']
    finalresults=pd.DataFrame({" 7 Days" :Days_7, " 14 Days" :Days_14, " 30 Days" :Days_30," 60 Days" :Days_60," 90 Days":Days_90  , 'NAMES':Names })
    finalresults=finalresults.set_index(['NAMES'])
    return finalresults


##########################  MAIN ##############################################


Epochs = []
LR = []
node = []
MAE_4 = []
MAPE_4 = []
MSE_4 = []
RMSE_4 = []
MAPE_4_3days = []
MAPE_4_7days = []
MAPE_4_Next_day = []
Features = []
MAPE_Next_day = []  #1 Day
MAPE_3days = []    ## Days
MAPE_7days = []    #7 Days
MAPE = []           #14 Days
loc="owid_dataset_fixed.csv"


seq_size = 3
epochs = 60
times = 10
pname= 'cases'
ctrl=[2,3,4,5,6,7,8,9,10]


##### Data  Creation #####
Greece_total=pd.read_csv(loc)

# Remove  ICU *& Hospital Data from original Dataset
titles = Greece_total.columns
titles.str.contains('adm')
admtitles = titles[titles.str.contains('adm')].to_list()
Greece_total = Greece_total.drop(admtitles, axis=1)

#######################################################################################################################
Greece_total=Greece_total.drop(columns=['date', 'Unnamed: 0'])

total_cases_cor=pd.DataFrame()
correlation_mat_p = Greece_total.corr()
total_cases_cor['Pearson'] = correlation_mat_p['total_cases']
correlation_mat_s = Greece_total.corr(method='spearman')

correlation_mat_s = Greece_total.corr(method='spearman')
total_cases_cor['Spearman'] = correlation_mat_s['total_cases']
Spearman=total_cases_cor['Spearman']
Spearman=Spearman[Spearman > 0.9]
Spearman=Spearman.sort_values(ascending=False)
Spearman=Spearman.index.to_list()

for i in range(len(ctrl)):
    control = ctrl[i]
    print(control)
    cor=Spearman[:control]

    ## Combinations ###
    flist = list(combinations(cor , len(cor)))
    flist=[ x for x in flist if "total_"+ pname in x ] # Must always contain total cases/ cases
    flist=flist*times
     ## Control length
    flist=sorted(flist)
    # flist=flist[:ctrl]
    # flist=Spearman
    dates = pd.DataFrame()
    print(flist)
#######################################################################################################################
    Greece_total=pd.read_csv(loc)

    for i in range(len(flist)):

        feature_list= flist[i]
        feature_list = list(itertools.chain(feature_list))
        feature_list.append('date')
        greece=Greece_total[feature_list]
        greece = greece.dropna(axis=0)

        dates['date'] = greece['date'].reset_index(drop=True)
        greece=greece.drop(columns=['date'])


        feature_list=(greece.columns).to_list()
        n_features = len(feature_list)


        train_set, validation_set, test_set = split_data( greece, seq_size)
        scaler = MinMaxScaler()
        scaler.fit(train_set)

        train_set=pd.DataFrame(scaler.transform(train_set))
        train_set=train_set.set_axis(feature_list, axis=1, inplace=False)

        validation_set=pd.DataFrame(scaler.transform(validation_set))
        validation_set=validation_set.set_axis(feature_list, axis=1, inplace=False)

        test_set=pd.DataFrame(scaler.transform(test_set))
        test_set=test_set.set_axis(feature_list, axis=1, inplace=False)

        train_generator, val_generator, test_generator = timeseries_gen(seq_size, n_features, train_set, validation_set,test_set)

        inv_train, inv_val, inv_test = inversesets(seq_size,feature_list, scaler, train_set, validation_set, test_set, greece,dates)

        experiments(i, 0, scaler, seq_size, epochs, n_features, train_generator, val_generator,validation_set, train_set, inv_val, inv_test, dates , 0.0001 )

    
    
metrics = pd.DataFrame(
{'Feat':Features  ,'MAE_4': MAE_4, 'MAPE_4 1 Day': MAPE_4_Next_day,
  'MAPE_4 3 Days': MAPE_4_3days,'MAPE_4 7 days': MAPE_4_7days, 'MAPE_4': MAPE_4,
 'MAPE 1 Day': MAPE_Next_day,
         'MAPE 3 Days': MAPE_3days, 'MAPE 7 days': MAPE_7days, 'MAPE': MAPE,
 'MSE_4': MSE_4, 'RMSE_4': RMSE_4 })

average = metrics.groupby(metrics['Feat'].map(tuple)).mean()

# #Save Results
metrics.to_csv("Results/Metrics_Valdation_Results_for_"+ str(len(feature_list))+ ".csv", float_format="%.5f",index=True, header=True)
average.to_csv("Results/Average_Valdation_Results_for__"+ str(len(feature_list)) +".csv", float_format="%.5f",index=True, header=True)



# bestmodel = find_best_model(MAPE_4)
# print(bestmodel)
#
# callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', restore_best_weights=True, patience=5)
# bestmodel.fit(val_generator, epochs=60 ,  callbacks=[callback], verbose=1)
#
#
# forecastf = predict(bestmodel, scaler, test_generator, test_set, inv_test, validation_set )
#
# finalresults=final_results(forecastf)
#
# finalresults.to_csv("Results\Final_Results_for_" +  str(len(feature_list)) +".csv", float_format="%.3f",index=True, header=True)

import winsound
winsound.Beep(800, 300)
winsound.Beep(800, 900)
winsound.Beep(800, 300)
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense



from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import itertools
from itertools  import product
from itertools  import combinations 

########################## Telegram Bot   ###################################

import requests

def telegram_bot_sendtext(bot_message):
    
    bot_token = '2062474091:AAGp1GiSrNw7DRds4qwLHBOkZ_Do9HlQ5V8'
    bot_chatID = '2013533042'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    
    response = requests.get(send_text)

    return response.json()



########################## Functions   ###################################


def readdata(location):
    
    data=pd.read_csv(location)
    Greece= data[data.location =='Greece'].reset_index(drop='True')
    Greece = Greece.dropna(how='all', axis=1)
    Greece_total = Greece.iloc[7:498, 3:40].reset_index(drop='True')
    titles =Greece_total.columns
    return Greece_total , titles



def featcombos(featurename ,titles , combin) :
    
    titles.str.contains(featurename)
    features = titles[titles.str.contains(featurename)].to_list()
    print(features)
    feature_list = list(combinations(features , combin))
    
    return feature_list

def createdata(dataset ,features):
    
    Greece=dataset[features]
    Greece["date"] = Greece_total['date']
    Greece=Greece.dropna(axis=0)
    
    dates=pd.DataFrame(Greece['date']).reset_index(drop=True)
    Greece=Greece[(features)].reset_index(drop=True)
    
    return dates , Greece



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def split_data(data, sequence):
    train_set = data[:355].reset_index(drop=True)
    validation_set = data[355 - sequence:369].reset_index(drop=True)
    test_set = data[369 - sequence:].reset_index(drop=True) 
    
    return train_set, validation_set, test_set


def timeseries_gen(seq_size, n_features, train, val, test):
    # Train Set
    train_generator = TimeseriesGenerator(train, train.iloc[:, 0], length=seq_size, batch_size=1)
    print("Total number of samples in the original training data = ", len(train))
    print("Total number of samples in the generated training data = ", len(train_generator))


    # Validation Set
    val_generator = TimeseriesGenerator(val, val.iloc[:, 0], length=seq_size, batch_size=1)
    print("Total number of samples in the original validation data = ", len(val))
    print("Total number of samples in the validation data = ", len(val_generator))

    # Test Set
    test_generator = TimeseriesGenerator(test, test.iloc[:, 0], length=seq_size, batch_size=1)
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


def plotprediction(ypredict , name=""):
    plt.figure(figsize=[12,10] , dpi=140 )
    plt.plot(ypredict.index, ypredict.iloc[:, 0], 'y', label='Prediction ')
    plt.plot(ypredict.index, ypredict.iloc[:, 1], 'r', label='Actual ')
    plt.title('Predicted vs  Actual Cases in Greece for ' +str(len(ypredict)) + ' days')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.savefig("Plots/pred" + name +".jpeg"  )
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


def model_create(nodes, seq_size , features,lrate):
    # opt = keras.optimizers.Adam(learning_rate=lrate)
    model = Sequential()
    model.add(LSTM(44, activation='relu', return_sequences=False, input_shape=(seq_size, features)))
    model.add(Dense(1))
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.summary()
    return model


def model_train(i, model, traingenerator, valgenerator, ep):
    history = model.fit(traingenerator, validation_data=valgenerator, epochs=ep, verbose=1)
    model.save('Models/model_' + str(i) + '.h5', overwrite=True)
    plotloss(history,str(i))
    return model


def predict(model, sc, valgenerator, validation_set, inverseval, trainset ):


    # Forecast   Predict using a for loop
    index = inverseval.index
    
    predictiondata = pd.DataFrame(inverseval[:seq_size])  # Empty list to populate later with predictions
    predictiondata = pd.DataFrame(trainset[-seq_size:]).reset_index(drop=True)
    
    
    A=[	1492, 1482, 1323, 1372, 1222, 662]
    newcasesprediction = pd.DataFrame(A)
    
    current_batch = trainset[-seq_size:]
    forecast = pd.DataFrame()

    # Predict future, beyond test dates
    future = len(validation_set) - seq_size  # Days
    for i in range(future): #instead of future
        
        current_batch = predictiondata[i:seq_size + i] #Create input for LSTM (Based on sequence size )
        current_batch = current_batch.to_numpy()  #Input to array 
        current_batch = current_batch.reshape(1, seq_size, n_features)  # Reshape

        ### Prediction ##
        
        current_pred = model.predict(current_batch) # Make a prediction 
        total_cases = float(current_pred[0]) #Convert Prediction to integer 
        total_cases= total_cases /5.80966e-06 #De-scale
        
        
        
        # ##### Create New Day Values #####
        
        #### Total Cases ####
        
        total_cases_per_million = total_cases * 0.096 #Calculate Total Caces per million 
        
        #### New Cases ####
        
        new_cases= total_cases-(predictiondata.iloc[len(predictiondata.index)-1,0])/5.80966e-06 # Calculate  new casesDe-scaled
        
        new_cases_per_million = new_cases*0.096  #Calculate New per million 
        
        
        newcasesprediction.loc[len(newcasesprediction.index)] = [new_cases] #append new cases 
        smoothednew = newcasesprediction.rolling(window=7).mean()
        new_cases_smoothed = float( smoothednew.iloc[6+i])
        
        new_cases_smoothed_pre_million= new_cases_smoothed * 0.096  #Calculate Smoothed Permillion New Cases 
        
        
        #Scale Back 
        
        total_cases = total_cases * 5.80966e-06
        new_cases = new_cases * 0.000301568
        new_cases_smoothed = new_cases_smoothed * 0.000374211
        total_cases_per_million = total_cases_per_million * 5.82583e-05
        new_cases_per_million = new_cases_per_million * 0.00314326
        new_cases_smoothed_pre_million = new_cases_smoothed_pre_million * 0.00389804
        

        
        
        #Add New Day Values 
        Featnames = ['total_cases','new_cases','new_cases_smoothed','total_cases_per_million','new_cases_per_million','new_cases_smoothed_per_million']
        featval = [total_cases,new_cases,new_cases_smoothed,total_cases_per_million,new_cases_per_million,new_cases_smoothed_pre_million]
        dictionary = dict(zip(Featnames, featval))

        usedval =[ dictionary[feature_list[0]] , dictionary[feature_list[1]] , dictionary[feature_list[2]]  ] # , dictionary[feature_list[4]] ,  dictionary[feature_list[5]] ]


        
        predictiondata.loc[len(predictiondata.index)] = usedval
    

    forecast = predictiondata[-(future):] #Save results in a dataframe 
    forecast = sc.inverse_transform(forecast)#Inverse Transform to get the actual cases 
    forecast = pd.DataFrame(forecast.round()) #Round results 
    forecast = forecast.set_index(index[seq_size:], 'Date').rename(columns={0: 'Prediction'})

    forecast = pd.concat([forecast['Prediction'], inverseval['total_cases'][seq_size:]], axis=1 ,ignore_index=True) #Concate the two dfs 

    forecast=forecast.set_axis(['Prediction', 'Actual'], axis=1, inplace=False)
    
    
    
    
    return forecast



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







########################## Cluster Fuck  LSTM ##############################################


def experiments(i, nodes, scaler, seq_size, epochs, n_features, train_generator, val_generator, validation_set,
                train_set, inv_val, inv_test, dates ,lrate):
    
    experimentmodel = model_create(nodes, seq_size ,n_features , lrate)

    experimentmodel = model_train(i, experimentmodel, train_generator, val_generator, epochs)  # Train Model

    forecast = predict(experimentmodel, scaler, val_generator, validation_set, inv_val, train_set)
    plotprediction(forecast ,str(i))
    
    
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

    LR.append(lrate)
        

    return 


def find_best_model(mape):
    mape = pd.DataFrame(mape)
    min = mape.idxmin()
    j = min[0]
    best_model = keras.models.load_model(r"Models/model_" + str(j) + ".h5")
    print("Best Model is :model_" + str(j) + ".h5")
    return best_model

##########################  MAIN ##############################################







seq_size =3
times =10
combos=3
nodes=2
lr = 0.0001
epochs=75






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
loc="owid-covid-data.csv"
Greece_total , titles =readdata(loc)
flist = featcombos('cases', titles, combos)


flist=flist*times

flist=[ x for x in flist if "total_cases"  in x ]
flist=[ x for x in flist if "new_cases_smoothed"   in x ]
flist=[ x for x in flist if "new_cases_smoothed_per_million"  in x ]


# flist=flist[:1]

for i in range(len(flist)):
    feature_list= flist[i]

    feature_list = list(itertools.chain(feature_list))
    n_features = len(feature_list)
    
    Greece_total['new_cases_smoothed']= Greece_total['new_cases'].rolling(window=7).mean()
    Greece_total['new_deaths_smoothed']= Greece_total['new_deaths'].rolling(window=7).mean()
    
    Greece_total['new_cases_smoothed_per_million']= Greece_total['new_cases_smoothed']*0.096
    Greece_total['new_deaths_smoothed_per_million']= Greece_total['new_deaths_smoothed']*0.096



    dates,greece = createdata(Greece_total ,feature_list )
    
    
    
    
    train_set, validation_set, test_set = split_data( greece, seq_size)
    
    #Scaling 
    scaler = MinMaxScaler() 
    scaler.fit(train_set)
    train_set=pd.DataFrame(scaler.transform(train_set))
    train_set=train_set.set_axis(feature_list, axis=1, inplace=False)
    
    validation_set=pd.DataFrame(scaler.transform(validation_set))
    validation_set=validation_set.set_axis(feature_list, axis=1, inplace=False)
    
    test_set=pd.DataFrame(scaler.transform(test_set))
    test_set=test_set.set_axis(feature_list, axis=1, inplace=False)
    
    
    train_generator, val_generator, test_generator = timeseries_gen(seq_size, n_features, train_set, validation_set,
                                                                    test_set)
    # a =train_generator[3]
    
    inv_train, inv_val, inv_test = inversesets(seq_size,feature_list, scaler, train_set, validation_set, test_set, greece,
                                                            dates)



# Hyperparameters= Hyper(learning_rate, epochs, nodes ,times )


    # nodes , lr , epochs = Hyperparameters[i]
    experiments(i, nodes, scaler, seq_size, epochs, n_features, train_generator, val_generator,
                      validation_set, train_set, inv_val, inv_test, dates , lr )









metrics = pd.DataFrame(
    {'Feat':Features  ,'MAE_4': MAE_4, 'MAPE_4 1 Day': MAPE_4_Next_day,
      'MAPE_4 3 Days': MAPE_4_3days,'MAPE_4 7 days': MAPE_4_7days, 'MAPE_4': MAPE_4, 'MSE_4': MSE_4, 'RMSE_4': RMSE_4 , 'Epochs' : Epochs})

metrics=metrics.sort_values(by=['Feat']).reset_index(drop=True)


metrics[['Feature 1','Feature 2' , 'Feature 3']] = pd.DataFrame(metrics.Feat.tolist(), index= metrics.index)
metrics1 = metrics.groupby(['Feature 1', 'Feature 2' , 'Feature 3' ]).mean()

 

# #Save Results
metrics.to_csv("Results/Valdation_Results_for_"+ str(len(feature_list)) +".csv", float_format="%.5f",index=True, header=True)
metrics1.to_csv("Results/AverageValdation_Results_for_"+ str(len(feature_list)) +".csv", float_format="%.5f",index=True, header=True)







bestmodel = find_best_model(MAPE_4)


bestmodel.fit_generator(val_generator, epochs=60, verbose=1) 
bestmodel.save(r"Models\Final_model_for_"+ str(feature_list) + ".h5")

forecastf = predict(bestmodel, scaler, test_generator, test_set, inv_test, validation_set )

plotprediction(forecastf[:7] , "iction_7_day_prediction")
plotprediction(forecastf[:14] , "iction_14_day_prediction")
plotprediction(forecastf[:30] , "iction_30_day_prediction")
plotprediction(forecastf[:60] , "iction_60_day_prediction")
plotprediction(forecastf[:90] , "iction_90_day_prediction")

Days_7= []
Days_14= []
Days_30= []
Days_60= []
Days_90= []


###############################################################################

mae = mean_absolute_error(forecastf['Actual'], forecastf['Prediction'])
mae= float("{:.3f}".format(mae))
Days_90.append(mae)

mae_7days = mean_absolute_error(forecastf['Actual'][:7], forecastf['Prediction'][:7])
mae_7days= float("{:.3f}".format(mae_7days))
Days_7.append(mae_7days)


mae_14days = mean_absolute_error(forecastf['Actual'][:14], forecastf['Prediction'][:14])
mae_14days= float("{:.3f}".format(mae_14days))
Days_14.append(mae_14days)

mae_30days = mean_absolute_error(forecastf['Actual'][:30], forecastf['Prediction'][:30])
mae_30days= float("{:.3f}".format(mae_30days))
Days_30.append(mae_30days)

mae_60days = mean_absolute_error(forecastf['Actual'][:60], forecastf['Prediction'][:60])
mae_60days= float("{:.3f}".format(mae_60days))
Days_60.append(mae_60days)

###############################################################################



###############################################################################

mape = mean_absolute_percentage_error(forecastf['Actual'], forecastf['Prediction'])
mape= float("{:.3f}".format(mape))
Days_90.append(mape)


mape_7days = mean_absolute_percentage_error(forecastf['Actual'][:7], forecastf['Prediction'][:7])
mape_7days= float("{:.3f}".format(mape_7days))
Days_7.append(mape_7days)

                  
mape_14days = mean_absolute_percentage_error(forecastf['Actual'][:14], forecastf['Prediction'][:14])
mape_14days= float("{:.3f}".format(mape_14days))
Days_14.append(mape_14days)


mape_30days = mean_absolute_percentage_error(forecastf['Actual'][:30], forecastf['Prediction'][:30])
mape_30days= float("{:.3f}".format(mape_30days))
Days_30.append(mape_30days)


mape_60days = mean_absolute_percentage_error(forecastf['Actual'][:60], forecastf['Prediction'][:60])
mape_60days= float("{:.3f}".format(mape_60days))
Days_60.append(mape_60days)


###############################################################################


###############################################################################

mse = mean_squared_error(forecastf['Actual'], forecastf['Prediction'])
mse= float("{:.3f}".format(mse))
Days_90.append(mse)


mse_7days = mean_squared_error(forecastf['Actual'][:7], forecastf['Prediction'][:7])
mse_7days= float("{:.3f}".format(mse_7days))
Days_7.append(mse_7days)

mse_14days = mean_squared_error(forecastf['Actual'][:14], forecastf['Prediction'][:14])
mse_14days= float("{:.3f}".format(mse_14days))
Days_14.append(mse_14days)


mse_30days = mean_squared_error(forecastf['Actual'][:30], forecastf['Prediction'][:30])
mse_30days= float("{:.3f}".format(mse_30days))
Days_30.append(mse_30days)


mse_60days = mean_squared_error(forecastf['Actual'][:60], forecastf['Prediction'][:60])
mse_60days= float("{:.3f}".format(mse_60days))
Days_60.append(mse_60days)

###############################################################################


###############################################################################

rmse = mean_squared_error(forecastf['Actual'], forecastf['Prediction'], squared=False)
rmse= float("{:.3f}".format(rmse))
Days_90.append(rmse)


rmse_7days = mean_squared_error(forecastf['Actual'][:7], forecastf['Prediction'][:7] , squared=False)
rmse_7days= float("{:.3f}".format(rmse_7days))
Days_7.append(rmse_7days)


rmse_14days = mean_squared_error(forecastf['Actual'][:14], forecastf['Prediction'][:14] , squared=False)
rmse_14days= float("{:.3f}".format(rmse_14days))
Days_14.append(rmse_14days)


rmse_30days = mean_squared_error(forecastf['Actual'][:30], forecastf['Prediction'][:30] , squared=False)
rmse_30days= float("{:.3f}".format(rmse_30days))
Days_30.append(rmse_30days)


rmse_60days = mean_squared_error(forecastf['Actual'][:60], forecastf['Prediction'][:60] , squared=False)
rmse_60days= float("{:.3f}".format(rmse_60days))
Days_60.append(rmse_60days)


###############################################################################



Names = ['MAE' , 'MAPE' , 'MSE'  , 'RMSE']
finalresults=pd.DataFrame({" 7 Days" :Days_7, " 14 Days" :Days_14, " 30 Days" :Days_30," 60 Days" :Days_60," 90 Days":Days_90  , 'NAMES':Names })
finalresults=finalresults.set_index(['NAMES'])

finalresults.to_csv("Results\Final_Results_for_" + str(feature_list) +".csv", float_format="%.3f",index=True, header=True)


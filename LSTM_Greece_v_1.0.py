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


########################## Functions   ###################################

def createdata(location , feature_list):
    
    data=pd.read_csv(location)
    Greece= data[data.location =='Greece'].reset_index(drop='True')
    Greece = Greece.dropna(how='all', axis=1)
    Greece_total = Greece.iloc[7:498, 3:40].reset_index(drop='True')
    dates=pd.DataFrame(Greece_total['date']).reset_index(drop=True)
    Greece=Greece_total[(feature_list)].reset_index(drop=True)
    
    return dates , Greece , Greece_total

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
    plt.savefig("Plots\loss_model_" + name +".jpeg"  )
    plt.show()


def plotprediction(ypredict , name=""):
    plt.figure(figsize=[12,10] , dpi=140 )
    plt.plot(ypredict.index, ypredict.iloc[:, 0], 'y', label='Prediction ')
    plt.plot(ypredict.index, ypredict.iloc[:, 1], 'r', label='Actual ')
    plt.title('Predicted vs  Actual Cases in Greece for ' +str(len(ypredict)) + ' days')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.savefig("Plots\pred_" + name +".jpeg"  )
    plt.show()
   


def inversesets(sequence,feature_list, sc, trainset, validationset, testset, ogdata, dates):
    
    date_index = pd.date_range('26/02/2020', periods=491, freq='D')
    
    
    set1 = pd.DataFrame(sc.inverse_transform(trainset),index=date_index[0:len(trainset)])

    set1=set1.set_axis(feature_list, axis=1, inplace=False)
    
    set2 = pd.DataFrame(sc.inverse_transform(validationset),index=date_index[len(trainset) - sequence:len(trainset) + len(validationset) - sequence])
    set2=set2.set_axis(feature_list, axis=1, inplace=False)

    set3 = pd.DataFrame(sc.inverse_transform(testset),index=date_index[-len(testset):])
    set3=set3.set_axis(feature_list, axis=1, inplace=False)
    return set1, set2, set3


def model_create(nodes, seq_size , features):
    model = Sequential()
    model.add(LSTM(nodes, activation='relu', return_sequences=False, input_shape=(seq_size, features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model


def model_train(i, model, traingenerator, valgenerator, ep):
    history = model.fit(traingenerator, validation_data=valgenerator, epochs=ep, verbose=1)
    model.save('Models\model_' + str(i) + '.h5', overwrite=True)
    plotloss(history,str(i))
    return model
    # Predict using the generator


def predict(model, sc, valgenerator, validation_set, inverseval, trainset ):


    # Forecast   Predict using a for loop
    index = inverseval.index
    predictiondata = pd.DataFrame(inverseval[:seq_size])  # Empty list to populate later with predictions
    predictiondata = pd.DataFrame(trainset[-seq_size:]).reset_index(drop=True)
    current_batch = trainset[-seq_size:]
    forecast = pd.DataFrame()

    # Predict future, beyond test dates
    future = len(validation_set) - seq_size  # Days
    for i in range(future):
        
        current_batch = predictiondata[i:seq_size + i] #Create input for LSTM (Based on sequence size )

        current_batch = current_batch.to_numpy()  #Input to array 

        current_batch = current_batch.reshape(1, seq_size, n_features)  # Reshape

        ### Prediction ##
        
        current_pred = model.predict(current_batch) # Make a prediction 
        
        current_pred = float(current_pred[0]) #Convert Prediction to integer 

        # ##### Create New Day Values #####
        newcases= current_pred-predictiondata.iloc[len(predictiondata.index)-1,0] # Calculate the new cases 
        # totalpm = current_pred * 0.096
        predictiondata.loc[len(predictiondata.index)] = [current_pred, newcases]#,totalpm ]  # Fill the two first collumns of the Dataframe 


        # predictiondata['Percentage'] = predictiondata['Daily_Confirmed_Cases'].pct_change() #Calculate Percentage 
        # predictiondata['Moving Average'] = predictiondata["New Cases"].rolling(3).mean() #Calculate Mean 
        # predictiondata=predictiondata.fillna(0.0051519) # Fill one missing value with the true value 

       
    forecast = predictiondata[-(future):] #Save results in a dataframe 
    forecast = sc.inverse_transform(forecast)#Inverse Transform to get the actual cases 
    forecast = pd.DataFrame(forecast.round()) #Round results 
    forecast = forecast.set_index(index[seq_size:], 'Date').rename(columns={0: 'Prediction'})

    forecast = pd.concat([forecast['Prediction'], inverseval['total_cases'][seq_size:]], axis=1 ,ignore_index=True) #Concate the two dfs 

    forecast=forecast.set_axis(['Prediction', 'Actual'], axis=1, inplace=False)
    
    
    return forecast


########################## Cluster Fuck  LSTM ##############################################


def experiments(times, nodes, scaler, seq_size, epochs, n_features, train_generator, val_generator, validation_set,
                train_set, inv_val, inv_test, dates):
    for i in range(times):
        experimentmodel = model_create(nodes, seq_size ,n_features)

        experimentmodel = model_train(i, experimentmodel, train_generator, val_generator, epochs)  # Train Model

        forecast = predict(experimentmodel, scaler, val_generator, validation_set, inv_val, train_set)
        plotprediction(forecast ,str(i))

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

    metrics = pd.DataFrame(
        {'MAE_4': MAE_4, 'MAPE_4 1 Day': MAPE_4_Next_day,
         'MAPE_4 3 Days': MAPE_4_3days,'MAPE_4 7 days': MAPE_4_7days, 'MAPE_4': MAPE_4, 'MSE_4': MSE_4, 'RMSE_4': RMSE_4, 'Nodes': node})

    metrics =metrics.append( metrics.groupby(['Nodes']).mean())

    return metrics


def find_best_model(mape):
    mape = pd.DataFrame(mape)
    min = mape.idxmin()
    j = min[0]
    best_model = keras.models.load_model(r"Models\model_" + str(j) + ".h5")
    print("Best Model is :model_" + str(j) + ".h5")
    return best_model




# def main():
    
Windeos_loc="owid-covid-data.csv"
feature_list=["total_cases", "new_cases"]#, "total_cases_per_million"]
a=str(feature_list)
n_features = len(feature_list)
print(a)
different_nodes = [20]
seq_size = 3
epochs = 60
rep = 2

dates,greece , Greece_total =createdata(Windeos_loc,feature_list)

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


inv_train, inv_val, inv_test = inversesets(seq_size,feature_list, scaler, train_set, validation_set, test_set, greece,
                                                       dates)

node = []
MAE_4 = []
MAPE_4 = []
MSE_4 = []
RMSE_4 = []
MAPE_4_3days = []
MAPE_4_7days = []
MAPE_4_Next_day = []

start = time.time()
for i in range(len(different_nodes)):
    
    nodes = different_nodes[i]
    times = rep
    metrics = experiments(times, nodes, scaler, seq_size, epochs, n_features, train_generator, val_generator,
                          validation_set, train_set, inv_val, inv_test, dates)
end = time.time()

hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))



# # #Save Results
metrics.to_csv("Results\Valdation_Results_for_"+ a +".csv", float_format="%.3f",index=True, header=True)

bestmodel = find_best_model(MAPE_4)

# lour, lour_1 = predict(bestmodel, scaler, val_generator, validation_set, inv_val, train_set) #Cotnrol Line

bestmodel.fit_generator(val_generator, epochs=epochs, verbose=1) 
bestmodel.save(r"Models\Final_model_for_"+ a + ".h5")

forecastf = predict(bestmodel, scaler, test_generator, test_set, inv_test, validation_set )

plotprediction(forecastf[:7] , "7_day_prediction")
plotprediction(forecastf[:14] , "14_day_prediction")
plotprediction(forecastf[:30] , "30_day_prediction")
plotprediction(forecastf[:60] , "60_day_prediction")
plotprediction(forecastf[:90] , "90_day_prediction")


mae = mean_absolute_error(forecastf['Actual'], forecastf['Prediction'])
mae= float("{:.3f}".format(mae))

mape = mean_absolute_percentage_error(forecastf['Actual'], forecastf['Prediction'])
mape= float("{:.3f}".format(mape))

mape_1day = mean_absolute_percentage_error(forecastf['Actual'][:1], forecastf['Prediction'][:1])
mape_1day= float("{:.3f}".format(mape_1day))


mape_3days = mean_absolute_percentage_error(forecastf['Actual'][:3], forecastf['Prediction'][:3])
mape_3days= float("{:.3f}".format(mape_3days))

mape_7days = mean_absolute_percentage_error(forecastf['Actual'][:7], forecastf['Prediction'][:7])
mape_7days= float("{:.3f}".format(mape_7days))
                  
mape_14days = mean_absolute_percentage_error(forecastf['Actual'][:14], forecastf['Prediction'][:14])
mape_14days= float("{:.3f}".format(mape_14days))

mape_30days = mean_absolute_percentage_error(forecastf['Actual'][:30], forecastf['Prediction'][:30])
mape_30days= float("{:.3f}".format(mape_30days))

mape_60days = mean_absolute_percentage_error(forecastf['Actual'][:60], forecastf['Prediction'][:60])
mape_60days= float("{:.3f}".format(mape_60days))

mse = mean_squared_error(forecastf['Actual'], forecastf['Prediction'])
mse= float("{:.3f}".format(mse))
rmse = mean_squared_error(forecastf['Actual'], forecastf['Prediction'], squared=False)
rmse= float("{:.3f}".format(rmse))

finalresults=pd.DataFrame({"MAE": [mae],"MAPE 1 Day" : [mape_1day] , "MAPE 3 Days" :[mape_3days],"MAPE 7 Days " :[mape_7days] , "MAPE 14 Days" :[mape_14days], "MAPE 30 Days" :[mape_30days],"MAPE 60 Days" :[mape_60days],"MAPE":[mape], "RMSE": [rmse], "MSE":[mse]})


finalresults.to_csv("Results\Final_Results_for_" + a +".csv", float_format="%.3f",index=True, header=True)






















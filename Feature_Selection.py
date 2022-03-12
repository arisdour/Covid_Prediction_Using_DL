import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    
import itertools
from itertools  import product
from itertools  import combinations 

def readdata(location):
    
    data=pd.read_csv(location)
    Greece= data[data.location =='Greece'].reset_index(drop='True')
    Greece = Greece.dropna(how='all', axis=1)
    Greece_total = Greece.iloc[7:498, 3:40].reset_index(drop='True')
    titles =Greece_total.columns
    return Greece_total , titles
loc="owid-covid-data.csv"


seq_size = 3
epochs = 60
times = 10
Κ= 1
nodes=0

##### Data  Creation #####
Greece_total , titles =readdata(loc)
Greece_total['new_deaths_smoothed']= Greece_total['new_deaths'].rolling(window=7).mean()
Greece_total['new_deaths_smoothed']= Greece_total['new_deaths'].rolling(window=7).mean()
Greece_total['new_deaths_smoothed_per_million']= Greece_total['new_deaths_smoothed']*0.096
Greece_total['new_deaths_smoothed_per_million']= Greece_total['new_deaths_smoothed']*0.096


def FeatureSelection(df,K):
    
    first_n_column  = df.iloc[:369 , :14]
    second_n_column = df.iloc[:369 , 22:26]


    first_n_column=pd.concat([first_n_column, second_n_column], axis=1)
    first_n_column['stringency_index'] = Greece_total['stringency_index']
    first_n_column = first_n_column.reindex(sorted(first_n_column.columns), axis=1)
    first_n_column = first_n_column.dropna()
    second_n_column= second_n_column.dropna()



    fs = SelectKBest(score_func=f_regression, k=K)

    y=first_n_column['total_cases'] # Set Total Cases as a Target
    dates = df['date']
    X=first_n_column.drop( columns=[ 'date' , 'total_cases']) # Remove Total Cases 


    # y=df['total_cases'] # Set Total Cases as a Target
    # dates = df['date']
    # X=df.drop( columns=[ 'date' , 'total_cases','tests_units']) # Remove Total Cases 
    X_selected =fs.fit_transform(X, y)

    # X_selected=pd.concat([X_selected, y] , axis=1)

    selected_col = fs.get_support(indices = True)
    Res = X.iloc[:, selected_col]
    
    Res=pd.concat([y,Res ] , axis=1)
    Final=pd.concat([dates,Res ] , axis=1)
    Final=Final.dropna()
    Final2=Final.columns
    return Final2 ,Final

lour , lour1 =FeatureSelection(Greece_total, 5)
lour2=Greece_total[lour]
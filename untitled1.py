# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 23:29:55 2021

@author: Aris_Dourdounas
"""
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
from itertools  import chain ,product




learning_rate = (0.001,0.0001,0.0005 )
epochs = (60 , 75 , 150)
nodes = (18,20,22,25,30,35,44,59,88)

hp1 = list(product(learning_rate , epochs ))
Hyperparameters = list (product(hp1 , nodes))

# test = Hyperparameters[0]

# Hyperparameters = list(chain.from_iterable(Hyperparameters))
Hyperparameters= pd.DataFrame(Hyperparameters).rename(columns={0: "A", 1: "Nodes"})




Hyperparameters[['Learning Rate' , 'Epochs']]= pd.DataFrame(Hyperparameters['A'].tolist(), index=Hyperparameters.index)
Hyperparameters =Hyperparameters.drop(['A'], axis=1)

Hyperparameters= list(Hyperparameters.itertuples(index=False, name=None))


nodes , lr , epochs = Hyperparameters[0]


# from itertools  import combinations 
# titles =Greece_total.columns
# titles.str.contains('cases')

# features = titles[titles.str.contains(r'cases')].to_list()

# features_2 = list(combinations(features , 2))



# print ("All the combination of list in sorted order(without replacement) is:")   
# print(list(combinations(['A', 2], 2)))  
# print()  


# params={'batch_size':[100, 20, 50, 25, 32], 
#         'nb_epoch':[200, 100, 300, 400],
#         'unit':[5,6, 10, 11, 12, 15],
           
#         }
# gs=GridSearchCV(estimator=model, param_grid=params, cv=10)
# # now fit the dataset to the GridSearchCV object. 
# gs = gs.fit(X, y)
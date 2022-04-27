import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

import itertools
from itertools  import product
from itertools  import combinations

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
    
import itertools
from itertools  import product
from itertools  import combinations

###############################################################
def FeatureSelection(df, K):
    # fs = SelectKBest(score_func=f_regression, k=K)            # Use F regression
    fs = SelectKBest(score_func=mutual_info_regression, k=K)  # use mutual info
    df = df.dropna()
    y = df['total_cases']  # Set Total cases as a Target
    dates = df['date']
    X = df.drop(columns=['date', 'total_cases'])  # Remove Total cases
    X_selected = fs.fit_transform(X, y)

    # X_selected=pd.concat([X_selected, y] , axis=1)

    selected_col = fs.get_support(indices=True)
    scores=fs.scores_

    Res = X.iloc[:, selected_col]
    Res = pd.concat([y, Res], axis=1)
    Final = pd.concat([dates, Res], axis=1)
    Final = Final.dropna()
    return Final.columns

###############################################################
###############################################################
##### Data  Creation #####
# Greece_total , titles =readdata(loc)
Greece_total=pd.read_csv(r"owid_dataset_fixed.csv")

# Remove  ICU *& Hospital Data from original Dataset
titles = Greece_total.columns

titles.str.contains('adm')
admtitles = titles[titles.str.contains('adm')].to_list()
Greece_total = Greece_total.drop(admtitles, axis=1)


# titles.str.contains('vac')
# admtitles = titles[titles.str.contains('vac')].to_list()
# Greece_total = Greece_total.drop(admtitles, axis=1)

###############################################################
###############################################################


###### Select K Best #####

# results,scores=FeatureSelection(Greece_total,2)
# results=pd.DataFrame(results)
fs = SelectKBest(score_func=f_regression, k='all')
# fs = SelectKBest(score_func=mutual_info_regression, k='all')  # use mutual info
df = Greece_total.dropna()
y = df['total_cases']  # Set Total cases as a Target
dates = df['date']
X = df.drop(columns=['date', 'total_cases','Unnamed: 0'])  # Remove Total cases
X_selected = fs.fit_transform(X, y)


selected_col = fs.get_support(indices=True)
# featnames = fs.get_feature_names_out(indices=True)
print(selected_col)
scores = fs.scores_
Res = X.iloc[:, selected_col]
Res_dataframe = X.iloc[:, selected_col]
Res.loc[len(Res)] = scores
Res=Res.reset_index(drop=True)
Res=Res.tail(1)
Res=Res.T

###############################################################
###############################################################


#Plot Correlation
Greece_total=Greece_total.drop(columns=['date', 'Unnamed: 0'])


total_cases_cor=pd.DataFrame()
#Plot Acctual Correlation (Pearson)

correlation_mat_p = Greece_total.corr()
f = plt.figure(figsize=(40, 40))
plt.title("Pearson's Correlation graph")
sns.heatmap(correlation_mat_p, annot = True)
# plt.show()
total_cases_cor['Pearson'] = correlation_mat_p['total_cases']

#Plot spearman 's Correlation

correlation_mat_s = Greece_total.corr(method='spearman')
f = plt.figure(figsize=(40, 40))
plt.title("Spearman's Correlation graph")
sns.heatmap(correlation_mat_s, annot = True)
# plt.show()
total_cases_cor['Spearman'] = correlation_mat_s['total_cases']
total_cases_cor['K Best'] = Res

ax = plt.gca()
total_cases_cor.plot(y='Pearson',ax=ax,figsize=(20,20))
total_cases_cor.plot(y='Spearman', ax=ax)
# total_cases_cor.plot(y='K Best', ax=ax)
plt.show()


Spearman=total_cases_cor['Spearman']
Spearman=Spearman[Spearman > 0.9]

Pearson=total_cases_cor['Pearson']
Pearson=Pearson[Pearson > 0.9]

K_Best=total_cases_cor['K Best']
# K_Best=K_Best[K_Best > 0]

Spearman=Spearman.index.to_list()
feature_list = list(combinations(Spearman , 5))
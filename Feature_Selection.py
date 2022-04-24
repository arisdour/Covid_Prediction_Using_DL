import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression



from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
    
import itertools
from itertools  import product
from itertools  import combinations


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
    Res = X.iloc[:, selected_col]

    Res = pd.concat([y, Res], axis=1)
    Final = pd.concat([dates, Res], axis=1)
    Final = Final.dropna()
    return Final.columns


##### Data  Creation #####
# Greece_total , titles =readdata(loc)
Greece_total=pd.read_csv(r"owid_dataset_fixed.csv")

# Remove  ICU *& Hospital Data from original Dataset
titles = Greece_total.columns
titles.str.contains('adm')
admtitles = titles[titles.str.contains('adm')].to_list()

Greece_total = Greece_total.drop(admtitles, axis=1)

# Greece_total=Greece_total.drop(columns=['date', 'Unnamed: 0'])





# cases = titles[titles.str.contains('cases')].to_list()
#
# deaths = titles[titles.str.contains('deaths')].to_list()
#
# tests = titles[titles.str.contains('tests')].to_list()
#
# date = titles[titles.str.contains('date')].to_list()
# Feature = cases+deaths+tests+date
# Greece_total=Greece_total[Feature]


results =FeatureSelection(Greece_total, 5)
results=results.to_list()
results=Greece_total[results]
# Greece=Greece_total[results]
#Plot Correlation

Greece_total=Greece_total.drop(columns=['date', 'Unnamed: 0'])

f = plt.figure(figsize=(40, 40))

plt.matshow(Greece_total.corr(), fignum=f.number)
plt.xticks(range(Greece_total.select_dtypes(['number']).shape[1]), Greece_total.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(Greece_total.select_dtypes(['number']).shape[1]), Greece_total.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.show()

correlation_mat = Greece_total.corr()
f = plt.figure(figsize=(40, 40))
sns.heatmap(correlation_mat, annot = True)
plt.show()
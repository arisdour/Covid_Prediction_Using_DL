import pandas as pd


def readdata(location):
    data = pd.read_csv(location)
    Greece = data[data.location == 'Greece'].reset_index(drop='True')
    Greece = Greece.dropna(how='all', axis=1)
    Greece_total = Greece.iloc[7:498, 3:40].reset_index(drop='True')
    titles = Greece_total.columns
    return Greece_total, titles


loc="owid-covid-data.csv"
Greece_total , titles =readdata(loc)


data=pd.read_csv(r"C:\Users\Aris_Dourdounas\Downloads\vaccinations.csv")
Greecevac = data[data.location =='Greece'].reset_index(drop='False')
Greecevac=Greecevac.fillna( axis=0, method="ffill" )
# Greecevac=Greecevac.set_index(Greecevac["date"], drop=False)
Greecevac=Greecevac.drop(['location','iso_code', 'total_boosters','total_boosters_per_hundred'], axis=1)

title=Greece_total.columns
titles.str.contains('vac')
features = titles[titles.str.contains('vac')].to_list()
print(features)
Greece_total=Greece_total.drop(features, axis=1)
Greece_total['new_cases_smoothed']= Greece_total['new_cases'].rolling(window=7).mean()
Greece_total['new_deaths_smoothed']= Greece_total['new_deaths'].rolling(window=7).mean()

Greece_total['new_cases_smoothed_per_million']= Greece_total['new_cases_smoothed']*0.096
Greece_total['new_deaths_smoothed_per_million']= Greece_total['new_deaths_smoothed']*0.096

Greece_total = Greece_total.merge(Greecevac,   how='left' , left_on='date', right_on='date')





######################## Tests ######################## Tests
title=Greece_total.columns
titles.str.contains('test')
tf = titles[titles.str.contains('test')].to_list()
test=Greece_total[tf]
Greece_total=Greece_total.drop(tf , axis=1)

test=test.drop(['tests_units'], axis=1)
test['date']=Greece_total['date']
test=test.set_index('date')


# pd.set_option("display.precision", 2)



df2 = pd.read_json(r"C:\Users\Aris_Dourdounas\Downloads\tests.json")
df2 = pd.DataFrame(df2.values.tolist(), index=df2.index)

res = pd.DataFrame([{'feature' : key, 'value' : value } for d in df2[0].tolist() for key, value in d.items()])


date = res[res['feature'].str.contains('date')].reset_index(drop=True)
rapid = res[res['feature'].str.contains('rapid')].reset_index(drop=True)
rapidtests = res[res['feature'].str.contains('^tests$')].reset_index(drop=True)

newtests=pd.DataFrame()
newtests['tests'] = rapidtests['value']
newtests['rapid_tests'] = rapid['value']
newtests['date'] = date['value']
newtests['total_tests'] = newtests["tests"] + newtests["rapid_tests"]
# newtests=newtests.set_index('date')

Total_Tests=pd.DataFrame()
Total_Tests['total_tests'] = newtests['total_tests']

Total_Tests['date'] = newtests['date']
Total_Tests =Total_Tests.set_index(['date'])

Total_Tests['new_tests'] = Total_Tests['total_tests'].diff()



#Per thousand
Total_Tests['total_tests_per_thousand']=Total_Tests['total_tests']* 0.000096
Total_Tests['new_tests_per_thousand']=Total_Tests['new_tests']* 0.000096

#Smoothed

Total_Tests['new_tests_smoothed']=Total_Tests['new_tests'].rolling(window=7,min_periods=5).mean()
Total_Tests['new_tests_per_thousand_smoothed']=Total_Tests['new_tests_per_thousand'].rolling(window=7, min_periods=5).mean()


Total_Tests['tests_per_case']=test['tests_per_case']
# Total_Tests['Original_total_tests']=test['total_tests']

Total_Tests=Total_Tests.replace('nan', 'lour')


Total_Tests = Total_Tests[Total_Tests['tests_per_case'].notna()]
Total_Tests=Total_Tests.reset_index()

Greece_total = Greece_total.merge(Total_Tests,   how='left' , left_on='date', right_on='date')
import pandas as pd
import numpy as np
def readdata(location):
    data = pd.read_csv(location)
    Greece = data[data.location == 'Greece'].reset_index(drop='True')
    Greece = Greece.dropna(how='all', axis=1)
    Greece_total = Greece.iloc[7:498, 3:40].reset_index(drop='True')
    titles = Greece_total.columns
    return Greece_total, titles


loc="owid-covid-data.csv"


########################  Original Dataset ##############################

Greece_total , titles =readdata(loc)

# Fix Smoothed Data
Greece_total['new_cases_smoothed']= Greece_total['new_cases'].rolling(window=7).mean()
Greece_total['new_deaths_smoothed']= Greece_total['new_deaths'].rolling(window=7).mean()

Greece_total['new_cases_smoothed_per_million']= Greece_total['new_cases_smoothed']*0.096
Greece_total['new_deaths_smoothed_per_million']= Greece_total['new_deaths_smoothed']*0.096


######################## Vaccinations ###################################

data=pd.read_csv(r"C:\Users\Aris_Dourdounas\Downloads\vaccinations.csv") #Read Vacciantion Data
Greecevac = data[data.location =='Greece'].reset_index(drop='False')
Greecevac=Greecevac.fillna( axis=0, method="ffill" ) #Fill NAN

# Greecevac=Greecevac.set_index(Greecevac["date"], drop=False)

Greecevac['new_vaccinations_smoothed']=Greecevac['new_vaccinations'].rolling(window=7).mean()
Greecevac['new_vaccinations_smoothed_per_million']=Greecevac['new_vaccinations_per_million'].rolling(window=7).mean()
Greecevac['new_people_vaccinated_smoothed_per_hundred']=Greecevac['new_people_vaccinated_per_hundred'].rolling(window=7).mean()

Greecevac=Greecevac.drop(['location','iso_code', 'total_boosters','total_boosters_per_hundred'], axis=1)  #Final Vaccination Dataset

######################## ########################  ########################

#Remove  Vaccination Data from original Dataset
title=Greece_total.columns
titles.str.contains('vac')
vacf = titles[titles.str.contains('vac')].to_list()
print(vacf)
Greece_total=Greece_total.drop(vacf, axis=1)

######################## ########################  ########################

########################    Tests       ###################################

#Remove  Test Data from original Dataset
title=Greece_total.columns
titles.str.contains('test')
testf = titles[titles.str.contains('test')].to_list()
test=Greece_total[testf]
Greece_total=Greece_total.drop(testf , axis=1)

#Save original Test Data in DataFrame

test=test.drop(['tests_units'], axis=1)
test['date']=Greece_total['date']
test=test.set_index('date')

New_Test_df = pd.read_json(r"C:\Users\Aris_Dourdounas\Downloads\tests.json")
New_Test_df = pd.DataFrame(New_Test_df.values.tolist(), index=New_Test_df.index)

res = pd.DataFrame([{'feature' : key, 'value' : value } for d in New_Test_df[0].tolist() for key, value in d.items()])


date = res[res['feature'].str.contains('date')].reset_index(drop=True)
rapid = res[res['feature'].str.contains('rapid')].reset_index(drop=True)
tottests = res[res['feature'].str.contains('^tests$')].reset_index(drop=True)

newtests=pd.DataFrame()
newtests['tests'] = tottests['value']
newtests['rapid_tests'] = rapid['value']
newtests['date'] = date['value']
missing_date = pd.DataFrame({"tests": np.NaN, "rapid_tests": np.NaN , "date": '2021-05-02'}, index=[431])
newtests = pd.concat([newtests.iloc[:431], missing_date, newtests.iloc[431:]]).reset_index(drop=True)
newtests['total_tests'] = newtests["tests"] + newtests["rapid_tests"]


# newtests=newtests.set_index('date')

GreeceTests=pd.DataFrame()
GreeceTests['total_tests'] = newtests['total_tests']
GreeceTests['date'] = newtests['date']
GreeceTests=GreeceTests.fillna( axis=0, method="ffill" , limit=2 )


GreeceTests =GreeceTests.set_index(['date'])

GreeceTests['new_tests'] = GreeceTests['total_tests'].diff()



#Per thousand
GreeceTests['total_tests_per_thousand']=GreeceTests['total_tests']* 0.000096
GreeceTests['new_tests_per_thousand']=GreeceTests['new_tests']* 0.000096

#Smoothed

GreeceTests['new_tests_smoothed']=GreeceTests['new_tests'].rolling(window=7,min_periods=5).mean()
GreeceTests['new_tests_per_thousand_smoothed']=GreeceTests['new_tests_per_thousand'].rolling(window=7, min_periods=5).mean()


GreeceTests['tests_per_case']=test['tests_per_case']
# GreeceTests['Original_total_tests']=test['total_tests']

GreeceTests=GreeceTests.replace('nan', np.NaN)


GreeceTests = GreeceTests[GreeceTests['tests_per_case'].notna()]
GreeceTests=GreeceTests.reset_index()

######################## ########################  ########################
########################    FINAL MERGE            ########################

Greece_total = Greece_total.merge(Greecevac,   how='left' , left_on='date', right_on='date')
Greece_total = Greece_total.merge(GreeceTests,   how='left' , left_on='date', right_on='date')
Greece_total.to_csv("owid_dataset_fixed" +".csv", float_format="%.3f",index=True, header=True)
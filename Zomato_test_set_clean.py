# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:39:28 2020

@author: AJINKYA
"""

#cleaning test dataset
# removing the ₹ sign from test set
test["Average_Cost"] = test["Average_Cost"].str.replace("₹","")
test["Minimum_Order"] = test["Minimum_Order"].str.replace("₹","")


#extracting cities from address
test['City.Pune'] = test['Location'].apply(lambda x: 'Pune' if 'Pune' in x else None)
test['City.Kolkata'] = test['Location'].apply(lambda x: 'Kolkata' if 'Kolkata' in x else None)
test['City.Mumbai'] = test['Location'].apply(lambda x: 'Mumbai' if 'Mumbai' in x else None)
test['City.Bangalore'] = test['Location'].apply(lambda x: 'Bangalore' if 'Bangalore' in x else None)
test['City.Delhi'] = test['Location'].apply(lambda x: 'Delhi' if 'Delhi' in x else None)
test['City.Hyderabad'] = test['Location'].apply(lambda x: 'Hyderabad' if 'Hyderabad' in x else None)
test['City.Noida'] = test['Location'].apply(lambda x: 'Noida' if 'Noida' in x else None)
test['City.Gurgaon'] = test['Location'].apply(lambda x: 'Gurgaon' if 'Gurgaon' in x else None)
test['City.Majestic'] = test['Location'].apply(lambda x: 'Bangalore' if 'Majestic' in x else None)
test['City.Marathalli'] = test['Location'].apply(lambda x: 'Bangalore' if 'Marathalli' in x else None)
test['City.Electronic'] = test['Location'].apply(lambda x: 'Bangalore' if 'Electronic' in x else None)
test['City.Gurgoan'] = test['Location'].apply(lambda x: 'Gurgaon' if 'Gurgoan' in x else None)
test['City.Whitefield'] = test['Location'].apply(lambda x: 'Bangalore' if 'Whitefield' in x else None)

test['City'] = test['City.Pune'].map(str)+test['City.Kolkata'].map(str)+test['City.Mumbai'].map(str)+test['City.Bangalore'].map(str)+test['City.Delhi'].map(str)+test['City.Hyderabad'].map(str)+test['City.Noida'].map(str)+test['City.Gurgaon'].map(str)+test['City.Majestic'].map(str)+test['City.Marathalli'].map(str)+test['City.Electronic'].map(str)+test['City.Gurgoan'].map(str)+test['City.Whitefield'].map(str)

test['City'] = test['City'].apply(lambda x: x.replace('None',''))

new_test = test[['Restaurant','Location','City','Cuisines','Average_Cost','Minimum_Order','Rating','Votes','Reviews']]

new_test['Count Cuisine'] = new_test['Cuisines'].apply(lambda x:countoc(x))
del(new_test['Restaurant'])
del(new_test['Location'])
del(new_test['Cuisines'])

new_test['City'], _ = pd.factorize(new_test['City'], sort=True)

new_test.dtypes

cols = ['Average_Cost','Minimum_Order','Rating','Votes','Reviews']
new_test[cols] = new_test[cols].apply(pd.to_numeric, errors = 'coerce')

new_test.dtypes

new_test.isna().sum()

# imputing missing values in training data
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=1, weights="uniform")
new_test=imputer.fit_transform(new_test)
new_test=pd.DataFrame(new_test)

#Assigning Coloumns names as per its index.
new_test.columns = ['City','Average_Cost','Minimum_Order','Rating','Votes','Reviews','Cuisine_Count']

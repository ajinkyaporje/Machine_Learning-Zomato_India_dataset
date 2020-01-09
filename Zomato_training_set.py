# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:50:31 2020

@author: AJINKYA
"""
#importing libraries
import pandas as pd
import numpy as np

#importing train and test data
train = pd.read_excel("F:\dataset\my project data\Participants Data/Data_Train.xlsx")
test = pd.read_excel("F:\dataset\my project data\Participants Data/Data_Test.xlsx")

# removing the ₹ sign from train set
train["Average_Cost"] = train["Average_Cost"].str.replace("₹","")
train["Minimum_Order"] = train["Minimum_Order"].str.replace("₹","")
train["Delivery_Time"] = train["Delivery_Time"].str.replace("minutes","")

#extracting cities from address
train['City.Pune'] = train['Location'].apply(lambda x: 'Pune' if 'Pune' in x else None)
train['City.Kolkata'] = train['Location'].apply(lambda x: 'Kolkata' if 'Kolkata' in x else None)
train['City.Mumbai'] = train['Location'].apply(lambda x: 'Mumbai' if 'Mumbai' in x else None)
train['City.Bangalore'] = train['Location'].apply(lambda x: 'Bangalore' if 'Bangalore' in x else None)
train['City.Delhi'] = train['Location'].apply(lambda x: 'Delhi' if 'Delhi' in x else None)
train['City.Hyderabad'] = train['Location'].apply(lambda x: 'Hyderabad' if 'Hyderabad' in x else None)
train['City.Noida'] = train['Location'].apply(lambda x: 'Noida' if 'Noida' in x else None)
train['City.Gurgaon'] = train['Location'].apply(lambda x: 'Gurgaon' if 'Gurgaon' in x else None)
train['City.Majestic'] = train['Location'].apply(lambda x: 'Bangalore' if 'Majestic' in x else None)
train['City.Marathalli'] = train['Location'].apply(lambda x: 'Bangalore' if 'Marathalli' in x else None)
train['City.Electronic'] = train['Location'].apply(lambda x: 'Bangalore' if 'Electronic' in x else None)
train['City.Gurgoan'] = train['Location'].apply(lambda x: 'Gurgaon' if 'Gurgoan' in x else None)
train['City.Whitefield'] = train['Location'].apply(lambda x: 'Bangalore' if 'Whitefield' in x else None)

train['City'] = train['City.Pune'].map(str)+train['City.Kolkata'].map(str)+train['City.Mumbai'].map(str)+train['City.Bangalore'].map(str)+train['City.Delhi'].map(str)+train['City.Hyderabad'].map(str)+train['City.Noida'].map(str)+train['City.Gurgaon'].map(str)+train['City.Majestic'].map(str)+train['City.Marathalli'].map(str)+train['City.Electronic'].map(str)+train['City.Gurgoan'].map(str)+train['City.Whitefield'].map(str)

train['City'] = train['City'].apply(lambda x: x.replace('None',''))

new_train = train[['Restaurant','Location','City','Cuisines','Average_Cost','Minimum_Order','Rating','Votes','Reviews','Delivery_Time']]

#writing file to path
new_train.to_csv("F:\dataset\my project data\Participants Data/citywisedata.csv")

#creating a function to count number of cuisines in particular restaurant
def countoc(s):
    a = s.split(',')
    return len(a)

new_train['Count Cuisine'] = new_train['Cuisines'].apply(lambda x:countoc(x))
del(new_train['Restaurant'])
del(new_train['Location'])
del(new_train['Cuisines'])

new_train['City'], _ = pd.factorize(new_train['City'], sort=True)

new_train.dtypes

cols = ['Average_Cost','Minimum_Order','Rating','Votes','Reviews','Delivery_Time']
new_train[cols] = new_train[cols].apply(pd.to_numeric, errors = 'coerce')

new_train.dtypes

new_train.isna().sum()

# imputing missing values in training data
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=1, weights="uniform")
new_train=imputer.fit_transform(new_train)
new_train=pd.DataFrame(new_train)

#Assigning Coloumns names as per its index.
new_train.columns = ['City','Average_Cost','Minimum_Order','Rating','Votes','Reviews','Delivery_Time','Cuisine_Count']

#seperating x and y variables
x = new_train.iloc[:,[0,1,2,3,4,5,7]].values
y = new_train.iloc[:, 6]


"""
#Fitting SVM model to training dataset using linear kernel
from sklearn.svm import SVC
classifier=SVC(kernel='linear',gamma='auto')
classifier.fit(x,y)

#checking the accuracy of SVM model on training dataset using linear as kernel
SVM_Accuracy = classifier.score(x,y)
print("Accuracy for SVM Classification is :", SVM_Accuracy)
"""
"""
Applying XGBoost Regressor to predict time

"""

from xgboost import XGBRegressor
import xgboost as xgb
xgb_regressor = XGBRegressor()
xgb_regressor.fit(x,y)

#checking accuracy of training model
XGBoost_Accuracy = xgb_regressor.score(x,y)
print("Accuracy for XGBoost Regressor is :", XGBoost_Accuracy)

test_new1 = np.array(new_test)

#predicting grade values on test data
y_predictions_xg = xgb_regressor.predict(test_new1)
new_test.columns = ['City','Average_Cost','Minimum_Order','Rating','Votes','Reviews','Cuisine_Count']

test_new1 = pd.DataFrame(test_new1)
test_new1['delivery time'] = y_predictions_xg

#Applying XGBoost Classifier to predict time
from xgboost import XGBClassifier
import xgboost as xgb
xgb_classifier = XGBClassifier()
xgb_classifier.fit(x,y)

#checking accuracy of training model
XGBoost_Accuracy = xgb_classifier.score(x,y)
print("Accuracy for XGBoost Classifier is :", XGBoost_Accuracy)


test_xgc = np.array(new_test)

#predicting grade values on test data
y_predictions_xgbclassifier = xgb_classifier.predict(test_xgc)
test_xgc.columns = ['City','Average_Cost','Minimum_Order','Rating','Votes','Reviews','Cuisine_Count', 'delivery time']

test_xgc = pd.DataFrame(test_xgc)
test_xgc['delivery time'] = y_predictions_xgbclassifier

test_city = pd.concat([city_merge,test_xgc], axis = 1)
city_merge = test.iloc[:,[0,1,2,-1]]
test_final = test_city.iloc[:,[0,1,2,3,5,6,7,8,9,10,11]]

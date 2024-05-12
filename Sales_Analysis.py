# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:19:44 2024

@author: Tom Tremerel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Walmart_sales.csv')

df.head()

hd = df.describe()

df.isnull().sum()
df.duplicated().sum()

df.Temperature = round((df.Temperature-32)/1.8, 2)

 

#Visualization // EDA

plt.figure(figsize=(10,5))
sns.histplot(df['Weekly_Sales'], kde=True)
plt.title(label = "Weeklysales")
sns.kdeplot(df['Fuel_Price'], fill=True) 
sns.regplot(df, x=df['Fuel_Price'], y= df['CPI'])


df['Date'] = pd.to_datetime(df['Date'], format = '%d-%m-%Y')
df['Year'] = df['Date'].dt.year

plt.figure(figsize =(15,15))
sns.lineplot(df, x=df['Year'], y=df['CPI']) 
plt.title('Consumer Price Index By Date')

df = df.drop(columns=['Date'])
df = df.drop(columns=['Year'])


correlation_matrix = df.corr()
plt.figure(figsize=(15,12))
sns.heatmap(correlation_matrix, annot=True, cmap='Oranges')

coul = sns.color_palette()

plt.figure(figsize=(15,11))
sns.barplot(df, x=df['Store'], y=df['Unemployment']<8,palette= coul )

#Preprocessing

scaler = StandardScaler()
df[['CPI','Temperature','Fuel_Price','Unemployment']] = scaler.fit_transform(df[['CPI','Temperature','Fuel_Price','Unemployment']])

#Split

X = df
X = X.drop(columns =['Weekly_Sales'])
X.shape
y = df['Weekly_Sales']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)

#Models 

#===> Random Forest
RF = RandomForestRegressor()
RF.fit(X_train, y_train)
#===> Linear Regression
LR = LinearRegression()
LR.fit(X_train, y_train)
#===>GradientBoostingRegressor
XGBR=GradientBoostingRegressor()
XGBR.fit(X_train,y_train)

#Models evaluation 

y_pred_RF = RF.predict(X_test)
y_pred_LR = LR.predict(X_test)
y_pred_XGBR = XGBR.predict(X_test)

plt.figure(figsize=(10, 7))
plt.Subplot(1,3,1)
plt.scatter(x=y_test, y=y_test, color='blue', label='Réalité')
plt.scatter(x=y_pred_RF, y=y_pred_RF, color='red', label='Prédiction')
plt.title('Comparaison des prédictions avec les valeurs réelles => RandomForest model')
plt.xlabel('Valeurs réelles')  
plt.ylabel('Prédictions')  
plt.legend()  

plt.Subplot(1,3,2)
plt.scatter(x=y_test, y=y_test, color='blue', label='Réalité')
plt.scatter(x=y_pred_LR, y=y_pred_LR, color='yellow', label='Prédiction')
plt.title('Comparaison des prédictions avec les valeurs réelles => LinearRegression model')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.legend()
 
plt.Subplot(1,3,3)
plt.scatter(x=y_test, y=y_test, color='blue', label='Réalité')
plt.scatter(x=y_pred_XGBR, y=y_pred_XGBR, color='green', label='Prédiction')
plt.title('Comparaison des prédictions avec les valeurs réelles => GradientBoostRegression model')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.legend()
plt.show()  

def cal_error(model_name,y_pred) :
    mae = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return pd.DataFrame({'model name':model_name,'MAE':mae,'MSE':mse, 'RMSE':rmse}, index=[0])


RF_result = cal_error('Random Forest Regressor',y_pred_RF)
LR_result = cal_error('Linear Regression', y_pred_LR)
XGBR_result = cal_error('XGBR', y_pred_XGBR)

models_result = pd.concat([RF_result,LR_result,XGBR_result])

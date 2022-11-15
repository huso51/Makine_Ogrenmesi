#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")


#veri on isleme

#encoder:  Kategorik -> Numeric
veriler2 = veriler.apply(LabelEncoder().fit_transform) #bütün kolonlar üzerine labelencoder'i uygulamış

c = veriler2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray() //ilk kolona onehotencoder'i uygula
print(c)

havadurumu = pd.DataFrame(data = c, index = range(14), columns=['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)#havadurumu ile 1 ila 3 kolonlarını birleştir axis=1 demek kolon bazında birleştir demek
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)#veriler 2'nin son iki kolonunu al sonveriler ile birleştir


#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)#son kolon hariç bizim bağımsız değişkenlerimiz yani son kolon bağımlı değişken oda humidity

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)#linear regressionu uygula


y_pred = regressor.predict(x_test)#tahmin et

print(y_pred)
#backward elimination
import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )#son kolon hariç hepsini al
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)#son kolonu baz al yani bağımlı değişkeni
r = r_ols.fit()
print(r.summary())

sonveriler = sonveriler.iloc[:,1:]#ilk satırı at çünkü p değeri yüksek çıktı yüksel olan değer regressionu olumusz yönde etkiler

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]#1. satırı attık çünkü p değeri yüksek çıktı yüksel olan değer regressionu olumusz yönde etkiler
x_test = x_test.iloc[:,1:]#1. satırı attık çünkü p değeri yüksek çıktı yüksel olan değer regressionu olumusz yönde etkiler

regressor.fit(x_train,y_train)#regression modelini oluşturduk


y_pred = regressor.predict(x_test)#ve tahmin ettik








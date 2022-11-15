# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]#eğitim seviyesi kolonunu al
y = veriler.iloc[:,2:]#maaş kolonunu al
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()#doğrusal regresyon nesnesi oluşturduk kansıtrakçırı çağırarak
lin_reg.fit(X,Y)#x'den y'e öğren

plt.scatter(X,Y,color='red')#x y boyutunda bir grafik oluştur
plt.plot(x,lin_reg.predict(X), color = 'blue')#model üzerinden tahmin et ve sonucu çiz
plt.show()#grafiği göster


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)#bir tane polynominal fetatures nesnesi oluşturduk 2. dereceden çünkü polinominal regressyon yapacaz
x_poly = poly_reg.fit_transform(X)#X değerini polinom boyutuna dönüştür
print(x_poly)
lin_reg2 = LinearRegression()#doğrusal regresyon nesnesi oluşturduk kansıtrakçırı çağırarak
lin_reg2.fit(x_poly,y)#x_poly'den y'e öğren
plt.scatter(X,Y,color = 'red')#x y boyutunda bir grafik oluştur
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')#önce X değerini polinom boyutuna dönüştür sonra model üzerinden tahmin et ve sonucu çiz
plt.show()#grafiği göster

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)#bir tane polynominal fetatures nesnesi oluşturduk 4. dereceden çünkü polinominal regressyon yapacaz
x_poly = poly_reg.fit_transform(X)#X değerini polinom boyutuna dönüştür
print(x_poly)
lin_reg2 = LinearRegression()#doğrusal regresyon nesnesi oluşturduk kansıtrakçırı çağırarak
lin_reg2.fit(x_poly,y)#x_poly'den y'e öğren
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')#önce X değerini polinom boyutuna dönüştür sonra model üzerinden tahmin et ve sonucu çiz
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))











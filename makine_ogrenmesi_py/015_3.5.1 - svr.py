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

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))




#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()#standart ölçek nesnesi tanımladık

x_olcekli = sc1.fit_transform(X)#X'i ölçekledik

sc2=StandardScaler()#standart ölçek nesnesi tanımladık
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')#support vector regression nesnesi tanımladık kernel function olarak rbf tanımladık. burayı istersek linear regression veya polinomal regression tanımlayabilirdik
svr_reg.fit(x_olcekli,y_olcekli)#iki değer arasındaki ilişkiyi bul(öğren) dedik

plt.scatter(x_olcekli,y_olcekli,color='red')X ve Y boyutu olarak grafik oluştur
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')#x_olcekli'ye göre tahmin et sonucu grafikte göster


#tahminler
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))










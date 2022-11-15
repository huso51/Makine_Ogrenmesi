# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.txt')
#pd.read_csv("veriler.csv")
#test
print(veriler)
Yas = veriler.iloc[:,1:4].values
print(Yas)

#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#encoder: Kategorik -> Numeric
c = veriler.iloc[:,-1:].values
print(c)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)


#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)


#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression #sklearn.linear_model dosyasından LinearRegression class'ını import ettik
regressor = LinearRegression() #bir adet linearregression constructoru çalıştırırak obje oluşturduk
regressor.fit(x_train,y_train) #x_trainde 6 boyut var. yani 6 kolon var. x_trainden y_train'e öğren yaptık(burada öğrenmeden kastımız aralarında linear bir model kurmak(istatiksel bir model oluşturmak))

y_pred = regressor.predict(x_test) #X'in test olarak ayrılmış kısmını daha önce öğrendiğin yönteme(fit(öğrenmek)) göre predict(tahmin) et. çıkan sonucuda y_pred'e yaz aktar 

'''
plt.title("Yas, ulke, kilo ve cinsiyete gore boy tahmini")
plt.xlabel("Yas ulke ve cinsiyet")
plt.ylabel("Boy")
plt.plot(x_train,y_train) #bu grafikdeki aşağı yukarı olan çizgiler
plt.plot(x_test, regressor.predict(x_test)) #bu çapraz düz çizgi
'''

boy = s2.iloc[:,3:4].values #boy kolonunu çektik 3. kolon 4 satır çektik
print(boy) #"boy değişkenini yazdık
sol = s2.iloc[:,:3] # bütün satırları al 3. kolona kadar
sag = s2.iloc[:,4:] #4. kolondan sonrakileri al

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0) # bağımsız değişken olan veri'yi kullanarak bağımlı değişken olan boy'a göre veri kümesini böl dedik. 


r2 = LinearRegression()
r2.fit(x_train,y_train) #yeni x_trainden yeni y_train'e öğren dedik (burada öğrenmeden kastımız aralarında linear bir model kurmak(istatiksel bir model oluşturmak))

y_pred = r2.predict(x_test) #yeni X'in test olarak ayrılmış kısmını daha önce öğrendiğin yönteme(fit(öğrenmek)) göre predict(tahmin) et. çıkan sonucuda y_pred'e yaz aktar 

'''
plt.title("Yas, ulke, kilo ve cinsiyete gore boy tahmini")
plt.xlabel("Yas ulke ve cinsiyet")
plt.ylabel("Boy")
plt.plot(x_train,y_train) #bu grafikdeki aşağı yukarı olan çizgiler
plt.plot(x_test, r2.predict(x_test)) #bu çapraz düz çizgi
'''















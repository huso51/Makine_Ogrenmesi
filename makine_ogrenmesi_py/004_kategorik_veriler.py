# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 13:20:21 2021

@author: husey
"""

import pandas as pd #veriler için verileri düzgün bir şekilde tutabilmemiz için
import numpy as np # büyük sayılar için ve hesaplamalar için kullandığımız kütüphane
import matplotlib.pyplot as plt # çizimler için kullandığımız kütüphane

#kategorik verilerin sayısal verilere çevrilmesi
veriler = pd.read_csv('eksikveriler.csv.txt')



print(veriler)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
yas = veriler.iloc[:,1:4].values #yas kolonunun 1den 4e kadar olan kolonları çektik iloc integer location demek 
imputer = imputer.fit(yas[:,1:4]) #fit öğreneme demek. yas kolonunun ortalamasını öğrendik(yas kolonunun 1 den 4 e kadar olan kolonlarını öğrendik)
yas[:,1:4] = imputer.transform(yas[:,1:4]) #transform ile öğrendiğimiz ortalamayı nan olan değerlere uyguladık
print(yas)

ulke = veriler.iloc[:,0:1].values #ulkee kolonunun bütün satırlarını çektik
print(ulke) #ulkee değişkenini ekrana yazdık
from sklearn import preprocessing #sklearn dosyasından preprocessing'i import ettik
le = preprocessing.LabelEncoder() #label encoder nesnesini tanımladık constructor'u çalıştırarak
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) #fit öğrenme demek. ulkee kolonunun verilerini öğreniyoruz ve transform(transform dönüşüm demek tr fr us kelimelerini sayısala dönüştürüyor.) ediyoruz. 0 ilk kolon demek.
print(ulke)
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray() #öğrenme işlemimiz öğrenecek ülkee kolonundan sonra transform edecek onuda array olarak ulke değişkenine atayacak
print(ulke)

print(range(22))
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values #bütün satırları içeren ve son kolonu içeren values'i alıyoruz
sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis = 1) #sonuç ve sonuç2 dataframelerini birleştirdik axis sonuc dataframesindeki 1. kolonu ve sonuç2 dataframesindeki 1. kolonu birleştirdik
s2 = pd.concat([s, sonuc3], axis = 1) #sonuc2 ve s dataframelerini birleştirdik axis sonuc2 dataframesindeki 1. kolonu ve s dataframesindeki 1. kolonu birleştirdik
print(s2)



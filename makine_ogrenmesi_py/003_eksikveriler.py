# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:21:35 2020

@author: husey
"""

#eksikveriler
import pandas as pd #veriler için verileri düzgün bir şekilde tutabilmemiz için
import numpy as np # büyük sayılar için ve hesaplamalar için kullandığımız kütüphane
import matplotlib.pyplot as plt # çizimler için kullandığımız kütüphane

veriler = pd.read_csv('eksikveriler.csv.txt')

print(veriler)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
yas = veriler.iloc[:,1:4].values #yas kolonunun 1den 4e kadar olan kolonları çektik iloc integer location demek 
imputer = imputer.fit(yas[:,1:4]) #fit öğreneme demek. yas kolonunun ortalamasını öğrendik(yas kolonunun 1 den 4 e kadar olan kolonlarını öğrendik)
yas[:,1:4] = imputer.transform(yas[:,1:4]) #transform ile öğrendiğimiz ortalamayı nan olan değerlere uyguladık
print(yas)


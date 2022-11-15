# -*- coding: utf-8 -*-
#ders 6 : kütüphanelerin yüklenmesi

#kütüphaneler
import pandas as pd #veriler için verileri düzgün bir şekilde tutabilmemiz için
import numpy as np # büyük sayılar için ve hesaplamalar için kullandığımız kütüphane
import matplotlib.pyplot as plt # çizimler için kullandığımız kütüphane

#kod bölümü

#veri yükleme
veriler = pd.read_csv('veriler.txt')


#veri ön işleme
print(veriler)
boy = veriler[['boy']]
boykilo = veriler[['boy','kilo']]
print('selam')

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:18:29 2024

@author: akdag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#veri yükleme
datas = pd.read_csv('../Datas/musteriler.csv')


 
X = datas.iloc[:,2:4].values # independent variables
Y = datas.iloc[:,4:].values # dependent  


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2 , init='k-means++')
kmeans.fit(X)

labels = kmeans.predict(X)
centers = kmeans.cluster_centers_

for i in labels:
    plt.scatter(X[labels == i , 0] , X[labels == i , 1] , label = i)
plt.scatter(centers[:,0] , centers[:,1] , s = 80, color = 'k')
plt.show()





print(kmeans.cluster_centers_)

sonuclar = []

for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123 )
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) # verilerin noktaya uzaklık değerleri
    
plt.plot(range(1,10), sonuclar)


"""
    ideal küme sayısı belirleme metotları : 
        Elbow method
        Silhouette method
        Gap statistic method
       
"""
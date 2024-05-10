# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:51:28 2024

@author: akdag
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv("../Datas/maaslar_yeni.csv")

X = dataset.iloc[:,2:]

plt.figure(figsize=(10, 6))
plt.scatter(X['Puan'], X['maas'])

plt.legend()

plt.show()


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=5, min_samples=3, metric='euclidean')

labels = dbscan.fit_predict(X)

plt.scatter(X['Puan'], X['maas'], c=labels, cmap='viridis')

plt.title('DBSCAN Clustering Results')
plt.colorbar(label='Cluster Label')
plt.show()






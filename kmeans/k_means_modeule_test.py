# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:23:03 2024

@author: akdag
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kmeans_module import K_Means_Alg


#veri yükleme
datas = pd.read_csv('../Datas/musteriler.csv')


 
X = datas.iloc[:,2:4].values # independent variables
Y = datas.iloc[:,4:].values # dependent  

kmeans = K_Means_Alg(10)

centroids = kmeans.fit(X)

colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(centroids)))

# Veri noktalarını ve küme merkezlerini farklı renklerle çizdirme
for label, centroid in centroids.items():
    data_points = X[kmeans.labels == label]
    plt.scatter(data_points[:, 0], data_points[:, 1], c=colors[label], label=f'Küme {label+1}')
    plt.scatter(centroid[0], centroid[1], marker='x', color='black', label=f'Merkez {label+1}')

# Veri setindeki tüm veri noktalarını farklı renklerle ekleme
plt.scatter(datas.iloc[:, 2], datas.iloc[:, 3], c=colors[kmeans.labels], alpha=0.3, label='Veri Noktaları')

# Etiketleme ve başlık ekleme
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.title('K-Means Kümeleme Sonucu (Renkli Kümeler)')
#plt.legend()

# Görselleştirmeyi gösterme
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:13:24 2024

@author: akdag
"""

import math
import numpy as np

class KNN_alg:
    
    def __init__(self, k, metric = 'euclidean' ):
        self.k = k
        self.metric = metric
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        
    def calculate_euclidean(self,point1, point2):
        col_len = point2.shape[0]

        euclidean_distance = 0;
        for i in range(0, col_len):
            euclidean_distance += (point2[i] - point1[i])**2
            
        euclidean_distance = math.sqrt(euclidean_distance)
        
        return euclidean_distance   
    
    
    def find_neighbors(self, x):
        neighbors = []
        print(self.x.shape[0])
        # tüm verilerin satır sayısı kadar dön ve tüm noktaların x e uzaklığını hesapla
        for i in range(0, self.x.shape[0]):
            euclidean = self.calculate_euclidean(x, self.x[i])
            # hesapladıklarını neighborsa ekle
            neighbors.append([euclidean, self.x[i], self.y[i]])
        
        # euclidean uzaklıına göre küçükten büyüğe sıralama
        neighbors_sorted = sorted(neighbors, key=lambda nb: nb[0])
        # daha sonra ilk k tanesi seçilecek
        return neighbors_sorted  
        
        
    def classificate(self, x):
        # komşular bulunur
        neighbors = self.find_neighbors(x)
        # k tanesi alınır
        neighbors = neighbors[:self.k]
        
        # komşuların labellerinin kaç tane olduğunu tutacak bir dict
        class_counts = {}
        
        for neighbor in neighbors:
            # komşunun labeli ne
            neighbor_class = neighbor[2][0]
            print(neighbor_class)
            # eğer class counts içinde label varsa sayısını artır, yoksa dict' ekle ve 2 yap
            if neighbor_class in class_counts:
                class_counts[neighbor_class] += 1  # sınıfın sayısını bir artır
            else:
                class_counts[neighbor_class] = 1
        
        # çoğunluk oylaması için dictionary içindeki en çok olan label
        most_common_class = max(class_counts, key=class_counts.get)
        
        return most_common_class  # en fazla tekrar eden sınıfı döndür
    
    def predict(self, x):
        result = []
        # labelleri bilinmeyenbir veri setini sınıflandırma
        for test_sample in x:
            predicted_class = self.classificate(test_sample)
            result.append(predicted_class)
        
        result = np.array(result)
        return result
    
    
        
        
        
        
        
    
    
    
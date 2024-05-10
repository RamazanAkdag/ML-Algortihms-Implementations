# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 00:31:11 2024

@author: akdag
"""

import numpy as np
import math

import matplotlib.pyplot as plt





class K_Means_Alg:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids = {}
        self.labels = []
        self.X = []
    
    
    def calculate_euclidean(self, point1, point2):
        col_len = point1.shape[0]
        print("point1 : ",point1)
        print("point2: ", point2)
        euclidean_distance = 0;
        for i in range(0, col_len):
            euclidean_distance += (point2[i] - point1[i])**2
            
        euclidean_distance = math.sqrt(euclidean_distance)
        return euclidean_distance   
    
    
    def calculate_min_centroid(self, point):
        min_distance = np.inf
        closest_centroid_label = None
        

        # Her bir küme merkezi için uzaklığı ve en yakın merkezi hesaplama
        for label, centroid in self.centroids.items():
            distance = self.calculate_euclidean(point, centroid)
            if distance < min_distance:
                min_distance = distance
                closest_centroid_label = label
        print("calculate min centroids, point:",point,"centroid: ", closest_centroid_label)
        return closest_centroid_label
        
    
    def update_centroid(self, label):
        points_on_label = []
        
        for i in range(len(self.labels)):
            # labellerin bulunduğu diziden labeli == parametredeki label olan elemanlar bulunur
            if self.labels[i] == label:
                points_on_label.append(self.X[i])
        
        # bulunan elemanların ortalaması alınarak yeni merkez bulunur
        means = np.mean(points_on_label, axis=0)
        print("means : ",means)
        # yeni merkez güncellenir
        self.centroids[label] = means
            
        
    def fit(self, X):
        data_points = X.shape[0]
        self.X = X
        
        # n_clusters sayısı kadar rastgele nokta seçilir
        if data_points >= self.n_clusters:
            centroid_indices = np.random.choice(data_points, size=self.n_clusters, replace=False)
            # centroidler ayrı bir diziye aktarılır
            for i in range(self.n_clusters):
                self.centroids.update({i: X[centroid_indices[i], :]}) 
            print("centroids : ", self.centroids)
        else:
            raise ValueError("Veri sayısı küme sayısından az olamaz.")
        
        print("------centroids : ", self.centroids)
        # uzaklıkları hesaplama
        for point in X:
          # Her bir küme merkezi için uzaklığı ve en yakın merkezi hesaplama
          closest_centroid_label = self.calculate_min_centroid(point)
          self.labels.append(closest_centroid_label)
          # her bir eleman bir kümeye eklenince o kümenin merkezi elemanların ortalaması oluyor
          self.update_centroid(closest_centroid_label)
        
        return self.centroids
          
        
       

          
          
          
          
        
        
      
        
        
        
                
                
                
                
            
        
        
        
        
        
        
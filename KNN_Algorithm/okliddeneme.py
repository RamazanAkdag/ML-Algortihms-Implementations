# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:41:59 2024

@author: akdag
"""

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

from knn_module import KNN_alg


datas = pd.read_csv('../Datas/veriler.csv')

x = datas.iloc[:,1:4].values

y = datas.iloc[:,-1:].values


deneme_data = np.array([130,20,5])




knn = KNN_alg(4)

#knn.fit(x, y)
#y_pred = knn.classificate(deneme_data)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0, test_size=0.33)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)







"""def calculate_euclidean(point1, point2):
    col_len = point1.shape[0]
    euclidean_distance = 0;
    for i in range(0, col_len):
        euclidean_distance += (point2[i] - point1[i])**2
        
    euclidean_distance = math.sqrt(euclidean_distance)
    return euclidean_distance   


for i in range(0,x_row):
        cur_x = x[i]
        euclidean = calculate_euclidean(deneme_data, cur_x)
        print(euclidean)"""
        
    

    
    
 
    
    
    
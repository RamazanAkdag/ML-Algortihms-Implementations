# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:05:24 2024

@author: akdag
"""

import numpy as np
import pandas as pd
import random


class Perceptron:
    
    def __init__(self):
        self.weights = []
        
    
    def step_func(self, x):
        return 1 if x > 1 else 0
        
    
    def learn(self, x, y, lr, epochs):
        
        # Başlangıçta rastgele ağırlıklar 
        self.weights = [round(random.uniform(0.0, 2.0), 1) for _ in range(x.shape[1])]
        
        
        total = 0
        for epoch in range(epochs):
            for i in range(x.shape[0]):
                total = np.dot(self.weights, x[i])
                step_val = self.step_func(total)
                
                if step_val != y[i]:
                    # Ağırlıkları güncelle
                    for k in range(len(self.weights)):
                        self.weights[k] += lr * (y[i] - step_val) * x[i][k]
                    
    
    def predict(self, x):
        total = np.dot(self.weights, x)  # Dot product kullanarak toplam değeri hesapla
        step_val = self.step_func(total)
        return step_val
    


            
        
  # tek perceptron                   
x_train = np.array([[0, 0], [0, 1], [1, 0]])
y_train = np.array([0, 1, 1])

per = Perceptron()
per.learn(x_train, y_train, lr=0.1, epochs=100)

x_test = np.array([0, 1])
prediction = per.predict(x_test)
print("Prediction:", prediction)    









# XOR Problemi Denemesi

perceptron1 = Perceptron()
perceptron2 = Perceptron()
output_perceptron = Perceptron()

"""
o----
     ----o-> output
o----

"""

# XOR Training Data
inputs = np.array([[0, 0], [0, 1], [1, 0]])
targets = np.array([0, 1, 1])

perceptron1.learn(inputs, targets, lr=0.01, epochs=100)
perceptron2.learn(inputs, targets, lr=0.01, epochs=100)


outputs = np.array([[]])
for x in inputs: 
    outputs += [perceptron1.predict(x), perceptron2.predict(x)]

print("outputs: ",outputs)





               
            
            
                
            
            
        
    
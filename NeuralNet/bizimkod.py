# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:35:48 2024

@author: akdag
"""

import numpy as np


"""def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative( net):
    return sigmoid(net)*(1 - sigmoid(net))

X = [[0,0],[0,1], [1,0], [1,1]]
y = [0,0,0,1]

X = np.array(X)
y = np.array(y)

weights = np.random.randn(2)

epochs = 100
learning_rate = 0.01

#print(len(X))

for epoch in range(epochs):
    for i in range(len(X)): 
        #print(X[i])
        toplam_fonk_degeri = np.dot(X[i], weights)
        cikti_degeri = sigmoid(toplam_fonk_degeri)
        
        hata_degeri = y[i] - cikti_degeri
        
        # agirlikalir güncelle
       
        for k in range(len(weights)):
            weights[k] += learning_rate * hata_degeri * X[i][k]
        #weights += learning_rate*hata_degeri*X
        
        print(f'Epoch {epoch}, Error: {np.mean(np.abs(hata_degeri))}')
        

   
pred = sigmoid(np.dot(X, weights))
print("predictions : ")
print(pred)"""

def sign(x):
    if x >= 0:
        return 1
    else:
        return 0

# Define training data (AND problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Initialize weights randomly
w = np.random.rand(2)  # Weights for input features x1 and x2
b = 0  # Bias term

# Learning rate
learning_rate = 0.1
epochs = 20

# Training loop
for epoch in range(5):  # Set the number of epochs (iterations)
    for i in range(len(X)):
        # Forward pass: Calculate predicted output
        toplam = np.dot(X[i], w) + b
        out = sign(toplam)

        # Error calculation
        error = y[i] - out

        # Update weights using the perceptron learning rule
        w += learning_rate * error * X[i]
        b += learning_rate * error
        print(f'Epoch {epoch}, Error: {np.mean(np.abs(error))}')
        

for x in X:
    toplam = np.dot(x, w) + b
    out = sign(toplam)
    print(out)

 
 
        
        
       
        








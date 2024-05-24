# -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:52:04 2024

@author: akdag
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Girdi verileri
y = np.array([[0], [1], [1], [0]])  


input_size = X.shape[1]
data_size = X.shape[0]

layer1_neuron_size = 2
output_neuron_size = 1

layer1_weights = np.random.randn(input_size,layer1_neuron_size) * 0.1
layer1_bias = np.random.randn(1, layer1_neuron_size) * 0.1
output_weights = np.random.randn(layer1_neuron_size, output_neuron_size) * 0.1
output_bias = np.random.randn(1, output_neuron_size) * 0.1

epochs = 10000
learning_rate = 0.6


layer1_net = np.dot(X, layer1_weights) + layer1_bias
layer1_fnet = sigmoid(layer1_net)

output_net = np.dot(layer1_fnet, output_weights) + output_bias
output_fnet = sigmoid(output_net)

    # hata hesaplama
error = y - output_fnet
delta_output = error*sigmoid_derivative(output_fnet)

error_layer1 = delta_output.dot(output_weights.T)
delta_layer1 = error_layer1*sigmoid_derivative(layer1_fnet)
    
    # backward
output_weights += learning_rate * layer1_fnet.T.dot(delta_output)
output_bias += learning_rate*np.sum(delta_output, axis=0, keepdims=True)
    
layer1_weights += learning_rate * X.T.dot(delta_layer1)
layer1_bias += learning_rate*np.sum(delta_layer1, axis=0, keepdims=True)
  
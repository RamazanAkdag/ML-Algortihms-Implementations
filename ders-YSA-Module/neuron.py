# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:41:50 2024

@author: akdag
"""
import numpy as np

class Neuron():
    
    def __init__(self, size, transfer):
        self.weight = np.random.randn(size)
        self.bias = np.random.randn(1)
        self.transfer = transfer
        
    
    def sum(self, x):
        return np.dot(x, self.weight) + self.bias
    
    def calculate(self, input):
        self.input = input
        self.output = self.transfer.calculate(self.sum(input))
        return self.output
    
    
    def update(self, input, error, output):
        delta = error * self.transfer.derivative(input)
        #self.weight += learing_rate * x.T.dot(delta)
    
    
        
    
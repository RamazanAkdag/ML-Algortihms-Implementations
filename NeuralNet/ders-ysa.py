# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:37:36 2024

@author: akdag
"""

import numpy as np



# ileri beslemede transfer fonksiyonu olarak


epochs = 10
learning_rate = 0.1

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([0,0,0,1])



weights = np.array([0.01, 0.01])
bias = -0.1


# keras kütüphanesinikullanmayı öğren
    

"""class NeuralNet:
    def __init__(self, neurons):
        #başlangıç 
        for i in range(len(neurons) - 1):
            # Katmandaki nöron sayısı
            n_neurons_in = neurons[i]
            # Sonraki katmandaki nöron sayısı
            n_neurons_out = neurons[i + 1]
            # Rastgele ağırlık matrisi oluşturun
            self.weights.append(np.random.rand(n_neurons_out, n_neurons_in))
            
        self.bias = np.random.ra
    
    def sigmoid(net):
        return 1 / (1 + np.exp(-net))


    def sigmoid_derivate(self, net):
        return self.sigmoid(net)*(1 - self.sigmoid(net))
    
    def learn(self, x, y, lr=0.01, epochs=100):
        for epoch in range(epochs):
            toplam_hata = 0
            
            for inp, gercek_cikis in zip(x, y):
                # toplam fonksiyonu
                net = np.dot(inp, weights) + bias
                ag_cikis = self.sigmoid(net)
                
                # hatanın karesi
                hata = gercek_cikis - ag_cikis
                toplam_hata += hata
                
                delta = learning_rate * hata * self.sigmoid_derivate(net) 
                delta_w = delta* inp
            
                self.weights += delta_w
                bias += delta
            
            print(f'{epoch} - epoch: {toplam_hata}')
    
    
"""







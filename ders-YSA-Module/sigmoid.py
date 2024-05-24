# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:19:41 2024

@author: akdag
"""
import numpy as np
from transfer import Transfer

class Sigmoid(Transfer):
    
    def calculate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return self.calculate(x)*(1 - self.calculate(x))
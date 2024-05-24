# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:22:05 2024

@author: akdag
"""

from abc import ABC, abstractmethod

class Transfer(ABC):
    
    @abstractmethod
    def calculate(self):
        pass
    
    @abstractmethod
    def derivative(self):
        pass
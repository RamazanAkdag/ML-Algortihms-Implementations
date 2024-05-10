# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:46:08 2024

@author: akdag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv("../Datas/musteriler.csv")

X = datas.iloc[:,2:4].values

from kmeans_module import K_Means_Alg

kmeans = K_Means_Alg(3)

kmeans.fit(X)
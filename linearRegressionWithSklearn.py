# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 19:32:19 2022

@author: espoir
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as ptl

x,y= make_regression(n_samples=100,n_features=1,noise=10)

y=y.reshape(100,1)

model = SGDRegressor(max_iter=100, eta0=0.01)
model.fit(x,y)
ptl.scatter(x, y)
ptl.plot(x, model.predict(x), c='red', lw = 3) 

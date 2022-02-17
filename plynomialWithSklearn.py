# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 19:23:20 2022

@author: Med

"""
from sklearn.datasets import make_regression
import matplotlib.pyplot as ptl

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor

x,y= make_regression(n_samples=100,n_features=1,noise=10)

y=y.reshape(100,1)

#make it polynomial
y=y**2
poly_features = PolynomialFeatures(degree=2, include_bias=False) 
x = poly_features.fit_transform(x) 

print(x)
model = SGDRegressor(max_iter=1000, eta0=0.001)
model.fit(x,y)
print('Coeff R2 =', model.score(x, y))

ptl.scatter(x[:,0], y, marker='o')
ptl.scatter(x[:,0], model.predict(x), c='red', marker='+')
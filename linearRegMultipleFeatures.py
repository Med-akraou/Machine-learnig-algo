# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 18:01:34 2022

@author: espoir
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 02:53:41 2022

@author: espoir
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as ptl

x,y= make_regression(n_samples=100,n_features=2,noise=10)

y=y.reshape(100,1)

X=np.hstack((x,np.ones((x.shape[0],1))))

def model(X,o):
    return X.dot(o)
theta= np.random.rand(3,1)



def cost_function(x,y,theta):
    m=len(y)
    return 1/(2*m)*np.sum((model(x, theta)-y)**2)

def grad(X,y,theta):
    m=len(y)
    return 1/m*X.T.dot(model(X, theta)-y)

def gradient_descente(X,y,theta,learning_rate,eps):
    cost_history=np.zeros(eps)
    for i in range(0,eps):
        theta=theta-learning_rate*grad(X, y, theta)
        cost_history[i]=cost_function(X, y, theta)
    return theta,cost_history

theta_finale,costs=gradient_descente(X, y, theta, 0.1, 1000)
print(theta_finale)

ptl.scatter(x[:,0],y)
ptl.scatter(x[:,0],model(X,theta_finale),c="r")
#ptl.plot(range(1000),costs)

def coeff_determination(y,pred):
    u=((y-pred)**2).sum()
    v=((y-y.mean())**2).sum()
    return 1-u/v
print(coeff_determination(y, model(X, theta_finale)))





    




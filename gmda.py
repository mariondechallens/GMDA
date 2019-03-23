# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:01:03 2019

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt

### Task 2 : generating a mixture of Gaussian
mean = [0, 100,9]
cov = [[1, 0,0], [0, 100,0],[0,0,1]]
m = np.random.multivariate_normal(mean, cov) # un point d'une gaussian mix 3D

def gaussian_mix(d,N,t):
    
    
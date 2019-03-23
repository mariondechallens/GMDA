# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:01:03 2019

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt

### Task 2 : generating a mixture of Gaussian

def gaussian_mix(d,N,t,s,n):
    cov = np.identity(d)
    mean = np.zeros(d)
    n_tot = N*n
    res = np.zeros((d,n_tot))
    for i in range(N):
        gaus = np.random.multivariate_normal(mean + t*i, cov*(s**i),n).T
        for j in range(d):
            res[j][i*n:(i+1)*n] = gaus[j]
            
    return(res)  
    
data = gaussian_mix(2,3,4,1.5,500)    
plt.plot(data[0],data[1])    

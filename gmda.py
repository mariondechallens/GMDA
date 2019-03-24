# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:01:03 2019

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt

### Task 2 : generating a mixture of Gaussian

def gaussian_mix(d,N,t,n,s=1):
    cov = np.identity(d)
    mean = np.zeros(d)
    n_tot = N*n
    res = np.zeros((d,n_tot))
    for i in range(N):
        gaus = np.random.multivariate_normal(mean + t*i, cov*(s**i),n).T
        for j in range(d):
            res[j][i*n:(i+1)*n] = gaus[j]
            
    return(res)  
    
data = gaussian_mix(d=2,N=3,t=4,n=500)    
plt.plot(data[0],data[1])    


### Task 3 : checking whether MMD is a test of level alpha
# The acceptance region for H0 at threshold alpha is given by
# MMDb^2 <= sqrt(2K/m) (1 + sqrt(2log(1/alpha)))

def gauss_kernel(x,y,s=1):
    k = np.exp(-sum((x-y)**2)/(2*(s**2)))
    return k

#MMD entre deux gauss mix
def MMD(g1,g2):         ### à optimiser !!
    n = len(g1[0])
    m = len(g2[0])
    MMD = 0
    for i in range(n):
        for j in range(n):
            MMD += gauss_kernel(g1.T[i],g1.T[j])/n**2
    
    for i in range(m):
        for j in range(m):
            MMD += gauss_kernel(g2.T[i],g2.T[j])/m**2
    
    for i in range(n):
        for j in range(m):
            MMD -= 2*gauss_kernel(g1.T[i],g2.T[j])/(m*n)
    MMD = np.sqrt(MMD)
    return(MMD)
  
g1 = gaussian_mix(d=1,N=1,t=4,n=500) 
g2 = gaussian_mix(d=1,N=1,t=4,n=500)     
MMD(g1,g2)  

def threshold(n,alpha):  # gauss kernel <=1, K = 1
    t = np.sqrt(2/n)*(1+np.sqrt(2*np.log(1/alpha)))  #need n=m
    return(t)

threshold(len(g1[0]),0.05)    

from itertools import product   

def MMD2(g1,g2): # moins long ?
    n = len(g1[0])
    m = len(g2[0])
    
    mmd1 = [gauss_kernel(g1.T[i],g1.T[j])/n**2 for i,j in product(range(n),range(n))]
    mmd2 = [gauss_kernel(g2.T[i],g2.T[j])/m**2 for i,j in product(range(m),range(m))]
    mmd3 = [-2*gauss_kernel(g1.T[i],g2.T[j])/(m*n) for i,j in product(range(n),range(m))]
    
    MMD = np.sqrt(sum(mmd1) + sum(mmd2) + sum(mmd3))
    
    return(MMD) 

MMD2(g1,g2)     
    
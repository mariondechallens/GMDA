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
    
data = gaussian_mix(2,3,4,1.1,500)    
plt.plot(data[0],data[1])    


### Task 3 : checking whether MMD is a test of level alpha
# The acceptance region for H0 at threshold alpha is given by
# MMDb^2 <= sqrt(2K/m) (1 + sqrt(2log(1/alpha)))

def gauss_kernel(x,y,s=1):
    k = np.exp(-sum((x-y)**2)/(2*(s**2)))
    return k

#MMD^2 entre deux gauss mix
def MMD2(g1,g2):         ### à optimiser !!
    n = len(g1[0])
    m = len(g2[0])
    MMD2 = 0
    for i in range(n):
        for j in range(n):
            MMD2 += gauss_kernel(g1.T[i],g1.T[j])
    MMD2 /= n**2
    
    for i in range(m):
        for j in range(m):
            MMD2 += gauss_kernel(g2.T[i],g2.T[j])
    MMD2 /= m**2
    
    for i in range(n):
        for j in range(m):
            MMD2 -= gauss_kernel(g1.T[i],g2.T[j])
    MMD2 /= 2/(m*n)
    
    return(MMD2)
  
MMD2(g1,g2)  

    
    
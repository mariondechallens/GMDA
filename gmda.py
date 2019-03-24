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
  
    
def threshold(n,alpha=0.05):  # gauss kernel <=1, K = 1
    t = np.sqrt(2/n)*(1+np.sqrt(2*np.log(1/alpha)))  #need n=m
    return(t)
   

from itertools import product   

def MMD2(g1,g2): # moins long ?
    n = len(g1[0])
    m = len(g2[0])
    
    mmd1 = [gauss_kernel(g1.T[i],g1.T[j])/n**2 for i,j in product(range(n),range(n))]
    mmd2 = [gauss_kernel(g2.T[i],g2.T[j])/m**2 for i,j in product(range(m),range(m))]
    mmd3 = [-2*gauss_kernel(g1.T[i],g2.T[j])/(m*n) for i,j in product(range(n),range(m))]
    
    MMD = np.sqrt(sum(mmd1) + sum(mmd2) + sum(mmd3))
    
    return(MMD) 

    
def TST_MMD(g1,g2,alpha=0.05):
    if len(g1) != len(g2):
        print('data must have the same dimension')
        mmd = 0
        t=0
        
    if len(g1[0]) != len(g1[0]):
        print('data must have the same number of points')
        mmd = 0
        t = 0

    if len(g1) == len(g2) and len(g1[0]) == len(g1[0]) :
        mmd = MMD2(g1,g2)
        t = threshold(len(g1[0]), alpha)
        if mmd <= t:
            print('H0 accepted at level',alpha)
        else:
            print('H0 rejected at level',alpha)
    
    return mmd, t

g1 = gaussian_mix(d=3,N=3,t=1,n=500) 
g2 = gaussian_mix(d=3,N=3,t=12,n=500) 
g3 = gaussian_mix(d=3,N=3,t=1,n=500)

TST_MMD(g1,g2)
TST_MMD(g1,g3)


### Task 4 : feedback with JS divergence
# permutation test
# https://github.com/skerit/cmusphinx/blob/master/SphinxTrain/python/cmusphinx/divergence.py

def JS(p, q):

    if (len(q.shape) == 1):
        axis = 0
    else:
        axis = 1

    # D_{JS}(P\|Q) = (D_{KL}(P\|Q) + D_{KL}(Q\|P)) / 2

    return 0.5 * ((q * (np.log(q.clip(1e-10,1))

                        - np.log(p.clip(1e-10,1)))).sum(axis)

                      + (p * (np.log(p.clip(1e-10,1))

                              - np.log(q.clip(1e-10,1)))).sum(axis))


g1 = gaussian_mix(d=1,N=3,t=1,n=500) 
g2 = gaussian_mix(d=1,N=3,t=1,n=500) 
JSobs = JS(g1,g2) 
JSobs

#permutation test / bootstrap
import random
g3 = np.concatenate([g1[0],g2[0]])
N=1000
res = []
for i in range(N):   
    g = random.sample(list(g3),len(g1[0])) #new g1
    gg = [x for x in g3 if x not in g]#new g2
    res.append(JS(np.array(g),np.array(gg)))

p_value = sum(res>JSobs)/N
p_value

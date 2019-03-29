# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:01:03 2019

@author: Admin
"""
import pandas as pd
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


### Task 4 : TST versus feedback under the null hypothesis : p=q
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
# H0 : p=q
g1 = gaussian_mix(d=1,N=3,t=1,n=500) 
g2 = gaussian_mix(d=1,N=3,t=1,n=500) 



#permutation test / bootstrap
import random
def permut(g1,g2,N):
    JSobs = JS(g1,g2) 
    g3 = np.concatenate([g1[0],g2[0]])
    n = len(g3)//2
    res = []
    for i in range(N):   
        g = random.sample(list(g3),len(g3))
        res.append(JS(np.array(g[:n]),np.array(g[n:])))
    p_value = sum(res>JSobs)/N  #ne fonctionne pas : pb avec la JS div 
    return res, p_value

permut(g1,g2,1000)  #H0 accepted
TST_MMD(g1,g2) #H0 accepted

### Task 5: TST versus feedback under the alternative  p!=q and order of magnitude
g1 = gaussian_mix(d=1,N=3,t=1,n=500) 
g2 = gaussian_mix(d=1,N=3,t=2,n=500) 
g3 = gaussian_mix(d=1,N=3,t=3,n=500) 
g4 = gaussian_mix(d=1,N=3,t=5,n=500) 
g5 = gaussian_mix(d=1,N=3,t=10,n=500) 
g6 = gaussian_mix(d=1,N=3,t=15,n=500) 

TST_MMD(g1,g6) # H0 rejected
permut(g1,g6,1000) # H0 accepted 


#### autre methode
import math

def f(pcond,param):
    if pcond == 0:
        return 0
    else:
        return math.log(pcond/param,2)
    
def JS2(pcond,param):
    js = pcond *  f(pcond,param) + (1-pcond) *  f(1-pcond,1-param)
    return js

# calculer proba conditionnelle avec knn
g1 = gaussian_mix(d=1,N=3,t=1,n=500) 
g2 = gaussian_mix(d=1,N=3,t=2,n=500) 
data =pd.DataFrame({'g1':g1[0], 'g2':g2[0]})

def labeling(data):

        counts = {0:0, 1:0}
        
        stop = False
        #generate sequence of labels until no more samples available from one population
        while not stop:
            label = random.randint(0,1)
            if counts[label] == data.iloc[:,label].shape[0]:
                stop = True
            else:
                counts[label] += 1

        #uniformly draw them from the two populations
        #create the arrays for points and labels
        nsampled = sum(counts.values())
        dim = 1
        points = np.ndarray(shape=(nsampled,dim))
        labels = np.ndarray(shape=(nsampled,))

        
        pt = np.array([data.iloc[:,0][i] for i in random.sample(range(data.iloc[:,0].shape[0]),counts[0]) ])
        points[0:counts[0],:] = pt.reshape(pt.shape[0],1)
        labels[0:counts[0]] = np.zeros((counts[0],))

        ptt = np.array([data.iloc[:,1][i] for i in random.sample(range(data.iloc[:,1].shape[0]),counts[1]) ])
        points[counts[0]:,:] = ptt.reshape(ptt.shape[0],1)
        labels[counts[0]:] = np.ones((counts[1],))


        return points, labels
    
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
points, labels = labeling(data)
knn.fit(points,labels)
d = data.iloc[:,0]
dd = data.iloc[:,1]
pcond1 = knn.predict_proba(np.array(d).reshape(len(d),1))
pcond2 = knn.predict_proba(np.array(dd).reshape(len(dd),1))
   
JSDTable = np.ndarray(shape=(data.iloc[:,0].shape[0]+data.iloc[:,1].shape[0],1))

def g(pcond):
    return JS2(pcond,1)

JSDTable[0:pcond1.shape[0],0] = list(map(g,pcond1[:,0]))
JSDTable[pcond2.shape[0]:,0] = list(map(g,pcond2[:,0]))

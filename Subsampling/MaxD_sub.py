#!/usr/bin/env python
# coding: utf-8

# In[17]:


import networkx as nx
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.spatial.distance as spsd
import scipy.linalg as spla
import sys
import random


# In[71]:


def gen_adj_from_expectation( P,seed=1):
    np.random.seed(seed)
    np.fill_diagonal(P,0)
    probs = spsd.squareform(P)
    coinflips = np.random.binomial(n=1,p=probs)
    # turn the vector coinflips into a symmetric hollow matrix
    # before returning it.
    return spsd.squareform(coinflips)

def gen_adj_from_posns(X,seed=1):
    '''
    Generate an adjacency matrix with expectation X X^T.
    '''
    X = np.matrix(X)
    # Avoid issue where Xhats are out of range
    #Another option would be
    #P = np.maximum(1/n, np.minimum(X * X.T,1-1/n))
    P = np.maximum(0, np.minimum(X@X.T,1))
    return gen_adj_from_expectation( P,seed=seed)


# In[176]:


def gen_beta( n,a,b,d=1,seed=1,with_X=False):
    '''
    Generate an n-vertex graph from RDPG(Beta(a,b),n).
    '''
    np.random.seed(seed)
    X = np.mat(np.random.beta(a=a, b=b, size=(n,d)))
    P = np.maximum(0, np.minimum(X@X.T,1))
    A = gen_adj_from_expectation(P)
    
    if with_X:
        return A,P,X
    else: return A,P
    

def gen_dir(n,params,seed=1,with_X=False):#params is the parameters for dirichlet distribution e.g. (1,2,3,4)
    '''
    Generate an n-vertex graph from RDPG(Dirichlet(params),n).
    params should be a tuple or list
    '''
    np.random.seed(seed)
    X = np.random.dirichlet(alpha=tuple(params),size=n)
    P = np.maximum(0, np.minimum(X@X.T,1))
    A = gen_adj_from_expectation(P)
    if with_X:
        return A,P,X
    else:return A,P


# In[117]:


def bootstrap_rows( X,seed=1 ):
    '''
    ordinary bootstrap
    draw sampels with replacement from X with same sample size as X
    '''
    np.random.seed(seed)
    n = X.shape[0]
    idxs = np.random.choice(np.arange(n), size=n, replace=True)
    return X[idxs,]


def get_bootsample(X,n_times,ifA=True):
    Max=[]
    if eval(str(ifA)):
        for seed in range(n_times):
            Xstar=bootstrap_rows(X,seed=seed)
            Ahat = gen_adj_from_posns(Xstar)
            value=np.amax(Ahat.sum(axis=1))
            Max.append(value)
        return Max
    
    else:
        for seed in range(n_times):
            Xstar=bootstrap_rows(X,seed=seed)
            Pstar = np.maximum(0, np.minimum(Xstar@Xstar.T,1))
            value=np.amax(Pstar.sum(axis=1))
            Max.append(value)
        return Max

def ase(A,d,reg=0.0):
    '''
    Compute the d-dimensional adjacency spectral embedding of A,
	with degree augmentation and
        (optional) regularization to the diagonal.
    '''

    # Diagonal augmentation.
    n = A.shape[0]
    degrees = np.sum(A,0)
    di = np.diag_indices(n)
    A[di] = degrees/n
    A = A + reg*np.ones(n)

    (S,U) = spla.eigh(A,subset_by_index=(n-d,n-1))
    return np.matrix(U)*np.matrix(np.diag(np.sqrt(S)))

#CI=cal_CI(real_mean_beta,observe_value,Max)

#3. Observe mean CI: Observe mean +- 1.96*standard error
def cal_CI(real,observe,bootsample,CI=False):
    
    #naive bootstrap
    ql=round(np.quantile(bootsample,q=0.025),4)
    qu=round(np.quantile(bootsample,q=0.975),4) 
    if qu>real>ql:
        CI_boot=1
    else:CI_boot=0
        
    #boot mean CI
    qu_boot=np.mean(bootsample)+1.96*np.sqrt(np.var(bootsample))
    ql_boot=np.mean(bootsample)-1.96*np.sqrt(np.var(bootsample))
    if qu_boot>real>ql_boot:
        CI_bootmean=1
    else:CI_bootmean=0
    
    #observe mean CI
    qu_ob=observe+1.96*np.sqrt(np.var(bootsample))
    ql_ob=observe-1.96*np.sqrt(np.var(bootsample))
    if qu_ob>real>ql_ob:
        CI_observe=1
    else:CI_observe=0
    if CI:
        return([ql,qu],[qu_boot,ql_boot],[qu_ob,ql_ob])
    else:
        return (CI_boot,CI_bootmean,CI_observe)


# In[103]:


def get_real_value(function,n_simulation=10000,ifA=True,**args):
    Max_real=[]
    for seed in range(int(n_simulation)):
        A,P=function(**args,seed=seed)
        if eval(str(ifA)):
            Max_real.append(np.amax((A.sum(axis=1))))
        else:
            Max_real.append(np.amax((P.sum(axis=1))))
    return np.mean(Max_real)


# # Set parameters

# In[101]:


n_trials=int(sys.argv[1])#number of CIs to calculate coverage rate
n_times=int(sys.argv[2])#number of bootstrap samples
n_nodes=int(sys.argv[3]);function=str(sys.argv[4])
ifA=str(sys.argv[5])
ifsub=str(sys.argv[6])
if ifsub:
    subsize=int(sys.argv[7])
    
#python3 MaxD.py 20 20 100 "beta" True
#python3 MaxD.py 300 300 100 "beta" True False
#python3 MaxD.py 300 300 100 "beta" True True 30


#function(beta or dir) 
#ifA(True,False) 
#ifsub(True False)


# In[194]:


if eval(str(ifA)):
    if eval(str(ifsub)):
        name=str(n_nodes)+'_'+function+'_'+'A_'+'subsize'+str(subsize)
    else:
        name=str(n_nodes)+'_'+function+'_'+'A_'+'.pkl'
else:
    if eval(str(ifsub)):
        name=str(n_nodes)+'_'+function+'_'+'P_'+'subsize'+str(subsize)
    else:
        name=str(n_nodes)+'_'+function+'_'+'P_'
print(name)


# In[141]:


#n_trials=20;n_times=20;n_nodes=100;function='beta';ifA=True
params=(1,2,3,4,5)


# In[142]:


print(function+' distribution')
if eval(str(ifA)):
    print('calculate real max degree using A') 
else:
    print('calculate expected max degree using P')


# In[143]:


if function=="beta":
    real_mean=get_real_value(gen_beta,n=n_nodes,a=2,b=3,d=1,with_X=False,ifA=ifA)
if function=="dir":
    real_mean=get_real_value(gen_dir,n=n_nodes,params=params,with_X=False,ifA=ifA)


# In[144]:


Naive=[];Bootmean=[];Observe=[]#save all results for all situations e.g. X_hat,X,diagonal augmentation

#Xhat
Cover_naive=[];Cover_bootmean=[];Cover_observe=[]

for seed in range(n_trials):#500 trials (500 CIs calculated)
    if function=='beta':
        #generate an initial network
        A,P=gen_beta(n=n_nodes,a=2,b=3,d=1,seed=seed)
        X_hat=ase(A,d=1) 
        
        #calculate observe value based on A of P
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    if function=='dir':
        A,P=gen_dir(n=n_nodes,params=params,seed=seed)
        X_hat=ase(A,d=len(params))
        
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
        
    Max=get_bootsample(X_hat,n_times=n_times,ifA=ifA)
    if eval(str(ifsub)):
        Max=np.random.choice(Max,size=subsize,replace=False)
        
    CI=cal_CI(real_mean,observe_value,Max)
    Cover_naive.append(CI[0]);Cover_bootmean.append(CI[1]);Cover_observe.append(CI[2])
Naive.append(np.mean(Cover_naive));Bootmean.append(np.mean(Cover_bootmean));Observe.append(np.mean(Cover_observe))


# In[145]:


#real X
Cover_naive=[];Cover_bootmean=[];Cover_observe=[]
for seed in range(n_trials):
    if function=='beta':
        #generate an initial network
        A,P,X=gen_beta(n=n_nodes,a=2,b=3,d=1,seed=seed,with_X=True)
        #calculate observe value based on A of P
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    if function=='dir':
        A,P,X=gen_dir(n=n_nodes,params=params,seed=seed,with_X=True)
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    Max=get_bootsample(X,n_times=n_times,ifA=ifA)
    if eval(str(ifsub)):
        Max=np.random.choice(Max,size=subsize,replace=False)
        
    CI=cal_CI(real_mean,observe_value,Max)
    Cover_naive.append(CI[0]);Cover_bootmean.append(CI[1]);Cover_observe.append(CI[2])
Naive.append(np.mean(Cover_naive));Bootmean.append(np.mean(Cover_bootmean));Observe.append(np.mean(Cover_observe))
            


# In[ ]:


#diagonal augmentation
#1. Average Degree
Cover_naive=[];Cover_bootmean=[];Cover_observe=[]
for seed in range(n_trials):
    if function=='beta':
        A,P=gen_beta(n=n_nodes,a=2,b=3,d=1,seed=seed)
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    if function=='dir':
        A,P=gen_dir(n=n_nodes,params=params,seed=seed)
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    ### diagonal augmentation
    ### fill diagonal entries of A with average degree
    avedeg=np.sum(A)/n_nodes
    np.fill_diagonal(A, avedeg)
    if function=='beta':
        X_hat=ase(A,d=1)
    if function=='dir':
        X_hat=ase(A,d=len(params))
    ###########################
    Max=get_bootsample(X_hat,n_times=n_times,ifA=ifA)
    if eval(str(ifsub)):
        Max=np.random.choice(Max,size=subsize,replace=False)
        
    CI=cal_CI(real_mean,observe_value,Max)
    Cover_naive.append(CI[0]);Cover_bootmean.append(CI[1]);Cover_observe.append(CI[2])
Naive.append(np.mean(Cover_naive));Bootmean.append(np.mean(Cover_bootmean));Observe.append(np.mean(Cover_observe))
            


# In[ ]:


#diagonal augmentation
#2. row sum /n
Cover_naive=[];Cover_bootmean=[];Cover_observe=[]
for seed in range(n_trials):
    if function=='beta':
        A,P=gen_beta(n=n_nodes,a=2,b=3,d=1,seed=seed)
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    if function=='dir':
        A,P=gen_dir(n=n_nodes,params=params,seed=seed)
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    np.fill_diagonal(A, np.sum(A,axis=1)/n_nodes)
    if function=='beta':
        X_hat=ase(A,d=1)
    if function=='dir':
        X_hat=ase(A,d=len(params))
    
    Max=get_bootsample(X_hat,n_times=n_times,ifA=ifA)#each trial generate 500 bootstrap samples
    if eval(str(ifsub)):
        Max=np.random.choice(Max,size=subsize,replace=False)
        
    CI=cal_CI(real_mean,observe_value,Max)
    Cover_naive.append(CI[0]);Cover_bootmean.append(CI[1]);Cover_observe.append(CI[2])
Naive.append(np.mean(Cover_naive));Bootmean.append(np.mean(Cover_bootmean));Observe.append(np.mean(Cover_observe))
            


# In[ ]:


#3. 1
Cover_naive=[];Cover_bootmean=[];Cover_observe=[]
for seed in range(n_trials):
    if function=='beta':
        A,P=gen_beta(n=n_nodes,a=2,b=3,d=1,seed=seed)
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    if function=='dir':
        A,P=gen_dir(n=n_nodes,params=params,seed=seed)
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    np.fill_diagonal(A,1)
    if function=='beta':
        X_hat=ase(A,d=1)
    if function=='dir':
        X_hat=ase(A,d=len(params))
    ###########################
    Max=get_bootsample(X_hat,n_times=n_times,ifA=ifA)
    if eval(str(ifsub)):
        Max=np.random.choice(Max,size=subsize,replace=False)
        
    CI=cal_CI(real_mean,observe_value,Max)
    Cover_naive.append(CI[0]);Cover_bootmean.append(CI[1]);Cover_observe.append(CI[2])
Naive.append(np.mean(Cover_naive));Bootmean.append(np.mean(Cover_bootmean));Observe.append(np.mean(Cover_observe))
    


# In[ ]:


#4. penalty
Cover_naive=[];Cover_bootmean=[];Cover_observe=[]
for seed in range(n_trials):
    if function=='beta':
        A,P=gen_beta(n=n_nodes,a=2,b=3,d=1,seed=seed)
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    if function=='dir':
        A,P=gen_dir(n=n_nodes,params=params,seed=seed)
        if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
        else: observe_value=np.amax(P.sum(axis=1))
            
    reg=1/n_nodes
    A=A+reg*np.ones(n_nodes)
    if function=='beta':
        X_hat=ase(A,d=1)
    if function=='dir':
        X_hat=ase(A,d=len(params))
    ###########################
    Max=get_bootsample(X_hat,n_times=n_times,ifA=ifA)
    if eval(str(ifsub)):
        Max=np.random.choice(Max,size=subsize,replace=False)
        
    CI=cal_CI(real_mean,observe_value,Max)
    Cover_naive.append(CI[0]);Cover_bootmean.append(CI[1]);Cover_observe.append(CI[2])
Naive.append(np.mean(Cover_naive));Bootmean.append(np.mean(Cover_bootmean));Observe.append(np.mean(Cover_observe))
    


# In[ ]:


Rate_naive=[];Rate_bootmean=[];Rate_observe=[]
for sigma in np.arange(0,0.1,0.01):#
    Cover_naive=[];Cover_bootmean=[];Cover_observe=[]
    mu=0
    for seed in range(n_trials):
        if function=='beta':
            #generate an initial network
            A,P,X=gen_beta(n=n_nodes,a=2,b=3,d=1,seed=seed,with_X=True)
            #calculate observe value based on A of P
            if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
            else: observe_value=np.amax(P.sum(axis=1))
        if function=='dir':
            A,P,X=gen_dir(n=n_nodes,params=params,seed=seed,with_X=True)
            if eval(str(ifA)): observe_value=np.amax(A.sum(axis=1))
            else: observe_value=np.amax(P.sum(axis=1))

        R=np.random.normal(mu,sigma,size=(X.shape[0],X.shape[1]))
        X_noise=X+R
        
        Max=get_bootsample(X_noise,n_times=n_times,ifA=ifA)
        if eval(str(ifsub)):
            Max=np.random.choice(Max,size=subsize,replace=False)
        
        CI=cal_CI(real_mean,observe_value,Max)
        Cover_naive.append(CI[0]);Cover_bootmean.append(CI[1]);Cover_observe.append(CI[2])
        
    Rate_naive.append(np.mean(Cover_naive))
    Rate_bootmean.append(np.mean(Cover_bootmean))
    Rate_observe.append(np.mean(Cover_observe))
Naive.append(Rate_naive);Bootmean.append(Rate_bootmean);Observe.append(Rate_observe)


# In[167]:


if eval(str(ifA)):
    if eval(str(ifsub)):
        filename=str(n_nodes)+'_'+function+'_'+'A_'+'subsize'+str(subsize)+'.pkl'
    else:
        filename=str(n_nodes)+'_'+function+'_'+'A_'+'.pkl'
else:
    if eval(str(ifsub)):
        filename=str(n_nodes)+'_'+function+'_'+'P_'+'subsize'+str(subsize)+'.pkl'
    else:
        filename=str(n_nodes)+'_'+function+'_'+'P_'+'.pkl'

with open(filename, 'wb') as f:  
    pickle.dump([Naive,Bootmean,Observe], f)


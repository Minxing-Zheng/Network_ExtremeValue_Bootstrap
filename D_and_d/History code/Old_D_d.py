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
import math


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
    return gen_adj_from_expectation(P,seed=seed)


# In[117]:


# In[194]:

n_nodes=int(sys.argv[1])#number of nodes
num_run=int(sys.argv[2])
num_param=int(sys.argv[3])#dim
function=str(sys.argv[4])#distribution
set_seed=int(sys.argv[5])
print(function)

# In[143]:

EX=math.sqrt(10)/10;VarX=(10**0.5-1)/110;n=1
#n=1,alpha,beta,m=2,3,10
Alpha_Beta=[];Beta_Beta=[]
for m in range(2,11):
    A=math.sqrt(1/m)*EX
    alpha = (m*A**2*(1-A)/(1*VarX))-A
    Alpha_Beta.append(alpha)
    beta=alpha*((1-A)/A)
    Beta_Beta.append(beta)


Beta_Dir=[2,]
for m in range(3,11):
    n=2;alpha=2
    a=1-1/m;b=(1-1/n)/(n*alpha+1)+(1/n)-(1/m)
    Beta_Dir.append((a/b-1)/m)

#0.1

N_distribution=[];N_10_distribution=[];N_30_distribution=[];N_50_distribution=[]
E_distribution=[];E_10_distribution=[];E_30_distribution=[];E_50_distribution=[]

if function=='beta':
    for times in range(num_run):
        D=[]
        for repeat in range(n_nodes):
            X = np.mat(np.random.beta(a=Alpha_Beta[num_param-2], b=Beta_Beta[num_param-2], size=(n_nodes,num_param)))
            probs=np.array(X[0,:]@X.T)
            probs=list(probs.flatten())
            for it,x in enumerate(probs):
                if x>1:probs[0,it]=1
            probs[0]=0
            Degree=np.sum(np.random.binomial(n=1,p=probs))
            D.append(Degree)
        D_10=np.random.choice(D,size=int(n_nodes/10),replace=False)
        D_30=np.random.choice(D,size=int(3*(n_nodes/10)),replace=False)
        D_50=np.random.choice(D,size=int(5*(n_nodes/10)),replace=False)
    
        N=max(D);N_distribution.append(N)
        N_10=max(D_10);N_10_distribution.append(N_10)
        N_30=max(D_30);N_30_distribution.append(N_30)
        N_50=max(D_50);N_50_distribution.append(N_50)
        
         
        X = np.mat(np.random.beta(a=Alpha_Beta[num_param-2], b=Beta_Beta[num_param-2], size=(n_nodes,num_param)))
        P = np.maximum(0, np.minimum(X@X.T,1))
        np.fill_diagonal(P,0)
        A = gen_adj_from_expectation(P,seed=times)
        d=A.sum(axis=1)
        d_10=np.random.choice(d,size=int(n_nodes/10),replace=False)
        d_30=np.random.choice(d,size=int(3*(n_nodes/10)),replace=False)
        d_50=np.random.choice(d,size=int(5*(n_nodes/10)),replace=False)
        
        E=max(d);E_distribution.append(E)
        E_10=max(d_10);E_10_distribution.append(E_10)
        E_30=max(d_30);E_30_distribution.append(E_30)
        E_50=max(d_50);E_50_distribution.append(E_50)
        

if function=='dir':
    N_distribution=[];E_distribution=[]
    for times in range(num_run):
        D=[]
        for repeat in range(n_nodes):
            X = np.random.dirichlet(alpha=[Beta_Dir[num_param-2]]*num_param,size=n_nodes)
            probs=np.array(X[0,:]@X.T)*num_param/2/5
            probs=list(probs.flatten())
            for it,x in enumerate(probs):
                if x>1:probs[0,it]=1
            probs[0]=0
            Degree=np.sum(np.random.binomial(n=1,p=probs))
            D.append(Degree)
        D_10=np.random.choice(D,size=int(n_nodes/10),replace=False)
        D_30=np.random.choice(D,size=int(3*(n_nodes/10)),replace=False)
        D_50=np.random.choice(D,size=int(5*(n_nodes/10)),replace=False)
    
        N=max(D);N_distribution.append(N)
        N_10=max(D_10);N_10_distribution.append(N_10)
        N_30=max(D_30);N_30_distribution.append(N_30)
        N_50=max(D_50);N_50_distribution.append(N_50)
        
        #subsample, max
        
        N_distribution.append(N)

        X = np.random.dirichlet(alpha=[Beta_Dir[num_param-2]]*num_param,size=n_nodes)
        P=(X@X.T)*num_param/2/5
        P = np.maximum(0, np.minimum(P,1))
        np.fill_diagonal(P,0)
        A = gen_adj_from_expectation(P,seed=times)
        d=A.sum(axis=1)

        d_10=np.random.choice(d,size=int(n_nodes/10),replace=False)
        d_30=np.random.choice(d,size=int(3*(n_nodes/10)),replace=False)
        d_50=np.random.choice(d,size=int(5*(n_nodes/10)),replace=False)
        
        E=max(d);E_distribution.append(E)
        E_10=max(d_10);E_10_distribution.append(E_10)
        E_30=max(d_30);E_30_distribution.append(E_30)
        E_50=max(d_50);E_50_distribution.append(E_50)


filename=f'{function}_{n_nodes}_node_{num_param}_dim_{set_seed}_seed.pkl'
with open(filename, 'wb') as f:  
    pickle.dump([[N_distribution,N_10_distribution,N_30_distribution,N_50_distribution],[E_distribution,E_10_distribution,E_30_distribution,E_50_distribution]], f)
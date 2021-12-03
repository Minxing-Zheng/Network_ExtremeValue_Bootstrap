#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

def gen_adj_from_expectation( P):
    np.fill_diagonal(P,0)
    probs = spsd.squareform(P)
    coinflips = np.random.binomial(n=1,p=probs)
    # turn the vector coinflips into a symmetric hollow matrix
    # before returning it.
    return spsd.squareform(coinflips)


def gen_adj_from_posns(X,scale=1):
    '''
    Generate an adjacency matrix with expectation X X^T.
    '''
    X = np.matrix(X)
    # Avoid issue where Xhats are out of range
    #Another option would be
    #P = np.maximum(1/n, np.minimum(X * X.T,1-1/n))
    P = np.maximum(0, np.minimum((X@X.T)*scale,1))
    return gen_adj_from_expectation(P)

def ase(A,d,aug_type='none'):
    '''
    Compute the d-dimensional adjacency spectral embedding of A,
    with degree augmentation and
        (optional) regularization to the diagonal.
    '''
    # Diagonal augmentation.
    n = A.shape[0]
    #degrees = np.sum(A,0)
    #di = np.diag_indices(n)
    #A[di] = degrees/n
    #A = A + reg*np.ones(n)

    if aug_type=='none':
        pass
    if aug_type=='rowsum':
        np.fill_diagonal(A, np.sum(A,axis=1)/n)
    if aug_type=='penalty':
        reg=1/n
        A=A+reg*np.ones(n)
    if aug_type=='avedegree':
        avedeg=np.sum(A)/(n*(n-1))
        np.fill_diagonal(A, avedeg)

    (S,U) = spla.eigh(A,subset_by_index=(n-d,n-1))
    return np.matrix(U)*np.matrix(np.diag(np.sqrt(S)))


# In[117]:


# In[194]:

n_nodes=int(sys.argv[1])#number of nodes
aug_type=str(sys.argv[2])
num_run=int(sys.argv[3])
num_param=int(sys.argv[4])#dim
function=str(sys.argv[5])#distribution
set_seed=int(sys.argv[6])
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

N_distribution=[];N_distribution_hat=[];N_distribution_expect=[];N_distribution_hat_expect=[]
E_distribution=[];E_distribution_hat=[];E_distribution_expect=[];E_distribution_hat_expect=[]

D_seq=[];D_hat_seq=[]
d_seq=[];d_hat_seq=[]
D_s=[];D_hat_s=[]
d_s=[];d_hat_s=[]

if function=='beta':
    for times in range(num_run):
        D=[];D_hat=[]
        D_expect=[];D_hat_expect=[]
        for repeat in range(n_nodes):
            X = np.mat(np.random.beta(a=Alpha_Beta[num_param-2], b=Beta_Beta[num_param-2], size=(n_nodes,num_param)))
            A=gen_adj_from_posns(X)
            P = np.maximum(0, np.minimum((X@X.T),1))

            X_hat=ase(A,d=num_param,aug_type=aug_type)
            A_hat=gen_adj_from_posns(X_hat)
            P_hat= np.maximum(0, np.minimum((X_hat@X_hat.T),1))

            D_hat_expect.append(np.sum(P_hat[0,]))
            D_expect.append(np.sum(P[0,]))
            
            D.append(np.sum(A[0,]))
            D_hat.append(np.sum(A_hat[0,]))

        D_seq=D_seq+D_expect;D_hat_seq=D_hat_seq+D_hat_expect
        D_s=D_s+D;D_hat_s=D_hat_s+D_hat

        N=[];N_hat=[];N_expect=[];N_hat_expect=[]
        N.append(max(D))
        N_hat.append(max(D_hat))
        
        N_expect.append(max(D_expect))
        N_hat_expect.append(max(D_hat_expect))
        

        #for sub in [1,3,5]:
            #D_sub=np.random.choice(D,size=int(int(sub)*n_nodes/10),replace=False)
            #D_hat_sub=np.random.choice(D_hat,size=int(int(sub)*n_nodes/10),replace=False)
            #D_sub_expect=np.random.choice(D_expect,size=int(int(sub)*n_nodes/10),replace=False)
            #D_hat_sub_expect=np.random.choice(D_hat_expect,size=int(int(sub)*n_nodes/10),replace=False)
            
            #N.append(max(D_sub))
            #N_hat.append(max(D_hat_sub))
            #N_expect.append(max(D_sub_expect))
            #N_hat_expect.append(max(D_hat_sub_expect))

        N_distribution.append(N)
        N_distribution_hat.append(N_hat)
        N_distribution_expect.append(N_expect)
        N_distribution_hat_expect.append(N_hat_expect)

         
        E=[];E_hat=[];E_expect=[];E_hat_expect=[]
        X = np.mat(np.random.beta(a=Alpha_Beta[num_param-2], b=Beta_Beta[num_param-2], size=(n_nodes,num_param)))
        A=gen_adj_from_posns(X)
        P = np.maximum(0, np.minimum((X@X.T),1))

        X_hat=ase(A,d=num_param,aug_type=aug_type)
        A_hat=gen_adj_from_posns(X_hat)
        P_hat= np.maximum(0, np.minimum((X_hat@X_hat.T),1))
        
        E.append(max(np.array(A).sum(axis=1)))
        E_hat.append(max(np.array(A_hat).sum(axis=1)))
        E_expect.append(max(np.array(P).sum(axis=1)))
        E_hat_expect.append(max(np.array(P_hat).sum(axis=1)))
        #expected degree seq
        d_seq=d_seq+list(np.array(P).sum(axis=1))
        d_hat_seq=d_hat_seq+list(np.array(P_hat).sum(axis=1))
        #degree seq
        d_s=d_s+list(np.array(A).sum(axis=1))
        d_hat_s=d_hat_s+list(np.array(A_hat).sum(axis=1))


        #for sub in [1,3,5]:

            #d_sub=np.random.choice(np.array(A).sum(axis=1),size=int(int(sub)*n_nodes/10),replace=False)
            
            #d_hat_sub=np.random.choice(np.array(A_hat).sum(axis=1),size=int(int(sub)*n_nodes/10),replace=False)
            #E.append(max(d_sub))
            #E_hat.append(max(d_hat_sub))

            #d_sub_expect=np.random.choice(np.array(P).sum(axis=1),size=int(int(sub)*n_nodes/10),replace=False)
            #d_hat_sub_expect=np.random.choice(np.array(P_hat).sum(axis=1),size=int(int(sub)*n_nodes/10),replace=False)
            #E_expect.append(max(d_sub_expect))
            #E_hat_expect.append(max(d_hat_sub_expect))

        E_distribution.append(E)
        E_distribution_hat.append(E_hat)
        E_distribution_expect.append(E_expect)
        E_distribution_hat_expect.append(E_hat_expect)




if function=='dir':
    for times in range(num_run):
        D=[];D_hat=[]
        D_expect=[];D_hat_expect=[]

        for repeat in range(n_nodes):
            X = np.random.dirichlet(alpha=[Beta_Dir[num_param-2]]*num_param,size=n_nodes)
            #A=gen_adj_from_posns(X,scale=num_param/2/5)

            A=gen_adj_from_posns(X,scale=num_param/2/5)
            P = np.maximum(0, np.minimum((X@X.T)*num_param/2/5,1))

            X_hat=ase(A,d=num_param,aug_type=aug_type)
            A_hat=gen_adj_from_posns(X_hat)
            P_hat= np.maximum(0, np.minimum((X_hat@X_hat.T),1))

            D_hat_expect.append(np.sum(P_hat[0,]))
            D_expect.append(np.sum(P[0,]))
            D.append(np.sum(A[0,]))
            D_hat.append(np.sum(A_hat[0,]))

        N=[];N_hat=[];N_expect=[];N_hat_expect=[]
        N.append(max(D))
        N_hat.append(max(D_hat))
        N_expect.append(max(D_expect))
        N_hat_expect.append(max(D_hat_expect))
        D_seq=D_seq+D_expect;D_hat_seq=D_hat_seq+D_hat_expect
        D_s=D_s+D;D_hat_s=D_hat_s+D_hat

        #for sub in [1,3,5]:
            #D_sub=np.random.choice(D,size=int(int(sub)*n_nodes/10),replace=False)
            #D_hat_sub=np.random.choice(D_hat,size=int(int(sub)*n_nodes/10),replace=False)
            #D_sub_expect=np.random.choice(D_expect,size=int(int(sub)*n_nodes/10),replace=False)
            #D_hat_sub_expect=np.random.choice(D_hat_expect,size=int(int(sub)*n_nodes/10),replace=False)
            
            #N.append(max(D_sub))
            #N_hat.append(max(D_hat_sub))
            #N_expect.append(max(D_sub_expect))
            #N_hat_expect.append(max(D_hat_sub_expect))

        N_distribution.append(N)
        N_distribution_hat.append(N_hat)
        N_distribution_expect.append(N_expect)
        N_distribution_hat_expect.append(N_hat_expect)


        E=[];E_hat=[];E_expect=[];E_hat_expect=[]

        X = np.random.dirichlet(alpha=[Beta_Dir[num_param-2]]*num_param,size=n_nodes)
        A=gen_adj_from_posns(X,scale=num_param/2/5)
        P = np.maximum(0, np.minimum((X@X.T)*num_param/2/5,1))
    
        X_hat=ase(A,d=num_param,aug_type=aug_type)
        A_hat=gen_adj_from_posns(X_hat)
        P_hat= np.maximum(0, np.minimum((X_hat@X_hat.T),1))

        E.append(max(np.array(A).sum(axis=1)))
        E_hat.append(max(np.array(A_hat).sum(axis=1)))
        E_expect.append(max(np.array(P).sum(axis=1)))
        E_hat_expect.append(max(np.array(P_hat).sum(axis=1)))
        
        #expected degree seq
        d_seq=d_seq+list(np.array(P).sum(axis=1))
        d_hat_seq=d_hat_seq+list(np.array(P_hat).sum(axis=1))

        #degree seq
        d_s=d_s+list(np.array(A).sum(axis=1))
        d_hat_s=d_hat_s+list(np.array(A_hat).sum(axis=1))

        #for sub in [1,3,5]:

            #d_sub=np.random.choice(np.array(A).sum(axis=1),size=int(int(sub)*n_nodes/10),replace=False)
            
            #d_hat_sub=np.random.choice(np.array(A_hat).sum(axis=1),size=int(int(sub)*n_nodes/10),replace=False)
            #E.append(max(d_sub))
            #E_hat.append(max(d_hat_sub))

            #d_sub_expect=np.random.choice(np.array(P).sum(axis=1),size=int(int(sub)*n_nodes/10),replace=False)
            #d_hat_sub_expect=np.random.choice(np.array(P_hat).sum(axis=1),size=int(int(sub)*n_nodes/10),replace=False)
            #E_expect.append(max(d_sub_expect))
            #E_hat_expect.append(max(d_hat_sub_expect))

        E_distribution.append(E)
        E_distribution_hat.append(E_hat)
        E_distribution_expect.append(E_expect)
        E_distribution_hat_expect.append(E_hat_expect)
        

        
filename=f'{function}_{n_nodes}_node_{num_param}_dim_{aug_type}_aug_{set_seed}_seed.pkl'
with open(filename, 'wb') as f:  
    pickle.dump([[N_distribution,N_distribution_hat,N_distribution_expect,N_distribution_hat_expect],[E_distribution,E_distribution_hat,E_distribution_expect,E_distribution_hat_expect],[D_seq,D_hat_seq,d_seq,d_hat_seq]], f)


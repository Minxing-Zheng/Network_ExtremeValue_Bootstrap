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
    
def bootstrap_rows(X):
    '''
    ordinary bootstrap
    draw sampels with replacement from X with same sample size as X
    '''
    n = X.shape[0]
    idxs = np.random.choice(np.arange(n), size=n, replace=True)
    return X[idxs,]

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

D_seq=[];D_Pseq=[]
D_hat_seq=[];D_hat_Pseq=[]
D_star_seq=[];D_star_Pseq=[]

d_seq=[];d_Pseq=[]
d_hat_seq=[];d_hat_Pseq=[]
d_star_seq=[];d_star_Pseq=[]

N=[];N_hat=[];N_expect=[];N_hat_expect=[];N_star=[];N_star_expect=[]
E=[];E_hat=[];E_expect=[];E_hat_expect=[];E_star=[];E_star_expect=[]


if function=='beta':
    for times in range(num_run):
        D=[];D_hat=[]
        D_expect=[];D_hat_expect=[]
        D_star=[];D_star_expect=[]
        for repeat in range(n_nodes):
            X = np.mat(np.random.beta(a=Alpha_Beta[num_param-2], b=Beta_Beta[num_param-2], size=(n_nodes,num_param)))
            A=gen_adj_from_posns(X)
            P = np.maximum(0, np.minimum((X@X.T),1))

            X_hat=ase(A,d=num_param,aug_type=aug_type)
            A_hat=gen_adj_from_posns(X_hat)
            P_hat= np.maximum(0, np.minimum((X_hat@X_hat.T),1))
            
            ### Bootstrap
            #X_star=bootstrap_rows(X)
            X_star=X
            A_star=gen_adj_from_posns(X_star)#generate an another A based on P
            P_star = np.maximum(0, np.minimum((X_star@X_star.T),1))
            
            D_expect.append(np.sum(P[0,]))
            D_hat_expect.append(np.sum(P_hat[0,]))
            D_star_expect.append(np.sum(P_star[0,]))
            
            D.append(np.sum(A[0,]))
            D_hat.append(np.sum(A_hat[0,]))
            D_star.append(np.sum(A_star[0,]))

        D_seq.append(D)
        D_hat_seq.append(D_hat)
        D_star_seq.append(D_star)
        
        D_Pseq.append(D_expect)
        D_hat_Pseq.append(D_hat_expect)
        D_star_Pseq.append(D_star_expect)

        N.append(max(D))
        N_hat.append(max(D_hat))
        N_star.append(max(D_star))
        
        N_expect.append(max(D_expect))
        N_hat_expect.append(max(D_hat_expect))
        N_star_expect.append(max(D_star_expect))
        
        
        #####################################
        # di sequence
        X = np.mat(np.random.beta(a=Alpha_Beta[num_param-2], b=Beta_Beta[num_param-2], size=(n_nodes,num_param)))
        A=gen_adj_from_posns(X)
        P = np.maximum(0, np.minimum((X@X.T),1))
        
        X_hat=ase(A,d=num_param,aug_type=aug_type)
        A_hat=gen_adj_from_posns(X_hat)
        P_hat= np.maximum(0, np.minimum((X_hat@X_hat.T),1))
        
        X_star=bootstrap_rows(X)
        X_star=X
        A_star=gen_adj_from_posns(X_star)
        P_star = np.maximum(0, np.minimum((X_star@X_star.T),1))
        #########################################
        
        E.append(max(np.array(A).sum(axis=1)))
        E_hat.append(max(np.array(A_hat).sum(axis=1)))
        E_star.append(max(np.array(A_star).sum(axis=1)))
        
        E_expect.append(max(np.array(P).sum(axis=1)))
        E_hat_expect.append(max(np.array(P_hat).sum(axis=1)))
        E_star_expect.append(max(np.array(P_star).sum(axis=1)))
        #expected degree seq
        d_Pseq.append(list(np.array(P).sum(axis=1)))
        d_star_Pseq.append(list(np.array(P_star).sum(axis=1)))
        d_hat_Pseq.append(list(np.array(P_hat).sum(axis=1)))
        #degree seq
        d_seq.append(list(np.array(A).sum(axis=1)))
        d_star_seq.append(list(np.array(A_star).sum(axis=1)))
        d_hat_seq.append(list(np.array(A_hat).sum(axis=1)))



if function=='dir':
    for times in range(num_run):
        D=[];D_hat=[]
        D_expect=[];D_hat_expect=[]
        D_star=[];D_star_expect=[]
        for repeat in range(n_nodes):
            X = np.random.dirichlet(alpha=[Beta_Dir[num_param-2]]*num_param,size=n_nodes)
            #A=gen_adj_from_posns(X,scale=num_param/2/5)

            A=gen_adj_from_posns(X,scale=num_param/2/5)
            P = np.maximum(0, np.minimum((X@X.T)*num_param/2/5,1))

            X_hat=ase(A,d=num_param,aug_type=aug_type)
            A_hat=gen_adj_from_posns(X_hat)
            P_hat= np.maximum(0, np.minimum((X_hat@X_hat.T),1))
            
            ### Bootstrap
            #X_star=bootstrap_rows(X)
            X_star=X
            A_star=gen_adj_from_posns(X_star)
            P_star = np.maximum(0, np.minimum((X_star@X_star.T),1))
            
            D_expect.append(np.sum(P[0,]))
            D_hat_expect.append(np.sum(P_hat[0,]))
            D_star_expect.append(np.sum(P_star[0,]))
            
            D.append(np.sum(A[0,]))
            D_hat.append(np.sum(A_hat[0,]))
            D_star.append(np.sum(A_star[0,]))

        D_seq.append(D)
        D_hat_seq.append(D_hat)
        D_star_seq.append(D_star)
        
        D_Pseq.append(D_expect)
        D_hat_Pseq.append(D_hat_expect)
        D_star_Pseq.append(D_star_expect)

        N.append(max(D))
        N_hat.append(max(D_hat))
        N_star.append(max(D_star))
        
        N_expect.append(max(D_expect))
        N_hat_expect.append(max(D_hat_expect))
        N_star_expect.append(max(D_star_expect))
        

        X = np.random.dirichlet(alpha=[Beta_Dir[num_param-2]]*num_param,size=n_nodes)
        A=gen_adj_from_posns(X,scale=num_param/2/5)
        P = np.maximum(0, np.minimum((X@X.T)*num_param/2/5,1))
    
        X_hat=ase(A,d=num_param,aug_type=aug_type)
        A_hat=gen_adj_from_posns(X_hat)
        P_hat= np.maximum(0, np.minimum((X_hat@X_hat.T),1))

        #X_star=bootstrap_rows(X)
        X_star=X
        A_star=gen_adj_from_posns(X_star)
        P_star = np.maximum(0, np.minimum((X_star@X_star.T),1))
        
        E.append(max(np.array(A).sum(axis=1)))
        E_hat.append(max(np.array(A_hat).sum(axis=1)))
        E_star.append(max(np.array(A_star).sum(axis=1)))
        
        E_expect.append(max(np.array(P).sum(axis=1)))
        E_hat_expect.append(max(np.array(P_hat).sum(axis=1)))
        E_star_expect.append(max(np.array(P_star).sum(axis=1)))
        #expected degree seq
        d_Pseq.append(list(np.array(P).sum(axis=1)))
        d_star_Pseq.append(list(np.array(P_star).sum(axis=1)))
        d_hat_Pseq.append(list(np.array(P_hat).sum(axis=1)))
        #degree seq
        d_seq.append(list(np.array(A).sum(axis=1)))
        d_star_seq.append(list(np.array(A_star).sum(axis=1)))
        d_hat_seq.append(list(np.array(A_hat).sum(axis=1)))

filename=f'{function}_{n_nodes}_node_{num_param}_dim_{aug_type}_aug_{set_seed}_seed.pkl'
with open(filename, 'wb') as f:  
    pickle.dump([[N,N_hat,N_expect,N_hat_expect,N_star,N_star_expect],[E,E_hat,E_expect,E_hat_expect,E_star,E_star_expect],[D_seq,D_Pseq,D_hat_seq,D_hat_Pseq,D_star_seq,D_star_Pseq],[d_seq,d_Pseq,d_hat_seq,d_hat_Pseq,d_star_seq,d_star_Pseq]], f)


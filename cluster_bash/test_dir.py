
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
from scipy.spatial import distance
from scipy import stats
from scipy.stats import wasserstein_distance
# In[71]:

def gen_adj_from_expectation( P):
    np.fill_diagonal(P,0)
    probs = spsd.squareform(P)
    coinflips = np.random.binomial(n=1,p=probs)
    # turn the vector coinflips into a symmetric hollow matrix
    # before returning it.
    return spsd.squareform(coinflips),P


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

def cal_CI(real_value,observe_value,bootsample):
    #quantile estimation
    ql=np.quantile(bootsample,q=0.025)
    qu=np.quantile(bootsample,q=0.975)
    CI_quantile=[]
    if qu>real_value>ql:
        CI_quantile.append(1)
    else:CI_quantile.append(0)

    if real_value>qu:
        CI_quantile.append(1)
    else:CI_quantile.append(0)
    if real_value<ql:
        CI_quantile.append(1)
    else:CI_quantile.append(0)


    #boot mean
    CI_bootmean=[]
    qu_boot=np.mean(bootsample)+1.96*np.sqrt(np.var(bootsample))
    ql_boot=np.mean(bootsample)-1.96*np.sqrt(np.var(bootsample))
    if qu_boot>real_value>ql_boot:
        CI_bootmean.append(1)
    else:CI_bootmean.append(0)

    if real_value>qu_boot:
        CI_bootmean.append(1)
    else:CI_bootmean.append(0)
    if real_value<ql_boot:
        CI_bootmean.append(1)
    else:CI_bootmean.append(0)

    #observe mean
    CI_normal=[]
    upper_bound=observe_value+1.96*np.sqrt(np.var(bootsample))
    lowwer_bound=observe_value-1.96*np.sqrt(np.var(bootsample))
    if upper_bound>real_value>lowwer_bound:
        CI_normal.append(1)
    else:CI_normal.append(0)

    if real_value>qu_boot:
        CI_normal.append(1)
    else:CI_normal.append(0)
    if real_value<ql_boot:
        CI_normal.append(1)
    else:CI_normal.append(0)

    return [CI_quantile,CI_bootmean,CI_normal]
    #return [CI_quantile,quantile_above,quantile_below],[CI_bootmean,bootmean_above,bootmean_below],[CI_normal,ob_above,ob_below]

n_nodes=int(sys.argv[1])#number of nodes 50 100 200 500 1000
aug_type=str(sys.argv[2])#none row_sum
function=str(sys.argv[3])#beta dir
set_seed=int(sys.argv[4])# 1 400
print(f'{n_nodes}-{aug_type}-{function}-{set_seed}')


Max_real=[]
for seed in range(20000):
    if function=='beta':
        X = np.mat(np.random.beta(a=2, b=3, size=(n_nodes,1)))
    if function=='dir':
        X = np.random.dirichlet(alpha=[10]*2,size=n_nodes)
    A,P=gen_adj_from_posns(X)
    Max_real.append(max(A.sum(axis=1)))
real_value=np.mean(Max_real)
real_sd=np.sqrt(np.var(Max_real))

DB_CI_total=[]
Boot_CI_total=[]
DB_sd_total=[]
Boot_sd_total=[]
for times in range(50): #1500

    if function=='beta':
        X = np.mat(np.random.beta(a=2, b=3, size=(n_nodes,1)))
    if function=='dir':
        X = np.random.dirichlet(alpha=[10]*2,size=n_nodes)

    A,P=gen_adj_from_posns(X)
    observe_value=max(A.sum(axis=1))

    #Bootstrap ASE
    if function=='beta':
        X_hat=ase(A,d=1,aug_type=aug_type)  
    if function=='dir':
        X_hat=ase(A,d=2,aug_type=aug_type)  
    A_hat,P_hat=gen_adj_from_posns(X_hat)


    Bootstrap=[];DB=[]#substitute

    for i in range(3000):#2000 for quantiles
        #Same X, bootstrap performance
        X_star=bootstrap_rows(X)
        A_star,P_star=gen_adj_from_posns(X_star)#generate an another A based on P
        DB.append(max(A_star.sum(axis=1)))


        X_star_hat=bootstrap_rows(X_hat)
        A_star_hat,P_star_hat=gen_adj_from_posns(X_star_hat)
        Bootstrap.append(max(A_star_hat.sum(axis=1)))

        DB_CI=cal_CI(real_value,observe_value,DB)
        Boot_CI=cal_CI(real_value,observe_value,Bootstrap)

        DB_sd=np.sqrt(np.var(DB))
        Boot_sd=np.sqrt(np.var(Bootstrap))

    
    # np.mean bootstrap
    # np.var bootstrap
    # 1-d beta distribution Quantile CI

    Boot_CI_total.append(Boot_CI)
    DB_CI_total.append(DB_CI)
    Boot_sd_total.append(Boot_sd)
    DB_sd_total.append(DB_sd)



filename=f'{function}_{n_nodes}_node_{aug_type}_aug_{set_seed}_seed.pkl'
with open(filename, 'wb') as f:  
    pickle.dump([[DB_CI_total,Boot_CI_total],real_value,[real_sd,DB_sd_total,Boot_sd_total]], f)

#larger parameters
#quantile CI, 2000 boot samples




        
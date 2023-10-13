import torch
import random
import numpy as np
import networkx as nx
import pandas as pd

def degrade_dataset(X, missingness, v):
    """
    Inputs:
        dataset to corrupt
        % of data to eliminate[0,1]
        'zero' or 'nan'
      Outputs:
        corrupted Dataset 
        binary mask
    """
    X_1d = X.flatten()
    n = len(X_1d)
    mask_1d = np.ones(n)
    corrupt_ids = random.sample(range(n), int(missingness * n))
    for i in corrupt_ids:
        X_1d[i] = v
        mask_1d[i] = 0

    cX = X_1d.reshape(X.shape)
    mask = mask_1d.reshape(X.shape)

    return cX, mask

def merl_dataset2nxg(nb_sensors=8):
    """
    Outputs:
       networkx MultiDiGraph from adjacency matrix
    """
    if nb_sensors==8:
        A = pd.read_csv('combined_floor8_8s_A.csv',sep=' ', header=None)
        weight= pd.read_csv('combined_floor8_8s_weight.csv', sep=' ', header=None, dtype=float)
    elif nb_sensors==16:
        A = pd.read_csv('combined_floor8_16s_A.csv', sep=' ', header=None)
        weight = pd.read_csv('combined_floor8_16s_weight.csv', sep=' ', header=None, dtype=float)
    elif nb_sensors == 32:
        A = pd.read_csv('combined_floor8_32s_A.csv', sep=' ', header=None)
        weight = pd.read_csv('combined_floor8_32s_weight.csv', sep=' ', header=None, dtype=float)

    weight=weight.iloc[:, :].values
    ngx = nx.DiGraph(A)

    i=0
    for  source, target in ngx.edges():
        ngx[source][target]['weight'] = weight[0,i]
        i+=1

    #print("Edges in G: ", ngx.nodes(data=True))
    return ngx
#merl_dataset2nxg()
def batch_mask(size, batch_size):
    """
    Inputs:
        lenght of the mask
        number of elements to mask
    Output: mask for the critic
    """

    b_mask = np.zeros(size)
    pos = random.sample(range(size), batch_size)
    b_mask[pos] = 1
    return b_mask

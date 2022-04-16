from collections import defaultdict
import numpy as np
from itertools import product
from scipy.special import gamma
from scipy.spatial.distance import pdist, squareform, euclidean
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
from sklearn.linear_model import LinearRegression
import math
from functions_for_clustering import *
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn import metrics
import threading

def make_z_vector_fixed(arr, pattern):
    z_vector = []
    for i in range(len(pattern)):
        z_vector.append(arr[pattern[i]])
    return z_vector

def motif_duckery(clustering_data, current_pattern, f):   
    cdzo = [generate_z_vector_fixed(x,current_pattern) for x in clustering_data]
    # cdzo = clustering_data_z_optimized
    clustering_result = get_clustering(cdzo, 2, 1) # clustering
    cru = list(set(clustering_result)) # clustering_result_unique
    
    for legend2 in len(cru):
        cluster = [] # choosing a cluster
        lcn = max(cru, key=list(clustering_result).count) # largest cluster number
        for i in range(len(clustering_result)):
            if clustering_result[i] == lcn:
                cluster.append(clustering_data[i])
        cru.remove(lcn)
        motif = generate_motif(cluster)
        f.write(motif)
        f.write('\n')
        
        
patterns_for_supercomp=[]
f = open("patterns/patterns_supercomp.txt", "r")

for line in f:
    line = line[1 : -2]
    temp = list(map(int, line.split(", ")))
    patterns_for_supercomp.append(temp)
f.close()


clustering_data = []
for legend in range(0,20):
    df.temp = read_csv('data/train/part_%s.csv' % str(legend))
    
    #extracting data
    df_temp = df.drop(labels=['Ticker', 'index', 'Date'], axis=1)
    key1 = -14 #the number of ticks we wish to observe
    key2 = min([int(x) for x in df_temp.columns])
    df_temp = df_temp.drop(labels=[str(x) for x in list(range(key2,key1))], axis=1) #drop all ticks but those we're observing
    df_temp = df_temp.dropna(axis=0) #drop rows with nan
    clustering_data_trended = df_temp.values.tolist() #convert resulting data to list
    clustering_data.append(detrend(clustering_data_trended)) #detrend resulting data
    
f = open('thamotifs.txt', 'w')
        
threads = [threading.Tread(target=motif_duckery, args=(clustering_data, x, f)) for x in patterns_for_supercomp[:24]]

for i in threads:
    i.start()
    i.join()
    
f.close()
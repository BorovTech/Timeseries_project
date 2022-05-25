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

def motif_duckery(clustering_data, current_pattern):
    cdzo = [make_z_vector_fixed(x, current_pattern) for x in clustering_data]
    # cdzo = clustering_data_z_optimized
    new_wishart = Wishart(2, 1)
    clustering_result = new_wishart.fit(np.array(cdzo), workers=4, batch_weight_in_gb=100) # clustering
    cru = list(set(clustering_result)) # clustering_result_unique
    
    for legend2 in range(150):
        cluster = [] # choosing a cluster
        lcn = max(cru, key=list(clustering_result).count) # largest cluster number
        for i in range(len(clustering_result)):
            if clustering_result[i] == lcn:
                cluster.append(clustering_data[i])
        cru.remove(lcn)
        motif = generate_motif(cluster)
        
        ot_f = open('thamotifs%s.txt' % ''.join(map(str, current_pattern)), 'a')
        ot_f.write('\n')
        ot_f.write(', '.join(map(str, motif)))
        ot_f.close()
        
        
patterns_for_supercomp=[]
fi = open("patterns/patterns_supercomp_7.txt", "r")
for line in fi:
    temp = list(map(int, line.split(", ")))
    patterns_for_supercomp.append(temp)
fi.close()


clustering_data = []
for legend in range(0,20):
    print("data %s loaded" % legend)
    df = pd.read_csv('data/train/part_%s.csv' % str(legend))
    #extracting data
    df_temp = df.drop(labels=['Ticker', 'index', 'Date'], axis=1)
    key1 = -6 #the number of ticks we wish to observe - 1
    key2 = min([int(x) for x in df_temp.columns[1:]])
    df_temp = df_temp.drop(labels=[str(x) for x in list(range(key2,key1))], axis=1) #drop all ticks but those we're observing
    df_temp = df_temp.dropna(axis=0) #drop rows with nan
    clustering_data_trended_temp = df_temp.values.tolist() #convert resulting data to list
    clustering_data_trended = [x[::-1] for x in clustering_data_trended_temp]
    for ser in clustering_data_trended:
        clustering_data.append(detrend_flat(ser)) #detrend resulting data

current_pattern = patterns_for_supercomp[21]
motif_duckery(clustering_data, current_pattern)
from collections import defaultdict
from itertools import product
from scipy.spatial.distance import pdist, squareform, euclidean
import pandas as pd
from random import randint
from sklearn.linear_model import LinearRegression
import math
from functions_for_clustering import *
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn import metrics


def make_z_vector_fixed(arr, pattern):
    z_vector = []
    for i in range(len(pattern)):
        z_vector.append(arr[pattern[i]])
    return z_vector


df = pd.read_csv('part_2.csv')

#extracting data
df_temp = df.drop(labels=['Ticker', 'index', 'Date', 'extremum'], axis=1)

key1 = -15
key2 = min([int(x) for x in df_temp.columns])

df_temp = df_temp.drop(labels=[str(x) for x in list(range(key2,key1))], axis=1)
df_temp = df_temp.loc[df_temp.index < 11150]  #it was likr 4000 (just in case)
df_temp = df_temp.dropna(axis=0)

clustering_data_trended = df_temp.values.tolist()
clustering_data = detrend(clustering_data_trended)

array_of_patterns = [[0, 3, 5, 7, 12], [0, 4, 10, 11, 14], [0, 2, 7, 9, 13], [3, 7, 9, 13, 14], [0, 3, 8, 11, 13], [2, 9, 10, 12, 14], [2, 4, 5, 10, 14], [3, 4, 8, 10, 12], [0, 1, 3, 7, 14], [0, 1, 6, 7, 14], [4, 5, 7, 12, 14], [3, 10, 12, 13, 14], [0, 5, 7, 10, 12], [1, 3, 7, 8, 10], [0, 2, 3, 6, 14], [0, 1, 2, 4, 14], [2, 4, 9, 12, 14], [0, 9, 11, 13, 14], [1, 3, 6, 8, 11], [0, 6, 7, 10, 11], [0, 2, 4, 6, 11], [0, 6, 7, 10, 14], [0, 3, 9, 11, 14], [0, 1, 3, 7, 14], [0, 5, 6, 8, 10], [1, 5, 6, 7, 13], [0, 2, 7, 11, 14], [0, 5, 8, 10, 14], [2, 3, 5, 10, 12], [2, 4, 7, 12, 14], [1, 3, 6, 10, 11], [1, 2, 6, 11, 12], [2, 4, 5, 7, 8], [0, 7, 12, 13, 14], [1, 6, 7, 8, 9], [1, 3, 8, 10, 11], [1, 2, 5, 11, 12], [0, 6, 7, 9, 12], [2, 3, 4, 6, 12], [6, 9, 12, 13, 14], [2, 3, 5, 6, 14], [1, 4, 10, 11, 14], [2, 4, 6, 8, 12], [4, 5, 6, 8, 14], [0, 1, 7, 12, 14], [0, 5, 8, 9, 10], [4, 5, 6, 9, 13], [1, 2, 3, 8, 13], [0, 4, 7, 9, 14], [1, 2, 5, 9, 10], [4, 7, 9, 11, 13]]

f = open("new_table2.txt", "w")

for pattern in array_of_patterns:
    clustering_data_z_optimized = []
    for i in clustering_data:
        clustering_data_z_optimized.append(make_z_vector_fixed(i, pattern))
    
    clustering_result = get_clustering(clustering_data_z_optimized, 2, 1)
    sil_score = metrics.silhouette_score(clustering_data_z_optimized, clustering_result, metric='euclidean')
    DB_score = metrics.davies_bouldin_score(clustering_data_z_optimized, clustering_result)
    f.write(str(pattern))
    f.write(" ")
    f.write(str(sil_score))
    f.write(" ")
    f.write(str(DB_score))
    f.write('\n')

f.close()
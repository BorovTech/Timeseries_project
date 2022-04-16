from collections import defaultdict
import numpy as np
from itertools import product
from scipy.special import gamma
from scipy.spatial.distance import pdist, squareform, euclidean
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from random import randint


def volume(r, m):
    return np.pi ** (m / 2) * r ** m / gamma(m / 2 + 1)


def significant(cluster, h, p):
    max_diff = max(abs(p[i] - p[j]) for i, j in product(cluster, cluster))

    return max_diff >= h


def partition(dist, l, r, order):
    if l == r:
        return l

    pivot = dist[order[(l + r) // 2]]
    left, right = l - 1, r + 1
    while True:
        while True:
            left += 1
            if dist[order[left]] >= pivot:
                break

        while True:
            right -= 1
            if dist[order[right]] <= pivot:
                break

        if left >= right:
            return right

        order[left], order[right] = order[right], order[left]
        


def detrend(data_matrix):
    ans = []
    for data_list in data_matrix:
        series = pd.Series(data_list)
        X = [i for i in range(0, len(series))]
        X = np.reshape(X, (len(X), 1))
        y = series.values
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        detrended = [y[i]-trend[i] for i in range(0, len(series))]
        ans.append(detrended)
    return ans


def detrend_flat(data_list):
    ans = []
    series = pd.Series(data_list)
    X = [i for i in range(0, len(series))]
    X = np.reshape(X, (len(X), 1))
    y = series.values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)
    detrended = [y[i]-trend[i] for i in range(0, len(series))]
    ans.append(detrended)
    return ans

        
def nth_element(dist, order, k):
    l = 0
    r = len(order) - 1
    while True:
        if l == r:
            break
        m = partition(dist, l, r, order)
        if m < k:
            l = m + 1
        elif m >= k:
            r = m

            
def get_clustering(x, k, h, verbose=True):
    n = len(x)
    if isinstance(x[0], list):
        m = len(x[0])
    else:
        m = 1
    dist = squareform(pdist(x)) #checkpoint №1

    dk = []
    for i in range(n):
        order = list(range(n))
        nth_element(dist[i], order, k - 1)
        dk.append(dist[i][order[k - 1]])

    p = [k / (volume(dk[i], m) * n) for i in range(n)]

    w = np.full(n, 0)
    completed = {0: False}
    last = 1
    vertices = set()
    for d, i in sorted(zip(dk, range(n))):
        neigh = set()
        neigh_w = set()
        clusters = defaultdict(list)
        for j in vertices:
            if dist[i][j] <= dk[i]:
                neigh.add(j)
                neigh_w.add(w[j])
                clusters[w[j]].append(j)

        vertices.add(i)
        if len(neigh) == 0:
            w[i] = last
            completed[last] = False
            last += 1
        elif len(neigh_w) == 1:
            wj = next(iter(neigh_w))
            if completed[wj]:
                w[i] = 0
            else:
                w[i] = wj
        else:
            if all(completed[wj] for wj in neigh_w):
                w[i] = 0
                continue
            significant_clusters = set(wj for wj in neigh_w if significant(clusters[wj], h, p))
            if len(significant_clusters) > 1:
                w[i] = 0
                for wj in neigh_w:
                    if wj in significant_clusters:
                        completed[wj] = (wj != 0)
                    else:
                        for j in clusters[wj]:
                            w[j] = 0
            else:
                if len(significant_clusters) == 0:
                    s = next(iter(neigh_w))
                else:
                    s = next(iter(significant_clusters))
                w[i] = s
                for wj in neigh_w:
                    for j in clusters[wj]:
                        w[j] = s
    return w


def index_element(arr, i, partition, identifier):
    if (identifier == "max"):
        return i+arr[i:i+partition-1].index(max(arr[i:i+partition-1]))
    else:
        return i+arr[i:i+partition-1].index(min(arr[i:i+partition-1]))
    

#TODO: поделить ряд на n // 2 участке и на каждом участке брать min и max 
def generate_z_vector_best(arr, n):
    z_vector = []
    partition = (len(arr) * 2) // n
    for i in range(0,len(arr),partition):
        z_vector.append(index_element(arr, i, partition, "max"))
        z_vector.append(index_element(arr, i, partition, "min"))
    return z_vector
#я тут немного пошаманил и сделал дополнительную функцию чтобы наш говнокод выглядел хоть чуть-чуть получше

def generate_motif(clustered_data):
    motif = []
    transposed_data = np.transpose(clustered_data)
    for point in transposed_data:
        motif.append(np.average(point))
    return motif

def find_boundaries(visualization_data, vdf_min, vdf_max, time_stamp):
    #distance2 = math.hypot(max(motif), min(motif))
    #distance3 = abs(max(motif))*math.sqrt(1 + (min(motif)/max(motif))**2)
    scale = np.arange(vdf_min, vdf_max, (vdf_max-vdf_min)/100) # divide the y-axis into 100 "cells"

    count_on_scale = [0]*len(scale)
    for sri in range(len(scale)-1): # sri = scale_range_index, here we insert in every cell the number of series passing through it
        for time_series in visualization_data:
            if scale[sri] < time_series[time_stamp] < scale[sri + 1]:
                count_on_scale[sri] += 1

    for i in range(len(count_on_scale)):
        if count_on_scale[i] <= 0.65*max(count_on_scale):
            count_on_scale[i] = 0

    boundaries = []
    index_max = count_on_scale.index(max(count_on_scale))
    for i in range(index_max, 0, -1):
        if count_on_scale[i] == 0:
            boundaries.append(scale[i])
            break
    for i in range(index_max, len(count_on_scale)):
        if count_on_scale[i] == 0:
            boundaries.append(scale[i])
            break
    return boundaries
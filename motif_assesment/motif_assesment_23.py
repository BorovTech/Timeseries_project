# -*- coding: utf-8 -*-
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
import threading


def assesment_out(filename, assesment, code):
    ot_f = open(filename, code)
    ot_f.write('\n')
    ot_f.write(', '.join(map(str, assesment['motif'])))
    ot_f.write('\n')
    ot_f.write('type_1_errors: ')
    ot_f.write(str(assesment['type_1_errors']))
    ot_f.write(' ; type_2_errors: ')
    ot_f.write(str(assesment['type_2_errors']))
    ot_f.write(' ; hits: ')
    ot_f.write(str(assesment['hits']))
    ot_f.write(' ; ratio: ')
    ot_f.write(str(assesment['ratio']))
    ot_f.close()

def small_patterns(arr):
    #if arr[i + 1] > arr[i] -> 1
    #else 0
    output = []
    for i in range(len(arr)-1):
        if arr[i + 1] >= arr[i]:
            output.append(1)
        else:
            output.append(0)
    return output

def check_directions(motif, series_window):
    if small_patterns(motif) == small_patterns(series_window):
        return True
    else:
        return False
    
def check_boundaries_euclidean(motif, series_window, EPS=0.05):
    dist = euclidean(motif, series_window)
    if dist < EPS:
        return True
    return False
    
def motif_assesment(motif1):

    type_1_errors = 0 # найден мотив, а точки смены тренда нет
    type_2_errors = 0 # точке смены тренда не предшествовал мотив
    hits = 0

    window_len = len(motif1) #m_len

    for legend in range(6):
        df = pd.read_csv('data/test/part_%s.csv' % str(legend))
        df = df.drop(labels=['Ticker', 'index', 'Date'], axis=1)
        for i in df.index:
            df_temp = df.iloc[[i]]
            df_temp = df_temp.dropna(axis=1)
            series = df_temp.values.tolist()
            if len(series[0]) >= window_len:
                series = detrend(series)
                series = list(np.array(series).flatten())
                series = series[::-1]

                # type 1
                for transposition in range(0, len(series) - window_len - 1):
                    # transposition is a pointer, determining what number we transpose the window by
                    window = series[transposition:transposition + window_len]
                    # мы приложили мотив к window начинающемуся с transposition, 
                    # теперь проходимся по точкам window и проверяем, попадают ли они под мотив
                    if check_directions(motif1, window) and check_boundaries_euclidean(motif1, window):
                        type_1_errors += 1


                # type 2 
                transposition = len(series) - window_len
                window = series[transposition:transposition + window_len]
                if check_directions(motif1, window) and check_boundaries_euclidean(motif1, window):
                    hits += 1
                else:
                    type_2_errors += 1
                    
    if type_1_errors == 0 and hits != 0:
        ratio = float("inf")
    elif type_1_errors == 0:
        ratio = 0
    else:
        ratio =  hits/type_1_errors # the more the ratio the more succesful the motif
    
    assesment = {'type_1_errors':type_1_errors,
                'type_2_errors':type_2_errors,
                'hits':hits,
                'ratio':ratio,
                'motif':motif1}
    
    return assesment


def assesment_duckery(pattern):
    f = open('thamotifs%s.txt' % ''.join(map(str, pattern)), 'r')
    f.readline()
    for line in f:
        current_motif = list(map(float, line.split(", "))) # up - upper_boundary, low - lower_boundary
        current_assesment = motif_assesment(current_motif)
        assesment_out('motifs_report_euc%s.txt' % ''.join(map(str, pattern)), current_assesment, 'a')
    f.close()
    


patterns_for_supercomp=[]
fi = open("patterns/patterns_supercomp_7.txt", "r")
for line in fi:
    temp = list(map(int, line.split(", ")))
    patterns_for_supercomp.append(temp)
fi.close()
assesment_duckery(patterns_for_supercomp[23])
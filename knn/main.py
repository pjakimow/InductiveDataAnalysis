# -*- coding: utf-8 -*-
import numpy as np
import tests
import data
import matplotlib

# %matplotlib auto #view plot in separated window #tap into the console
matplotlib.rcParams.update({'font.size': 30})

folder='datasets'
file_names=['iris', 'wine', 'glass', 'pima', 'seeds']

metrics=['euclidean','manhattan']
weights=['uniform','distance']
scoring = ['accuracy', 'f1_macro']

# Prepare data
X, y=data.read_data(folder, file_names[4])
X=data.standardize(X)

# Run tests
k_test_set = np.arange(1,21)
weight='uniform'
#tests.test_metrics_and_k(X, y, scoring, k_test_set, metrics, weight, True, 5)
#tests.test_weights_and_k(X, y, scoring, k_test_set, weights, 'manhattan', True, 5)

n_test_set=np.arange(2,15)
tests.test_ncrossvalidation_with_const_k(X, y, n_test_set, scoring, 9, 'manhattan', 'uniform')
#tests.test_ncrossvalidation_with_const_n(X, y, k_test_set, scoring, n, metric, weight)
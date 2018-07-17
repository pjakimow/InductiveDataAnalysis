import numpy as np
import pandas as pd

def constant_range_step(X, bins_count): 
    #for each feature
    for i in range(len(X[0])): 
        feature = X[:, i]
        bins = np.linspace(min(feature), max(feature), bins_count)
        X[:,i] = np.digitize(feature, bins)
    return X

def equal_elements_number(X, bins_count): 
    #for each feature
    for i in range(len(X[0])): 
        feature = X[:, i]
        digitized, _ = pd.qcut(feature, bins_count, labels=False, duplicates='drop', retbins=True)
        X[:,i] = digitized
    return X

def log_step(X, bins_count):
    #for each feature
    for i in range(len(X[0])): 
        feature = X[:, i]
        bins = np.geomspace(max(min(feature), 1), max(feature),num=bins_count)
        X[:,i] = np.digitize(feature, bins)
    return X

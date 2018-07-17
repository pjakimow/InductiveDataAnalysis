# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import preprocessing

def read_data(file_folder, file_name, sep=',', header_nr=0):
    if file_name=='seeds':
        path=file_folder+'/'+file_name+'.txt'
        df = pd.read_table(path, sep=sep, header=None)
    else:
        path=file_folder+'/'+file_name+'.data'
        df = pd.read_table(path, sep=sep, header=header_nr)
        
    df = pd.read_table(path, sep=sep, header=header_nr)
    if file_name=='iris':
        X=df.iloc[:,0:4]
        y=df.iloc[:,4]
    elif file_name=='wine':
        X=df.iloc[:,1:14]
        y=df.iloc[:,0]
    elif file_name=='glass':
        X=df.iloc[:,1:10]
        y=df.iloc[:,10]
    elif file_name=='pima':
        X=df.iloc[:,0:8]
        y=df.iloc[:,8]
    elif file_name=='seeds':
        X=df.iloc[:,0:7]
        y=df.iloc[:,7]
    return X, y

def standardize(X):
    return preprocessing.scale(X)

import pandas as pd
import numpy as np
import seaborn as sns

sns.set_style("white")

def read_data(file_folder, file_name, sep=',', header_nr=0): 
    path=file_folder+'/'+file_name+'.data'
    df = pd.read_table(path, sep=sep, header=header_nr)
        
    if file_name=='iris':
        X=df.iloc[:,0:4].as_matrix()
        y=np.asarray(df.iloc[:,4].tolist())
    elif file_name=='wine':
        X=df.iloc[:,1:14].as_matrix()
        y=np.asarray(df.iloc[:,0].tolist())
    elif file_name=='glass':
        X=df.iloc[:,1:10].as_matrix()
        y=np.asarray(df.iloc[:,10].tolist())
    elif file_name=='pima-indians-diabetes':
        X=df.iloc[:,0:8].as_matrix()
        y=np.asarray(df.iloc[:,8].tolist())
    return X, y
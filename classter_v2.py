
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import ujson as json
from tqdm import tqdm
from  scipy.stats import skew,kurtosis,entropy

traint_dir=pathlib.Path(r'D:\Ботать\Работа\dcase\dev_data\csv_2\train')
files = list(traint_dir.glob('*.csv'))
sizex=len(files)
for num,file in tqdm(enumerate(files),total=len(files)):
    fits=pd.read_csv(file,index_col=0).values.T[0]
    if num==0:
        X=np.zeros((sizex,fits.shape[0]))
        y=np.zeros((sizex))
    X[num,:]=fits
    name=file.stem.split('_')[0]
    if name=='normal':
        y[num]=1
    else:
        y[num]=0

#%%

dataframe=pd.DataFrame(X).to_csv('X_fits_max_from_wavlet.csv')
dataframey=pd.DataFrame(y).to_csv('y_fits_max_from_wavlet.csv')


#%%

traint_dir=pathlib.Path(r'D:\Ботать\Работа\dcase\dev_data\csv_2\test')
files = list(traint_dir.glob('*.csv'))
sizex=len(files)
for num,file in tqdm(enumerate(files),total=len(files)):
    fits=pd.read_csv(file,index_col=0).values.T[0]
    if num==0:
        X_test=np.zeros((sizex,fits.shape[0]))
        y_test=np.zeros((sizex))
    X_test[num,:]=fits
    name=file.stem.split('_')[0]

    if name=='normal':
        y_test[num]=1
    else:
        y_test[num]=0

#%%

dataframe=pd.DataFrame(X_test).to_csv('X_test_fits_max_from_wavlet.csv')
dataframey=pd.DataFrame(y_test).to_csv('y_test_fits_max_from_wavlet.csv')

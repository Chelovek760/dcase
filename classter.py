
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pathlib
import ujson as json
from tqdm import tqdm
from  scipy.stats import skew,kurtosis,entropy


#%%

def FeatureSpectralDecrease(X):
    # compute index vector
    kinv = np.arange(0, X.shape[0])
    kinv[0] = 1
    kinv = 1 / kinv

    norm = X.sum(axis=0, keepdims=True)
    ind = np.argwhere(norm == 0)
    if ind.size:
        norm[norm == 0] = 1 + X[0, ind[0, 1]]  # hack because I am not sure how to sum subarrays
    norm = norm - X[0, :]

    # compute slope
    vsc = np.dot(kinv, X - X[0, :]) / norm

    return (vsc)
def FeatureSpectralSlope(X):

    # compute mean
    mu_x = X.mean(axis=0, keepdims=True)

    # compute index vector
    kmu = np.arange(0, X.shape[0]) - X.shape[0] / 2

    # compute slope
    X = X - mu_x
    vssl = np.dot(kmu, X) / np.dot(kmu, kmu)

    return (vssl)
def FeatureSpectralFlux(X):
    # difference spectrum (set first diff to zero)
    X = np.c_[X[:, 0], X]
    # X = np.concatenate(X[:,0],X, axis=1)
    afDeltaX = np.diff(X, 1, axis=1)
    # flux
    vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]
    return (vsf)
def FeatureSpectralRolloff(X):
    total_energy = np.cumsum(X, axis=0)
    threshold = 0.85 * total_energy[-1]
    ind = np.where(total_energy < threshold, np.nan, 1)
    return np.nanmin(ind * freq, axis=0, keepdims=True)
def FeatureSpectralFlatness(X):
    norm = X.mean(axis=0)
    norm[norm == 0] = 1
    X = np.log(X + 1e-20)
    vtf = np.exp(X.mean(axis=0)) / norm
    return (vtf)

def simpleFeats(x):
    feats = []
    feats.append(np.mean(x))
    feats.append(np.median(x))
    feats.append(np.std(x))
    feats.append(np.max(x))
    feats.append(np.min(x))
    feats.append(skew(x))
    feats.append(kurtosis(x))
    feats.append(entropy(x))
    return feats
traint_dir=pathlib.Path(r'D:\Ботать\Работа\dcase\dev_data\fan\train')
files = list(traint_dir.glob('*.json'))
sizex=len(files)
for num,file in tqdm(enumerate(files),total=len(files)):
    with open(file) as json_file:
        data=json.load(json_file)
        bulat=np.array(simpleFeats(data['coef_list']))
        coef = np.array(data['coef_list'])
        koef=np.mean(np.array(data['disbal_coef']))
        freq=np.std(np.array(data['main_tone_list']))
        spectal=coef.reshape(data['size'])
        fsd=np.mean(FeatureSpectralDecrease(spectal)[0])
        fsf=np.mean(FeatureSpectralFlux(spectal))
        fsr=np.mean(FeatureSpectralRolloff(spectal)[0])
        fss=np.mean(FeatureSpectralSlope(spectal))
        fsflat=np.mean(FeatureSpectralFlatness(spectal))
        fits=np.hstack((bulat,koef,freq,fsd,fsf,fsr,fss,fsflat))

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

dataframe=pandas.DataFrame(X).to_csv('X_fits.csv')
dataframey=pandas.DataFrame(y).to_csv('y_fits.csv')


#%%

traint_dir=pathlib.Path(r'D:\Ботать\Работа\dcase\dev_data\fan\test')
files = list(traint_dir.glob('*.json'))
sizex=len(files)
for num,file in tqdm(enumerate(files),total=len(files)):
    with open(file) as json_file:
        data = json.load(json_file)
        bulat = np.array(simpleFeats(data['coef_list']))
        coef = np.array(data['coef_list'])
        koef = np.mean(np.array(data['disbal_coef']))
        freq = np.std(np.array(data['main_tone_list']))
        spectal = coef.reshape(data['size'])
        fsd = np.mean(FeatureSpectralDecrease(spectal)[0])
        fsf = np.mean(FeatureSpectralFlux(spectal))
        fsr = np.mean(FeatureSpectralRolloff(spectal)[0])
        fss = np.mean(FeatureSpectralSlope(spectal))
        fsflat = np.mean(FeatureSpectralFlatness(spectal))
        fits = np.hstack((bulat, koef, freq, fsd, fsf, fsr, fss, fsflat))
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

dataframe=pandas.DataFrame(X_test).to_csv('X_test_fits.csv')
dataframey=pandas.DataFrame(y_test).to_csv('y_test_fits.csv')

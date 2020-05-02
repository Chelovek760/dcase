import pathlib
from concurrent.futures import ThreadPoolExecutor
import F5signal as f5s
import numpy as np
import pywt
from scipy.stats import skew, kurtosis, entropy
from tqdm import tqdm
import pandas as pd
from config import DCASE_JSON_DIR, TEST_WAV_DIR


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


def FeatureSpectralDecrease(X):
    # compute index vector
    kinv = np.arange(0, X.shape[0])
    kinv[0] = 1
    kinv = 1 / kinv
    norm = X.sum()
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
    vsf = np.sqrt((afDeltaX ** 2).sum(axis=0)) / X.shape[0]
    return (vsf)


def FeatureSpectralRolloff(X):
    total_energy = np.cumsum(X, axis=0)
    threshold = 0.85 * total_energy[-1]
    ind = np.where(total_energy < threshold, np.nan, 1)
    return np.nanmin(ind * 20000, axis=0, keepdims=True)


def FeatureSpectralFlatness(X):
    norm = X.mean(axis=0)
    norm[norm == 0] = 1
    X = np.log(X + 1e-20)
    vtf = np.exp(X.mean(axis=0)) / norm
    return (vtf)


def gen_wavelet_feats(x, n_level):
    feats = []
    coeffs = pywt.wavedec(x, 'db4', level=n_level)

    for coeff in coeffs:
        coeff=np.array(coeff)
        feats.append(simpleFeats(coeff))

    feats = np.concatenate(feats)
    return np.reshape(feats, (1, len(feats)))


directory = TEST_WAV_DIR
files = pathlib.Path(directory)
files = list(files.glob('*.wav'))
sizex=len(files)

def visual_finc(wav):
    if pathlib.Path(DCASE_JSON_DIR + wav.stem + '.csv').exists():
        return print(str(wav.stem) + 'Alredy Exist')
    path_str = TEST_WAV_DIR + wav.stem + '.wav'
    path = pathlib.Path(path_str)
    wave_good = f5s.read_wave(str(path))
    return gen_wavelet_feats(wave_good.ys,7)

for ix, wav in tqdm(enumerate(files),total=sizex):
    fists= visual_finc(wav)

    if ix==0:
        matrix=np.zeros((sizex,fists.shape[1]))
    matrix[ix,:]=fists
df=pd.DataFrame(matrix).to_csv('feature/Train_Dcase_Bulat_feats_tran.csv')
# with ThreadPoolExecutor(4) as executor:
#     for _ in tqdm(executor.map(visual_finc, files)):
#         pass

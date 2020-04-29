from scipy import signal

import numpy as np
import F5report as f5r
import pathlib
import pandas as pd
import json
import numpy as np
import pathlib
import ujson as json
from  scipy.stats import skew,kurtosis,entropy
from config import DCASE_CSV_DIR
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
def FeatureSpectralRolloff(X,freq):
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

class Saver():
    def __init__(self, path_in, dir_out, Frequency_Analysis_obj):
        self.na = Frequency_Analysis_obj
        self.path = path_in
        self.dir_out = dir_out
        self.data_from_isotone()
        self.saver_json()

    def data_from_isotone(self):
        self.na.isotone_n()
        self.xcoord = np.array(self.na.harmony[0]['time'])
        self.ycoord_mid = self.na.harmony[0]['filtered_mid_from']
        dict_list = {}

        for t, freq in enumerate(self.ycoord_mid):
            heroistic = f5r.shaft_shaking(freq)['Main']
            dict_list[t] = {}

            main_coef = self.na.c_wavlet_coef[np.abs(self.na.y_axis_freq-heroistic[0]).argmin(), t]
            for i,fsq in enumerate(heroistic[1:]):

                if main_coef == 0:
                    dict_list[t][i] = 0
                else:

                    dict_list[t][i] = self.na.c_wavlet_coef[(np.abs(self.na.y_axis_freq-fsq)).argmin(), t]
        coefq = []
        for t in dict_list:
            s = 0
            for freq in dict_list[t]:
                s += dict_list[t][freq]
            coefq.append(s)
        self.dist_coef = coefq

    def saver_json(self):
        coef_list = self.na.c_wavlet_coef.flatten().tolist()
        time_list = self.xcoord.tolist()
        main_tone_list = self.ycoord_mid
        freqs_line_list = self.na.y_axis_freq.tolist()

        disbal_coef = np.nan_to_num(self.dist_coef, nan=0).tolist()
        size = self.na.c_wavlet_coef.shape
        path = pathlib.Path(self.path)
        dir_out = pathlib.Path(self.dir_out)
        if not dir_out.exists():
            dir_out.mkdir(parents=True, exist_ok=True)
        # save_data_dict = {'coef_list': coef_list, 'time_list': time_list, 'main_tone_list': main_tone_list,
        #                   'disbal_coef': disbal_coef, 'freqs_line_list': freqs_line_list, 'size': size}
        # with open(str(dir_out.joinpath(path.stem)) + '.json', "w") as write_file:
        #     json.dump(save_data_dict, write_file)
        bulat = np.array(simpleFeats(coef_list))
        coef = np.array(coef_list)
        koef = np.max(disbal_coef)
        freq = np.std(freqs_line_list)
        spectal = coef.reshape(size)
        pd.DataFrame(spectal).to_csv(r'D:\Ботать\Работа\dcase\dev_data\coefs\\' + path.stem + '.csv')
        corr=np.sum(signal.convolve(spectal, spectal, mode='same'))
        fsd = np.max(FeatureSpectralDecrease(spectal)[0])
        fsf = np.max(FeatureSpectralFlux(spectal))
        fsr = np.max(FeatureSpectralRolloff(spectal,1/self.na.dt)[0])
        fss = np.max(FeatureSpectralSlope(spectal))
        fsflat = np.max(FeatureSpectralFlatness(spectal))
        fits = np.hstack((bulat, koef, freq, fsd, fsf, fsr, fss, fsflat,corr))
        pd.DataFrame(fits).to_csv(DCASE_CSV_DIR+path.stem+'.csv')
        print(path.stem, ' OK')

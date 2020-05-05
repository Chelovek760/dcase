import F5signal as f5s
import F5analys as f5a
import pathlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
import pathlib
from  scipy.stats import skew,kurtosis,entropy
from config import BAD_CASE_WAV, GOOD_CASE_WAV,TEST_WAV_DIR
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns

directory = TEST_WAV_DIR
files = pathlib.Path(directory)
files = files.glob('*.wav')
files=list(files)
class  Buono_Brutto_Cattivo:
    def __init__(self,file_name,segment_number=99):
        self.segment_number=segment_number
        self.filename=file_name
    def separate(self):
        file=self.filename
        segment_number=self.segment_number
        wave = f5s.read_wave(str(file))
        wavlet = f5a.Wavlet(wave)
        dur=wavlet.x_axis_time[-1]/segment_number
        bad_dict = {'freq': wavlet.y_axis_freq,'dur_part':dur}
        good_dict = {'freq': wavlet.y_axis_freq,'dur_part':dur}
        # fig, ax1, ax2 = wavlet.wavlet_plot()
        newtimeshape=wavlet.c_wavlet_coef.shape[1]//segment_number*segment_number
        #print(wavlet.c_wavlet_coef.shape[1],newtimeshape)
        wavlet_list=np.hsplit(wavlet.c_wavlet_coef[:,:newtimeshape], segment_number)
        #print(wavlet_list[0].shape[0]*wavlet_list[0].shape[1])
        X=np.zeros((segment_number,wavlet_list[0].shape[0]*wavlet_list[0].shape[1]))
        for id,wavelet_part in enumerate(wavlet_list):
            X[id,:]=wavelet_part.flatten()
        pca = PCA(n_components=2)
        Xnew = pca.fit_transform(X)
        model = IsolationForest(n_estimators=500)
        res=model.fit_predict(Xnew)
        countminus=np.sum(res==-1)
        if countminus>segment_number//2:
            res=res*-1
        for i in np.argwhere(res==-1):
            x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
            # ax2.axvspan(x1,x2,alpha=0.3, color='red')
            # ax2.text((x1+x2)/2,wavlet.y_axis_freq[wavlet.y_axis_freq.shape[0]//2],str(i),color='blue')
        for i in np.argwhere(res==1):
            x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
            # ax2.text((x1+x2)/2,wavlet.y_axis_freq[wavlet.y_axis_freq.shape[0]//2],str(i),color='red')
        allfile=pd.DataFrame(X)
        allfile['y']=res
        good_frames=allfile.loc[allfile['y'] == 1].T
        #print(good_frames.shape)
        bad_frames=allfile.loc[allfile['y'] == -1].T
        #print(bad_frames.shape)
        col_list_bad=bad_frames.columns.tolist()
        col_list = good_frames.columns.tolist()
        corr=pd.concat([good_frames, bad_frames], axis=1).corr()
        c=[]
        cor_bad_good= np.abs(corr[col_list].loc[col_list_bad])
        #print(col_list,col_list_bad)
        #print(cor_bad_good)
        # plt.figure()
        # sns.heatmap(cor_bad_good)
        for ind,col in enumerate(col_list):
            if ind==0:
                c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind+1]])))
            elif col_list[ind]==col_list[-1]:
                 c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind-1]])))
            else:
                c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind-1]])))
                c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind+1]])))
        lim=np.median(c)
        #print(lim)
        good_another=cor_bad_good>lim*0.95
        for index, row in good_another.iterrows():
            if row[col_list].mean()>lim:
                res[index]=1
        repared=allfile.copy()
        repared['y']=res
        # print('Bad_time:')

        for i in np.argwhere(res==-1):
            x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
            time1 = float(wavlet.x_axis_time[i*newtimeshape//segment_number])
            time2 = float(wavlet.x_axis_time[(i+1)*newtimeshape//segment_number])
            # print(time1)
            #print(time2)
            bad_dict[i[0]]=wavlet_list[i[0]]
            # ax2.axvspan(x1,x2,alpha=0.3, color='black')

        for i in np.argwhere(res==1):
            x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
            time1 = float(wavlet.x_axis_time[i*newtimeshape//segment_number])
            time2 = float(wavlet.x_axis_time[(i+1)*newtimeshape//segment_number])
            # print(time1)
            #print(time2)
            good_dict[i[0]] = wavlet_list[i[0]]
            #ax2.axvspan(x1,x2,alpha=0.3, color='black')
        # plt.figure()
        # sns.heatmap(np.abs(allfile.loc[allfile['y'] == 1].T.corr()))
        # plt.figure()
        # sns.heatmap(allfile.T.corr())
        #
        #plt.show()
        return good_dict,bad_dict

    def FeatureSpectralDecrease(self,X):
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

    def FeatureSpectralSlope(self,X):
        # compute mean
        mu_x = X.mean(axis=0, keepdims=True)

        # compute index vector
        kmu = np.arange(0, X.shape[0]) - X.shape[0] / 2

        # compute slope
        X = X - mu_x
        vssl = np.dot(kmu, X) / np.dot(kmu, kmu)

        return (vssl)

    def FeatureSpectralFlux(self,X):
        # difference spectrum (set first diff to zero)
        X = np.c_[X[:, 0], X]
        # X = np.concatenate(X[:,0],X, axis=1)
        afDeltaX = np.diff(X, 1, axis=1)
        # flux
        vsf = np.sqrt((afDeltaX ** 2).sum(axis=0)) / X.shape[0]
        return (vsf)

    def FeatureSpectralRolloff(self,X, freq):
        total_energy = np.cumsum(X, axis=0)
        threshold = 0.85 * total_energy[-1]
        ind = np.where(total_energy < threshold, np.nan, 1)
        return np.nanmin(ind * freq, axis=0, keepdims=True)

    def FeatureSpectralFlatness(self,X):
        norm = X.mean(axis=0)
        norm[norm == 0] = 1
        X = np.log(X + 1e-20)
        vtf = np.exp(X.mean(axis=0)) / norm
        return (vtf)

    def simpleFeats(self,x):
        feats = []
        feats.append(np.mean(x))
        # feats.append(np.median(x))
        feats.append(np.std(x))
        feats.append(np.max(x))
        feats.append(np.min(x))
        feats.append(skew(x))
        feats.append(kurtosis(x))
        feats.append(entropy(x))
        return feats

    def features_generator(self,wavlet_matrix=None):
        if not wavlet_matrix:
            good_features_dict={}
            bad_features_dict={}
            good_dict,bad_dict=self.separate()

            good_siment=list(filter(lambda x: isinstance(x,np.int64),good_dict.keys()))
            bad_siment=list(filter(lambda x: isinstance(x,np.int64),bad_dict.keys()))
            for ind in good_siment:
                coef_list = good_dict[ind]
                bulat = np.array(self.simpleFeats(coef_list.flatten()))
                fsd = np.max(self.FeatureSpectralDecrease(coef_list)[0])
                fsf = np.max(self.FeatureSpectralFlux(coef_list))
                fss = np.max(self.FeatureSpectralSlope(coef_list))
                fsflat = np.max(self.FeatureSpectralFlatness(coef_list))
                features_dict = {'mean': bulat[0], 'std': bulat[1], 'max': bulat[2], 'min': bulat[3],'skew': bulat[4], 'kurtosis': bulat[5], 'entropy': bulat[6], 'fsd': fsd, 'fsf': fsf, 'fss': fss, 'fsflat': fsflat}
                good_features_dict[ind]= features_dict
            for ind in bad_siment:
                coef_list = bad_dict[ind]
                bulat = np.array(self.simpleFeats(coef_list.flatten()))
                fsd = np.max(self.FeatureSpectralDecrease(coef_list)[0])
                fsf = np.max(self.FeatureSpectralFlux(coef_list))
                fss = np.max(self.FeatureSpectralSlope(coef_list))
                fsflat = np.max(self.FeatureSpectralFlatness(coef_list))
                features_dict = {'mean': bulat[0], 'std': bulat[1], 'max': bulat[2], 'min': bulat[3], 'skew': bulat[4],
                                 'kurtosis': bulat[5], 'entropy': bulat[6], 'fsd': fsd, 'fsf': fsf, 'fss': fss,
                                 'fsflat': fsflat}
                bad_features_dict[ind] = features_dict
            return good_dict,good_features_dict,bad_dict,bad_features_dict
        else:
            coef_list = wavlet_matrix
            bulat = np.array(self.simpleFeats(coef_list.flatten()))
            fsd = np.max(self.FeatureSpectralDecrease(coef_list)[0])
            fsf = np.max(self.FeatureSpectralFlux(coef_list))
            fss = np.max(self.FeatureSpectralSlope(coef_list))
            fsflat = np.max(self.FeatureSpectralFlatness(coef_list))
            features_dict = {'mean': bulat[0], 'std': bulat[1], 'max': bulat[2], 'min': bulat[3], 'skew': bulat[4],
                             'kurtosis': bulat[5], 'entropy': bulat[6], 'fsd': fsd, 'fsf': fsf, 'fss': fss,
                             'fsflat': fsflat}
            return wavlet_matrix,features_dict
        pass

if __name__=='__main__':
    bbc=Buono_Brutto_Cattivo(r'normal_id_00_00000000.wav')
    print(len(bbc.features_generator()[0]))

import F5signal as f5s
import F5analys as f5a

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from collections import Counter

from config import BAD_CASE_WAV, GOOD_CASE_WAV


directory = TEST_WAV_DIR
files = pathlib.Path(directory)
files = files.glob('*.wav')
files=list(files)
segment_number=99

def find_ext(dir, ext):

    for root, dirs, files in os.walk("/mydir"):
        for file in files:
            if file.endswith(ext):
             print(os.path.join(root, file))
import os
for root, dirs, files in os.walk("/mydir"):
    for file in files:
        if file.endswith(".txt"):
             print(os.path.join(root, file))

def seporator(dir_wav, GOOD_CASE_WAV, BAD_CASE_WAV):
    pass


files=[r'normal_id_00_00000000.wav', r'anomaly_id_00_00000002.wav']
segment_number=99
for num,file in tqdm(enumerate(files),total=len(files)):
    wave = f5s.read_wave(str(file))
    wavlet = f5a.Wavlet(wave) 
    fig, ax1, ax2 = wavlet.wavlet_plot()
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
        ax2.axvspan(x1,x2,alpha=0.3, color='red')
        ax2.text((x1+x2)/2,wavlet.y_axis_freq[wavlet.y_axis_freq.shape[0]//2],str(i),color='blue')
    for i in np.argwhere(res==1):
        x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
        ax2.text((x1+x2)/2,wavlet.y_axis_freq[wavlet.y_axis_freq.shape[0]//2],str(i),color='red')
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
    plt.figure()
    sns.heatmap(cor_bad_good)
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
        countminus = np.sum(res == -1)
        if row[col_list].mean()>lim:
            res[index]=1
    repared=allfile.copy()
    repared['y']=res

    dir_bad_out = pathlib.Path(BAD_CASE_WAV + str(file)[:-4] + '\\')

    if not dir_bad_out.exists():
        dir_bad_out.mkdir(parents=True, exist_ok=True)

    print('Bad_time:')

    for i in np.argwhere(res==-1):
        x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
        time1 = float(wavlet.x_axis_time[i*newtimeshape//segment_number])
        time2 = float(wavlet.x_axis_time[(i+1)*newtimeshape//segment_number])
        wave_tmp = f5s.Wave(wave.segment(time1, time2 - time1).ys, framerate=wave.framerate)
        wave_tmp.write(str(dir_bad_out) + '\\' + str(i) + '.wav')

        ax2.axvspan(x1,x2,alpha=0.3, color='black')

    dir_good_out = pathlib.Path(GOOD_CASE_WAV + str(file.stem)[:-4] + '\\')

    if not dir_good_out.exists():
        dir_good_out.mkdir(parents=True, exist_ok=True)

    print('Good_time:')
    wave_good = f5s.Wave([0], framerate=16000)
    for i in np.argwhere(res==1):

        x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
        time1 = float(wavlet.x_axis_time[i*newtimeshape//segment_number])
        #time2 = float(wavlet.x_axis_time[(i+1)*newtimeshape//segment_number])
        print(time1)
        #print(time2)

        wave_tmp = f5s.Wave(wave.segment(time1, 0.1).ys, framerate=wave.framerate)
        wave_tmp.write(str(dir_good_out) + '\\' + str(i) + '.wav')
        #ax2.axvspan(x1,x2,alpha=0.3, color='black')


    plt.figure()
    sns.heatmap(np.abs(allfile.loc[allfile['y'] == 1].T.corr()))
    plt.figure()
    sns.heatmap(allfile.T.corr())



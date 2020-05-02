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
#from config import DCASE_JSON_DIR, TEST_WAV_DIR,DCASE_FITS_CSV_DIR,DCASE_COEF_CSV_DIR
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
#directory = TEST_WAV_DIR
# files = pathlib.Path(directory)
# files = files.glob('*.wav')
# files=list(files)[0:1]
files=[r'normal_id_00_00000000.wav']
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
    model = IsolationForest()
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

    print('Bad_time:')
    wave_bad = f5s.Wave([0], framerate=16000)
    for i in np.argwhere(res==-1):
        x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
        time1 = float(wavlet.x_axis_time[i*newtimeshape//segment_number])
        #time2 = float(wavlet.x_axis_time[(i+1)*newtimeshape//segment_number])
        print(time1)
        #print(time2)
        wave_tmp = wave.segment(time1, 0.1)
        
        wave_bad = wave_bad + wave_tmp
        ax2.axvspan(x1,x2,alpha=0.3, color='black')

    print('Good_time:')
    wave_good = f5s.Wave([0], framerate=16000)
    for i in np.argwhere(res==1):

        x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
        time1 = float(wavlet.x_axis_time[i*newtimeshape//segment_number])
        #time2 = float(wavlet.x_axis_time[(i+1)*newtimeshape//segment_number])
        #print(time1)
        #print(time1)
        wave_tmp = wave.segment(time1, 0.1)

        wave_good = wave_good + wave_tmp
        #ax2.axvspan(x1,x2,alpha=0.3, color='black')


    plt.figure()
    sns.heatmap(np.abs(allfile.loc[allfile['y'] == 1].T.corr()))
    plt.figure()
    sns.heatmap(allfile.T.corr())

plt.show()

wave_bad.plot()
plt.show()

wave_good.plot()
plt.show()



# name = r'normal_id_00_00000000.wav'
# def return_good_wav(name, segment_number):

#     wave = f5s.read_wave(name)
#     wavlet = f5a.Wavlet(wave)
#     newtimeshape=wavlet.c_wavlet_coef.shape[1]//(segment_number**2)
#     print(wavlet.c_wavlet_coef.shape[1],newtimeshape)
#     wavlet_list=np.hsplit(wavlet.c_wavlet_coef[:,:newtimeshape], segment_number)
#     print(wavlet_list[0].shape[0]*wavlet_list[0].shape[1])
#     X=np.zeros((segment_number,wavlet_list[0].shape[0]*wavlet_list[0].shape[1]))
#     for id,wavelet_part in enumerate(wavlet_list):
#         X[id,:]=wavelet_part.flatten()
#     pca = PCA(n_components=2)
#     Xnew = pca.fit_transform(X)
#     model = IsolationForest()
#     res=model.fit_predict(Xnew)
#     countminus=np.sum(res==-1)
#     if countminus>segment_number//2:
#         res=res*-1
#     for i in np.argwhere(res==-1):
#         x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
#         ax2.axvspan(x1,x2,alpha=0.3, color='red')
#         ax2.text((x1+x2)/2,wavlet.y_axis_freq[wavlet.y_axis_freq.shape[0]//2],str(i),color='blue')
#     for i in np.argwhere(res==1):
#         x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
#         ax2.text((x1+x2)/2,wavlet.y_axis_freq[wavlet.y_axis_freq.shape[0]//2],str(i),color='red')
#     allfile=pd.DataFrame(X)
#     allfile['y']=res
#     good_frames=allfile.loc[allfile['y'] == 1].T
#     bad_frames=allfile.loc[allfile['y'] == -1].T
#     col_list_bad=bad_frames.columns.tolist()
#     col_list = good_frames.columns.tolist()
#     corr=pd.concat([good_frames, bad_frames], axis=1).corr()
#     c=[]
#     cor_bad_good= np.abs(corr[col_list].loc[col_list_bad])
#     print(col_list,col_list_bad)
#     print(cor_bad_good)
#     plt.figure()
#     sns.heatmap(cor_bad_good)
#     for ind,col in enumerate(col_list):
#         if ind==0:
#             c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind+1]])))
#         elif col_list[ind]==col_list[-1]:
#              c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind-1]])))
#         else:
#             c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind-1]])))
#             c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind+1]])))
#     lim=np.median(c)
#     print(lim)
#     good_another=cor_bad_good>lim*0.95
#     for index, row in good_another.iterrows():
#         countminus = np.sum(res == -1)
#         if row[col_list].mean()>lim:
#             res[index]=1
#     repared=allfile.copy()
#     repared['y']=res
#     for i in np.argwhere(res==-1):
#         x1,x2=wavlet.x_axis_time[i*newtimeshape//segment_number],wavlet.x_axis_time[(i+1)*newtimeshape//segment_number]
#         print(x1)
#         ax2.axvspan(x1,x2,alpha=0.3, color='black')
#     plt.figure()
#     sns.heatmap(np.abs(allfile.loc[allfile['y'] == 1].T.corr()))
#     plt.figure()
#     sns.heatmap(allfile.T.corr())

#     return 0


    # kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # pred_y = kmeans.fit_predict(X)
    # print(pred_y.labels_)
# path_str = TEST_WAV_DIR + wav.stem + '.wav'
# path = pathlib.Path(path_str)
# print(path)
# wave = f5s.read_wave(str(path))
# wavlet = f5a.Wavlet(wave)
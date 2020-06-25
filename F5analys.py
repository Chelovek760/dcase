from collections import defaultdict
import numpy as np
from pykalman import KalmanFilter
from scipy import signal, interpolate
from scipy.fftpack import fft
from scipy.signal import argrelextrema
from sklearn.isotonic import IsotonicRegression
from scipy.signal import savgol_filter
from F5signal import razryad, razryad_2d
from Fquality import support_auto_corr_freq, support_fft_freq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tftb.processing.cohen import MargenauHillDistribution

class Wavlet():
    """
    Class if description of wavelet
    """

    def __init__(self, wave):
        print('CALCULATE T-F MAP')
        self.sig = wave.ys
        self.duration = wave.duration
        self.N = np.shape(self.sig)[0]
        self.dt = wave.duration / self.N
        self.title = 'Wave'
        self.t = wave.ts
        tp = 0.5
        waveframe = wave.segment(0, tp)
        arsf, std = support_auto_corr_freq(waveframe)
        fftsf, fftstd = support_fft_freq(waveframe)
        tsarsf = waveframe.ts
        list_of_fftsf = [fftsf] * tsarsf.shape[0]
        list_of_fftsf_std = [fftstd] * tsarsf.shape[0]
        list_of_arsf = [arsf] * tsarsf.shape[0]
        list_of_arsf_std = [std] * tsarsf.shape[0]
        tfr = MargenauHillDistribution(waveframe.ys)
        tf, ts, freq = tfr.run()
        # print(tf.shape)
        tps = np.arange(0, self.t[-1], tp)
        for frame in tps[1:-2]:
            waveframe = wave.segment(frame, tp)
            arsf, std = support_auto_corr_freq(waveframe)
            fftsf, fftstd = support_fft_freq(waveframe)
            tsarsf = np.hstack((tsarsf, waveframe.ts + tsarsf[-1]))
            list_of_arsf = list_of_arsf + [arsf] * waveframe.ts.shape[0]
            list_of_arsf_std = list_of_arsf_std + [std] * waveframe.ts.shape[0]
            list_of_fftsf = list_of_fftsf + [fftsf] * waveframe.ts.shape[0]
            list_of_fftsf_std = list_of_fftsf_std + [fftstd] * waveframe.ts.shape[0]
            tfr = MargenauHillDistribution(waveframe.ys)
            tff, tsf, freqf = tfr.run()
            # print(tff.shape)
            tf = np.hstack((tf, tff))
            ts = np.hstack((ts, tsf))
        # tfr = MargenauHillDistribution(wave.ys)
        # tf, ts, freq = tfr.run()
        n_fbins = len(freq)
        tf = tf[:int(n_fbins / 2.0), :]
        freq = freq[:int(n_fbins / 2.0)]
        self.x_axis_time = np.arange(0, ts.shape[0]) * self.dt
        self.y_axis_freq = freq * wave.framerate
        self.c_wavlet_coef = np.abs(tf).astype(int)
        self.arsf = np.array(list_of_arsf)
        self.arsf_std = np.array(list_of_arsf_std)
        self.fftsf = np.array(list_of_fftsf)
        self.fftsf_std = np.array(list_of_fftsf_std)
        self.tp = tp
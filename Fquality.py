import matplotlib.pyplot as plt
import numpy as np
import F5analys as f5a
import F5signal as f5s

from scipy.signal import argrelextrema, convolve
from pykalman import KalmanFilter
from scipy.fftpack import fft
from scipy.signal import savgol_filter


class Find_quality:

    def __init__(self, wave, range_find, step):
        self.wave = wave
        self.find_freq_list = list(range(range_find[0], range_find[1], step))

    def find_max_quality(self, corr_matrix):
        return np.mean(corr_matrix)

    def kalman_filter_for_corr(self, liney):
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=liney[0],
                          initial_state_covariance=[1],
                          observation_covariance=[1],
                          transition_covariance=0.001)
        state_means, state_covariances = kf.filter(liney)
        state_means = [w[0] for w in state_means]
        return np.array(state_means)

    @classmethod
    def support_auto_corr_freq(self, wave):
        order = 6
        near_max = int(order / 2)
        wave = wave

        sig_corr = np.correlate(wave.ys, wave.ys, mode='same')

        sig_corr = sig_corr[int(len(wave.ys) / 2):]
        sig_corr[sig_corr < 0] = 0
        sig_corr = self.kalman_filter_for_corr(sig_corr)
        time = wave.ts[int(len(wave.ys) / 2):]
        # max_corr_args = argrelextrema(sig_corr, np.greater, order=2)[0]
        # func_max_corr = sig_corr[max_corr_args]
        realtime = []

        # for s in max_corr_args:
        #     realtime.append((s, time[s]))

        realtime = {}

        arg_func_max_corr = argrelextrema(sig_corr, np.greater, order=int(self.wave.framerate / 88))[0]
        for s in arg_func_max_corr:
            realtime[s] = time[s]

        for loc in arg_func_max_corr:
            max = sig_corr[loc]
            argm = loc
            try:
                for arg_near_false_max in range(loc - near_max, loc + near_max):
                    if sig_corr[arg_near_false_max] > max:
                        max = sig_corr[arg_near_false_max]
                        argm = arg_near_false_max
                arg_func_max_corr[arg_func_max_corr.index(loc)] = argm
            except:
                continue
        # print(realtime)
        realtime = [realtime[h] for h in realtime]
        # print(realtime)
        sorted_func_max_corr = sorted(list(zip(sig_corr[arg_func_max_corr], realtime)), key=lambda x: x[1],
                                      reverse=False)
        find_corr_freq = []
        for i in range(0, len(sorted_func_max_corr) - 1):
            find_corr_freq.append(1 / abs(sorted_func_max_corr[i][1] - sorted_func_max_corr[i + 1][1]))
        find_corr_freq = np.array(find_corr_freq)
        mean = np.mean(find_corr_freq)
        mid = np.median(find_corr_freq)
        std = np.std(find_corr_freq)
        # find_corr_freq /= len(sorted_func_max_corr)

        # print(mid, mean, std)
        # plt.plot(time, sig_corr)
        # plt.plot(time[arg_func_max_corr], sig_corr[arg_func_max_corr])
        # plt.plot(time[arg_func_max_corr], sig_corr[arg_func_max_corr], 'o')
        # plt.plot(sorted_func_max_corr[0][1], sorted_func_max_corr[0][0], 'o')
        # plt.plot(sorted_func_max_corr[1][1], sorted_func_max_corr[1][0], 'o')
        find_corr_freq = mean
        return find_corr_freq, std

    def create_correlation(self, freq):
        amp = np.max(self.wave.ys)
        signal_sin = f5s.SinSignal(freq, amp=amp)
        wave_model = signal_sin.make_wave(0.5, self.wave.framerate)
        corr = convolve(self.wave.ys, wave_model.ys, mode='same')
        return corr, signal_sin

    def chose_next_gen(self):
        quality_per_freq = []
        for freq in self.find_freq_list:
            corr, signal_sin = self.create_correlation(freq)
            quality_per_freq.append((freq, self.find_max_quality(corr), signal_sin))
        quality_per_freq = sorted(quality_per_freq, key=lambda x: x[1], reverse=True)
        #print(quality_per_freq)
        return quality_per_freq[0]

    def find_great_corr(self, num_lim):
        len_freqs = len(self.find_freq_list)
        if num_lim > len_freqs:
            num_lim = len_freqs
        great_corr_freq_list = []
        for i in range(num_lim):
            ch_freq = self.chose_next_gen()
            self.find_freq_list.pop(self.find_freq_list.index(ch_freq[0]))
            great_corr_freq_list.append(ch_freq[2])
        sig = f5s.SinSignal(0, amp=0)
        for gs in great_corr_freq_list:
            sig += gs
        wave_model = sig.make_wave(0.5, self.wave.framerate)
        corr = convolve(self.wave.ys, wave_model.ys, mode='same')
        corr = f5s.Wave(corr, framerate=self.wave.framerate)
        return f5a.Wavlet(corr)

    def up_sup_freq(self):
        freq = self.support_auto_corr_freq()
        print(freq)
        sig = f5s.SinSignal(freq, amp=np.max(self.wave.ys))
        wave_model = sig.make_wave(0.5, self.wave.framerate)
        corr = convolve(self.wave.ys, wave_model.ys, mode='same')
        corr = f5s.Wave(corr, framerate=self.wave.framerate)
        corr = corr.downscalefreq(10)
        return f5a.Wavlet(corr)


def support_fft_freq(wave):
    order = 6
    near_max = int(order / 2)
    wave = wave
    sig_corr = np.correlate(wave.ys, wave.ys, mode='same')
    sig_corr = sig_corr[int(len(wave.ys) / 2):]
    sig_corr[sig_corr < 0] = 0
    # kf = KalmanFilter(transition_matrices=[1],
    #                   observation_matrices=[1],
    #                   initial_state_mean=sig_corr[0],
    #                   initial_state_covariance=[1],
    #                   observation_covariance=[1],
    #                   transition_covariance=0.01)
    # state_means, state_covariances = kf.filter(sig_corr)
    # state_means = [w[0] for w in state_means]
    # sig_corr = np.array(state_means)
    # sig_corr = savgol_filter(sig_corr, 101, 3)
    yf = np.abs(fft(sig_corr))
    # print(yf)
    xf = np.linspace(0.0, wave.framerate / 2.0, sig_corr.shape[0] // 2)
    ampline = yf[:sig_corr.shape[0] // 2]
    ampline[0:2] = 0
    # plt.figure(figsize=(14, 6))
    # plt.plot(freqline, ampline)
    # plt.xlabel('Freq,[Hz]')
    # plt.grid()
    return xf[np.argmax(ampline)], 1


def support_auto_corr_freq(wave):
    order = 6
    near_max = int(order / 2)
    wave = wave

    sig_corr = np.correlate(wave.ys, wave.ys, mode='same')

    sig_corr = sig_corr[int(len(wave.ys) / 2):]
    sig_corr[sig_corr < 0] = 0
    # kf = KalmanFilter(transition_matrices=[1],
    #                   observation_matrices=[1],
    #                   initial_state_mean=sig_corr[0],
    #                   initial_state_covariance=[1],
    #                   observation_covariance=[1],
    #                   transition_covariance=0.001)
    # state_means, state_covariances = kf.filter(sig_corr)
    # state_means = [w[0] for w in state_means]
    # sig_corr = np.array(state_means)
    # max_corr_args = argrelextrema(sig_corr, np.greater, order=2)[0]
    # func_max_corr = sig_corr[max_corr_args]
    realtime = []
    sig_corr = savgol_filter(sig_corr, 101, 2)
    time = wave.ts[int(len(wave.ys) / 2):]
    # for s in max_corr_args:
    #     realtime.append((s, time[s]))

    realtime = {}

    arg_func_max_corr = argrelextrema(sig_corr, np.greater, order=order)[0]
    for s in arg_func_max_corr:
        realtime[s] = time[s]

    for loc in arg_func_max_corr:
        max = sig_corr[loc]
        argm = loc
        try:
            for arg_near_false_max in range(loc - near_max, loc + near_max):
                if sig_corr[arg_near_false_max] > max:
                    max = sig_corr[arg_near_false_max]
                    argm = arg_near_false_max
            arg_func_max_corr[arg_func_max_corr.index(loc)] = argm
        except:
            continue
    # print(realtime)
    realtime = [realtime[h] for h in realtime]
    # print(realtime)
    sorted_func_max_corr = sorted(list(zip(sig_corr[arg_func_max_corr], realtime)), key=lambda x: x[1],
                                  reverse=False)
    find_corr_freq = []
    for i in range(0, len(sorted_func_max_corr) - 1):
        find_corr_freq.append(1 / abs(sorted_func_max_corr[i][1] - sorted_func_max_corr[i + 1][1]))
    find_corr_freq = np.array(find_corr_freq)
    mean = np.mean(find_corr_freq)
    mid = np.median(find_corr_freq)
    std = np.std(find_corr_freq)
    # find_corr_freq /= len(sorted_func_max_corr)

    # print(mid, mean, std)
    # plt.plot(time, sig_corr)
    # plt.plot(time[arg_func_max_corr], sig_corr[arg_func_max_corr])
    # plt.plot(time[arg_func_max_corr], sig_corr[arg_func_max_corr], 'o')
    # plt.plot(sorted_func_max_corr[0][1], sorted_func_max_corr[0][0], 'o')
    # plt.plot(sorted_func_max_corr[1][1], sorted_func_max_corr[1][0], 'o')
    find_corr_freq = mean
    return mean, std


if __name__ == "__main__":
    wave_good = f5s.read_wave(r'D:\Ботать\Работа\dg.a.core\data\wav\zatoshka\24_1m_1.wav')
    wave = wave_good.downscalefreq(1)
    spec = wave.make_spectrum()
    spec.low_pass(200)
    # wave = spec.make_wave()
    wave = wave.downscalefreq(13)
    # wave = wave.segment(2, 10)
    print(support_auto_corr_freq(wave))
    # wavlet = ffq.find_great_corr(5)
    # fig, ax1, ax2 = wavlet.wavlet_plot()
    # na1 = f5a.Frequency_Analysis(wavlet, wave)
    # na1.isotone_n()
    # # # # fig, ax1, ax2 = wavelet.wavlet_plot()
    # # # # # # na1.corr_an(wavelet.correlation)
    # # # # # # plt.figure()
    # line_plots = []
    # labels = []
    # for tone in list(na1.harmony.keys())[:5]:
    #     xcoord = na1.harmony[tone]['time']
    #     ycoord = na1.harmony[tone]['filtered']
    #     yold = na1.harmony[tone]['freq']
    #     plot_line, = ax2.plot(xcoord, ycoord, label=tone + 1)
    #     line_plots.append(plot_line)
    #     labels.append(tone + 1)
    # plt.legend(handles=line_plots)
    plt.show()

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
            # print(tff.shape)
        # tfr = MargenauHillDistribution(wave.ys)
        # tf, ts, freq = tfr.run()
        nperseg = int(round(20 * wave.framerate/1000))
        noverlap = int(round(10 * wave.framerate/1000))
        freq, times, spec = signal.spectrogram(self.sig,
                                                fs=wave.framerate,
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False)
        tf = np.log(spec.astype(np.float32) + 10e-10)
        n_fbins = len(freq)
        ts=times
        # tf = tf[:int(n_fbins / 2.0), :]
        # freq = freq[:int(n_fbins / 2.0)]
        self.x_axis_time = ts
        self.y_axis_freq = freq
        self.c_wavlet_coef = tf**2
        self.arsf = np.array(list_of_arsf)
        self.arsf_std = np.array(list_of_arsf_std)
        self.fftsf = np.array(list_of_fftsf)
        self.fftsf_std = np.array(list_of_fftsf_std)
        self.tp = tp

    def wavlet_plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
        fig.subplots_adjust(hspace=0.3)
        ax1.plot(self.t, self.sig)
        ax1.set_xlim(self.t[0],self.t[-1])
        ax1.set_title('Исходный')
        xred = 1
        yred = 1
        xplot = razryad(self.x_axis_time, xred)
        yplot = razryad(self.y_axis_freq, yred)
        tfplot = razryad_2d(self.c_wavlet_coef, yred, xred)
        ts, freq = np.meshgrid(xplot, yplot)
        print('PLOT, DOWNSCALE ', xred, yred)
        ax2.set_title(self.title)
        clim = 50.
        clim = np.max(tfplot)
        nlevels = 50
        levels = np.linspace(0, clim)
        ax2.pcolormesh(ts, freq, tfplot)
        # divider = make_axes_locatable(ax2)
        # cax = divider.append_axes("top", size="1%", pad="1%")
        # fig.colorbar(im, cax=cax, orientation="horizontal", extend='max')
        # cax.xaxis.set_ticks_position("top")
        # ax2.set_ylim([0, 300])
        return fig, ax1, ax2
        pass

    def wavlet_plot_3d(self):
        Y, X = np.meshgrid(self.y_axis_freq, self.x_axis_time)
        Z = np.transpose(self.c_wavlet_coef)
        fig = plt.figure(figsize=(13, 7))
        ax = fig.gca(projection='3d')
        plt.title('Wavlet')
        ax.set_xlabel('Time,[sec]', fontsize=10)
        ax.set_ylabel('Freq,[Hz]', fontsize=10)
        surf = ax.plot_surface(X, Y, Z, linewidth=0, cmap='viridis', antialiased=False)
        fig.colorbar(surf, ax=ax)
        return fig
        pass

    def correlation(self, c_koef):
        corr = signal.convolve(self.c_wavlet_coef, c_koef, mode='same')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
        fig.subplots_adjust(hspace=0.3)
        ax1.set_title('Исходный')
        xred = 1
        yred = 1
        xplot = razryad(self.x_axis_time, xred)
        yplot = razryad(self.y_axis_freq, yred)
        tfplot = razryad_2d(corr, yred, xred)
        ts, freq = np.meshgrid(xplot, yplot)
        print('Рисую, прореживание ', xred, yred)
        ax2.set_title('Corr')
        clim = 50.
        climmax = np.max(tfplot)
        climmin = np.min(tfplot)
        nlevels = 100
        levels = np.linspace(climmin, climmax, nlevels)
        im = ax2.contour(ts, freq, tfplot, levels)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("top", size="1%", pad="1%")
        fig.colorbar(im, cax=cax, orientation="horizontal", extend='max')
        cax.xaxis.set_ticks_position("top")
        ax2.set_ylim([0, 40])
        self.corr = corr
        return fig, ax1, ax2

    def autocorr(self):
        return self.correlation(self.c_wavlet_coef)

    def alignment_afc(self, freq_afc=[10, 14.25, 18, 22.5, 28.5, 35.5, 45, 56, 60],
                      amp_afc=[50, 53, 59, 66, 68, 72, 75, 84, 86]
                      ):
        f = interpolate.interp1d(freq_afc, amp_afc, kind='cubic')
        xnew = np.arange(min(freq_afc), max(freq_afc), 0.001)
        ynew = f(xnew)

        f_freq = interpolate.interp1d(xnew, (max(ynew)) / ynew, kind='cubic')

        freq_dict = {}
        for freq in range(self.y_axis_freq.shape[0]):
            if self.y_axis_freq[freq] < max(freq_afc) and self.y_axis_freq[freq] > min(freq_afc):
                freq_dict[self.y_axis_freq[freq]] = freq

        for index, freq in zip(freq_dict.values(), freq_dict.keys()):
            self.c_wavlet_coef[index] *= f_freq(freq)


class Frequency_Analysis():
    """
    Make some magic CWT analys
    """

    def __init__(self, wavelet, wave):

        self.x_axis_time = wavelet.x_axis_time
        self.y_axis_freq = wavelet.y_axis_freq
        self.c_wavlet_coef = wavelet.c_wavlet_coef
        self.arsf = wavelet.arsf
        self.arsf_std = wavelet.arsf_std
        self.fftsf = wavelet.fftsf
        self.fftsf_std = wavelet.fftsf_std
        self.sig = wave.ys
        self.N = np.shape(self.sig)[0]
        self.t = wave.ts
        self.dt = wave.duration / self.N
        self.lenkadr = self.N // 100
        self.comp_tones = None

    def change_c_wavlet_coef(self, c):
        self.c_wavlet_coef = c

    def corr_an(self, corealated_from_wavlet):
        fig, ax1, ax2 = corealated_from_wavlet()
        order = 6
        near_max = int(order / 2)
        sig_corr = np.correlate(self.sig, self.sig, mode='same')
        max_corr_args = argrelextrema(sig_corr, np.greater, order=order)[0]

        func_max_corr = sig_corr[max_corr_args]
        realtime = []
        for s in max_corr_args:
            realtime.append((s, self.t[s]))
        realtime = np.array(realtime)
        arg_func_max_corr = argrelextrema(func_max_corr, np.greater, order=order)[0]
        for loc in arg_func_max_corr:
            max = func_max_corr[loc]
            argm = loc
            try:
                for arg_near_false_max in range(loc - near_max, loc + near_max):
                    if func_max_corr[arg_near_false_max] > max:
                        max = func_max_corr[arg_near_false_max]
                        argm = arg_near_false_max
                arg_func_max_corr[arg_func_max_corr.index(loc)] = argm
            except:
                continue
        realtime = [h[1] for h in realtime[arg_func_max_corr]]
        sorted_func_max_corr = sorted(list(zip(func_max_corr[arg_func_max_corr], realtime)), key=lambda x: x[0],
                                      reverse=True)
        find_corr_freq = 1 / abs(sorted_func_max_corr[0][1] - sorted_func_max_corr[1][1])
        self.main_freq_from_corr = find_corr_freq

        ax1.plot(self.t, sig_corr)
        ax1.plot(self.t[max_corr_args], func_max_corr)
        ax1.plot(realtime, func_max_corr[arg_func_max_corr], 'o')
        ax1.plot(sorted_func_max_corr[0][1], sorted_func_max_corr[0][0], 'o')
        ax1.plot(sorted_func_max_corr[1][1], sorted_func_max_corr[1][0], 'o')
        return fig, ax1, ax2

    def fft_an(self, plot):

        yf = fft(self.sig)
        xf = np.linspace(0.0, 1.0 / (2.0 * self.dt), self.N // 2)
        freqline = xf
        ampline = 2.0 / self.N * np.abs(yf[0:self.N // 2])
        if plot:
            plt.figure(figsize=(14, 6))
            plt.plot(freqline[10:], ampline[10:])
            plt.xlabel('Freq,[Hz]')
            plt.grid()
            # plt.savefig(os.getcwd() + config.SOURSE_DIR + config.SOURSE_DIR_PIC + os.path.splitext(filename)[
            #     0] + 'fft' + '.png')
            # plt.show()
        else:
            return freqline, ampline
        pass

    def filter(self, liney, k):
        window = k - 1
        window = signal.general_gaussian(window, p=0.5, sig=20)
        filtered = signal.fftconvolve(window, liney[k:])
        filtered = (np.average(liney[k:]) / np.average(filtered)) * filtered
        roll = -(k - 1) // 2
        filteredfull = [*liney[:k], *np.roll(filtered, roll)][:-k]

        return np.array(filteredfull)
        pass

    def easy_gaus_filter(self, liney):
        g = window = 5  # len(liney)//200 if len(liney)//200%2==1 else len(liney)//200-1
        # print(window)
        window = signal.general_gaussian(window, p=0.5, sig=20)
        filtered = signal.fftconvolve(window, liney)
        roll = -g // 2 + 1
        filtered = (np.average(liney) / np.average(filtered)) * filtered
        filtered = np.roll(filtered, roll)
        return np.array(filtered[:-g])

    def saveoy_filter(self, liney, window_size=11, order=3):
        return signal.savgol_filter(liney, window_size, order)

    def kalman_filter(self, liney):
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=np.mean(liney[0]),
                          initial_state_covariance=[1],
                          observation_covariance=[1],
                          transition_covariance=0.0001)
        state_means, state_covariances = kf.filter(liney)
        state_means = [w[0] for w in state_means]
        return np.array(state_means)

    def window_size(self, y):
        g = [y[i] - y[i + 1] for i in range(0, len(y) - 1)]
        for i in range(0, len(g) - 1):
            if g[i] > 1.03 * g[i + 1]:
                k = i
        return k
        pass

    def autocorr(self, x):
        result = np.correlate(x, x, mode='same')
        return result[len(result) // 2:]

    def mainharmony_n(self, n=-1):
        print('FIND MAIN')
        colvomin = np.zeros(np.array(self.y_axis_freq).shape[0])
        self.harmony = {}
        self.arsf_std_first_harm = []
        self.arsf_first_harm = []
        self.fftsf_std_first_harm = []
        self.fftsf_first_harm = []
        order = 2
        near_max = int(order / 2)
        for i in range(self.c_wavlet_coef.shape[1]):
            column = list(self.c_wavlet_coef[:, i])
            # column=[ 2*(180-self.y_axis_freq[column.index(i)])/180*i if (self.y_axis_freq[column.index(i)]<200 and self.y_axis_freq[column.index(i)]>20) else i for i in column]
            column = np.array(column)
            filt = self.saveoy_filter(column)
            locminarg = argrelextrema(filt, np.greater, order=order)[0]
            # if len(locminarg) > 0:
            #     locminarg = list(filter(
            #         lambda x: (self.y_axis_freq[x] < 100) and (self.y_axis_freq[x] > 10) and column[x] > 0.1 * np.max(
            #             column[locminarg]), locminarg))

            # print(locminarg)
            for loc in locminarg:
                max = column[loc]
                argm = loc
                try:
                    for arg_near_false_max in range(loc - near_max, loc + near_max):
                        if column[arg_near_false_max] > max:
                            max = column[arg_near_false_max]
                            argm = arg_near_false_max
                    locminarg[locminarg.index(loc)] = argm
                except:
                    continue
            # print(locminarg)
            first3 = sorted(list(zip(column[locminarg], locminarg)), key=lambda x: x[0], reverse=True)

            # first3 = list(filter(lambda x: (first3[0][0] * 0.1 < x[0]), first3))
            # print(first3)
            # plt.figure()
            # plt.plot(self.y_axis_freq,column)
            # plt.plot(self.y_axis_freq,filt)
            # plt.plot(self.y_axis_freq[locminarg], column[locminarg], 'o')
            # plt.show()
            if len(first3) < n:
                n = len(first3)
            first3 = first3[:n]

            first3ind = [d[1] for d in first3]
            for ind in range(len(first3ind)):
                if ind in self.harmony:
                    self.harmony[ind]['freq'].append(self.y_axis_freq[first3ind[ind]])
                    self.harmony[ind]['cwav'].append(self.c_wavlet_coef[first3ind[ind]][i])
                    self.harmony[ind]['time'].append(self.x_axis_time[i])
                else:
                    self.harmony.update({ind: {'freq': [self.y_axis_freq[first3ind[ind]]],
                                               'cwav': [self.c_wavlet_coef[first3ind[ind]][i]],
                                               'time': [self.x_axis_time[i]]}})
                if ind == 0:
                    self.arsf_std_first_harm.append(self.arsf_std[i])
                    self.arsf_first_harm.append(self.arsf[i])
                    self.fftsf_std_first_harm.append(self.fftsf_std[i])
                    self.fftsf_first_harm.append(self.fftsf[i])
            colvomin[first3ind] += 1
        colvomin_filt = self.saveoy_filter(colvomin)
        moninglobal_ind = argrelextrema(colvomin_filt, np.greater, order=order)[0]
        moninglobal_ind = list(moninglobal_ind)
        for loc in moninglobal_ind:
            max = colvomin[loc]
            argm = loc
            try:
                for arg_near_false_max in range(loc - near_max, loc + near_max):
                    if colvomin[arg_near_false_max] > max:
                        max = colvomin[arg_near_false_max]
                        argm = arg_near_false_max
                moninglobal_ind[moninglobal_ind.index(loc)] = argm
            except:
                continue
        moninglobal_val = sorted(list(zip(moninglobal_ind, colvomin[moninglobal_ind])), key=lambda x: x[1],
                                 reverse=True)
        coords = [list(t) for t in zip(*moninglobal_val)]
        # plt.figure()
        #
        # plt.plot(self.y_axis_freq, colvomin)
        # plt.plot(self.y_axis_freq, colvomin_filt)
        # plt.plot(self.y_axis_freq[moninglobal_ind], colvomin[moninglobal_ind], 'o')
        # plt.show()
        if len(self.harmony) > len(moninglobal_val):
            newharm = defaultdict(dict)
            for ind in range(len(moninglobal_ind)):
                newharm[ind] = self.harmony[ind]
            self.harmony = newharm

        return self.harmony, self.y_axis_freq[coords[0]]
        pass

    def isotone_n(self, n=-1):

        harmony, line = self.mainharmony_n(n)
        self.arsf_first_harm = np.array(self.arsf_first_harm)
        self.arsf_std_first_harm = np.array(self.arsf_std_first_harm)
        print('ISOTONE')

        alltones = []
        order = 5
        near_max = order // 2
        # self.harmony[0].update({'filtered': self.saveoy_filter(self.kalman_filter(self.harmony[0]['freq']),window_size=101,order=1)})
        for top in list(self.harmony.keys()):
            # print('-----------------------------------------------')
            lineytop = np.array(harmony[top]['freq'])
            filtered = lineytop  # self.easy_gaus_filter(lineytop)
            # if len(lineytop)>200:
            #     filtered = self.saveoy_filter(lineytop,window_size=101,order=1)
            meadtop = np.mean(lineytop)
            isotone = []
            # mindistanceline = sorted(list(zip(line, abs(line - meadtop))), key=lambda x: x[1])
            lenkadr = 100  # len(filtered)//50e
            # plt.figure()
            # plt.scatter(harmony[top]['time'],filtered,marker='.')
            # plt.show()
            filt_std = []
            for kard in range(0, len(filtered), lenkadr):
                filt_std = filt_std + [np.std(filtered[kard:kard + lenkadr])] * lenkadr
                mindistanceline = sorted(list(zip(line, abs(line - filtered[kard]))), key=lambda x: x[1])
                # print(mindistanceline,filtered[kard])
                dictr = defaultdict(int)
                for f in filtered[kard:kard + lenkadr]:
                    dictr[f] += 1
                dictr = sorted(dictr.items(), key=lambda x: x[1], reverse=True)
                if top > 0 and abs(dictr[0][0] - self.harmony[top - 1]['filtered'][kard:kard + lenkadr] < 2).all():
                    try:
                        filtered[kard:kard + lenkadr] = dictr[1][0]
                    except:
                        filtered[kard:kard + lenkadr] = dictr[0][0]
                else:
                    filtered[kard:kard + lenkadr] = dictr[0][0]
                # if (filtered] < 1.1 * mindistanceline[0][0]).any() and (
                #         filtered[kard:kard + lenkadr] > 0.9 * mindistanceline[0][0]).any():
                #     filtered[kard] = line[top]
                # print(sorted(list(zip(line, abs(line - np.mean(filtered[:kard])))), key=lambda x: x[1]),
                #       filtered[kard])  # mindistanceline[0][0]

            # filtered=savgol_filter(filtered, self.N//100 if self.N%2==1 else self.N//100-1, 1)

            # filtred_from_kalman=self.kalman_filter(lineytop)
            # isotone = ir.fit_transform(self.x_axis_time[:len(filtred_from_kalman)], filtred_from_kalman)

            if len(filtered) < len(lineytop):
                dopzero = [0] * (len(lineytop) - len(filtered))
                filtered = np.concatenate((filtered, dopzero), axis=None)
            self.harmony[top]['filtered'] = filtered.copy()
            filtered_with_corr = filtered.copy()
            filtered_with_fft = filtered.copy()
            # print(self.arsf_first_harm.shape, self.x_axis_time.shape, filtered.shape, )
            for indf, freq in enumerate(filtered_with_corr):
                if filt_std[indf] > self.arsf_std_first_harm[indf]:
                    # print(filtered[indf], self.arsf_first_harm[indf], self.harmony[top]['time'][indf], top)
                    filtered_with_corr[indf] = self.arsf_first_harm[indf]
            self.harmony[top]['filtered_with_sup_corr'] = filtered_with_corr
            for indf, freq in enumerate(filtered_with_fft):
                if filt_std[indf] > self.fftsf_std_first_harm[indf]:
                    # print(filtered[indf], self.arsf_first_harm[indf], self.harmony[top]['time'][indf], top)
                    filtered_with_fft[indf] = self.fftsf_first_harm[indf]
            self.harmony[top]['filtered_with_sup_fft'] = filtered_with_fft
            alltones.append(list(zip(self.x_axis_time[:len(filtered)], filtered_with_corr)))
            mid = []
            for indf, freq in enumerate(filtered_with_fft):
                mid.append(np.median([filtered_with_corr[indf], filtered[indf], filtered_with_fft[indf]]))
            len_kard = lenkadr * 3
            len_kard -= 1
            try:
                self.harmony[top]['filtered_mid_from'] = savgol_filter(np.array(mid).astype(float), len_kard,
                                                                       1).tolist()
            except:
                self.harmony[top]['filtered_mid_from'] = mid
        mass = []
        for el in line:
            mass.append([el for i in range(len(self.t))])
        # print(list(self.harmony[0]['filtered_with_sup_corr']))
        # print(list(self.harmony[0]['filtered']))
        self.dropx2 = self.harmony[0]['filtered'] / 2
        return self.harmony, mass

    def swap_to_probably(self, to_what):

        self.normalize_freq()
        print('SWAP')
        if np.mean(self.harmony[0]['normal']) > to_what:
            for harm in list(self.harmony.keys())[1:]:
                if np.mean(self.harmony[harm]['normal']) < to_what:
                    self.harmony[harm]['normal'], self.harmony[0]['normal'] = self.harmony[0]['normal'], \
                                                                              self.harmony[harm]['normal']
                    break
        pass

    def find_max_koef(self, comp_def):

        harmony = self.harmony.copy()
        print('FIND MAX_KOEF')
        max_harm = 0
        bestdestroy = harmony[0]

        for harm in list(self.harmony.keys()):
            self.harmony[harm]['normal'], self.harmony[0]['normal'] = self.harmony[0]['normal'], \
                                                                      self.harmony[harm]['normal']
            self.find_tones_from_main_for_model(comp_def)
            _, val_koef = self.find_balance_v2()
            koef = 0
            for koef_pertime in val_koef:
                koef += koef_pertime[1]
            if max_harm < koef:
                bestdestroy = harmony[harm]
                max_harm = koef

            print(np.mean(bestdestroy['normal']), np.mean(harmony[harm]['normal']), koef)
            self.harmony = harmony
        print('TO CHTO NADO', np.mean(bestdestroy['normal']), max_harm)

    def normalize_freq(self, n=-1):
        self.isotone_n()
        print('NORMALIZE')
        # self.swap_to_probably(30)
        normalize_freq = []
        for tone in self.harmony:
            newtone = self.harmony[tone]['filtered']
            normalize_freq.append(newtone)
            self.harmony[tone].update({'normal': newtone})
        return normalize_freq
        pass

    def find_nagruz(self):
        if self.comp_tones:
            comp_tones = self.comp_tones
        else:
            comp_tones = self.find_tones_from_main()
        nagruz = {'time': [], 'koef': []}
        padenie = 0
        for one_time in comp_tones:
            if len(comp_tones[one_time]) > 2:
                if one_time > 0 and self.harmony[0]['normal'][one_time - 1] < self.harmony[0]['normal'][one_time]:
                    padenie += 1
                    if padenie * self.dt > 0.2:
                        if nagruz['time']:
                            if nagruz['time'][-1][1] == self.harmony[0]['time'][one_time - 1]:
                                nagruz['time'][-1][1] = self.harmony[0]['time'][one_time]

                            else:
                                nagruz['time'].append(
                                    [self.harmony[0]['time'][one_time], self.harmony[0]['time'][one_time - 1]])
                        else:
                            nagruz['time'].append(
                                [self.harmony[0]['time'][one_time], self.harmony[0]['time'][one_time]])
                else:
                    padenie = 0
        print(nagruz)
        return nagruz
        pass

    def find_touch(self):
        if self.comp_tones:
            comp_tones = self.comp_tones
        else:
            comp_tones = self.find_tones_from_main()
        koeflist = []
        touch = {'time': [], 'koef': []}
        for one_time in comp_tones:
            count = 0
            if len(comp_tones[one_time]) > 2:
                koef = 0
                for dop_tone in comp_tones[one_time]:
                    koef = self.harmony[dop_tone]['cwav'][one_time]
                koef = koef / self.harmony[0]['cwav'][one_time]
                koeflist.append(koef)
                if koef > 0.4:
                    count = +1
            if count > 3:
                if touch['time']:
                    if touch['time'][-1][1] == self.harmony[0]['time'][one_time - 1]:
                        touch['time'][-1][1] = self.harmony[0]['time'][one_time]

                    else:
                        touch['time'].append([self.harmony[0]['time'][one_time], self.harmony[0]['time'][one_time - 1]])
                else:
                    touch['time'].append([self.harmony[0]['time'][one_time], self.harmony[0]['time'][one_time]])

        print(touch)
        return touch
        pass

    def find_banalce(self):
        lenkadr = self.lenkadr
        normfreq = self.normalize_freq()
        balance = {'time': [], 'koef': []}
        koeflist = []
        for kard in range(0, self.N - 50 * lenkadr, lenkadr):
            if len(normfreq) >= 3:
                main = self.harmony[0]['normal'][kard:kard + lenkadr]
                if len(main) != lenkadr:
                    main = np.append(main, [0] * (lenkadr - len(main)))
                sec = self.harmony[1]['normal'][kard:kard + lenkadr]
                if len(sec) != lenkadr:
                    sec = np.append(sec, [0] * (lenkadr - len(sec)))
                th = self.harmony[2]['normal'][kard:kard + lenkadr]
                if len(th) != lenkadr:
                    th = np.append(th, [0] * (lenkadr - len(th)))
                if (sec > (2 * main * 0.95)).any() and (sec < (2 * main * 1.1)).any() or (
                        (th > (2 * main * 0.95)).any() and (th < (2 * main * 1.1)).any()):
                    print('Banance at ', kard)
                    koef = (np.mean(self.harmony[1]['cwav'][kard:kard + lenkadr] + self.harmony[2]['cwav'][
                                                                                   kard:kard + lenkadr]) / np.mean(
                        self.harmony[0]['cwav'][kard:kard + lenkadr]))
                    print('Koefi=', koef)
                    koeflist.append(koef)
                    if balance['time']:
                        if balance['time'][-1][1] == self.harmony[0]['time'][kard]:
                            print(np.shape(self.harmony[0]['time']), kard + lenkadr)
                            balance['time'][-1][1] = self.harmony[0]['time'][kard + lenkadr]

                        else:
                            balance['time'].append(
                                [self.harmony[0]['time'][kard], self.harmony[0]['time'][kard + lenkadr]])
                    else:
                        balance['time'].append([self.harmony[0]['time'][kard], self.harmony[0]['time'][kard + lenkadr]])
                    balance['koef'].append(koef)
            if len(normfreq) == 2:
                main = self.harmony[0]['normal'][kard:kard + lenkadr]
                if len(main) != lenkadr:
                    main = np.append(main, [0] * (lenkadr - len(main)))
                sec = self.harmony[1]['normal'][kard:kard + lenkadr]
                if len(sec) != lenkadr:
                    sec = np.append(sec, [0] * (lenkadr - len(sec)))
                if (sec > main).all():
                    # print(1)
                    if (sec > (main * 1.9)).any() and (sec < (main * 2.1)).any():
                        print('Banance off at ', kard * self.dt)
                        koef = (np.mean(self.harmony[1]['cwav'][kard:kard + lenkadr]) / np.mean(
                            self.harmony[0]['cwav'][kard:kard + lenkadr]))
                        print('Koefi=', koef)
                        koeflist.append(koef)

                else:
                    # print(2)
                    if (main > (sec * 1.9)).any() and (main < (sec * 2.1)).any():
                        print('Banance off at ', kard * self.dt)
                        koef = (np.mean(self.harmony[1]['cwav'][kard:kard + lenkadr]) / np.mean(
                            self.harmony[0]['cwav'][kard:kard + lenkadr]))
                        print('Koef=', koef)
                        koeflist.append(koef)
        return balance
        pass

    def find_balance_v2(self):
        comp_tones = self.comp_tones
        # print(comp_tones)
        koeflist = []
        balance = {'time': [], 'koef': []}
        for one_time in comp_tones:
            if len(comp_tones[one_time]) > 1:
                koef = 0
                # print(np.mean(comp_tones[one_time]))
                for dop_tone in comp_tones[one_time]:
                    koef = koef + self.harmony[dop_tone]['cwav'][one_time]
                koef = (koef - self.harmony[0]['cwav'][one_time]) / self.harmony[0]['cwav'][one_time]
                koeflist.append((one_time, koef))
                if koef > 0.15:
                    if balance['time']:
                        if balance['time'][-1][1] == self.harmony[0]['time'][one_time - 1]:
                            balance['time'][-1][1] = self.harmony[0]['time'][one_time]
                        else:
                            balance['time'].append(
                                [self.harmony[0]['time'][one_time], self.harmony[0]['time'][one_time]])
                    else:
                        balance['time'].append([self.harmony[0]['time'][one_time], self.harmony[0]['time'][one_time]])
                    balance['koef'].append(koef)
        print(balance)
        print(koeflist)
        return balance, koeflist

    # def find_tones_from_main(self, touchharm=[0.5, 1.5, 2, 2.5, 3, 3.5, 4]):
    #     self.normalize_freq()
    #     touchharm = np.array(touchharm)
    #     compltone = defaultdict(dict)
    #     if len(self.harmony) > 1:
    #         for ind in list(self.harmony.keys())[1:]:
    #             for t in self.harmony[0]['time']:
    #                 if t in self.harmony[ind]['time']:
    #                     indexfreq = self.harmony[ind]['time'].index(t)
    #                     if (self.harmony[0]['normal'][indexfreq] * touchharm * 1.05 < self.harmony[ind]['normal'][
    #                         indexfreq]).any() and (
    #                             self.harmony[0]['normal'][indexfreq] * touchharm * 0.95 > self.harmony[ind]['normal'][
    #                         indexfreq]).any():
    #                         if not compltone[indexfreq]:
    #                             compltone[indexfreq] = []
    #                         compltone[indexfreq].append(ind)
    #     self.comp_tones = compltone
    #     return compltone

    def find_tones_from_main_for_model(self, func):
        try:
            f = self.harmony[0]['normal'][0]
        except:
            print('alarm')
            self.normalize_freq()
        compltone = defaultdict(dict)
        if len(self.harmony) > 1:
            for ind in list(self.harmony.keys())[1:]:
                flag = 0
                for t in self.harmony[0]['time']:
                    if t in self.harmony[ind]['time']:
                        indexfreq = self.harmony[ind]['time'].index(t)
                        freq_values = np.array(list(func(self.harmony[0]['normal'][indexfreq]).values()))
                        if (freq_values > 0).all():
                            for value in freq_values:
                                if ((value - 2) < self.harmony[ind]['normal'][indexfreq]) and (
                                        (value + 2) > self.harmony[ind]['normal'][indexfreq]):
                                    flag = 1
                                    break
                            if flag:
                                # print(freq_values,self.harmony[ind]['normal'][indexfreq])
                                if not compltone[indexfreq]:
                                    compltone[indexfreq] = []
                                compltone[indexfreq].append(ind)
                        # print(self.harmony[ind]['time'][indexfreq],self.harmony[ind]['normal'][indexfreq],self.harmony[0]['normal'][indexfreq])
        self.comp_tones = compltone
        return compltone

    def heuristic_analis(self, freq_dict):
        print(freq_dict)
import matplotlib.pyplot as plt
import numpy as np
import F5analys as f5a
import F5signal as f5s
import pathlib
import pandas as pd
from config import TEST_PIC_DIR, TEST_WAV_DIR

plt.rcParams['figure.figsize'] = 22, 4

# wave_good = signal.make_wave(10, framerate = 44000)
path_str = TEST_WAV_DIR + r'normal_id_00_00000000.wav'
path = pathlib.Path(path_str)
wave_good = f5s.read_wave(path_str)
wave = wave_good
# wave=wave_good
spec = wave.make_spectrum()
# spec.low_pass(200)
wave = spec.make_wave()
# wave = wave.segment(0, 1)
# ffq = Find_quality(wave, [10, 70], 1)
# wavlet = ffq.up_sup_freq()
wavlet = f5a.Wavlet(wave)
# wavlet.alignment_afc()
na1 = f5a.Frequency_Analysis(wavlet, wave)
na1.isotone_n()
fig, ax1, ax2 = wavlet.wavlet_plot()
# na1.corr_an(wavelet.correlation)
# plt.figure()
line_plots = []
labels = []
miss = {}
proc = {}
for tone in list(na1.harmony.keys())[:1]:
    xcoord = np.array(na1.harmony[tone]['time'])
    ycoord = na1.harmony[tone]['filtered']
    ycoord_cor = na1.harmony[tone]['filtered_with_sup_corr']
    ycoord_fft = na1.harmony[tone]['filtered_with_sup_fft']
    ycoord_mid = na1.harmony[tone]['filtered_mid_from']
    # print([int(wav.split('_')[1])] * len(ycoord))
    # tryval = np.array([int(wav.split('_')[1]) / 2] * len(ycoord))
    # miss[tone] = np.sum(np.abs(tryval - ycoord))
    # proc[tone] = ycoord[abs(ycoord - tryval[0]) < 2].shape[0] / xcoord.shape[0]
    ax2.plot(xcoord, ycoord, label='F' + str(tone + 1) + ' algoritm', color='r')
    ax2.plot(xcoord, ycoord_cor, label='F_cor' + str(tone + 1) + ' algoritm', color='g')
    ax2.plot(xcoord, ycoord_fft, label='F_fft' + str(tone + 1) + ' algoritm', color='b')
    ax2.plot(xcoord, ycoord_mid, label='F_sum' + str(tone + 1) + ' algoritm', color='orange')
    # ax2.plot(xcoord, tryval, label='Ground True')
    # line_plots.append(plot_line)
    # line_plots.append(plot_line1)
    # labels.append(tone + 1)
#
# # plt.title(str(round(proc[0], 2)) + ' ' + str(miss[0]))
ax2.legend()
plt.title(path.stem)
# plt.savefig(TEST_PIC_DIR + path.stem)
print('end')
plt.show()

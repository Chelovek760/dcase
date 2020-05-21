import librosa
from saparator import Buono_Brutto_Cattivo
from config import TEST_WAV_DIR
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def shuffle_generator(signal, len_out):
    bbc = Buono_Brutto_Cattivo(segment_number=99)
    _, gg, _, _ = bbc.separate(signal)
    segment_ind = list(gg.keys())
    big_wave = gg[segment_ind[0]]

    for one_wavlet in segment_ind[1:]:
        big_wave = np.hstack((big_wave, gg[one_wavlet]))
    size_wave = big_wave.shape
    out_ys = np.zeros((len_out))
    window_size = 2
    for ind in range(0, len_out, window_size):
        rind = np.random.randint(0, size_wave[0] - window_size)
        out_ys[ind:ind + window_size] = big_wave[rind:rind + window_size]

    return out_ys


if __name__ == '__main__':
    directory = TEST_WAV_DIR
    files = pathlib.Path(directory)
    files = files.glob('*.wav')
    files = list(files)[:1]
    signal, fr = librosa.load(files[0], sr=None, mono=True)
    signal = np.array([0] * 10000 + [1] * 1000)
    signal_s = shuffle_generator(signal, fr * 10)
    plt.figure()
    plt.plot(signal_s)
    plt.show()

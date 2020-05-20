import librosa
from saparator import Buono_Brutto_Cattivo
from config import TEST_WAV_DIR
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def shuffle_generator(signal, len_out):
    bbc = Buono_Brutto_Cattivo(segment_number=99)
    _, gg, _, _ = bbc.separate(signal)
    big_wave = gg[0]

    for one_wavlet in list(gg.keys())[1:]:
        big_wave = np.hstack((big_wave, gg[one_wavlet]))
    size_wave = big_wave.shape
    out_ys = np.zeros((len_out))
    for ind in range(0, len_out, 2):
        out_ys[ind] = big_wave[np.random.randint(0, size_wave[0])]
        out_ys[ind + 1] = big_wave[np.random.randint(0, size_wave[0])]
    return out_ys


if __name__ == '__main__':
    directory = TEST_WAV_DIR
    files = pathlib.Path(directory)
    files = files.glob('*.wav')
    files = list(files)[:1]
    signal, fr = librosa.load(files[0], sr=None, mono=True)
    signal_s = shuffle_generator(signal, fr * 10)
    plt.figure()
    plt.plot(signal_s)
    plt.show()

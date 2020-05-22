"""
 @file   common.py
 @brief  Commonly used script
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os
import matplotlib.pyplot as plt
# additional
import numpy
import pandas
import statistics
import librosa
import librosa.core
import librosa.feature
import librosa.display
import yaml

from F5signal import razryad_2d
from saparator import Buono_Brutto_Cattivo
from shuffle_augmentator import shuffle_generator

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.0"


########################################################################


########################################################################
# argparse
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2020 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.eval ^ args.dev:
        if args.dev:
            flag = True
        else:
            flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag


########################################################################


########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param


########################################################################


########################################################################
# file I/O
########################################################################
# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# feature extractor
########################################################################
def spectrogramm_augmentation(y,
                              sr,
                              method,
                              n_mels=64,
                              frames=5,
                              n_fft=1024,
                              hop_length=512,
                              power=2.0):
    if method == 'normal':
        pass
    if method == 'revers':
        y = y[::-1]
    if method == 'white_noise':
        df = pandas.Series(y)
        amp = df.quantile([0.75])
        y = y + numpy.random.uniform(-amp, amp, len(y))
    if method == 'shuffle':
        y = shuffle_generator(y, len(y))

    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    return log_mel_spectrogram


def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2,
                         method='normal'):
    """
    convert file_name to a vector array.
    
    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    # dims = n_mels * frames
    # 02 generate melspectrogram using librosa
    disc = 5
    y, sr = file_load(file_name)
    bbc = Buono_Brutto_Cattivo(segment_number=99)
    _, _, y_list_bad, _ = bbc.separate(y, sr)
    dims = bbc.newshape[0] * bbc.newshape[1] // disc
    segment_ind = list(y_list_bad.keys())[2:]
    if len(segment_ind) < 1:
        print(0)
        spectrogram = numpy.zeros((2, 2))

    else:
        big_wave = y_list_bad[segment_ind[0]]

        for one_wavlet in segment_ind[1:]:
            big_wave = numpy.hstack((big_wave, y_list_bad[one_wavlet]))
        spectrogram = razryad_2d(big_wave, 1, disc)
    # 03 generate spectrogramm with augmentstion or not. Depend on method param
    # spectrogram = spectrogramm_augmentation(y=y,
    #                                         sr=sr,
    #                                         method=method,
    #                                         n_mels=n_mels,
    #                                         frames=frames,
    #                                         n_fft=n_fft,
    #                                         hop_length=hop_length,
    #                                         power=power)

    # 04 add some new features
    # features = add_new_feature(mel_spectrogram, log_mel_spectrogram)
    features = spectrogram
    # 05 calculate total vector size
    vector_array_size = len(features[0, :]) - frames + 1
    # 06 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims)), dims

    # 07 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, bbc.newshape[0] * t: bbc.newshape[0] * (t + 1)] = features[:, t: t + vector_array_size].T

    return vector_array[:, :], dims


# load dataset
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
        dirs = sorted(glob.glob(dir_path))
    else:
        logger.info("load_directory <- evaluation")
        dir_path = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
        dirs = sorted(glob.glob(dir_path))
    return dirs


def add_new_feature(mel_spectrogram, log_mel_spectrogram):
    rms = librosa.feature.rmse(S=mel_spectrogram)
    features = numpy.append(log_mel_spectrogram, rms, axis=0)

    spectral_centroid = librosa.feature.spectral_centroid(S=mel_spectrogram)
    features = numpy.append(features, spectral_centroid, axis=0)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=mel_spectrogram)
    features = numpy.append(features, spectral_bandwidth, axis=0)

    spectral_flatness = librosa.feature.spectral_flatness(S=mel_spectrogram)
    features = numpy.append(features, spectral_flatness, axis=0)

    spectral_contrast = librosa.feature.spectral_contrast(S=mel_spectrogram)
    features = numpy.append(features, spectral_contrast, axis=0)

    spectral_rolloff = librosa.feature.spectral_rolloff(S=mel_spectrogram)
    features = numpy.append(features, spectral_rolloff, axis=0)

    poly_features = librosa.feature.poly_features(S=mel_spectrogram)
    features = numpy.append(features, poly_features, axis=0)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)
    features = numpy.append(features, zero_crossing_rate, axis=0)

    return features


########################################################################


if __name__ == "__main__":
    file_name = r'dev_data\fan\train\normal_id_00_00000000.wav'
    n_mels = 128
    frames = 5
    n_fft = 1024
    hop_length = 512
    power = 2
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)
    df = pandas.Series(y)
    amp = df.quantile([0.75])
    y_n = y + numpy.random.uniform(-amp, amp, len(y))

    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    mel_spectrogram_n = librosa.feature.melspectrogram(y=y_n,
                                                       sr=sr,
                                                       n_fft=n_fft,
                                                       hop_length=hop_length,
                                                       n_mels=n_mels,
                                                       power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram_n = 20.0 / power * numpy.log10(mel_spectrogram_n + sys.float_info.epsilon)

    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.specshow(log_mel_spectrogram, x_axis='time',
                             y_axis='mel', sr=sr,
                             fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(log_mel_spectrogram_n, x_axis='time',
                             y_axis='mel', sr=sr,
                             fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram - noise')
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    librosa.display.specshow(numpy.append(log_mel_spectrogram_n, log_mel_spectrogram, axis=1), x_axis='time',
                             y_axis='mel', sr=sr,
                             fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram - noise')
    plt.tight_layout()

    plt.show()

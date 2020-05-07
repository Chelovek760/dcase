from saparator import Buono_Brutto_Cattivo
import matplotlib.pyplot as plt
from config import TEST_WAV_DIR
import pathlib

directory = TEST_WAV_DIR
files = pathlib.Path(directory)
files = files.glob('*.wav')
files=list(files)

bbc = Buono_Brutto_Cattivo(files[0], 99)
bcc_keys = bbc.separate()[1]
bcc_keys.pop('freq', None)
bcc_keys.pop('dur_part', None)

l = list(bcc_keys.keys())

for part in l:
    print(bcc_keys[part])
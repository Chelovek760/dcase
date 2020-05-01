import F5signal as f5s
import F5analys as f5a
from Dcase_saver import Saver
from config import DCASE_JSON_DIR, TEST_WAV_DIR,DCASE_FITS_CSV_DIR,DCASE_COEF_CSV_DIR
import pathlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
directory = TEST_WAV_DIR
files = pathlib.Path(directory)
files = files.glob('*.wav')


def visual_finc(wav):
    if pathlib.Path(DCASE_JSON_DIR + wav.stem + '.json').exists():
        return print(str(wav.stem) + 'Alredy Exist')
    path_str = TEST_WAV_DIR + wav.stem + '.wav'
    path = pathlib.Path(path_str)
    print(path)
    wave = f5s.read_wave(str(path))
    wavlet = f5a.Wavlet(wave)
    na1 = f5a.Frequency_Analysis(wavlet, wave)
    Saver(path, DCASE_JSON_DIR, na1)


with ThreadPoolExecutor(2) as executor:
    for _ in executor.map(visual_finc, files):
        pass

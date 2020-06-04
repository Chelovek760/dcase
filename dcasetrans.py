import F5signal as f5s
import F5analys as f5a
from Dcase_saver import Saver
from config import TEST_DCASE_JSON_DIR, TRAIN_DCASE_JSON_DIR, TRAIN_WAV_DIR, TEST_WAV_DIR
import pathlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

directory = TRAIN_WAV_DIR
directory_out = TRAIN_DCASE_JSON_DIR
files = pathlib.Path(directory)
files = list(files.glob('*.wav'))

from tqdm import tqdm
def visual_finc(wav):
    if pathlib.Path(directory_out + wav.stem + '.json').exists():
        return print(str(wav.stem) + 'Alredy Exist')
    path_str = directory + wav.stem + '.wav'
    path = pathlib.Path(path_str)
    print(path)
    wave = f5s.read_wave(str(path))
    wave = wave.downscalefreq(8)
    wavlet = f5a.Wavlet(wave)
    na1 = f5a.Frequency_Analysis(wavlet, wave)
    Saver(path, directory_out, na1)


for file in tqdm(files, total=len(files)):
    visual_finc(file)
# with ThreadPoolExecutor(2) as executor:
#     for _ in executor.map(visual_finc, files):
#         pass

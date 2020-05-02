import F5signal as f5s
import F5analys as f5a
from Dcase_saver import Saver
import matplotlib.pyplot as plt
from config import DCASE_JSON_DIR, TEST_WAV_DIR,DCASE_CSV_DIR
import pathlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
directory = TEST_WAV_DIR
files = pathlib.Path(directory)
files = files.glob('*.wav')
files=list(files)

def visual_finc(wav):
    if pathlib.Path(r'D:\Ботать\Работа\dcase\pic\test\\' + wav.stem + '.png').exists():
        return print(str(wav.stem) + 'Alredy Exist')
    path_str = TEST_WAV_DIR + wav.stem + '.wav'
    path = pathlib.Path(path_str)
    print(path)
    wave = f5s.read_wave(str(path))
    wavlet = f5a.Wavlet(wave)
    wavlet.wavlet_plot()
    plt.savefig(r'D:\Ботать\Работа\dcase\pic\test\\'+path.stem+'.png')
    plt.close()


for _, wav in tqdm(enumerate(files),total=len(files)):
    visual_finc(wav)

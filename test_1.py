import pandas as pd
import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('sdfs')
S_full, phase = librosa.magphase(librosa.stft(y))

librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                        y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()
plt.show()
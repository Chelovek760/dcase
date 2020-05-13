import librosa


y, sr = librosa.load(r'dev_data\fan\test\anomaly_id_00_00000000.wav')
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=512)

print(mfccs.shape)
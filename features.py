
import librosa

import numpy as np

import matplotlib.pyplot as plt

SR = 16000

N_MFCC = 13

MAX_LEN = 100

def extract_mfcc(path):

    y, _ = librosa.load(path, sr=SR)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=400,
        hop_length=160
    ).T

    if len(mfcc) < MAX_LEN:

        pad = np.zeros((MAX_LEN - len(mfcc), N_MFCC))

        mfcc = np.vstack([mfcc, pad])
        
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc.astype(np.float32)

# mfcc = extract_mfcc("data/positive/positive_039.wav")

# plt.imshow(mfcc.T, aspect="auto", origin="lower")

# plt.colorbar()

# plt.show()


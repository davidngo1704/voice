import os
import numpy as np
import torch
from features import extract_mfcc

def load_data():
    X, y = [], []

    for label, cls in [("positive", 1), ("negative", 0)]:
        for f in os.listdir(f"data/{label}"):
            path = f"data/{label}/{f}"
            X.append(extract_mfcc(path))
            y.append(cls)

    X = torch.tensor(np.array(X))
    y = torch.tensor(y).float()
    return X, y

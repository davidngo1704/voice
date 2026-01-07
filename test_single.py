import torch
import numpy as np
from model import WakeWordNet
from features import extract_mfcc

ckpt = torch.load("wakeword.pt", map_location="cpu")

model = WakeWordNet()
model.load_state_dict(ckpt["model"])
model.eval()

mean = ckpt["mean"]
std = ckpt["std"]


#mfcc = extract_mfcc("data/positive/positive_001.wav")

mfcc = extract_mfcc("data/negative/negative_000.wav")

mfcc = (torch.from_numpy(mfcc) - mean.squeeze(0)) / std.squeeze(0)

x = mfcc.unsqueeze(0)


with torch.no_grad():
    logits = model(x)
    score = torch.sigmoid(logits).item()

print("score =", score)



import sounddevice as sd
import numpy as np
import torch
import time
import librosa
from collections import deque

from model import WakeWordNet

from voice_service import speak

import asyncio

# ============ CONFIG ============
SR = 16000
WINDOW = int(1.0 * SR)
STEP = int(0.2 * SR)

THRESHOLD = 0.4     # chỉ cần vượt là bắn
COOLDOWN = 1.0      # cho phép trigger lại nhanh

# ============ STATE ============
audio_buffer = deque(maxlen=WINDOW)
last_trigger = 0.0

# ============ LOAD MODEL ============
ckpt = torch.load("wakeword.pt", map_location="cpu", weights_only=True)

model = WakeWordNet()
model.load_state_dict(ckpt["model"])
model.eval()

mean = ckpt["mean"].squeeze(0)
std = ckpt["std"].squeeze(0)

print("Listening... Say: hey jarvis")

# ============ CALLBACK ============
def callback(indata, frames, time_info, status):
    global last_trigger

    audio_buffer.extend(indata[:, 0])

    if len(audio_buffer) < WINDOW:
        return

    audio = np.array(audio_buffer, dtype=np.float32)
    audio /= (np.max(np.abs(audio)) + 1e-6)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=13,
        n_fft=400,
        hop_length=160
    ).T[:100]

    if mfcc.shape[0] < 100:
        mfcc = np.pad(mfcc, ((0, 100 - mfcc.shape[0]), (0, 0)))

    mfcc = torch.from_numpy(mfcc)
    mfcc = (mfcc - mean) / std
    x = mfcc.unsqueeze(0)

    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()

    print(f"score={score:.3f}", end="\r")

    now = time.time()
    if score > THRESHOLD and now - last_trigger > COOLDOWN:
        print("\n🔥 HEY JARVIS 🔥")

        asyncio.run(speak("Tôi xin lắng nghe"))

        last_trigger = now

# ============ STREAM ============
with sd.InputStream(
    channels=1,
    samplerate=SR,
    blocksize=STEP,
    callback=callback
):
    while True:
        time.sleep(0.1)

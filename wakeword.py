# wakeword.py
import sounddevice as sd
import numpy as np
import torch
import time
import librosa
from collections import deque

from model import WakeWordNet
from voice_service import speak
import asyncio

SR = 16000
WINDOW = int(1.0 * SR)
STEP = int(0.2 * SR)

THRESHOLD = 0.4
COOLDOWN = 1.0

def wait_for_wakeword():
    audio_buffer = deque(maxlen=WINDOW)
    last_trigger = 0.0

    ckpt = torch.load("wakeword.pt", map_location="cpu", weights_only=True)

    model = WakeWordNet()
    model.load_state_dict(ckpt["model"])
    model.eval()

    mean = ckpt["mean"].squeeze(0)
    std = ckpt["std"].squeeze(0)

    triggered = False

    def callback(indata, frames, time_info, status):
        nonlocal last_trigger, triggered

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

        now = time.time()
        if score > THRESHOLD and now - last_trigger > COOLDOWN:
            triggered = True
            last_trigger = now

    print("👂 Đang chờ wake word: hey jarvis")

    with sd.InputStream(
        channels=1,
        samplerate=SR,
        blocksize=STEP,
        callback=callback
    ):
        while not triggered:
            time.sleep(0.05)

    asyncio.run(speak("Tôi xin lắng nghe"))

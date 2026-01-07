import sounddevice as sd
import numpy as np
import soundfile as sf
import os
import time

SR = 16000
DURATION = 1.2

def record(out_path):
    print("Recording...")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    sf.write(out_path, audio, SR)
    print("Saved:", out_path)

label = input("positive / negative? ").strip()
os.makedirs(f"data/{label}", exist_ok=True)

for i in range(50 if label == "positive" else 200):
    input(f"Press Enter to record {i+1}")
    record(f"data/{label}/{label}_{i:03d}.wav")
    time.sleep(0.3)


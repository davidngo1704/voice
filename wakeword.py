import sounddevice as sd
import numpy as np
import torch
import time
import librosa
import threading
from collections import deque
import asyncio

from model import WakeWordNet
from voice_service import speak

# ================= CONFIG =================
SR = 16000
WINDOW_SEC = 0.8
WINDOW = int(WINDOW_SEC * SR)
STEP = int(0.2 * SR)

INFER_INTERVAL = 0.6
THRESHOLD = 0.4
COOLDOWN = 1.0
SILENCE_GUARD = 0.5   # 🔥 cực kỳ quan trọng

# ================= TORCH OPT =================
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# ================= LOAD MODEL =================
print("🧠 Loading WakeWord model...")

ckpt = torch.load("wakeword.pt", map_location="cpu", weights_only=True)

model = WakeWordNet()
model.load_state_dict(ckpt["model"])
model.eval()

mean = ckpt["mean"].squeeze(0)
std = ckpt["std"].squeeze(0)

print("✅ WakeWord sẵn sàng.")

# ================= SHARED STATE =================
audio_buffer = deque(maxlen=WINDOW)
buffer_lock = threading.Lock()


# ================= AUDIO CALLBACK =================
def audio_callback(indata, frames, time_info, status):
    with buffer_lock:
        audio_buffer.extend(indata[:, 0])


# ================= WORKER =================
def inference_worker(trigger_flag, stop_flag):
    last_infer = 0.0
    last_trigger = 0.0

    while not stop_flag.is_set():
        now = time.time()
        if now - last_infer < INFER_INTERVAL:
            time.sleep(0.02)
            continue

        with buffer_lock:
            if len(audio_buffer) < WINDOW:
                continue
            audio = np.array(audio_buffer, dtype=np.float32)

        last_infer = now
        audio /= (np.max(np.abs(audio)) + 1e-6)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SR,
            n_mfcc=13,
            n_fft=400,
            hop_length=160
        ).T

        mfcc = mfcc[:80] if mfcc.shape[0] >= 80 else np.pad(
            mfcc, ((0, 80 - mfcc.shape[0]), (0, 0))
        )

        x = torch.from_numpy(mfcc)
        x = (x - mean) / std
        x = x.unsqueeze(0)

        score = torch.sigmoid(model(x)).item()

        if score > THRESHOLD and now - last_trigger > COOLDOWN:
            trigger_flag.set()
            last_trigger = now
            return


# ================= PUBLIC API =================
def wait_for_wakeword():
    # 🔥 RESET TOÀN BỘ TRẠNG THÁI
    with buffer_lock:
        audio_buffer.clear()

    trigger_flag = threading.Event()
    stop_flag = threading.Event()

    print("👂 Đang chờ: hey jarvis")

    worker = threading.Thread(
        target=inference_worker,
        args=(trigger_flag, stop_flag),
        daemon=True
    )
    worker.start()

    with sd.InputStream(
        channels=1,
        samplerate=SR,
        blocksize=STEP,
        callback=audio_callback
    ):
        while not trigger_flag.is_set():
            time.sleep(0.05)

    # 🧹 dọn sạch
    stop_flag.set()
    worker.join(timeout=0.2)

    with buffer_lock:
        audio_buffer.clear()

    asyncio.run(speak("Tôi xin lắng nghe"))

    # 🛑 guard silence
    time.sleep(SILENCE_GUARD)
    print("✅ WakeWord detected.")
import sounddevice as sd
import webrtcvad
import queue
import time
import numpy as np
from faster_whisper import WhisperModel

# ================= CONFIG =================
SAMPLE_RATE = 16000
CHANNELS = 1

FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

VAD_AGGRESSIVENESS = 2
SILENCE_TIMEOUT = 1.0
MAX_RECORD_TIME = 15.0

PRE_ROLL_FRAMES = 5     # ~150ms giữ đầu câu
POST_ROLL_FRAMES = 2   # ~60ms giữ cuối câu
MIN_AUDIO_SEC = 0.3    # lọc audio quá ngắn


class SpeechToText:
    def __init__(self):
        print("🧠 Loading Whisper model (1 lần duy nhất)...")
        self.model = WhisperModel(
            "large",
            device="cuda",
            compute_type="int8"
        )
        print("✅ Whisper sẵn sàng.")

    def record_until_silence(self):
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        audio_queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            audio_queue.put(bytes(indata))

        print("🎙️ Nói đi...")

        pre_roll = []
        voiced_bytes = []

        triggered = False
        last_voice_time = None
        start_time = time.time()

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            dtype="int16",
            channels=CHANNELS,
            callback=callback,
        ):
            # 🔥 warm-up mic (xả frame rác)
            time.sleep(0.1)

            while True:
                try:
                    frame = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    if time.time() - start_time > MAX_RECORD_TIME:
                        break
                    continue

                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                now = time.time()

                # giữ pre-roll
                if not triggered:
                    pre_roll.append(frame)
                    if len(pre_roll) > PRE_ROLL_FRAMES:
                        pre_roll.pop(0)

                if is_speech:
                    if not triggered:
                        triggered = True
                        voiced_bytes.extend(pre_roll)  # 🔥 ghép đầu câu
                    voiced_bytes.append(frame)
                    last_voice_time = now
                elif triggered:
                    voiced_bytes.append(frame)

                # silence detected → giữ thêm post-roll
                if triggered and last_voice_time and now - last_voice_time > SILENCE_TIMEOUT:
                    for _ in range(POST_ROLL_FRAMES):
                        try:
                            voiced_bytes.append(audio_queue.get(timeout=0.1))
                        except queue.Empty:
                            pass
                    break

                if now - start_time > MAX_RECORD_TIME:
                    break

        if not voiced_bytes:
            return None

        # ===== convert audio =====
        audio = b"".join(voiced_bytes)
        audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # ===== lọc audio quá ngắn =====
        if len(audio) < SAMPLE_RATE * MIN_AUDIO_SEC:
            return None

        # ===== RMS normalization =====
        rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
        audio = audio / max(rms, 0.01)

        # ===== clip nhẹ chống spike =====
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio,
            language="vi",
            task="transcribe",
            beam_size=7,
            temperature=0.0,
            vad_filter=True,
            initial_prompt="Đây là tiếng Việt nói tự nhiên, không phải tiếng Anh."
        )

        text = ""
        for seg in segments:
            text += seg.text

        return text.strip()

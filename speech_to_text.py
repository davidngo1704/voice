import sounddevice as sd
import webrtcvad
import queue
import time
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

VAD_AGGRESSIVENESS = 2
SILENCE_TIMEOUT = 1.0
MAX_RECORD_TIME = 15.0


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
            while True:
                try:
                    frame = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    if time.time() - start_time > MAX_RECORD_TIME:
                        break
                    continue

                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                now = time.time()

                if is_speech:
                    triggered = True
                    voiced_bytes.append(frame)
                    last_voice_time = now
                elif triggered:
                    voiced_bytes.append(frame)

                if triggered and last_voice_time and now - last_voice_time > SILENCE_TIMEOUT:
                    break

                if now - start_time > MAX_RECORD_TIME:
                    break

        if not voiced_bytes:
            return None

        audio = b"".join(voiced_bytes)
        return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio,
            language="vi",
            task="transcribe",
            beam_size=5,
            temperature=0.0,
            vad_filter=False,
        )

        text = ""
        for seg in segments:
            text += seg.text

        return text.strip()

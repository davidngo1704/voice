import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_WAV = "record_ok.wav"


def record_until_enter():
    print("Nhấn Enter để BẮT ĐẦU thu âm...")
    input()

    print("Đang thu âm... Nhấn Enter để DỪNG.")
    frames = []

    def callback(indata, frames_count, time, status):
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    ):
        input()

    audio = np.concatenate(frames, axis=0)
    write(OUTPUT_WAV, SAMPLE_RATE, audio)
    print(f"Đã lưu file: {OUTPUT_WAV}")


def transcribe():
    model = WhisperModel(
        "large",
        device="cuda",
        compute_type="int8"
    )

    segments, info = model.transcribe(
        OUTPUT_WAV,
        language="vi",
        task="transcribe",
        beam_size=5,
        temperature=0.0,
        vad_filter=True,
    )

    for seg in segments:
        print(f"[{seg.start:.2f}s → {seg.end:.2f}s] {seg.text}")


if __name__ == "__main__":
    record_until_enter()
    transcribe()

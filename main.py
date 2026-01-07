from wakeword import wait_for_wakeword
from speech_to_text import SpeechToText
import time

stt = SpeechToText()

while True:
    wait_for_wakeword()

    # tránh wake word bleed
    time.sleep(0.5)

    audio = stt.record_until_silence()
    if audio is None:
        print("🤫 Không phát hiện giọng nói.")
        continue

    text = stt.transcribe(audio)
    if not text:
        print("🤫 Không đủ tự tin để dịch.")
        continue

    print(f"🧠 Bạn nói: {text}")
    print("🔁 Quay lại chờ wake word...\n")

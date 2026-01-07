from wakeword import wait_for_wakeword
from speech_to_text import SpeechToText

stt = SpeechToText()

while True:
    wait_for_wakeword()

    audio = stt.record_until_silence()
    if audio is None:
        print("❌ Không thu được giọng nói.")
        continue

    text = stt.transcribe(audio)
    print(f"🧠 Bạn nói: {text}")

    print("🔁 Quay lại chờ wake word...\n")

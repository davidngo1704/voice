# main.py
from wakeword import wait_for_wakeword
from speech_to_text import record_until_silence, transcribe

while True:
    # 1. Chờ wake word
    wait_for_wakeword()

    # 2. Ghi âm đến khi im lặng
    audio = record_until_silence()

    if audio is None:
        print("❌ Không nghe thấy gì.")
        continue

    # 3. Speech to text
    print("🧠 Kết quả STT:")
    transcribe(audio)

    print("\n🔁 Quay lại chờ wake word...\n")

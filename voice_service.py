import os

import asyncio

import edge_tts

import sounddevice as sd

import soundfile as sf

import numpy as np

import librosa

import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_NAME = "vinai/PhoWhisper-medium"

MODEL_DIR = r"D:\SourceCode\agents\tro_ly\database\models"

WAV_FILE = "test_prompt.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR
)

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()



async def speak_to_wav(text: str, output_path: str):

    communicate = edge_tts.Communicate(
        text=text,
        voice="vi-VN-HoaiMyNeural"
    )

    await communicate.save(output_path)

    data, samplerate = sf.read(output_path, dtype="float32")

    sd.play(data, samplerate)

    sd.wait()


def load_audio_16k(path):
    
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio = librosa.resample(
            y=audio,
            orig_sr=sr,
            target_sr=16000
        )

    return audio


def speech_to_text(wav_path: str) -> str:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Load audio
    audio = load_audio_16k(wav_path)

    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    input_features = input_features.to(dtype=model.dtype)


    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="vi",
            task="transcribe"
        )

    text = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    return text.strip()

def record_audio(seconds=5, sr=16000):
    audio = sd.rec(
        int(seconds * sr),
        samplerate=sr,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.squeeze()

def speech_to_text_from_buffer(audio_16k: np.ndarray) -> str:
    input_features = processor(
        audio_16k,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        ids = model.generate(
            input_features,
            language="vi",
            task="transcribe"
        )

    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


async def main():
    prompt = "anh rất yêu em"

    print("▶ TTS prompt:", prompt)

    await speak_to_wav(prompt, WAV_FILE)

    print("▶ STT processing...")

    result = speech_to_text(WAV_FILE)

    print("▶ STT result: ")

    print(result)

    if os.path.exists(WAV_FILE):

        os.remove(WAV_FILE)

async def speak(text: str):
    tmp = "_tts_tmp.wav"
    await speak_to_wav(text, tmp)
    if os.path.exists(tmp):
        os.remove(tmp)

if __name__ == "__main__":

    asyncio.run(main())
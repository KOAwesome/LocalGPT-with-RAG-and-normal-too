from distutils import util 
import torch
import os
import argparse
import wave
import pyaudio
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

import soundfile as sf
from sentence_transformers import SentenceTransformer, util
from faster_whisper import WhisperModel
from openai1 import OpenAI
import speech_recognition as sr


PINK = "\033[95m"
CYAN = "\033[96m"
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

model_size = "medium.en"
# what is cuda
# whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
whisper_model = WhisperModel(model_size, device="cpu", compute_type="float32")


def open_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
client = OpenAI(base_url="http:localhost:1234/v1", api_key="not-needed")

def play_audio(file_path):
    wf = wave.open(file_path, "rb")
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    #47
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()

play_audio("vault_audio.wav")
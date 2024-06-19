
import torch
import os
import argparse
import wave
import pyaudio
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

import soundfile as sf




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

parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true", default=False, help="make link public")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "/Users/nani/git/LocalGPTwithRAG/outputs"
os.makedirs(output_dir, exist_ok=True)

xtts_config = XttsConfig()
xtts_config.load_json("/Users/nani/git/XTTS-v2/config.json")#no idea where to get this config file

xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(xtts_config, checkpoint_dir="/Users/nani/git/XTTS-v2", eval=True)


def process_and_play(prompt, audio_file_path):
    tts_model = xtts_model
    try:
        output = tts_model.synthesize(prompt,xtts_config, speaker_wav=audio_file_path, gpt_cond_len=24,
                                      temperature=0.6,  language="en", speed=1.2)
        synthesized_audio = output["wav"]
        src_path = f'./outputs/output_audio.wav'
        sample_rate = xtts_config.audio.sample_rate
        sf.write(src_path, synthesized_audio, sample_rate)

        print("Audio generated successfully")
        play_audio(src_path)
    except Exception as e:
        print(f"Error during audio generation: {e}")

process_and_play("I love you baby you are my sweet crush", "./en_sample.wav")
# text_to_speak = "I love you baby"
# reference_audios = ["./en_sample.wav"]

# outputs = xtts_model.synthesize(
#     text_to_speak,
#     xtts_config,
#     speaker_wav=reference_audios,
#     gpt_cond_len=3,
#     language="en",
# )

# output_file_path = f'./outputs/output_audio.wav'
# sf.write(output_file_path, outputs['wav'], 24000)
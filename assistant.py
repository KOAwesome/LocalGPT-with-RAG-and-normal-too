from distutils import util #hoping this will work
import torch
import os
import argparse
import wave
import pyaudio
from TTS.tts.xtts import Xtts#doubtfulk
from TTS.tts.configs.xtts_config import XTTSConfig

import soundfile as sf
from sentence_transformers import SentenceTransformer, util
from faster_whisper import WhisperModel
from openai import OpenAI
import speech_recognition as sr


PINK = "\033[95m"
CYAN = "\033[96m"
NEON_GREEN = "\033[92m"
RESET_COLOR = "\033[0m"

model_size = "medium.en"
# what is cuda
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

def open_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
client = OpenAI(base_url="http:localhos:1234/v1", api_key="not-needed")

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
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

xtts_config = XTTSConfig()#device=device)
xtts_config.load_json("TTS/tts/configs/config.json")#no idea where to get this config file

xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint("TTS/tts/weights/checkpoint_262000.pth", eval=True)# again no idea
xtts_model.cuda()

def process_and_play(prompt, audio_file_path):
    tts_model = xtts_model
    try:
        output = tts_model.synthesize(prompt,xtts_config, speaker_wav=audio_file_path, gpt_cond_len=24,
                                      temperature=0.6,  language="en", speed=1.2)
        synthesized_audio = output["wav"]
        src_path = f'{output_dir}/output.wav'
        sample_rate = xtts_config.audio.sample_rate
        sf.write(src_path, synthesized_audio, sample_rate)

        print("Audio generated successfully")
        play_audio(src_path)
    except Exception as e:
        print(f"Error during audio generation: {e}")

def get_relevant_context(user_input, vault_embeddings, vault_context, model, top_k=3):
    if vault_embeddings.nelement() == 0: # if tensor has any elements
        return []
    
    input_embedding = model.encode([user_input]) #encode user context
    cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0] #compute cosine similarity b/w input and vault embvedding
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, top_k)[1].tolist() #get top k indices
    relevant_context = [vault_context[i].strip() for i in top_indices] #get top k context
    return relevant_context

def chatgpt_streamed(user_input, system_message, conversation_history, bot_name, vault_embeddings, vault_context, model):
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_context, model)
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = "\n".join(relevant_context) + "\n\n" + user_input
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input_with_context}]
    temperature = 1
    streamed_completion = client.chat.completions.create(
        model="local-model",
        messages=messages,
        stream=True)
    full_response = ""
    line_buffer = ""
    for chunk in streamed_completion:
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            line_buffer += delta_content
            if '\n' in line_buffer:
                lines = line_buffer.split('\n')
                for line in lines[:-1]:
                    print(NEON_GREEN + line + RESET_COLOR)
                    full_response += line + "\n"
                line_buffer = lines[-1]
    if line_buffer:
        print(NEON_GREEN + line_buffer + RESET_COLOR)
        full_response += line_buffer
    return full_response

def transcribe_with_whisper(audio_file):
    segments,info = whisper_model.transcribe(audio_file, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment["text"] + ""#segment.text
    return transcription.strip()

def record_audio(file_path):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    print("Recording...")
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    print("Recording stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b"".join(frames))
    wf.close()


def user_chatbot_conversation():
    conversation_history = []
    system_message = open_file("chatbot2.txt")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vault_content = []
    if os.path.exists("vault.txt"):
        vault_content = vault_file.readlines()
    vault_embeddings = model.encode(vault_content) if vault_content else []
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    while True:
        audio_file = "vault_audio.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)
        if user_input.lower() == "exit":
            break
        elif user_input.lower().startswith(("print info", "Print Info")):
            print("Info contents:")
            if os.path.exists("vault.txt"):
                with open ("vault.txt", "r", encoding="utf-8") as vault_file:
                    print(NEON_GREEN+ vault_file.read() + RESET_COLOR)
            else:
                print("Vault is empty")
            continue
        elif user_input.lower().startswith(("clear vault", "Clear Vault")):
            confirm = input("Are you sure you want to clear the vault? (yes/no): ")
            if confirm.lower() == "yes":
                with open("vault.txt", "w", encoding="utf-8") as vault_file:
                    vault_file.write("")
                print("Vault cleared successfully")
            else:
                print("Operation cancelled")
            continue
        elif user_input.lower().startswith(("insert info", "Insert Info")):
            print("Recording for info...")
            audio_file ="vault_audio.wav"
            record_audio(audio_file)
            vault_input = transcribe_with_whisper(audio_file)
            os.remove(audio_file)
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                vault_file.write(vault_input + "\n")
            print("Info inserted successfully")
            vault_content = open("vault.txt", "r", encoding="utf-8").readLines()
            vault_embeddings= model.encode(vault_content)
            vault_embeddings_tensor = torch.tensor(vault_embeddings)
            continue
        print(CYAN + "You:" + user_input + RESET_COLOR) #idk indentation
        conversation_history.append({"role": "user", "content" : user_input})
    #246
    print(PINK + "Emma:" + RESET_COLOR)
    chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "chatbot", vault_embeddings_tensor, vault_content, model)
    conversation_history.append({"role": "assistant", "content": chatbot_response})
    prompt2 = chatbot_response
    audio_file_pth2 = "/Users/nani/git/LocalGPTwithRAG/emma2.wav"
    process_and_play(prompt2, audio_file_pth2)
    if  len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
user_chatbot_conversation()
### ended program
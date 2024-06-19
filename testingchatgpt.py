from distutils import util 
import torch
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
client = OpenAI(base_url="http:localhost:1234/v1", api_key="not-needed")


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
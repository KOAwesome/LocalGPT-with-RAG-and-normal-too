import jax
import jax.numpy as jnp
import whisper_jax

def transcribe_with_whisper(audio_file):
    # Load the Whisper JAX model
    model = whisper_jax.load_model("base")

    # Transcribe the audio file using Whisper JAX
    transcription = model.transcribe(audio_file)

    return transcription.text

print(transcribe_with_whisper("./en_sample.wav"))
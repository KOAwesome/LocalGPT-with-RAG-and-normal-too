
from faster_whisper import WhisperModel
model_size = "medium.en"
whisper_model = WhisperModel(model_size, device="cpu", compute_type="float32")

def transcribe_with_whisper(audio_file):
    segments,info = whisper_model.transcribe(audio_file, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + ""#segment.text
    return transcription.strip()

print(transcribe_with_whisper("vault_audio.wav"))
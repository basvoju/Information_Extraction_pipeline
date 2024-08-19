import whisper
import resampy
import numpy as np
from jiwer import wer

# Load Whisper model
model = whisper.load_model("small")

# Function to transcribe audio
def transcribe_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text

# Calculate WER
def calculate_wer(reference, hypothesis):
    return wer(reference, hypothesis)

# Example usage with a LibriSpeech test-clean sample

reference_text = r'C:/Users/BASVOJU/Desktop/Master_thesis/Transcriptions/ground_truth.txt'
hypothesis_text = r'C:/Users/BASVOJU/Desktop/Master_thesis/Transcriptions/transcribed_text.txt'

wer_score = calculate_wer(reference_text, hypothesis_text)
print(f"WER: {wer_score:.3f}")

# Compare with SOTA
sota_wer = 0.016  # 1.6%
print(f"Your model's WER: {wer_score:.3f}")
print(f"SOTA WER: {sota_wer:.3f}")


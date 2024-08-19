import numpy as np
from audio_utils import load_audio, audio_to_numpy
from transcribe import transcribe_audio
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    audio_file = "C:/Users/BASVOJU/Desktop/Master_thesis/Dataset/1_FALL.mp4"

    try:
        # Load audio bytes from file
        audio_bytes = load_audio(audio_file)

        # Convert audio bytes to NumPy array and samplerate
        audio_np, samplerate = audio_to_numpy(audio_bytes)

        print(f"Audio shape: {audio_np.shape}, dtype: {audio_np.dtype}, samplerate: {samplerate}")
        print("Starting transcription...")

        try:
            # Choose the model to use ('whisper', 'azure', or 'ibm')
            transcription = transcribe_audio(audio_np, samplerate, model_name='ibm')
            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"An error occurred during transcription: {e}")

    except Exception as e:
        print(f"An error occurred while loading audio: {e}")

if __name__ == "__main__":
    main()

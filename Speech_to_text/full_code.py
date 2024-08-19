import subprocess
import whisper
import imageio_ffmpeg as ffmpeg
import numpy as np
import soundfile as sf
import io
import sys
import warnings
import resampy
import pandas as pd


def load_audio(file_path):
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_path, "-i", file_path, "-f", "wav", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error while running FFmpeg:\n{e.stderr.decode()}", file=sys.stderr)
        raise


def audio_to_numpy(audio_bytes):
    audio_file = io.BytesIO(audio_bytes)
    audio_np, samplerate = sf.read(audio_file)
    return audio_np, samplerate


def transcribe_audio(audio_np, samplerate):
    model = whisper.load_model("large")

    # Resample the audio to 16000 Hz if necessary
    if samplerate != 16000:
        audio_np = resampy.resample(audio_np, samplerate, 16000)
        samplerate = 16000

    # Convert audio data to float32
    audio_np = audio_np.astype(np.float32)

    # Prepare audio for transcription
    audio_np = np.array(audio_np)  # Ensure it's a NumPy array

    # Transcribe the audio
    result = model.transcribe(audio_np, language="de")
    return result


# Load and convert audio file
audio_path = "C:/Users/BASVOJU/Desktop/Master_thesis/Dataset/1_FALL.mp4"

# Extract audio bytes from the video file
audio_bytes = load_audio(audio_path)

# Convert audio bytes to NumPy array
audio_np, samplerate = audio_to_numpy(audio_bytes)

# Convert audio data to float32 (ensure dtype is correct)
audio_np = audio_np.astype(np.float32)

# Check the shape and dtype of the audio array
print(f"Audio shape: {audio_np.shape}, dtype: {audio_np.dtype}, samplerate: {samplerate}")

# Suppress specific warnings and transcribe the audio
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

    print("Starting transcription...")
    try:
        result = transcribe_audio(audio_np, samplerate)
        print("Transcription completed.")
        transcription_text = result['text']
        print(transcription_text)

        # Prepare data for Excel
        data = {
            'File Name': [audio_path],
            'Transcription': [transcription_text]
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Define the path to save the Excel file
        excel_path = "C:/Users/BASVOJU/Desktop/Master_thesis/Transcriptions/transcriptions.xlsx"

        # Save DataFrame to Excel
        df.to_excel(excel_path, index=False)

        print(f"Transcription saved to {excel_path}")

    except Exception as e:
        print(f"An error occurred during transcription: {e}")


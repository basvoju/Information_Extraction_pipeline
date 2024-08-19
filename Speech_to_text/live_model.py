import whisper
import pyaudio
import numpy as np
import resampy
import time

# Initialize Whisper model
model = whisper.load_model("large")

# Audio stream settings
RATE = 16000  # Sampling rate
CHUNK = 1024  # Number of frames per chunk

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
print("Recording...")

try:
    while True:
        data = stream.read(CHUNK)
        audio_np = np.frombuffer(data, dtype=np.int16)
        # print(f"Audio chunk: {audio_np[:10]}")  # Print the first 10 samples
except KeyboardInterrupt:
    print("Stopped recording.")


def process_audio_chunk(chunk):
    # Convert chunk to numpy array
    audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)

    # Resample if necessary
    if RATE != 16000:
        audio_np = resampy.resample(audio_np, RATE, 16000)

    # Transcribe audio
    result = model.transcribe(audio_np, language="de")  # Set language to German
    return result['text']


print("Starting real-time transcription...")

try:
    while True:
        # Read audio chunk
        data = stream.read(CHUNK)
        text = process_audio_chunk(data)
        print(f"Transcribed Text: {text}")
        time.sleep(0.1)  # Adjust sleep time as necessary

except KeyboardInterrupt:
    print("Stopping real-time transcription...")

finally:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

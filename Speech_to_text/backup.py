import whisper
import pyaudio
import numpy as np
import resampy
import queue
import threading

# Initialize Whisper model
model = whisper.load_model("small")

# Audio settings
RATE = 16000
CHUNK = 1024

# Queue for audio chunks
audio_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def process_audio_queue():
    audio_data = []
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break

        # Convert audio bytes to numpy array
        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        audio_data.extend(audio_np)

        if len(audio_data) >= RATE * 5:  # Process every 5 seconds of audio
            # Resample if necessary
            if RATE != 16000:
                audio_data = resampy.resample(np.array(audio_data), RATE, 16000)

            # Transcribe audio
            result = model.transcribe(np.array(audio_data), language="de")
            print(f"Transcribed Text: {result['text']}")

            # Reset audio data
            audio_data = []

    # Process any remaining audio data at the end
    if len(audio_data) > 0:
        if RATE != 16000:
            audio_data = resampy.resample(np.array(audio_data), RATE, 16000)
        result = model.transcribe(np.array(audio_data), language="de")
        print(f"Transcribed Text: {result['text']}")

def main():
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    print("Starting real-time transcription...")

    # Start processing thread
    processing_thread = threading.Thread(target=process_audio_queue, daemon=True)
    processing_thread.start()

    try:
        stream.start_stream()
        while processing_thread.is_alive():
            processing_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("Stopping real-time transcription...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        audio_queue.put(None)
        processing_thread.join()

if __name__ == "__main__":
    main()

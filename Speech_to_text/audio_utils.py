import subprocess
import imageio_ffmpeg as ffmpeg
import soundfile as sf
import io
import sys

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

import numpy as np
import resampy
import whisper
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import io
import os

load_dotenv()

# Whisper Transcription
def transcribe_whisper(audio_np, samplerate):
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
    return result['text']

# Azure Speech-to-Text Transcription
def transcribe_azure(audio_np, samplerate):
    subscription_key = 'YOUR_AZURE_SUBSCRIPTION_KEY'
    region = 'YOUR_AZURE_REGION'

    if samplerate != 16000:
        audio_np = resampy.resample(audio_np, samplerate, 16000)
        samplerate = 16000

    audio_np = (audio_np * 32767).astype(np.int16)
    with open('temp.wav', 'wb') as f:
        f.write(audio_np.tobytes())

    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.speech_recognition_language = "de-DE"
    audio_config = speechsdk.AudioConfig(filename='temp.wav')
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        raise Exception(f"Azure Speech Service Error: {result.reason}")

# IBM Watson Speech-to-Text Transcription
def transcribe_ibm(audio_np, samplerate):
    apikey = os.getenv('IBM_API_KEY')
    url = os.getenv('IBM_URL')

    if samplerate != 16000:
        audio_np = resampy.resample(audio_np, samplerate, 16000)
        samplerate = 16000

    audio_np = (audio_np * 32767).astype(np.int16)
    audio_data = io.BytesIO(audio_np.tobytes())

    authenticator = IAMAuthenticator(apikey)
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(url)

    result = speech_to_text.recognize(
        audio=audio_data,
        content_type='audio/wav',
        model='de-DE_BroadbandModel'
    ).get_result()

    return result['results'][0]['alternatives'][0]['transcript']

# Wrapper Function for Transcription
def transcribe_audio(audio_np, samplerate, model_name='whisper'):
    if model_name == 'whisper':
        return transcribe_whisper(audio_np, samplerate)
    elif model_name == 'azure':
        return transcribe_azure(audio_np, samplerate)
    elif model_name == 'ibm':
        return transcribe_ibm(audio_np, samplerate)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

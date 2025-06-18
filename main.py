# from fastapi import FastAPI, File, UploadFile
# from diarize_transcribe import diarize_and_transcribe
# import shutil
# import os

# app = FastAPI()


# @app.post("/diarize/")
# async def diarize_endpoint(file: UploadFile = File(...)):
#     temp_filename = "input.wav"
#     with open(temp_filename, "wb") as f:
#         shutil.copyfileobj(file.file, f)

#     try:
#         result = diarize_and_transcribe(temp_filename)
#         return result
#     finally:
#         if os.path.exists(temp_filename):
#             os.remove(temp_filename)


# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# import os
# import time
# import shutil
# import json
# import wave
# import pyaudio
# from diarize_transcribe import diarize_and_transcribe

# app = FastAPI()

# os.makedirs("output", exist_ok=True)
# os.makedirs("realtime_chunks", exist_ok=True)

# # === API 1: Upload audio file ===
# @app.post("/upload-audio/")
# async def upload_audio(file: UploadFile = File(...)):
#     temp_filename = "input.wav"
#     with open(temp_filename, "wb") as f:
#         shutil.copyfileobj(file.file, f)

#     try:
#         result = diarize_and_transcribe(temp_filename)
#         out_path = f"output/transcript_{int(time.time())}.json"
#         with open(out_path, "w") as f:
#             json.dump(result, f, indent=4)
#         return JSONResponse(content=result)
#     finally:
#         if os.path.exists(temp_filename):
#             os.remove(temp_filename)

# # === API 2: Record audio from microphone ===
# @app.post("/record-mic/")
# async def record_mic():
#     duration = 10  # seconds
#     timestamp = int(time.time())
#     filename = f"mic_record_{timestamp}.wav"
#     filepath = os.path.join("realtime_chunks", filename)

#     # Mic recording config
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 16000
#     CHUNK = 1024

#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=FORMAT, channels=CHANNELS,
#                         rate=RATE, input=True,
#                         frames_per_buffer=CHUNK)

#     frames = []
#     print(f"Recording {duration}s from mic...")
#     for _ in range(0, int(RATE / CHUNK * duration)):
#         data = stream.read(CHUNK)
#         frames.append(data)

#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     # Save .wav file
#     with wave.open(filepath, 'wb') as wf:
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(audio.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))

#     try:
#         result = diarize_and_transcribe(filepath)
#         out_path = f"output/transcript_{timestamp}.json"
#         with open(out_path, "w") as f:
#             json.dump(result, f, indent=4)
#         return JSONResponse(content=result)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)


import os
import json
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline
from datetime import datetime, timedelta
import requests


app = FastAPI()

# Load models
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
pipeline_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
whisper_model = WhisperModel("base", device="cpu")
sentiment_analyzer = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = ["satisfaction", "frustration", "neutral", "confusion", "helpfulness", "rudeness", "urgency"]

def convert_to_mono(wav_path: str) -> str:
    audio = AudioSegment.from_file(wav_path)
    mono_audio = audio.set_channels(1)
    mono_audio = mono_audio.set_frame_rate(16000)
    mono_file = NamedTemporaryFile(delete=False, suffix=".wav")
    mono_audio.export(mono_file.name, format="wav")
    return mono_file.name

def analyze_sentiment(text: str) -> str:
    prompt = f"""
You are an expert in analyzing customer support conversations.

Classify the following utterance into one of these categories:
- satisfaction
- frustration
- neutral
- confusion
- helpfulness
- rudeness
- urgency

Utterance: "{text}"

Only respond with the most relevant label.
"""
    response = requests.post(
        "http://localhost:11434/api/generate",  
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    if response.status_code == 200:
        return response.json()["response"].strip().lower()
    else:
        return "unknown"

def diarize_and_transcribe(file_path: str) -> list:
    base_time = datetime.now()
    mono_path = convert_to_mono(file_path)
    diarization = pipeline_diarization(mono_path)
    segments, _ = whisper_model.transcribe(mono_path, beam_size=5)

    speaker_first_spoken = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_str = str(speaker)
        if speaker_str not in speaker_first_spoken:
            speaker_first_spoken[speaker_str] = turn.start

    ordered_speakers = [s for s, _ in sorted(speaker_first_spoken.items(), key=lambda x: x[1])]
    speaker_labels = {ordered_speakers[0]: "Agent", ordered_speakers[1]: "Customer"}

    output = []
    for segment in segments:
        seg_start, seg_end = segment.start, segment.end
        text = segment.text.strip()
        midpoint = (seg_start + seg_end) / 2

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_str = str(speaker)
            if turn.start <= midpoint <= turn.end:
                output.append({
                    "speaker": speaker_labels.get(speaker_str, speaker_str),
                    "timestamp": f"from {(base_time + timedelta(seconds=seg_start)).isoformat(timespec='milliseconds')} to {(base_time + timedelta(seconds=seg_end)).isoformat(timespec='milliseconds')}",
                    "complete_trans": text,
                    "sentiment": analyze_sentiment(text)
                })
                break

    os.makedirs("output", exist_ok=True)
    output_path = f"output/transcript_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    os.remove(mono_path)
    return output

@app.post("/diarize/")
async def diarize_endpoint(file: UploadFile = File(...)):
    temp = NamedTemporaryFile(delete=False, suffix=".wav")
    temp.write(await file.read())
    temp.close()

    try:
        result = diarize_and_transcribe(temp.name)
        return JSONResponse(content=result)
    finally:
        os.remove(temp.name)

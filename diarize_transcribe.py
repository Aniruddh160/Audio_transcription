

import os
import json
import time
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from datetime import datetime, timedelta


# Load models
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
whisper_model = WhisperModel("base", device="cpu")
base_time = datetime.now()
def convert_to_mono(wav_path: str) -> str:
    audio = AudioSegment.from_file(wav_path)
    mono_audio = audio.set_channels(1)
    mono_audio = mono_audio.set_frame_rate(16000)
    mono_file = NamedTemporaryFile(delete=False, suffix=".wav")
    mono_audio.export(mono_file.name, format="wav")
    return mono_file.name

def diarize_and_transcribe(file_path: str) -> list:
    mono_path = convert_to_mono(file_path)
    diarization = pipeline(mono_path)
    segments, _ = whisper_model.transcribe(mono_path, beam_size=5)

    # Step 1: Map segment midpoints to speakers
    speaker_turns = []  # ← This will be our final output list

    # Step 1.1: Determine speaker order
    speaker_first_spoken = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_str = str(speaker)
        if speaker_str not in speaker_first_spoken:
            speaker_first_spoken[speaker_str] = turn.start

    ordered_speakers = [s for s, _ in sorted(speaker_first_spoken.items(), key=lambda x: x[1])]
    speaker_labels = {ordered_speakers[0]: "Agent", ordered_speakers[1]: "Customer"}

    # Step 2: Align Whisper segments to speaker turns
    for segment in segments:
        seg_start, seg_end = segment.start, segment.end
        text = segment.text.strip()
        midpoint = (seg_start + seg_end) / 2

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_str = str(speaker)
            if turn.start <= midpoint <= turn.end:
                speaker_turns.append({
    "speaker": speaker_labels.get(speaker_str, speaker_str),
    "start_timestamp": (base_time + timedelta(seconds=seg_start)).isoformat(timespec='milliseconds'),
    "end_timestamp": (base_time + timedelta(seconds=seg_end)).isoformat(timespec='milliseconds'),
    "complete_trans": text,
    "additional_info": {}
})
                break

    # Step 3: Save JSON to output/
    os.makedirs("output", exist_ok=True)
    file_path = f"output/transcript_{int(time.time())}.json"
    with open(file_path, "w") as f:
        json.dump(speaker_turns, f, indent=4)

    os.remove(mono_path)
    return speaker_turns

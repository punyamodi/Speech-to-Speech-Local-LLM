import os
import glob
import torch
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel
from whisper_timestamped.transcribe import get_audio_tensor, get_vad_segments

_whisper_model_size = "medium"
_model = None


def _get_whisper_model(device="cuda"):
    global _model
    if _model is None:
        compute_type = "float16" if device == "cuda" and torch.cuda.is_available() else "int8"
        _model = WhisperModel(_whisper_model_size, device=device, compute_type=compute_type)
    return _model


def split_audio_vad(audio_path, target_dir, split_seconds=10.0):
    SAMPLE_RATE = 16000
    audio_vad = get_audio_tensor(audio_path)
    segments = get_vad_segments(
        audio_vad,
        output_sample=True,
        min_speech_duration=0.1,
        min_silence_duration=1,
        method="silero",
    )
    segments = [(float(s["start"]) / SAMPLE_RATE, float(s["end"]) / SAMPLE_RATE) for s in segments]

    audio = AudioSegment.from_file(audio_path)
    audio_active = AudioSegment.silent(duration=0)
    for start_time, end_time in segments:
        audio_active += audio[int(start_time * 1000): int(end_time * 1000)]

    audio_dur = audio_active.duration_seconds
    audio_name = os.path.basename(audio_path).rsplit(".", 1)[0]
    target_folder = os.path.join(target_dir, audio_name)
    wavs_folder = os.path.join(target_folder, "wavs")
    os.makedirs(wavs_folder, exist_ok=True)

    num_splits = max(1, int(np.round(audio_dur / split_seconds)))
    interval = audio_dur / num_splits
    start_time = 0.0

    for i in range(num_splits):
        end_time = min(start_time + interval, audio_dur)
        if i == num_splits - 1:
            end_time = audio_dur
        output_file = f"{wavs_folder}/{audio_name}_seg{i}.wav"
        audio_active[int(start_time * 1000): int(end_time * 1000)].export(output_file, format="wav")
        start_time = end_time

    return wavs_folder


def split_audio_whisper(audio_path, target_dir="processed"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_whisper_model(device=device)
    audio = AudioSegment.from_file(audio_path)
    max_len = len(audio)
    audio_name = os.path.basename(audio_path).rsplit(".", 1)[0]
    target_folder = os.path.join(target_dir, audio_name)
    wavs_folder = os.path.join(target_folder, "wavs")
    os.makedirs(wavs_folder, exist_ok=True)

    segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = list(segments)

    s_ind = 0
    start_time = None
    for k, w in enumerate(segments):
        if k == 0:
            start_time = max(0, w.start)
        end_time = w.end
        confidence = (sum(s.probability for s in w.words) / len(w.words)) if w.words else 0.0
        text = w.text.replace("...", "")

        audio_seg = audio[int(start_time * 1000): min(max_len, int(end_time * 1000) + 80)]
        fname = f"{audio_name}_seg{s_ind}.wav"

        if (audio_seg.duration_seconds > 1.5 and audio_seg.duration_seconds < 20.0
                and 2 <= len(text) < 200):
            audio_seg.export(os.path.join(wavs_folder, fname), format="wav")

        if k < len(segments) - 1:
            start_time = max(0, segments[k + 1].start - 0.08)
        s_ind += 1

    return wavs_folder


def get_se(audio_path, vc_model, target_dir="processed", vad=True):
    device = vc_model.device
    audio_name = os.path.basename(audio_path).rsplit(".", 1)[0]
    se_path = os.path.join(target_dir, audio_name, "se.pth")

    if os.path.isfile(se_path):
        return torch.load(se_path).to(device), audio_name

    if os.path.isdir(audio_path):
        wavs_folder = audio_path
    elif vad:
        wavs_folder = split_audio_vad(audio_path, target_dir)
    else:
        wavs_folder = split_audio_whisper(audio_path, target_dir)

    audio_segs = glob.glob(f"{wavs_folder}/*.wav")
    if not audio_segs:
        raise RuntimeError("No audio segments found for speaker embedding extraction.")

    return vc_model.extract_se(audio_segs, se_save_path=se_path), audio_name

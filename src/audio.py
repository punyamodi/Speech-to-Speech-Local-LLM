import wave
import pyaudio
import numpy as np
import soundfile as sf
import tempfile
import os


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_SIZE = 1024


def play_wav_file(file_path: str) -> None:
    wf = wave.open(file_path, "rb")
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )
    data = wf.readframes(CHUNK_SIZE)
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK_SIZE)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


def numpy_to_wav(audio_array: np.ndarray, sample_rate: int, output_path: str) -> str:
    sf.write(output_path, audio_array, sample_rate)
    return output_path


def get_audio_duration(file_path: str) -> float:
    with sf.SoundFile(file_path) as f:
        return len(f) / f.samplerate


def ensure_wav(audio_path: str) -> str:
    if audio_path.lower().endswith(".wav"):
        return audio_path
    data, sr = sf.read(audio_path)
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, data, sr)
    return tmp

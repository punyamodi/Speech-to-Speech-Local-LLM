import scipy.io.wavfile
import numpy as np
from typing import Optional
from .base import BaseTTS


VOICE_PRESETS = [f"v2/en_speaker_{i}" for i in range(10)]


class BarkTTS(BaseTTS):
    def __init__(self, model_name: str = "suno/bark-small"):
        self._model_name = model_name
        self._pipeline = None

    @property
    def name(self) -> str:
        return "bark"

    def _load(self) -> None:
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline("text-to-speech", self._model_name)

    def synthesize(self, text: str, output_path: str, voice_preset: str = "v2/en_speaker_6", **kwargs) -> str:
        self._load()
        speech = self._pipeline(
            text,
            forward_params={"do_sample": True, "voice_preset": voice_preset},
        )
        audio_data = speech["audio"]
        sample_rate = speech["sampling_rate"]

        if audio_data.ndim == 2:
            audio_data = audio_data[0]
        audio_int16 = (audio_data * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_path, sample_rate, audio_int16)
        return output_path

import whisper
from typing import Optional


class WhisperTranscriber:
    def __init__(self, model_size: str = "base.en"):
        self._model_size = model_size
        self._model: Optional[whisper.Whisper] = None

    def _load(self) -> None:
        if self._model is None:
            self._model = whisper.load_model(self._model_size)

    def transcribe(self, audio_file_path: str) -> str:
        self._load()
        result = self._model.transcribe(audio_file_path)
        return result["text"].strip()

    def reload(self, model_size: str) -> None:
        if model_size != self._model_size:
            self._model_size = model_size
            self._model = None

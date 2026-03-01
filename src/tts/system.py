import os
import tempfile
from .base import BaseTTS


class SystemTTS(BaseTTS):
    def __init__(self, rate: int = 150, voice_index: int = 0):
        self._rate = rate
        self._voice_index = voice_index
        self._engine = None

    @property
    def name(self) -> str:
        return "system"

    def _init_engine(self) -> None:
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            voices = self._engine.getProperty("voices")
            if voices and self._voice_index < len(voices):
                self._engine.setProperty("voice", voices[self._voice_index].id)
            self._engine.setProperty("rate", self._rate)

    def synthesize(self, text: str, output_path: str, **kwargs) -> str:
        self._init_engine()
        self._engine.save_to_file(text, output_path)
        self._engine.runAndWait()
        return output_path

    def speak(self, text: str) -> None:
        self._init_engine()
        self._engine.say(text)
        self._engine.runAndWait()

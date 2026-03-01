import os
import logging
from typing import Iterator, List, Dict, Tuple, Optional

from src.config import AppConfig, LLMConfig, STTConfig, TTSConfig
from src.stt import WhisperTranscriber
from src.llm import LMStudioClient, OllamaClient
from src.tts.base import BaseTTS

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: AppConfig):
        self._config = config
        self._transcriber = WhisperTranscriber(config.stt.model_size)
        self._llm = self._build_llm(config.llm)
        self._tts: Optional[BaseTTS] = None
        os.makedirs(config.output_dir, exist_ok=True)

    def _build_llm(self, cfg: LLMConfig):
        if cfg.backend == "lmstudio":
            return LMStudioClient(
                base_url=cfg.base_url,
                model=cfg.model,
                temperature=cfg.temperature,
            )
        if cfg.backend == "ollama":
            return OllamaClient(model=cfg.ollama_model, temperature=cfg.temperature)
        raise ValueError(f"Unknown LLM backend: {cfg.backend!r}")

    def _build_tts(self, cfg: TTSConfig) -> BaseTTS:
        if cfg.backend == "bark":
            from src.tts.bark import BarkTTS
            return BarkTTS()
        if cfg.backend == "system":
            from src.tts.system import SystemTTS
            return SystemTTS(rate=cfg.system_rate, voice_index=cfg.system_voice_index)
        if cfg.backend == "openvoice":
            from src.tts.openvoice import OpenVoiceTTS
            return OpenVoiceTTS()
        raise ValueError(f"Unknown TTS backend: {cfg.backend!r}")

    def _get_tts(self) -> BaseTTS:
        if self._tts is None:
            self._tts = self._build_tts(self._config.tts)
        return self._tts

    def transcribe(self, audio_path: str) -> str:
        return self._transcriber.transcribe(audio_path)

    def stream_response(self, user_input: str, history: List[Dict]) -> Iterator[str]:
        trimmed = history[-self._config.llm.max_history:]
        messages = trimmed + [{"role": "user", "content": user_input}]
        return self._llm.stream(messages, self._config.system_prompt)

    def get_response(self, user_input: str, history: List[Dict]) -> str:
        return "".join(self.stream_response(user_input, history))

    def synthesize(self, text: str) -> str:
        tts = self._get_tts()
        output_path = os.path.join(self._config.output_dir, "response.wav")
        kwargs = {}
        cfg = self._config.tts
        if cfg.backend == "openvoice":
            kwargs["style"] = cfg.style
            kwargs["reference_audio"] = cfg.reference_audio
        elif cfg.backend == "bark":
            kwargs["voice_preset"] = cfg.bark_voice
        return tts.synthesize(text, output_path, **kwargs)

    def log_exchange(self, user_input: str, assistant_response: str) -> None:
        try:
            with open(self._config.log_file, "a", encoding="utf-8") as f:
                f.write(f"User: {user_input}\n")
                f.write(f"Assistant: {assistant_response}\n\n")
        except OSError as exc:
            logger.warning("Could not write to log file: %s", exc)

    def update_config(self, config: AppConfig) -> None:
        old_llm_backend = self._config.llm.backend
        old_tts_backend = self._config.tts.backend
        self._config = config

        if (
            config.llm.backend != old_llm_backend
            or config.llm.base_url != getattr(self._llm, "_client", None) and True
        ):
            self._llm = self._build_llm(config.llm)

        if config.tts.backend != old_tts_backend:
            self._tts = None

        self._transcriber.reload(config.stt.model_size)
        os.makedirs(config.output_dir, exist_ok=True)

    @property
    def config(self) -> AppConfig:
        return self._config

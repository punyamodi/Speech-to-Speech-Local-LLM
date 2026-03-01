from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    backend: str = "lmstudio"
    base_url: str = "http://localhost:1234/v1"
    model: str = "local-model"
    ollama_model: str = "llama2"
    temperature: float = 0.7
    max_history: int = 20


@dataclass
class STTConfig:
    model_size: str = "base.en"


@dataclass
class TTSConfig:
    backend: str = "bark"
    style: str = "default"
    reference_audio: Optional[str] = None
    bark_voice: str = "v2/en_speaker_6"
    system_rate: int = 150
    system_voice_index: int = 0


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    system_prompt: str = (
        "You are a helpful, concise, and friendly AI assistant. "
        "Keep your responses conversational and to the point since they will be spoken aloud."
    )
    output_dir: str = "outputs"
    log_file: str = "conversation_log.txt"

import argparse
import logging
import sys

from src.config import AppConfig
from src.pipeline import Pipeline
from ui.app import create_ui

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoiceChat — Local speech-to-speech AI assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host address")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public shareable Gradio link")
    parser.add_argument("--tts", choices=["bark", "system", "openvoice"], default="bark", help="TTS backend")
    parser.add_argument("--llm", choices=["lmstudio", "ollama"], default="lmstudio", help="LLM backend")
    parser.add_argument("--lmstudio-url", default="http://localhost:1234/v1", help="LM Studio server URL")
    parser.add_argument("--ollama-model", default="llama2", help="Ollama model name")
    parser.add_argument("--whisper-model", default="base.en", help="Whisper model size")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    from src.config import LLMConfig, STTConfig, TTSConfig

    config = AppConfig(
        llm=LLMConfig(
            backend=args.llm,
            base_url=args.lmstudio_url,
            ollama_model=args.ollama_model,
        ),
        stt=STTConfig(model_size=args.whisper_model),
        tts=TTSConfig(backend=args.tts),
    )

    logger.info("Initializing pipeline (LLM=%s, TTS=%s, STT=%s)", args.llm, args.tts, args.whisper_model)
    pipeline = Pipeline(config)

    app = create_ui(pipeline)

    logger.info("Starting VoiceChat at http://%s:%d", args.host, args.port)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()

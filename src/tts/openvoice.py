import os
import sys
import torch
import shutil
from .base import BaseTTS

OPENVOICE_STYLES = [
    "default", "whispering", "shouting", "excited",
    "cheerful", "terrified", "angry", "sad", "friendly",
]

_EN_CKPT_BASE = os.path.join("checkpoints", "base_speakers", "EN")
_CKPT_CONVERTER = os.path.join("checkpoints", "converter")


class OpenVoiceTTS(BaseTTS):
    def __init__(
        self,
        en_ckpt_base: str = _EN_CKPT_BASE,
        ckpt_converter: str = _CKPT_CONVERTER,
        device: str = None,
    ):
        self._en_ckpt_base = en_ckpt_base
        self._ckpt_converter = ckpt_converter
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._base_tts = None
        self._converter = None
        self._default_se = None
        self._style_se = None

    @property
    def name(self) -> str:
        return "openvoice"

    @staticmethod
    def checkpoints_available(en_ckpt_base: str = _EN_CKPT_BASE, ckpt_converter: str = _CKPT_CONVERTER) -> bool:
        required = [
            os.path.join(en_ckpt_base, "config.json"),
            os.path.join(en_ckpt_base, "checkpoint.pth"),
            os.path.join(ckpt_converter, "config.json"),
            os.path.join(ckpt_converter, "checkpoint.pth"),
        ]
        return all(os.path.isfile(p) for p in required)

    def _load_models(self) -> None:
        if self._base_tts is not None:
            return

        from openvoice import BaseSpeakerTTS, ToneColorConverter
        from openvoice import se_extractor as _se_extractor

        self._se_extractor = _se_extractor
        self._base_tts = BaseSpeakerTTS(
            os.path.join(self._en_ckpt_base, "config.json"),
            device=self._device,
        )
        self._base_tts.load_ckpt(os.path.join(self._en_ckpt_base, "checkpoint.pth"))

        self._converter = ToneColorConverter(
            os.path.join(self._ckpt_converter, "config.json"),
            device=self._device,
        )
        self._converter.load_ckpt(os.path.join(self._ckpt_converter, "checkpoint.pth"))

        self._default_se = torch.load(
            os.path.join(self._en_ckpt_base, "en_default_se.pth"),
            map_location=self._device,
        )
        self._style_se = torch.load(
            os.path.join(self._en_ckpt_base, "en_style_se.pth"),
            map_location=self._device,
        )

    def synthesize(
        self,
        text: str,
        output_path: str,
        style: str = "default",
        reference_audio: str = None,
        **kwargs,
    ) -> str:
        if not self.checkpoints_available(self._en_ckpt_base, self._ckpt_converter):
            raise RuntimeError(
                "OpenVoice checkpoints not found. Download them from "
                "https://myshell-public-repo-hosting.s3.amazonaws.com/checkpoints_1226.zip "
                "and extract to the 'checkpoints/' directory."
            )

        self._load_models()

        source_se = self._default_se if style == "default" else self._style_se
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        tmp_path = output_path.replace(".wav", "_tmp.wav")

        self._base_tts.tts(text, tmp_path, speaker=style, language="English")

        if reference_audio and os.path.isfile(reference_audio):
            target_se, _ = self._se_extractor.get_se(
                reference_audio, self._converter, target_dir="processed", vad=True
            )
            self._converter.convert(
                audio_src_path=tmp_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path,
                message="@MyShell",
            )
        else:
            shutil.copy(tmp_path, output_path)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return output_path

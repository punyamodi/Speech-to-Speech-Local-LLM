import torch
import numpy as np
import re
import soundfile
import os
import librosa
from .text import text_to_sequence
from .mel_processing import spectrogram_torch
from .models import SynthesizerTrn
from . import utils
from . import commons


class OpenVoiceBaseClass:
    def __init__(self, config_path, device="cuda:0"):
        if "cuda" in device:
            if not torch.cuda.is_available():
                device = "cpu"

        hps = utils.get_hparams_from_file(config_path)

        model = SynthesizerTrn(
            len(getattr(hps, "symbols", [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.hps = hps
        self.device = device

    def load_ckpt(self, ckpt_path):
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint_dict["model"], strict=False)


class BaseSpeakerTTS(OpenVoiceBaseClass):
    language_marks = {"english": "EN"}

    @staticmethod
    def get_text(text, hps, is_symbol):
        text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.0):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        return np.array(audio_segments).astype(np.float32)

    @staticmethod
    def split_sentences_into_pieces(text, language_str):
        return utils.split_sentence(text, language_str=language_str)

    def tts(self, text, output_path, speaker, language="English", speed=1.2):
        mark = self.language_marks.get(language.lower())
        assert mark is not None, f"language {language} is not supported"

        texts = self.split_sentences_into_pieces(text, mark)
        audio_list = []
        for t in texts:
            t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
            t = f"[{mark}]{t}[{mark}]"
            stn_tst = self.get_text(t, self.hps, False)
            speaker_id = self.hps.speakers[speaker]
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0).to(self.device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
                sid = torch.LongTensor([speaker_id]).to(self.device)
                audio = self.model.infer(
                    x_tst, x_tst_lengths, sid=sid,
                    noise_scale=0.667, noise_scale_w=0.6,
                    length_scale=1.0 / speed
                )[0][0, 0].data.cpu().float().numpy()
            audio_list.append(audio)

        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)
        if output_path is None:
            return audio
        soundfile.write(output_path, audio, self.hps.data.sampling_rate)


class ToneColorConverter(OpenVoiceBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.watermark_model = None

    def extract_se(self, ref_wav_list, se_save_path=None):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]

        hps = self.hps
        gs = []
        for fname in ref_wav_list:
            audio_ref, _ = librosa.load(fname, sr=hps.data.sampling_rate)
            y = torch.FloatTensor(audio_ref).to(self.device).unsqueeze(0)
            y = spectrogram_torch(
                y, hps.data.filter_length, hps.data.sampling_rate,
                hps.data.hop_length, hps.data.win_length, center=False
            ).to(self.device)
            with torch.no_grad():
                g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                gs.append(g.detach())
        gs = torch.stack(gs).mean(0)

        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(gs.cpu(), se_save_path)
        return gs

    def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3, message="default"):
        hps = self.hps
        audio, _ = librosa.load(audio_src_path, sr=hps.data.sampling_rate)
        audio = torch.tensor(audio).float()

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device).unsqueeze(0)
            spec = spectrogram_torch(
                y, hps.data.filter_length, hps.data.sampling_rate,
                hps.data.hop_length, hps.data.win_length, center=False
            ).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            audio = self.model.voice_conversion(
                spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau
            )[0][0, 0].data.cpu().float().numpy()

        if output_path is None:
            return audio
        soundfile.write(output_path, audio, hps.data.sampling_rate)

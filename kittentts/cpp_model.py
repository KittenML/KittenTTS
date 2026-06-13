import os

import espeakng_loader
import numpy as np
import phonemizer
import soundfile as sf
from phonemizer.backend.espeak.wrapper import EspeakWrapper

from .preprocess import TextPreprocessor, chunk_text, normalize_text


EspeakWrapper.set_library(espeakng_loader.get_library_path())
os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()

SAMPLE_RATE = 24000
STYLE_DIM = 256
DEFAULT_VOICE = "expr-voice-5-m"
DEFAULT_NATIVE_VOICE = "expr-voice-2-f"

DEFAULT_VOICE_ALIASES = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}


def basic_english_tokenize(text):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re

    return re.findall(r"\w+|[^\w\s]", text)


class TextCleaner:
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        self.word_index_dictionary = {symbols[i]: i for i in range(len(symbols))}

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes


def _import_model_inference():
    try:
        import model_inference as mi
    except ImportError as exc:
        raise ImportError(
            "KittenTTS native inference requires the kitten-inference wheel. "
            "Install the wheel that matches this Python/platform before loading a model."
        ) from exc
    return mi


def _coerce_voice_pack(values):
    pack = np.asarray(values, dtype=np.float32)
    if pack.size % STYLE_DIM != 0:
        raise ValueError(
            f"Voice style data has {pack.size} floats, which is not divisible by {STYLE_DIM}."
        )
    return pack.reshape(-1, STYLE_DIM)


def _load_voice_bin(path):
    return _coerce_voice_pack(np.fromfile(path, dtype=np.float32))


class KittenTTS_1_Cpp:
    def __init__(
        self,
        arch_path,
        weights_path,
        voices_path=None,
        voice_paths=None,
        speed_priors=None,
        voice_aliases=None,
        backend=None,
        style_index="token_count",
        sample_rate=SAMPLE_RATE,
    ):
        """Initialize KittenTTS with the native model_inference engine."""
        self.arch_path = arch_path
        self.weights_path = weights_path
        self.sample_rate = sample_rate
        self.style_index = style_index

        aliases = dict(DEFAULT_VOICE_ALIASES)
        aliases.update(voice_aliases or {})
        self.voice_aliases = aliases
        self.speed_priors = speed_priors or {}
        self.preprocessor = TextPreprocessor(remove_punctuation=False)
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.text_cleaner = TextCleaner()

        self._voices_npz = None
        self._voice_packs = {}
        self._load_voices(voices_path, voice_paths or {})
        if self._voices_npz is None and not self._voice_packs:
            raise ValueError(
                "Native KittenTTS config must provide voices as a .npz file or voice_files."
            )

        self.available_voices = sorted(self._available_voice_names())
        self.all_voice_names = [
            alias for alias, target in self.voice_aliases.items()
            if target in self.available_voices
        ] or self.available_voices

        self._mi = _import_model_inference()
        selected_backend = backend or "cpu"
        if selected_backend == "amd_gpu":
            raise ValueError("The native engine does not support backend='amd_gpu'.")
        if not self._mi.set_backend(selected_backend):
            raise ValueError(
                f"Backend {selected_backend!r} is not available in the installed kitten-inference wheel."
            )
        self.session = self._mi.InferenceModel(arch_path, weights_path)

    def _load_voices(self, voices_path, voice_paths):
        if voices_path:
            if str(voices_path).endswith(".npz"):
                self._voices_npz = np.load(voices_path)
            else:
                self._voice_packs[DEFAULT_NATIVE_VOICE] = _load_voice_bin(voices_path)

        for voice, path in voice_paths.items():
            self._voice_packs[voice] = _load_voice_bin(path)

    def _available_voice_names(self):
        names = set(self._voice_packs.keys())
        if self._voices_npz is not None:
            names.update(self._voices_npz.files)
        return names

    def _resolve_voice(self, voice):
        voice = self.voice_aliases.get(voice, voice)
        if voice not in self.available_voices:
            choices = sorted(set(self.all_voice_names) | set(self.available_voices))
            raise ValueError(f"Voice {voice!r} not available. Choose from: {choices}")
        return voice

    def _phoneme_ids(self, text):
        phonemes_list = self.phonemizer.phonemize([text])
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = " ".join(phonemes)
        tokens = self.text_cleaner(phonemes)
        tokens.insert(0, 0)
        tokens.append(0)
        return np.asarray(tokens, dtype=np.float32)

    def _style_row(self, text, n_ids, rows):
        if rows <= 0:
            return 0
        if self.style_index == "text_length":
            return min(len(text), rows - 1)
        if self.style_index == "phoneme_count":
            return min(max(0, n_ids - 2), rows - 1)
        if self.style_index != "token_count":
            raise ValueError(
                "style_index must be one of: 'token_count', 'phoneme_count', 'text_length'."
            )
        return min(max(0, n_ids), rows - 1)

    def _style_for_voice(self, voice, text, n_ids):
        if voice in self._voice_packs:
            pack = self._voice_packs[voice]
        elif self._voices_npz is not None and voice in self._voices_npz:
            pack = _coerce_voice_pack(self._voices_npz[voice])
        else:
            raise ValueError(f"Voice {voice!r} has no style data.")

        row = self._style_row(text, n_ids, pack.shape[0])
        style = pack[row]
        return style[:128].copy(), style[128:].copy()

    def _effective_speed(self, voice, speed):
        return float(speed) * float(self.speed_priors.get(voice, 1.0))

    def _set_native_speed(self, speed):
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed must be a positive finite value")

        set_speed = getattr(self.session, "set_speed", None)
        if set_speed is None:
            if np.isclose(speed, 1.0):
                return
            raise RuntimeError(
                "The installed kitten-inference wheel does not support speed control. "
                "Install the latest kitten-inference wheel for this Python/platform."
            )
        set_speed(speed)

    def normalize_text(self, text, locale="en-US", return_spans=False):
        return normalize_text(text, locale=locale, return_spans=return_spans)

    def generate(self, text, voice=DEFAULT_VOICE, speed=1.0, clean_text=True):
        out_chunks = []
        if clean_text:
            text = self.preprocessor(text)
        for text_chunk in chunk_text(text):
            out_chunks.append(self.generate_single_chunk(text_chunk, voice, speed))
        if not out_chunks:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(out_chunks, axis=-1)

    def generate_stream(self, text, voice=DEFAULT_VOICE, speed=1.0, clean_text=True):
        if clean_text:
            text = self.preprocessor(text)
        for text_chunk in chunk_text(text):
            yield self.generate_single_chunk(text_chunk, voice, speed)

    def generate_single_chunk(self, text, voice=DEFAULT_VOICE, speed=1.0):
        voice = self._resolve_voice(voice)
        effective_speed = self._effective_speed(voice, speed)
        self._set_native_speed(effective_speed)

        ids = self._phoneme_ids(text)
        style_dec, style_pred = self._style_for_voice(voice, text, len(ids))
        outputs = self.session.forward(
            {
                "phoneme_ids": ids,
                "style_dec": style_dec,
                "style_pred": style_pred,
            }
        )
        return np.asarray(outputs["t_audio"], dtype=np.float32).reshape(-1)

    def generate_to_file(
        self,
        text,
        output_path,
        voice=DEFAULT_VOICE,
        speed=1.0,
        sample_rate=SAMPLE_RATE,
        clean_text=True,
    ):
        audio = self.generate(text, voice, speed, clean_text=clean_text)
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")

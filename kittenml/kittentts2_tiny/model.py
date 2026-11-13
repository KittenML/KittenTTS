"""KittenTTS 2 nano / pico / mini: small multilingual TTS models, run with ONNX Runtime (int8 weights, the
default, CPU-first) or with PyTorch (the float32 weights).

A repository is one model family. `kitten-tts-2-nano` and `kitten-tts-2-pico` hold one model per language,
`kitten-tts-2-mini` one model for every language; config.json says which files serve which language:

    {"type": "KITTEN2_TINY", "phonemizer": "v1", "vocab": "vocab.json", "g2p_dir": "g2p",
     "languages": {"en": {"model": "kitten_tts_2_nano_en", "model_file": "kitten_tts_2_nano_en.onnx",
                          "weights_file": "kitten_tts_2_nano_en.safetensors", "voices": "voices_en.npz"}, ...},
     "models": {"kitten_tts_2_nano_en": {"architecture": {...}, ...}, ...},
     "voices": {"en": [{"key": "eleanor", "name": "Eleanor", "description": "Somber historian (female)"}, ...]},
     ...}

Each voice is a pack of style vectors, one per input length (float32 [510, 256]); a chunk uses the row of its
token count. Both runtimes compute `input_ids [1, T], style [1, 256], speed [1] -> waveform, durations` and
synthesis follows the training recipe: tokens [0] + phonemes + [10] + [0]; the start pause of the first chunk is
kept, the end token's frames are cut, 5 ms fades, 0.15 s of silence between chunks.
"""
from __future__ import annotations

import os

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from ..repo import resolve_repo

SAMPLE_RATE = 24000
GAP_S = 0.15
END_TOKEN = 10
PROVIDERS = {None: None, "cpu": ["CPUExecutionProvider"], "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
             "amd_gpu": ["ROCMExecutionProvider", "CPUExecutionProvider"]}
BACKENDS = (None, "onnx", "torch", *[k for k in PROVIDERS if k])     # "cpu" / "cuda" / "amd_gpu": ONNX on that device


class KittenTTS2Tiny:
    """A KittenTTS 2 nano / pico / mini repository.

        m = KittenTTS("KittenML/kitten-tts-2-nano")                    # ONNX Runtime, int8 weights
        m = KittenTTS("KittenML/kitten-tts-2-nano", backend="torch")   # PyTorch, float32 weights (device="cuda")
        audio = m.generate("Guten Morgen.", language="de")             # the language's first voice
        audio = m.generate("Hello there.", voice="Eleanor")             # a name (any case) or a description
        m.voices("de")                                                  # [{"name", "description", ...}, ...]
    """

    def __init__(self, model_name="KittenML/kitten-tts-2-nano", cache_dir=None, backend=None, device=None,
                 hf_token=None, *, vllm_options=None, mlx_options=None, threads=None):
        if vllm_options is not None:
            raise ValueError("the vLLM backend requires a KittenTTS 2 model")
        if mlx_options is not None:
            raise ValueError("the MLX backend requires a KittenTTS 2 model")
        if backend not in BACKENDS:
            raise ValueError(f"Unsupported backend {backend!r}; expected one of {list(BACKENDS)}")
        local_dir, repo_id, config = getattr(self, "_resolved", None) or resolve_repo(model_name, cache_dir)
        self.config, self.local_dir, self.repo_id = config, local_dir, repo_id
        self.cache_dir, self.hf_token = cache_dir, hf_token
        self.runtime = "torch" if backend == "torch" else "onnx"
        if self.runtime == "onnx":
            onnx_device = backend if backend not in (None, "onnx") else device
            if onnx_device not in PROVIDERS:
                raise ValueError(f"Unsupported ONNX device {onnx_device!r}; expected one of {list(PROVIDERS)}")
            self.providers, self.device = PROVIDERS[onnx_device], onnx_device
        else:
            self.providers, self.device = None, device
        self.threads = threads
        self.sample_rate = int(config.get("sample_rate", SAMPLE_RATE))
        self.languages = list(config["languages"])
        self.default_language = config.get("default_language", self.languages[0])
        self._voices = {lang: list(vs) for lang, vs in config["voices"].items()}
        import json
        with open(self._fetch(config.get("vocab", "vocab.json")), encoding="utf-8") as f:
            self.vocab = json.load(f)
        self._frontend = None
        self._sessions, self._torch, self._packs = {}, {}, {}

    # ---- files ------------------------------------------------------------------------------------
    def _fetch(self, filename):
        if self.local_dir is not None:
            return os.path.join(self.local_dir, filename)
        return hf_hub_download(repo_id=self.repo_id, filename=filename, cache_dir=self.cache_dir,
                               token=self.hf_token)

    def _g2p_dir(self):
        d = self.config.get("g2p_dir", "g2p")
        if self.local_dir is not None:
            return os.path.join(self.local_dir, d)
        root = snapshot_download(repo_id=self.repo_id, cache_dir=self.cache_dir, token=self.hf_token,
                                 allow_patterns=[f"{d}/*", f"{d}/**"])
        return os.path.join(root, d)

    @property
    def frontend(self):
        if self._frontend is None:
            from .frontend import Frontend
            self._frontend = Frontend(self.config.get("phonemizer", "v1"), self.vocab, self._g2p_dir())
        return self._frontend

    def _session(self, lang):
        import onnxruntime as ort
        path = self.config["languages"][lang]["model_file"]
        if path not in self._sessions:
            so = ort.SessionOptions()
            if self.threads:
                so.intra_op_num_threads = self.threads
            kw = {} if self.providers is None else {"providers": self.providers}
            self._sessions[path] = ort.InferenceSession(self._fetch(path), so, **kw)
        return self._sessions[path]

    def _torch_model(self, lang):
        entry = self.config["languages"][lang]
        path = entry.get("weights_file")
        if path is None:
            raise ValueError(f"{self.config.get('name', 'this repository')} has no PyTorch weights for {lang!r}")
        if path not in self._torch:
            from .torch_model import TorchModel
            arch = self.config["models"][entry["model"]]["architecture"]
            self._torch[path] = TorchModel(arch, self._fetch(path), self.device, self.threads)
        return self._torch[path]

    def _run(self, lang, tokens, style, speed):
        """tokens -> (waveform float32, durations int64) with the selected runtime."""
        if self.runtime == "torch":
            return self._torch_model(lang)(tokens, style, speed)
        wav, dur = self._session(lang).run(None, {
            "input_ids": np.array([tokens], dtype=np.int64), "style": style.astype(np.float32),
            "speed": np.array([speed], dtype=np.float32)})
        return wav, dur

    def _pack(self, lang, key):
        path = self.config["languages"][lang]["voices"]
        if path not in self._packs:
            with np.load(self._fetch(path)) as z:
                self._packs[path] = {k: z[k].astype(np.float32) for k in z.files}
        return self._packs[path][key]

    # ---- voices -----------------------------------------------------------------------------------
    @property
    def available_voices(self):
        """Voice names, per language: {language: [name, ...]}."""
        return {lang: [v["name"] for v in vs] for lang, vs in self._voices.items()}

    def voices(self, language=None):
        """The voices of one language (default: all), as dicts with "key", "name", "description" and "language"."""
        langs = [self._language(language)] if language else self.languages
        return [dict(v, language=lang) for lang in langs for v in self._voices[lang]]

    def _language(self, language):
        if language is None:
            return self.default_language
        if language not in self.config["languages"]:
            raise ValueError(f"{self.config.get('name', 'this model')} does not speak {language!r}; "
                             f"languages: {self.languages}")
        return language

    def resolve_voice(self, voice=None, language=None):
        """-> (language, voice key). `voice` is a name in any case ("Anna", "anna"), a description, or None
        (the language's first voice); without `language`, a name that only one language has decides it,
        and a name several languages share means the default language's voice."""
        if voice is None:
            lang = self._language(language)
            return lang, self._voices[lang][0]["key"]
        langs = [self._language(language)] if language else self.languages
        want = str(voice).casefold()
        hits = [(lang, v["key"]) for lang in langs for v in self._voices[lang]
                if want in {str(v.get(f, "")).casefold() for f in ("key", "name", "description", "id") if f in v}]
        if not hits:
            raise ValueError(f"no voice {voice!r}" + (f" for {language!r}" if language else "")
                             + f"; voices: {self.available_voices if not language else self.available_voices[langs[0]]}")
        if len(hits) > 1:
            if language is None and any(lang == self.default_language for lang, _ in hits):
                return next(h for h in hits if h[0] == self.default_language)
            raise ValueError(f"voice {voice!r} exists in {[lang for lang, _ in hits]}: pass language=")
        return hits[0]

    # ---- synthesis --------------------------------------------------------------------------------
    def phonemize(self, text, language=None):
        """text -> the phoneme chunks the model will speak."""
        return self.frontend(text, self._language(language))[0]

    def _tokens(self, phonemes):
        ids = []
        for ch in phonemes:
            if ch in self.vocab:
                ids.append(self.vocab[ch])
            elif ch in "()":
                ids.append(self.vocab[","])
        return [0] + ids + [END_TOKEN, 0]

    def _chunk(self, lang, key, phonemes, speed, keep_start):
        tok = self._tokens(phonemes)
        pack = self._pack(lang, key)
        style = pack[min(len(tok) - 1, pack.shape[0] - 1)][None]
        wav, dur = self._run(lang, tok, style, speed)
        w = np.clip(wav.astype(np.float32), -1, 1)
        total = int(dur.sum())
        spf = max(1, round(len(w) / max(total, 1)))             # samples per duration frame
        lead = 0 if keep_start else max(0, int(dur[0]) - 1) * spf
        trail = max(0, int(dur[-1]) - 1) * spf
        if lead + trail + spf < len(w):
            w = w[lead: len(w) - trail] if trail else w[lead:]
        w = np.ascontiguousarray(w)
        f = min(120, len(w) // 8)                                # ~5 ms fades
        if f:
            w[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            w[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
        return w

    def generate_stream(self, text, voice=None, language=None, speed=1.0):
        """Yield the audio of each chunk (long text is cut at sentence ends) as it is produced."""
        lang, key = self.resolve_voice(voice, language)
        for i, ps in enumerate(self.frontend(text, lang)[0]):
            yield self._chunk(lang, key, ps, float(speed), keep_start=(i == 0))

    def generate(self, text, voice=None, language=None, speed=1.0):
        """Synthesize `text`. Returns float32 audio at `self.sample_rate` (24 kHz).

        Args:
            text: what to say.
            voice: a voice name from `voices()` (any case); default: the language's first voice.
            language: a code from `languages`; default: the voice's language, else "en".
            speed: > 1 speaks faster.
        """
        gap = np.zeros(int(GAP_S * self.sample_rate), dtype=np.float32)
        out = []
        for w in self.generate_stream(text, voice, language, speed):
            out += [w, gap]
        return np.concatenate(out[:-1]) if out else np.zeros(0, dtype=np.float32)

    def generate_to_file(self, text, output_path, voice=None, language=None, speed=1.0, sample_rate=None):
        import soundfile as sf
        audio = self.generate(text, voice=voice, language=language, speed=speed)
        sf.write(str(output_path), audio, sample_rate or self.sample_rate)
        return str(output_path)

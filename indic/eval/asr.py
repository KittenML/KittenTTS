"""The recognizers every intelligibility number is measured through.

TTS intelligibility is scored by asking an ASR model what it heard: synthesize the
reference text, transcribe the result, measure the distance. That makes the
recognizer part of the measuring instrument, and this instrument is not neutral
across languages — IndicConformer's own error rate on Telugu is several times its
rate on Hindi. So a bare CER of 18% says nothing until you know what the *human*
recording of the same sentence scores through the same model. That is why
`testsets.py` carries the ground-truth wav and `score.py` reports the floor in the
same row, and why both passes go through this one module: if the floor and the TTS
number came from different decoders, different language codes or different
resampling, subtracting one from the other would be meaningless.

Two backends, for different jobs:

  * `IndicConformer` (ai4bharat/indic-conformer-600m-multilingual) is the primary
    scorer for hi/bn/ta/te/mr. It is trained on Indian speech, it writes native
    script, and it is what the IndicTTS/IndicF5 line of work scores against — so
    our numbers are at least comparable in kind to published ones.
  * `Whisper` (openai/whisper-large-v3) scores English and cross-checks the Indic
    languages. Whisper's Indic error is high enough that it would flatter a bad
    model and punish a good one, so it is never the headline for the five; it is
    here to catch the case where a CER regression is IndicConformer having an
    opinion rather than the model getting worse.

Nothing is downloaded at import or construction. A scoring run shards across many
processes and most of them need only one backend, so the weights load on the first
`transcribe`.

AI4Bharat's repositories are gated (auto-approved): accept the terms on the model
page once and export `HF_TOKEN`, or every download is a 401. AI4Bharat now points
at a newer `indic-transcribe` model; IndicConformer stays the default here because
switching the scorer invalidates every number taken before the switch — change it
deliberately, as a new `--asr`, not as an upgrade.
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.langs import LANGUAGES

# Both backends are 16 kHz models. Feeding 24 kHz TTS output at its native rate
# does not error, it just decodes a sped-up clip, so the resample is not optional.
SAMPLE_RATE = 16000


def load_audio(source, sample_rate=SAMPLE_RATE):
    """A clip as a mono float32 array at `sample_rate`.

    `source` is a path, or an array that is *already* at `sample_rate` — a caller
    holding a waveform got it from a model whose rate it knows, and guessing from
    the array would be a silent way to score the wrong thing. 2-D arrays are
    channels-first, as torch and `speaker.py` use them.

    Reading and resampling follow `kittenml/kittentts2/speaker.py:_load_waveform`
    deliberately: a different resampler shifts a speaker embedding slightly, and
    `metrics.speaker_similarity` compares clips that were read by both paths.
    """
    import numpy as np

    if not isinstance(source, (str, pathlib.Path)):
        wav = np.asarray(source, dtype=np.float32)
        return wav.mean(axis=0) if wav.ndim > 1 else wav

    import soundfile as sf
    audio, file_rate = sf.read(str(source), dtype="float32", always_2d=True)
    wav = audio.T.mean(axis=0)
    if int(file_rate) != sample_rate:
        import torch
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)), int(file_rate), sample_rate).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


class IndicConformer:
    """AI4Bharat's multilingual Conformer, CTC or RNNT decoded.

    The released checkpoint runs its encoder through onnxruntime behind
    `trust_remote_code`, and is called as `model(wav, lang, decoding)` with a
    (1, samples) tensor at 16 kHz.

    `decoding` is a real fork in the number, not a performance knob: the RNNT head
    has a language model baked into its predictor and reads numbers and rare words
    more fluently than the CTC head, so the two disagree by whole points of CER.
    RNNT is the default and `describe()` records which was used, so a CER measured
    one way is never quietly compared with one measured the other.
    """

    MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
    # The checkpoint's own language set. These are ISO 639-1 codes — the same key
    # `LANGUAGES` uses — so no translation table is needed, unlike Whisper. English
    # is absent, which is the whole reason the English arm is scored by Whisper.
    SUPPORTED = frozenset(
        "as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur".split())

    def __init__(self, device="cpu", decoding="rnnt", model_id=MODEL_ID,
                 hf_token=None, cache_dir=None):
        if decoding not in ("rnnt", "ctc"):
            raise ValueError(f"decoding must be 'rnnt' or 'ctc', got {decoding!r}")
        self.model_id = model_id
        self.decoding = decoding
        self.device = device
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        from huggingface_hub.errors import GatedRepoError
        from transformers import AutoModel

        token = (self.hf_token or os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        try:
            model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True, token=token,
                **({"cache_dir": self.cache_dir} if self.cache_dir else {}))
        except GatedRepoError as exc:
            raise RuntimeError(
                f"{self.model_id} is gated. Accept the terms on its model page and "
                "export HF_TOKEN; approval is automatic, but without the token every "
                "file download is a 401.") from exc

        if hasattr(model, "eval"):
            model.eval()
        # Device placement is best-effort: the encoder is an ONNX graph, so
        # onnxruntime chooses its own execution provider and `.to()` moves only
        # whatever torch modules the remote code happens to keep.
        try:
            model = model.to(self.device)
        except Exception:
            pass
        self._model = model
        return model

    def transcribe(self, audio, lang):
        """One clip -> its transcription in `lang`'s native script."""
        import torch

        if lang not in self.SUPPORTED:
            raise ValueError(
                f"{self.model_id} does not cover {lang!r}; score it with Whisper "
                f"(supported: {sorted(self.SUPPORTED)})")
        wav = load_audio(audio)
        # A model that generated nothing should score as a total miss, not crash the
        # run three hours in — the empty hypothesis is what CER wants anyway.
        if wav.size == 0:
            return ""
        with torch.inference_mode():
            out = self._load()(torch.from_numpy(wav).unsqueeze(0), lang, self.decoding)
        return _as_text(out)

    def transcribe_many(self, audios, lang):
        """Transcribe a list of clips, one at a time.

        Not batched on purpose. The encoder is convolutional, so padding a batch to
        the longest clip changes the activations near every shorter clip's tail, and
        the transcription of a given wav would then depend on what it was batched
        with — which breaks the one property the resumable scorer needs, that
        re-scoring an utterance reproduces its earlier row.
        """
        return [self.transcribe(a, lang) for a in audios]

    def describe(self):
        return {"backend": "indicconformer", "asr_model": self.model_id,
                "asr_decoding": self.decoding}


class Whisper:
    """openai/whisper-large-v3 through the transformers ASR pipeline.

    Built the same way `kittenml/kittentts2/speaker.py:ReferenceTranscriber` builds
    it, so the clone-time transcript and the scoring transcript come from one
    configuration — except that this one is told the language, which that one is
    not (it hardcodes English, which is a bug for an Indic reference clip).
    """

    MODEL_ID = "openai/whisper-large-v3"

    def __init__(self, device="cpu", model_id=MODEL_ID, batch_size=8,
                 hf_token=None, cache_dir=None):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline

            on_cuda = str(self.device).startswith("cuda")
            self._pipeline = pipeline(
                "automatic-speech-recognition", model=self.model_id,
                chunk_length_s=30, device=self.device,
                dtype=torch.float16 if on_cuda else torch.float32,
                token=self.hf_token or os.environ.get("HF_TOKEN"),
                model_kwargs={"cache_dir": self.cache_dir} if self.cache_dir else {})
        return self._pipeline

    def _generate_kwargs(self, lang):
        """Pinned language, greedy decoding.

        Both matter for the numbers. Left to identify the language itself, Whisper
        routinely hears a poorly synthesized Hindi clip as Urdu and writes it in
        Arabic script, and the CER then measures script choice rather than
        pronunciation. And sampling would make the ASR floor drift between the week
        the baseline was scored and the week the arm was, which is exactly the
        comparison this harness exists to protect.
        """
        return {"language": LANGUAGES[lang].whisper, "task": "transcribe",
                "num_beams": 1, "do_sample": False}

    def transcribe(self, audio, lang):
        wav = load_audio(audio)
        if wav.size == 0:
            return ""
        return self._load()(wav, generate_kwargs=self._generate_kwargs(lang))["text"].strip()

    def transcribe_many(self, audios, lang):
        """Batched: the pipeline pads and masks, so a batch decodes as singles do."""
        waves = [load_audio(a) for a in audios]
        out = self._load()(waves, batch_size=self.batch_size,
                           generate_kwargs=self._generate_kwargs(lang))
        return [r["text"].strip() for r in out]

    def describe(self):
        return {"backend": "whisper", "asr_model": self.model_id, "asr_decoding": "greedy"}


BACKENDS = {"indicconformer": IndicConformer, "whisper": Whisper}


def default_backend(lang):
    """Which recognizer owns a language by default.

    IndicConformer cannot transcribe English and Whisper is weak on the five, so
    the default is per language rather than per run. A run that wants one scorer
    across everything has to say so, because that choice changes every number.
    """
    return "whisper" if lang == "en" else "indicconformer"


def get_asr(name, **kwargs):
    if name not in BACKENDS:
        raise ValueError(f"unknown ASR backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)


def _as_text(result):
    """Flatten whatever the remote code returned into one string.

    The released IndicConformer returns a plain string for a single clip, but the
    per-language and batched variants of the same repo return a one-element list,
    and a list that reaches CER as a string scores every bracket as an error.
    """
    while isinstance(result, (list, tuple)):
        if not result:
            return ""
        result = result[0]
    return str(result).strip()

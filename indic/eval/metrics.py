"""Scoring: character and word error rates, and speaker similarity.

The arithmetic here is standard. What is not standard, and what this module exists
for, is the normalization that happens first — on Indic text the normalizer decides
more of the number than the edit distance does:

  * **Danda.** `।` is punctuation, and an ASR model emits it or does not depending
    on its training transcripts. Scored naively, a TTS that pronounced the sentence
    perfectly loses a character per sentence to a mark nobody says out loud.
  * **Digits.** The reference may hold `२०२४` and the hypothesis `2024`, or the
    other way round. Those are the same number; both go to ASCII before scoring,
    reusing `text/indic_text.py` so the eval and the TTS front end agree on what a
    digit is.
  * **ZWJ / ZWNJ.** Bengali and Devanagari conjuncts can be spelled with or without
    U+200C/U+200D. The choice is orthographic, the ASR makes it by habit, and the
    TTS never had a say — charging it would be scoring the recognizer's convention.
  * **Normalization form.** FLEURS ships mixed NFC/NFD. The same Devanagari syllable
    decomposed differently is a different character string, and an edit distance
    over it reports a loss that does not exist.

Spaces are kept and do count as characters. That is a defensible choice only
because of the floor: the human recording goes through the identical pipeline, so
whatever the recognizer does about word boundaries is already priced into the
number the TTS is compared against.

WER is reported but is the weaker metric here. Telugu and Tamil compound freely, so
a single wrong suffix destroys a whole "word", and Indic ASR word segmentation is
inconsistent. CER is the headline; WER is for spotting the case where CER stayed
flat because the model substituted a plausible wrong word.

**On MOS predictors.** Automatic MOS (UTMOS, NISQA, DNSMOS and friends) is trained
almost entirely on English listening tests. Its absolute values do not transfer to
Hindi or Tamil — the model has never heard the phone inventory, and it scores
accented or unfamiliar-prosody speech low whether or not a native listener would.
Any such number in this project is usable for ranking two of our own checkpoints on
the same text, never as a quantity to report, to compare across languages, or to
put beside a published English MOS. Human MOS with native listeners is the only
version of that claim we can actually make.
"""

import pathlib
import sys
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.text.indic_text import indic_digits_to_ascii


def normalize_for_scoring(text):
    """The canonical form both sides of an edit distance are reduced to.

    Order matters: NFC first so that composed and decomposed spellings become one
    string, then digits, then the character filter — a decomposed Devanagari digit
    would otherwise survive the digit map and be dropped as punctuation.
    """
    s = unicodedata.normalize("NFC", str(text or ""))
    s = indic_digits_to_ascii(s)
    kept = []
    for ch in s:
        category = unicodedata.category(ch)
        if ch.isspace():
            kept.append(" ")
        elif category[0] in ("P", "S", "C"):
            # P covers the danda (U+0964 is Po) along with Latin punctuation, S the
            # currency symbols an ASR never writes, C the ZWJ/ZWNJ and stray control
            # characters. Marks (Mn/Mc) are *not* touched: those are the vowel signs,
            # and dropping them would discard most of what Indic pronunciation is.
            continue
        else:
            kept.append(ch)
    return " ".join("".join(kept).split()).casefold()


def edit_distance(reference, hypothesis):
    """Levenshtein distance over any two sequences.

    Written out rather than taken from `jiwer`/`rapidfuzz` because it is fifteen
    lines, the sequences here are one sentence long, and a scoring dependency is a
    thing that has to still install the week the numbers are reproduced.
    """
    if not reference:
        return len(hypothesis)
    previous = list(range(len(reference) + 1))
    for i, h in enumerate(hypothesis, start=1):
        current = [i]
        for j, r in enumerate(reference, start=1):
            current.append(min(previous[j] + 1,          # deletion from hypothesis
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (h != r)))
        previous = current
    return previous[-1]


def error_counts(reference, hypothesis, unit="char"):
    """(errors, reference length) for one utterance, normalized first.

    Returned as counts, not a rate, because a corpus rate is total errors over
    total length — averaging per-utterance rates lets a three-word robustness item
    outweigh a forty-word read-speech one.
    """
    ref = normalize_for_scoring(reference)
    hyp = normalize_for_scoring(hypothesis)
    if unit == "word":
        ref, hyp = ref.split(), hyp.split()
    elif unit != "char":
        raise ValueError(f"unit must be 'char' or 'word', got {unit!r}")
    return edit_distance(ref, hyp), len(ref)


def rate(errors, length, hypothesis):
    """Turn counts into a rate, with the empty-reference case decided once.

    Public because `score.py` needs the counts *and* the rate and should not pay
    for the edit distance twice, and because an empty reference scored 0.0 in one
    place and 1.0 in another would quietly bend a corpus average.
    """
    if length == 0:
        return 0.0 if not normalize_for_scoring(hypothesis) else 1.0
    return errors / length


def cer(reference, hypothesis):
    """Character error rate, after Indic-aware normalization. Unbounded above."""
    errors, length = error_counts(reference, hypothesis, "char")
    return rate(errors, length, hypothesis)


def wer(reference, hypothesis):
    """Word error rate. See the module docstring on why CER leads on Indic."""
    errors, length = error_counts(reference, hypothesis, "word")
    return rate(errors, length, hypothesis)


def corpus_rate(pairs, unit="char"):
    """Total errors over total reference length across (reference, hypothesis) pairs.

    This is the number to report. The mean of per-utterance CERs is a different
    quantity and is not comparable with published corpus CERs.
    """
    errors = length = 0
    for reference, hypothesis in pairs:
        e, n = error_counts(reference, hypothesis, unit)
        errors += e
        length += n
    return errors / length if length else 0.0


class WavLMSpeakerEmbedder:
    """microsoft/wavlm-base-plus-sv — the default embedder for speaker similarity.

    Independent of anything in the TTS stack, which is the point. It is ungated,
    runs on CPU, and is the x-vector head the speaker-verification literature (and
    the SECS numbers in the XTTS/F5 papers) is reported against.
    """

    MODEL_ID = "microsoft/wavlm-base-plus-sv"

    def __init__(self, device="cpu", model_id=MODEL_ID, cache_dir=None):
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._extractor = None

    def _load(self):
        if self._model is None:
            from transformers import AutoFeatureExtractor, WavLMForXVector
            kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
            self._extractor = AutoFeatureExtractor.from_pretrained(self.model_id, **kwargs)
            self._model = WavLMForXVector.from_pretrained(self.model_id, **kwargs)
            self._model = self._model.to(self.device).eval()
        return self._extractor, self._model

    def embed(self, source):
        """A clip (path or 16 kHz array) -> a unit-norm embedding."""
        import numpy as np
        import torch

        from .asr import SAMPLE_RATE, load_audio

        wav = load_audio(source)
        extractor, model = self._load()
        inputs = extractor([wav], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           padding=True)
        with torch.inference_mode():
            vector = model(**{k: v.to(self.device) for k, v in inputs.items()}).embeddings
        vector = vector.squeeze(0).float().cpu().numpy()
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def describe(self):
        return {"speaker_embedder": self.model_id}


class KittenSpeakerEmbedder:
    """The pyannote x-vector vendored in `kittenml/kittentts2/speaker_embedding.py`.

    Available for diagnosis, and deliberately not the default. This is the exact
    model whose 512-dim output is fed to the checkpoint's `spk_proj` head, so a
    similarity computed with it is partly asking "did the conditioning vector
    survive the generation" rather than "does this sound like the speaker". A model
    that learns to satisfy this embedding without matching the voice scores well
    here and badly anywhere else, and an Indic fine-tune is exactly the setting
    where that divergence would appear. Use it to check that cloning is wired up
    at all; report similarity from `WavLMSpeakerEmbedder`.

    It also needs a path rather than an array: `SpeakerEncoder` owns its own file
    reading, and routing an array through a temporary file would read it back with
    a different resampler than the one it was trained against.
    """

    def __init__(self, weights_path, device="cpu"):
        self.weights_path = str(weights_path)
        self.device = device
        self._encoder = None

    def embed(self, source):
        if not isinstance(source, (str, pathlib.Path)):
            raise TypeError("KittenSpeakerEmbedder takes a path, not an array "
                            "(see the class docstring)")
        if self._encoder is None:
            from kittenml.kittentts2.speaker import SpeakerEncoder
            self._encoder = SpeakerEncoder(self.weights_path, device=self.device)
        return self._encoder(str(source))       # already unit-norm

    def describe(self):
        return {"speaker_embedder": f"kitten-pyannote:{self.weights_path}"}


def speaker_similarity(wav_a, wav_b, embedder=None):
    """Cosine similarity between two clips' speaker embeddings, in [-1, 1].

    With no embedder, a `WavLMSpeakerEmbedder` is built and thrown away — fine for
    a one-off, wasteful in a loop, so `score.py` passes one in and keeps it.

    The number is only interpretable against its own distribution: compare it to
    the similarity between two *different* real recordings of the same speaker
    (the ceiling) and between two different speakers (the floor), both of which
    FLEURS can supply. On its own, 0.78 means nothing.
    """
    import numpy as np

    embedder = embedder or WavLMSpeakerEmbedder()
    a = np.asarray(embedder.embed(wav_a), dtype=np.float64)
    b = np.asarray(embedder.embed(wav_b), dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else 0.0

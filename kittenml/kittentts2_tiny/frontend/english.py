#!/usr/bin/env python
# Copied from the Kitten training code (g2p_noespeak.py). Vendored into kittenml: only import paths and asset paths changed.
"""English G2P for the model's phoneme vocab with no espeak anywhere in the process.

Training phonemised with misaki's lexicon plus an espeak-ng fallback for the words the
lexicon lacks (phonemize_multi.get_g2p). This is the same lexicon path with only the
fallback swapped, so a sentence made of lexicon words gives the identical string --
checked on 1,628 corpus transcripts: 0 rows differ unless a fallback fired.

The fallback is NOT rare: it fires in 37.8% of those rows (1.75% of words), mostly on
names misaki's dictionaries lack ("Kevin", "Paul", "Diego"). Two stages replace espeak:

  1. CMUdict, converted to misaki's alphabet by build_cmu_lexicon.py. Hand-curated, so
     names come out right. Holds 58% of the fallback word tokens.
  2. PeterReid/graphemes_to_phonemes_en_us for the rest: a 0.75 M-parameter
     character-level BART, Apache-2.0, trained on misaki's own dictionaries, so it
     emits misaki's alphabet directly. It guesses names badly on its own ("Kevin" ->
     kˈAvɪn, "Klein" -> klˈAn), which is why it is the second stage and not the first.

Whatever still cannot be said -- other scripts, emoji, bare symbols -- is skipped and
REPORTED in G2PResult, never dropped silently.

block_espeak() makes "no espeak" a checked property instead of a claim: it fails if
espeak is already loaded and makes any later import of it raise. assert_no_espeak()
re-checks at any point, including for the shared library itself.

"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field

import os

# The CMUdict lexicon and the BART live in the model repository: g2p/cmu_misaki_us.json, g2p/g2p_bart_en_us/

ESPEAK_MODULES = ("espeakng_loader", "phonemizer", "misaki.espeak")
DIGITS = "zero one two three four five six seven eight nine".split()
# Latin letters NFKD does not decompose; without this "Søren" reaches the lookup as
# "S" and "ren".
TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "Æ": "AE",
                          "œ": "oe", "Œ": "OE", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
                          "’": "'"})


def _is_espeak(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in ESPEAK_MODULES)


def block_espeak() -> None:
    """Fail if espeak is already in the process, and stop it from ever being imported.

    A None entry in sys.modules makes `import x` raise while find_spec(x) returns None,
    so an availability probe -- transformers runs one for phonemizer at import -- sees
    "not installed" instead of crashing."""
    assert_no_espeak()
    for m in ESPEAK_MODULES:
        sys.modules[m] = None


def assert_no_espeak() -> None:
    """No espeak python module and no espeak shared library mapped into this process."""
    loaded = sorted(m for m in sys.modules if _is_espeak(m) and sys.modules[m] is not None)
    if loaded:
        raise RuntimeError(f"espeak modules are imported: {loaded}")
    libs = sorted({ln.split()[-1] for ln in open("/proc/self/maps")
                   if "espeak" in ln.lower()})
    if libs:
        raise RuntimeError(f"espeak shared libraries are loaded: {libs}")


class Fallback:
    """Drop-in for misaki's `fallback=` hook: called with a token misaki's lexicon
    cannot say, returns (phonemes, rating). CMUdict first, the BART for the rest."""

    RATING = 2          # what misaki's own espeak fallback reports for its output

    def __init__(self, model_dir: str, cmu_lexicon: str):
        import torch
        from transformers import BartForConditionalGeneration
        self._torch = torch
        with open(cmu_lexicon, encoding="utf-8") as f:
            self.cmu = json.load(f)
        self.model = BartForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True).eval()
        cfg = self.model.config
        # Both strings open with four "_" placeholders for <pad> <s> </s> <unk>, so a
        # character's index IS its token id.
        self.g2i = {c: i for i, c in enumerate(cfg.grapheme_chars) if i > 3}
        self.i2p = {i: c for i, c in enumerate(cfg.phoneme_chars) if i > 3}
        self.max_len = cfg.max_position_embeddings
        letters = re.escape("".join(self.g2i))
        self._pieces = re.compile(rf"(\d+)|([{letters}]+)|(.)", re.S)
        self._memo: dict[str, str] = {}
        self.used: list[tuple[str, str, str]] = []   # (word, phonemes, source)
        self.skipped: list[str] = []                 # text with nothing sayable in it

    def reset(self) -> None:
        self.used, self.skipped = [], []

    def _bart(self, word: str) -> str:
        if word not in self._memo:
            torch, out = self._torch, []
            step = self.max_len - 2               # room for <s> and </s>
            for i in range(0, len(word), step):
                ids = torch.tensor([[1] + [self.g2i[c] for c in word[i:i + step]] + [2]])
                with torch.no_grad():
                    # generate() defaults to max_length=20, which cuts long words short
                    gen = self.model.generate(input_ids=ids, max_length=self.max_len,
                                              num_beams=1, do_sample=False)
                out.append("".join(self.i2p.get(int(t), "") for t in gen[0]))
            # same final substitution misaki applies to its own lexicon output
            self._memo[word] = "".join(out).replace("ɾ", "T").replace("ʔ", "t")
        return self._memo[word]

    def _cmu(self, word: str):
        w = word.lower()
        if w in self.cmu:
            return self.cmu[w]
        if w.endswith("'s") and w[:-2] in self.cmu:      # possessive, misaki's -s rule
            stem = self.cmu[w[:-2]]
            return stem + ("s" if stem[-1] in "ptkfθ" else
                           "ᵻz" if stem[-1] in "szʃʒʧʤ" else "z")
        return None

    def _word(self, word: str, sources: set) -> str:
        """One spelled word: CMUdict whole, else its hyphen parts, else the BART."""
        ps = self._cmu(word)
        if ps is None and "-" in word:
            return " ".join(self._word(p, sources) for p in word.split("-") if p)
        sources.add("CMUdict" if ps else "BART")
        return ps or self._bart(word)

    def say(self, text: str):
        # fold accents and the typographic apostrophe so "café" and "it’s" stay sayable
        t = unicodedata.normalize("NFKD", text.translate(TRANSLIT))
        t = "".join(c for c in t if not unicodedata.combining(c))
        out, sources = [], set()
        for digits, letters, other in self._pieces.findall(t):
            if digits:
                out.append(" ".join(self._word(DIGITS[int(d)], sources) for d in digits))
            elif letters.strip("'-."):
                out.append(self._word(letters.strip("'-."), sources))
            elif other.strip():
                self.skipped.append(other)
        return " ".join(p for p in out if p), "+".join(sorted(sources))

    def __call__(self, token):
        ps, source = self.say(token.text)
        if not ps:
            self.skipped.append(token.text)
            return None, None
        self.used.append((token.text, ps, source))
        return ps, self.RATING


@dataclass
class G2PResult:
    text: str                                   # after tts_norm, what the G2P saw
    phonemes: str                               # only symbols the model has a row for
    fallback: list = field(default_factory=list)   # (word, phonemes, source)
    skipped: list = field(default_factory=list)    # text nothing could pronounce
    dropped: list = field(default_factory=list)    # phoneme symbols outside the vocab


class EnglishG2P:
    """tts_norm -> misaki lexicon -> CMUdict -> BART -> the model's phoneme vocab."""

    def __init__(self, vocab: dict, g2p_dir: str):
        from . import phonemize_v1 as pm
        from misaki import en
        from .tts_norm import normalize
        self._pm, self._normalize = pm, normalize
        self.vocab = dict(vocab)
        self.fallback = Fallback(os.path.join(g2p_dir, "g2p_bart_en_us"), os.path.join(g2p_dir, "cmu_misaki_us.json"))
        # unk="" because an unsayable token is reported through G2PResult.skipped
        # rather than marked in the phoneme string with a symbol the vocab lacks.
        self.g2p = en.G2P(trf=False, british=False, fallback=self.fallback, unk="")
        self._unsaid: list[str] = []

    def _run(self, text: str):
        ps, tokens = self.g2p(text)
        self._unsaid = [tk.text for tk in tokens if tk.phonemes is None]
        return ps, tokens

    def __call__(self, text: str, normalise: bool = True) -> G2PResult:
        text = self._normalize(text) if normalise else text
        self.fallback.reset()
        # phonemize_lang applies the same FOLD table the training lists went through
        ps = self._pm.phonemize_lang(self._run, text, "en")
        dropped = sorted({c for c in ps if c not in self.vocab})
        if dropped:
            ps = "".join(c for c in ps if c in self.vocab)
        skipped = list(dict.fromkeys(self.fallback.skipped + self._unsaid))
        return G2PResult(text, ps, list(self.fallback.used), skipped, dropped)

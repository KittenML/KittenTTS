"""Text -> phoneme chunks for the KittenTTS 2 nano / pico / mini models, exactly as in their training.

English: tts_norm text normalisation -> misaki lexicon -> CMUdict -> a small BART G2P for the words neither
knows (english.EnglishG2P), with the v1 vocab; the mini (phonemizer v2) then applies phonemize_v2.fold_extra.
Other languages: espeak-ng through misaki's EspeakG2P (misaki's own G2P for Japanese), phonemizer v1 (nano,
pico) or v2 (mini), no text normalisation; symbols outside the vocab are dropped. Text of more than MAX_PHON
phonemes is cut at sentence ends (then at spaces) into chunks.
"""
from __future__ import annotations

import re

from .english import EnglishG2P, G2PResult

MAX_PHON = 350                   # p90 phoneme length of the training lists
SENT_MULTI = re.compile(r"(?<=[.!?…।])\s+|(?<=[。！？])")


class LangG2P:
    """One non-English language as in training."""

    def __init__(self, lang, phon, vocab):
        from . import phonemize_v1, phonemize_v2
        self.lang, self.mod = lang, (phonemize_v1 if phon == "v1" else phonemize_v2)
        self.vocab = vocab
        self.g2p = self.mod.get_g2p(lang)

    def __call__(self, text, normalise=True):
        ps = self.mod.phonemize_lang(self.g2p, text, self.lang)
        dropped = sorted({c for c in ps if c not in self.vocab})
        if dropped:
            ps = "".join(c for c in ps if c in self.vocab)
        return G2PResult(text, ps, [], [], dropped)


def _wrap(ps):
    """Cut one over-long phoneme string at spaces."""
    while len(ps) > MAX_PHON:
        cut = ps.rfind(" ", 0, MAX_PHON)
        cut = cut if cut > 0 else MAX_PHON
        yield ps[:cut]
        ps = ps[cut:].strip()
    if ps:
        yield ps


def front_end(g2p, text):
    """text -> (phoneme chunks, G2PResult). Text of up to MAX_PHON phonemes is one G2P call."""
    whole = g2p(text)
    if len(whole.phonemes) <= MAX_PHON:
        return ([whole.phonemes] if whole.phonemes else []), whole
    chunks, cur = [], ""
    rep = dict(fallback=[], skipped=[], dropped=[])
    for sent in SENT_MULTI.split(whole.text):
        if not sent.strip():
            continue
        r = g2p(sent, normalise=False)
        for k in rep:
            rep[k] += [x for x in getattr(r, k) if x not in rep[k]]
        for ps in _wrap(r.phonemes):
            if cur and len(cur) + 1 + len(ps) > MAX_PHON:
                chunks.append(cur)
                cur = ps
            else:
                cur = f"{cur} {ps}".strip()
    if cur:
        chunks.append(cur)
    whole.phonemes = " | ".join(chunks)
    whole.fallback, whole.skipped, whole.dropped = (rep[k] for k in rep)
    return chunks, whole


class Frontend:
    """The text front end of one model repository: its phonemizer version and vocab, and (for English) the
    CMUdict lexicon and BART G2P in <repo>/g2p. G2P objects are built on first use of each language."""

    def __init__(self, phonemizer_version, vocab, g2p_dir):
        self.phon, self.vocab, self.g2p_dir = phonemizer_version, dict(vocab), g2p_dir
        self._g2p = {}

    def g2p(self, lang):
        if lang not in self._g2p:
            if lang == "en":
                # English always filters with the v1 symbol set, as in training (v2 = v1 + U+0289 on the
                # unused id 29), then the v2 models fold with phonemize_v2.fold_extra in __call__.
                v1 = {k: v for k, v in self.vocab.items() if k != "\u0289"}
                self._g2p[lang] = EnglishG2P(v1, self.g2p_dir)
            else:
                self._g2p[lang] = LangG2P(lang, self.phon, self.vocab)
        return self._g2p[lang]

    def __call__(self, text, lang):
        """text -> (phoneme chunks, G2PResult)."""
        chunks, rep = front_end(self.g2p(lang), text or "")
        if lang == "en" and self.phon == "v2":
            from . import phonemize_v2
            chunks = [phonemize_v2.fold_extra(c) for c in chunks]
        return chunks, rep


__all__ = ["Frontend", "G2PResult", "MAX_PHON"]

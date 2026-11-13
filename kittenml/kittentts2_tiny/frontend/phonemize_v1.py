#!/usr/bin/env python
# Copied from the Kitten training code (phonemize_multi.py). Vendored into kittenml: only import paths and asset paths changed.
"""Phonemize a multilingual manifest into the model's exact 114-symbol IPA vocab.

Writes the training list format:  wav_path|phonemes|speaker_id

WHY A COVERAGE GATE. The vocab is 114 IPA symbols. A language whose G2P emits
symbols outside that set cannot be trained without either dropping those rows or
growing the embedding -- and growing it breaks the transfer we are hot-starting from. So `--probe` measures per-language
OOV on real text BEFORE anything commits to a full build, and the per-language OOV
counters stay on during the build.

ROUTING. misaki ships dedicated G2P for en/zh/ja/ko/vi; everything else goes through
espeak-ng with the language's own voice. espeak emits a few marks the vocab
lacks (U+0329 syllabic, U+030D vertical line) which carry no segmental content for
us -- strip, do not drop the utterance over them.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

# vocab_ext.json adds 5 IPA symbols on previously-unused embedding ids so Arabic,
# Russian and German keep contrasts that folding would destroy. n_token is unchanged,
# so every pretrained weight still loads; only the 5 new rows start untrained.
def load_vocab(path) -> dict:
    """A model repository's vocab.json (v1: vocab_ext.json of the training repo): symbol -> id."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Languages the cbox_multi dump carries, mapped onto an espeak voice.
ESPEAK_VOICE = {
    "es": "es", "fr": "fr-fr", "it": "it", "pt": "pt-br", "hi": "hi",
    "de": "de", "nl": "nl", "pl": "pl", "ru": "ru", "tr": "tr", "ar": "ar",
    "he": "he", "sv": "sv", "da": "da", "no": "nb", "fi": "fi", "el": "el",
    "cs": "cs", "hu": "hu", "ro": "ro", "uk": "uk", "ms": "ms", "sw": "sw",
    "en": "en-us", "en-us": "en-us", "en-gb": "en-gb",
}

# Symbols espeak emits that the 114-symbol vocab has no row for, folded onto the
# nearest symbol it DOES have. Measured, not guessed: a probe over one sentence per
# language found exactly these three outside the vocab, 1-2 occurrences each.
#   U+032A dental mark (ar)  -- a diacritic on t/d, no row, drop it
#   U+026D retroflex l  (ru) -- fold to plain l
#   U+026B velarized l  (tr) -- fold to plain l ("dark l")
FOLD = {
    # --- measured on ONE probe sentence per language (first pass) --------------
    "\u032a": "",   # U+032A dental mark (ar) -- diacritic on t/d, no row
    "\u0329": "",   # U+0329 syllabic (en espeak fallback)
    "\u030d": "",   # U+030D vertical line above, the other syllabic form
    "\u032f": "",   # U+032F non-syllabic (da) -- marks a glide, no row
    "\u026d": "l",  # retroflex l (ru) -> plain l
    "\u026b": "l",  # velarized "dark" l (tr) -> plain l
    "\u0275": "\u00f8",  # close-mid central rounded (sv) -> o-slash
    "\u03b5": "\u025b",  # espeak emits GREEK small epsilon for da; IPA wants U+025B
    # --- measured on 60 REAL rows per language (second pass) -------------------
    # ar pharyngeals (ʕ ħ ˤ), ru ʑ and de ʏ are NOT folded any more -- they were given
    # real embedding rows in vocab_ext.json. Folding them merged minimal pairs that
    # those languages actually use (ʕ/ʔ and ħ/h in Arabic, щ/ш in Russian).
    # zh text normalisation leaks raw control bytes into the phoneme string.
    "\u0005": "", "\u0006": "", "\u0007": "", "\u0008": "",
    "{": "", "}": "", "[": "", "]": "",
    # --- third pass, residuals on 80 real rows per language -------------------
    "\u00bf": "", "\u00a1": "",   # Spanish inverted ? and ! -- the trailing mark carries it
    "\u202f": " ",                # French narrow no-break space -> plain space (in vocab)
}

# The Chinese G2P passes ASCII through unconverted (digits, latin letters, hyphens from
# mixed-script source text). They are leftovers rather than phonemes, so strip them for
# zh only -- doing it globally would destroy the Latin languages' phoneme strings.
ZH_STRIP = set("-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

_CACHE: dict[str, object] = {}


def get_g2p(lang: str):
    """Build (and memoise) the right G2P for a language code."""
    if lang in _CACHE:
        return _CACHE[lang]
    from . import espeak

    if lang in ("en", "en-us", "en-gb"):
        from misaki import en

        british = lang == "en-gb"
        # Without a fallback misaki emits U+2753 for any OOV word -- 38% of rows.
        g = en.G2P(trf=False, british=british,
                   fallback=espeak.EspeakFallback(british=british))
    elif lang == "zh":
        from misaki import zh

        g = zh.ZHG2P()
    elif lang == "ja":
        from misaki import ja

        g = ja.JAG2P()
    elif lang == "ko":
        from misaki import ko

        g = ko.KOG2P()
    elif lang == "vi":
        from misaki import vi

        g = vi.VIG2P()
    else:
        voice = ESPEAK_VOICE.get(lang)
        if voice is None:
            raise KeyError(f"no espeak voice mapped for lang {lang!r}")
        g = espeak.EspeakG2P(language=voice)
    _CACHE[lang] = g
    return g


def phonemize(g2p, text: str) -> str:
    out = g2p(text)
    ps = out[0] if isinstance(out, tuple) else out
    ps = (ps or "").strip()
    for src, dst in FOLD.items():
        if src in ps:
            ps = ps.replace(src, dst)
    return ps


def phonemize_lang(g2p, text: str, lang: str) -> str:
    ps = phonemize(g2p, text)
    if lang == "zh" and ps:
        ps = "".join(c for c in ps if c not in ZH_STRIP)
    return ps

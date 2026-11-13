#!/usr/bin/env python
# Copied from the Kitten training code (phonemize_v2.py). Vendored into kittenml: only import paths and asset paths changed.
"""Phonemizer v2 -- for the 18-language mini (user decision 2026-10-15).

v1 is phonemize_multi.py + vocab_ext.json. Every checkpoint trained before
2026-10-15 (mini, mini7, nano/pico, the language nanos) used v1, and both files are
FROZEN: this module imports v1 read-only and never changes it. phonemizer_golden.py
proves v1's output is unchanged against experiments/phonemizer_golden/.

v2 = v1 plus, all measured on 200 downloaded rows per language:
  routing   misaki only where its output fits the vocab (en, ja, zh); espeak for every
            other language. ko moves to espeak: misaki ko emits Hangul jamo, not IPA
            (0 of 100 sentences usable), espeak ko gives 0 unknown symbols. espeak is NOT
            a fallback for ja/zh: it reads kanji as "Chinese letter" and writes zh tones
            as digits, so a failing ja/zh row is dropped instead.
  vocab     vocab_v2.json = vocab_ext.json + U+0289 (sv, 118/200 rows) on
            the unused id 29. No existing id moves.
  FOLD      v1's table plus: thin space (fr) -> space, tab -> space, and the marks
            « » ‘ ’ ' ~ · # (ja/zh) -> removed; ASCII g -> IPA ɡ (zh).
  ja + zh   ASCII leftovers (Latin capitals, digits, "-") are stripped for both. v1 did
            this for zh only, and v1's CLI never applied it (main() calls phonemize()).
  hi        (2026-10-16) the danda । and ॥ become "." BEFORE espeak: v1 lost the
            sentence end in 77.8% of Hindi rows and the pause between sentences. espeak's
            ASCII "r." / "r.h" for ड़ / ढ़ (a period inside the word, 20.2% of rows)
            become ɽ / ɽʰ AFTER espeak; ɽ is already in the vocab.
  ar        (2026-10-16) "؟" becomes "?" before espeak (v1 dropped it, ~4% of rows).

For en/de/es/it/pt/ru, v2(text) == apply FOLD_EXTRA to v1(text): the routing is the
same and the extra folds run after v1's. build_mini18_lists.py uses this to carry
mini7's stored v1 rows into v2 without phonemizing them again. Hindi is NOT in that
set any more: its rows are phonemized again with v2.

CLI: same flags and log format as phonemize_multi.py (phonemize_chunked.py parses it).
"""
from __future__ import annotations

import json
import os
import re
import sys

from . import phonemize_v1 as v1  # noqa: E402  (frozen; read only)

VERSION = "v2"
MISAKI_LANGS = ("en", "en-us", "en-gb", "ja", "zh")
ESPEAK_VOICE = {**v1.ESPEAK_VOICE, "ko": "ko"}
FOLD_EXTRA = {
    " ": " ",                  # thin space (fr)
    "\t": " ",
    "«": "", "»": "",     # « » (ja)
    "‘": "", "’": "",     # ‘ ’ (zh)
    "'": "", "~": "", "·": "", "#": "",   # zh pinyin leftovers
    "g": "ɡ",                  # ASCII g -> IPA ɡ (zh)
}
FOLD = {**v1.FOLD, **FOLD_EXTRA}
CJK_STRIP = v1.ZH_STRIP
# main() rebinds v1.get_g2p for v1's CLI; keep the original, or the misaki path below
# would call the rebound one (2026-10-15 dry run: every ja/zh row -> g2p_error).
_V1_GET_G2P = v1.get_g2p

_CACHE: dict[str, object] = {}


def load_vocab(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_g2p(lang: str):
    if lang in _CACHE:
        return _CACHE[lang]
    if lang in MISAKI_LANGS:
        g = _V1_GET_G2P(lang)           # the same misaki objects v1 builds
    else:
        voice = ESPEAK_VOICE.get(lang)
        if voice is None:
            raise KeyError(f"no espeak voice mapped for lang {lang!r}")
        from . import espeak
        g = espeak.EspeakG2P(language=voice)
    _CACHE[lang] = g
    return g


def fold_extra(ps: str) -> str:
    for src, dst in FOLD_EXTRA.items():
        if src in ps:
            ps = ps.replace(src, dst)
    return ps


def phonemize(g2p, text: str) -> str:
    out = g2p(text)
    ps = out[0] if isinstance(out, tuple) else out
    ps = (ps or "").strip()
    for src, dst in FOLD.items():
        if src in ps:
            ps = ps.replace(src, dst)
    return ps


TEXT_MAP = {"hi": {"।": ".", "॥": "."},        # । ॥ -> .
            "ar": {"؟": "?"}}                         # ؟ -> ?
_HI_FLAP = re.compile("[ड़ढ़]|[डढ]़")   # ड़ ढ़ (both encodings)


def phonemize_lang(g2p, text: str, lang: str) -> str:
    for src, dst in TEXT_MAP.get(lang, {}).items():
        text = text.replace(src, dst)
    ps = phonemize(g2p, text)
    if lang == "hi":
        # espeak also says ड़ inside its own number words (67 -> sˈʌr.səʈʰ), so an "r."
        # INSIDE a word is always the flap; a word-final one (भीड़) only when the text
        # holds ड़/ढ़, since there "r." could also be r + a real full stop.
        ps = ps.replace("r.h", "ɽʰ")
        ps = re.sub(r"r\.(?=[^\s.,?!;:…])", "ɽ", ps)
        if _HI_FLAP.search(text):
            ps = ps.replace("r.", "ɽ")
    if lang in ("zh", "ja") and ps:
        ps = "".join(c for c in ps if c not in CJK_STRIP)
    return ps

"""Indic-script fixes for the text pipeline that feeds KittenTTS 2.

`kittenml/kittentts2/text.py` assumes a Latin-script sentence ends in `.!?` and
that digits are ASCII. Neither holds for Devanagari, Bengali, Tamil or Telugu, and
the failures are not graceful:

  * **Sentences never end.** `_SENT_END` only breaks on `.!?`, so a Hindi
    paragraph punctuated with danda (`।`) is one unbroken string. Once it passes
    the 380-character budget the splitter falls through to `sentence[:max_chars]`
    — a hard cut that lands mid-word, mid-grapheme-cluster, and the model is asked
    to pronounce the fragment.
  * **Doubled terminators.** `।` is absent from `_ALREADY_TERMINATED`, so
    `_capitalize_and_terminate` appends a full stop to text that already ended,
    giving `…है।.`, and `_ensure_terminal_punct` appends a comma, giving `…है।,`.
  * **Digits are read in English.** The normalizer expands `2024` but not `२०२४`,
    which reaches the model as four rare codepoints.

This module owns the script-aware half of that. It is deliberately separate from
`text.py` rather than tangled into it: the rules here are per-script, they are the
part that needs native-speaker review, and keeping them addressable means the
benchmark can report "with fixes" against "without" on the same checkpoint.

Why it must land before any baseline number is taken: an Indic CER measured
through the unpatched splitter is scoring the chunker, not the model.
"""

import re
import unicodedata

# ── script-aware punctuation ──────────────────────────────────────────────────
# U+0964 DEVANAGARI DANDA and U+0965 DOUBLE DANDA are shared across Brahmic
# scripts — Bengali and Marathi use the same codepoints, not script-local clones.
# Tamil and Telugu modern orthography uses the Latin full stop instead, so they
# need no addition here; they were already broken only by the digit rules below.
DANDA = "।"
DOUBLE_DANDA = "॥"
INDIC_TERMINATORS = DANDA + DOUBLE_DANDA

# Sentence-final marks across the scripts we target, Latin included because Indic
# text mixes in `.` and `?` freely.
SENTENCE_END_CHARS = ".!?" + INDIC_TERMINATORS

# Mirrors text.py's `_SENT_END` but with the dandas added. The trailing-quote class
# is kept identical so behaviour on Latin text is unchanged.
SENT_END = re.compile(r'(?<=[.!?' + INDIC_TERMINATORS + r'])["’”\')\]]*\s+')


# ── Indic digits ──────────────────────────────────────────────────────────────
# Each Brahmic script has its own digit block, laid out 0-9 contiguously from the
# base codepoint. Mapping them to ASCII lets the existing number grammar — and any
# Indic number-to-words backend — see a number at all.
DIGIT_BLOCK_STARTS = {
    "Devanagari": 0x0966,   # hi, mr
    "Bengali": 0x09E6,      # bn
    "Gurmukhi": 0x0A66,
    "Gujarati": 0x0AE6,
    "Oriya": 0x0B66,
    "Tamil": 0x0BE6,
    "Telugu": 0x0C66,
    "Kannada": 0x0CE6,
    "Malayalam": 0x0D66,
}

INDIC_DIGIT_TO_ASCII = {
    chr(start + d): str(d)
    for start in DIGIT_BLOCK_STARTS.values()
    for d in range(10)
}
_INDIC_DIGIT_RE = re.compile("[" + "".join(INDIC_DIGIT_TO_ASCII) + "]")


def indic_digits_to_ascii(text):
    """Rewrite Devanagari/Bengali/Tamil/Telugu/... digits as ASCII digits.

    Purely a script change: `२०२४` becomes `2024`, which the downstream number
    grammar can then expand into words. It does not decide *which language's*
    words — that is the normalizer's job, and today it only knows English.
    """
    return _INDIC_DIGIT_RE.sub(lambda m: INDIC_DIGIT_TO_ASCII[m.group(0)], str(text or ""))


# ── script detection ──────────────────────────────────────────────────────────
_SCRIPT_CACHE = {}


def char_script(ch):
    """Unicode script name for one character, via its Unicode name prefix.

    `unicodedata` exposes no script property, but every Brahmic codepoint's name
    begins with its script ("DEVANAGARI LETTER KA"), which is enough to tell the
    blocks apart without pulling in a dependency.
    """
    if ch in _SCRIPT_CACHE:
        return _SCRIPT_CACHE[ch]
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = ""
    script = next((s for s in DIGIT_BLOCK_STARTS if name.startswith(s.upper())), None)
    if script is None and (ch.isascii() and ch.isalpha()):
        script = "Latin"
    _SCRIPT_CACHE[ch] = script
    return script


def dominant_script(text):
    """The script most of the letters in `text` belong to, or None.

    Used to route a chunk to the right number-to-words backend, and to decide
    whether a danda fix is even relevant.
    """
    counts = {}
    for ch in str(text or ""):
        script = char_script(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1
    return max(counts, key=counts.get) if counts else None


def has_indic(text):
    """True when the text contains any Brahmic-script letter."""
    return any(char_script(ch) not in (None, "Latin") for ch in str(text or ""))


# ── normalization entry point ─────────────────────────────────────────────────
def normalize_indic(text):
    """The script-level cleanup every Indic string should get before synthesis.

    NFC first, because the same Devanagari syllable has several decomposed forms
    and the tokenizer treats them as different pieces — a corpus mixing them
    silently splits the vocabulary's mass across duplicates.
    """
    s = unicodedata.normalize("NFC", str(text or ""))
    s = indic_digits_to_ascii(s)
    s = collapse_danda_punctuation(s)
    return s


# `।.` / `।,` / `। .` — a danda followed by Latin terminal punctuation, which is
# what text.py produces today. The danda wins; the appended mark is dropped.
_DANDA_THEN_LATIN = re.compile("([" + INDIC_TERMINATORS + r"])\s*[.,;:]+")
# The reverse order, from text appending a danda after an existing full stop.
_LATIN_THEN_DANDA = re.compile(r"[.,;:]+\s*([" + INDIC_TERMINATORS + "])")


def collapse_danda_punctuation(text):
    """Remove the doubled terminator `text.py` leaves behind.

    Safe to run even on already-correct text: with nothing doubled, both patterns
    fail to match and the string is returned unchanged.
    """
    s = _DANDA_THEN_LATIN.sub(r"\1", str(text or ""))
    return _LATIN_THEN_DANDA.sub(r"\1", s)


def split_sentences(text):
    """Sentence split that honours the danda as well as `.!?`.

    Returned pieces keep their terminator, since the model is sensitive to whether
    a chunk is closed.
    """
    stripped = (text or "").strip()
    return [s.strip() for s in SENT_END.split(stripped) if s and s.strip()]

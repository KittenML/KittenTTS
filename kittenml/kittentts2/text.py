"""Text handling for KittenTTS 2: expression markup, normalization, chunking, joining.

KittenTTS 2 accepts three expression controls that the ONNX models do not:

    [joyful] We won the grant <laugh> I can (((hardly))) believe it!

A leading `[emotion]` tag, inline `<event>` tags, and `(((emphasis)))` spans. They
are director's notes for the model, not text to pronounce, so they must reach it
verbatim — which means protecting them before normalization and restoring them
after. Any *other* bracketed span is rewritten to plain comma-separated text,
since the model destabilises on bracket punctuation it was not trained to read.

Long text is split on sentence boundaries and synthesized chunk by chunk. One
generate() call for a whole script is where the model falls apart — it truncates
or drifts into repetition — while it is reliable on short inputs, so chunking
keeps every generation inside the range it handles well.

The written-to-spoken step — expanding numbers, currency, dates and abbreviations
— is delegated to `_backend`, and this module only owns what wraps it. Several
regexes below exist to steer that backend away from known misreadings (paired
quotes read as inches, fused version strings, digit-adjacent ?/!), and are kept
so the text reaching the model matches what it heard in serving.
"""

import re

import numpy as np


# ── expression markup ─────────────────────────────────────────────────────────
# Restricted to a known tag vocabulary rather than "any bracketed span", so we know
# exactly when a span is a real tag. Text that merely looks bracketed — a citation
# like "section [3]", a comparison like "x < 5" — is normalized like ordinary text
# instead of being silently exempted. Extend these sets to protect more tags.
EXPRESSION_TAGS = frozenset({
    "mundane", "nervous", "tender", "angry", "excited",
    "stern", "sad", "contemplative", "surprised", "joyful",
})
VOCAL_EVENT_TAGS = frozenset({
    "pause", "sigh", "gasp", "laugh", "giggle",
    "sob", "scoff", "growl", "um", "gulp",
})

_TAG_SPAN = re.compile(
    r"\[(?:" + "|".join(re.escape(t) for t in EXPRESSION_TAGS) + r")\]"
    r"|<(?:" + "|".join(re.escape(t) for t in VOCAL_EVENT_TAGS) + r")>",
    re.IGNORECASE,
)
_EMPHASIS_SPAN = re.compile(r"\(\(\(([^()\n]{1,80})\)\)\)")
_PLACEHOLDER = "\ue000{}\ue001"          # private-use sentinels; cannot occur in real input


def _protect_tags(text):
    """Swap every tag and emphasis span for a sentinel-wrapped index.

    The normalizer must not see them, and must not be able to reorder or reword
    them. The default backend passes private-use sentinels through untouched, which
    makes this safe; the index inside is restored verbatim afterwards.
    """
    tags = []

    def _swap(match):
        tags.append(match.group(0))
        return _PLACEHOLDER.format(len(tags) - 1)

    return _EMPHASIS_SPAN.sub(_swap, _TAG_SPAN.sub(_swap, text)), tags


def _restore_tags(text, tags):
    for i, tag in enumerate(tags):
        text = text.replace(_PLACEHOLDER.format(i), tag)
    return text


def has_expression_tag(text):
    """True when the text uses any expression, vocal-event or emphasis control.

    Drives the `{emo: 1}` conditioning prefix: the control is only spliced into the
    prompt when the user actually reaches for one of these markers.
    """
    s = str(text or "")
    return bool(_TAG_SPAN.search(s) or _EMPHASIS_SPAN.search(s))


# ── written -> spoken ─────────────────────────────────────────────────────────
_UI_GLYPHS = re.compile(r"[\u2022\u2023\u2043\u25e6\u25b8\u21b3\u2192\u2190\u2191\u2193]+")

# Raw math glyphs reach the model as rare tokens and destabilise it, and the normalizer
# either passes them through or picks stilted readings ("=" -> "equal sign"). Spoken
# forms are substituted first so any digits they produce still go through its number
# grammar. Readings follow how papers are read aloud ("y ~ pi" -> "distributed as")
# rather than notation-theoretic precision.
_MATH_GLYPH_WORDS = {
    "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03b4": "delta", "\u03b5": "epsilon", "\u03b6": "zeta",
    "\u03b7": "eta", "\u03b8": "theta", "\u03b9": "iota", "\u03ba": "kappa", "\u03bb": "lambda", "\u03bc": "mu",
    "\u03bd": "nu", "\u03be": "xi", "\u03c0": "pi", "\u03c1": "rho", "\u03c3": "sigma", "\u03c4": "tau",
    "\u03c5": "upsilon", "\u03c6": "phi", "\u03d5": "phi", "\u03c7": "chi", "\u03c8": "psi", "\u03c9": "omega",
    "\u0393": "gamma", "\u0394": "delta", "\u0398": "theta", "\u039b": "lambda", "\u039e": "xi", "\u03a0": "pi",
    "\u03a3": "sigma", "\u03a6": "phi", "\u03a8": "psi", "\u03a9": "omega",
    "=": "equals", "\u2260": "is not equal to", "\u2248": "approximately equals",
    "\u2264": "is less than or equal to", "\u2265": "is greater than or equal to",
    "\u00b1": "plus or minus", "\u00d7": "times", "\u00f7": "divided by", "\u00b7": "dot",
    "\u2212": "minus", "\u2217": "star", "\u2218": "composed with", "\u22a4": "transpose",
    "\u223c": "distributed as", "\u221d": "proportional to", "\u2261": "is equivalent to",
    "\u2208": "in", "\u2209": "not in", "\u2282": "subset of", "\u2286": "subset of", "\u222a": "union",
    "\u2229": "intersect", "\u2295": "plus", "\u2297": "times", "^": "to the",
    "\u2211": "sum of", "\u220f": "product of", "\u222b": "integral of", "\u2202": "partial",
    "\u2207": "gradient of", "\u221a": "square root of", "\u221e": "infinity", "\u2206": "delta",
    "\u2200": "for all", "\u2203": "there exists", "\u21d2": "implies", "\u21d4": "if and only if",
    "\u2225": "norm", "\u2016": "norm", "\u230a": "floor of", "\u230b": "", "\u2308": "ceiling of", "\u2309": "",
    "\u211d": "R", "\u2115": "N", "\u2124": "Z", "\u211a": "Q", "\u2102": "C", "\U0001d53c": "the expectation of",
}
# Super/subscript digits -> plain digits, so "x²" becomes "x 2" and normalizes onward.
_MATH_GLYPH_WORDS.update({c: str(d) for d, pair in
                          enumerate(zip("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")) for c in pair})
_MATH_GLYPH_RE = re.compile("|".join(re.escape(g) for g in
                                     sorted(_MATH_GLYPH_WORDS, key=len, reverse=True)))

# PDF pastes render "…" as spaced periods; either form is a pause, not three spoken dots.
_ELLIPSIS = re.compile(r"(?:\.\s+){2,}\.|\u2026|\.\.\.")

# A straight double quote right after a digit is ambiguous between a closing quotation
# mark and inches, and the measure grammar always reads it as inches — silently
# eating the closing quote. Retyping PAIRED straight quotes as curly disambiguates the
# common case while leaving a genuinely unpaired one ('the TV is 50" wide') as inches.
_STRAIGHT_QUOTE_PAIR = re.compile(r'"([^"\n]*)"')

_NONTAG_BRACKET_SPAN = re.compile(r"\[([^\[\]\n]{1,400})\]|<([^<>\n]{1,60})>|\(([^()\n]{1,400})\)")
# ", VPO,," — a rewritten span's trailing comma colliding with punctuation that already
# followed the bracket; keep whichever mark came after it.
_COMMA_COLLISION = re.compile(r",\s*([,.!?;:])")

_TRAILING_NONTERMINAL = re.compile(r"[,;:\-\u2013\u2014]+\s*$")
_ALREADY_TERMINATED = set('.!?"\u2019\u201d\')]')

# Em/en dashes -> commas. The checkpoint has little dash-delimited training data, and
# nested em-dashes were the most reliable way to make a chunk run away in testing. A
# comma carries the same prosodic break with far more support in the data. Hyphens are
# deliberately left alone: the normalizer writes 21-99 as hyphenated words, so converting
# them would speak "twenty-six miles" as "twenty, six miles".
_DASH = re.compile(r"\s*(?:[\u2014\u2013]|--)\s*")
_DASH_FIX = [
    (re.compile(r",\s*,+"), ", "),
    (re.compile(r"\s+([,.!?;:])"), r"\1"),
    (re.compile(r",\s*([.!?])"), r"\1"),
    (re.compile(r"\s{2,}"), " "),
    (re.compile(r"^\s*,\s*"), ""),
]


def _speak_math_glyphs(text):
    text = _ELLIPSIS.sub(", ", text)
    text = _MATH_GLYPH_RE.sub(lambda m: f" {_MATH_GLYPH_WORDS[m.group(0)]} ", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def _reformat_nontag_brackets(text):
    def _swap(match):
        inner = next(g for g in match.groups() if g is not None).strip()
        return f", {inner},"

    text = _NONTAG_BRACKET_SPAN.sub(_swap, text)
    while True:                     # ", , ," needs repeated passes to collapse
        collapsed = _COMMA_COLLISION.sub(r"\1", text)
        if collapsed == text:
            return collapsed
        text = collapsed


def _smart_quote_pairs(text):
    return _STRAIGHT_QUOTE_PAIR.sub(lambda m: "\u201c" + m.group(1) + "\u201d", text)


def _dashes_to_commas(text):
    if not text:
        return text
    text = _DASH.sub(", ", text)
    for pattern, replacement in _DASH_FIX:
        text = pattern.sub(replacement, text)
    return text.strip()


def _capitalize_and_terminate(text):
    """Capitalize the opening letter and guarantee sentence-final punctuation.

    Runs on the tag-protected text, so the first alphabetic character it finds is
    always the first real word — a placeholder carries no letters, so this can
    never reach inside a tag and capitalize its deliberately lowercase contents.
    """
    if not text:
        return text
    text = re.sub(r"\s+([?!])", r"\1", text)    # re-attach marks detached for the number grammar
    for i, ch in enumerate(text):
        if ch.isalpha():
            text = text[:i] + ch.upper() + text[i + 1:]
            break
    stripped = text.rstrip()
    if not stripped:
        return text
    if stripped[-1] in _ALREADY_TERMINATED:
        return stripped
    stripped = _TRAILING_NONTERMINAL.sub("", stripped).rstrip()
    return stripped + "." if stripped else stripped


# ── the written -> spoken backend ─────────────────────────────────────────────
# kitten_text_processing reproduces the normalization this checkpoint was served
# with, which is what makes the text the model hears match production. It is pure
# Python with no compiled dependencies, so it installs everywhere and is an
# ordinary requirement rather than something to detect at run time.
import kitten_text_processing as _backend

# Tags are held out of normalization by wrapping them in private-use sentinels and
# sending the whole string through in one call — the backend must therefore leave
# those codepoints alone. It declares that it does; checked here because the
# failure is silent and ugly if it ever stops being true (the sentinels vanish and
# the model speaks the placeholder index, so "[joyful]" comes out as "Zero").
if not getattr(_backend, "PRESERVES_SENTINELS", False):
    raise RuntimeError(
        "kitten_text_processing no longer preserves private-use sentinels; "
        "expression tags would be corrupted. Pin an earlier version.")


def warm_normalizer():
    """Build the backend's grammars now rather than inside the first request."""
    _backend.warm()


def normalize_text(text, normalize=True):
    """Turn written text into the spoken form the model should hear.

    Never raises: every failure path falls back to the lightly-cleaned text.
    The written-to-spoken step itself comes from `_backend`.
    Expression markup is protected before anything else runs and restored on
    every return path, so it survives identically whether normalization
    succeeds, fails, or is switched off.
    """
    protected, tags = _protect_tags(str(text or ""))
    # Hard line wraps otherwise reach the sentence splitter as missing spaces:
    # "require.\nWe" collapses to "require.We", the boundary regex cannot see it, and
    # the fused pseudo-sentence falls into mid-clause splits that are audible as seams.
    protected = re.sub(r"\s*\n+\s*", " ", protected)
    protected = re.sub(r"(?<=[a-z])([.!?;:])(?=[A-Z])", r"\1 ", protected)
    protected = _speak_math_glyphs(protected)
    # "V0.8" -> "V 0 point 8": fused version tokens land in the serial grammar and
    # read as "dot"; split out, the separators stay "point".
    protected = re.sub(r"\b([vV])(\d+(?:\.\d+)+)",
                       lambda m: m.group(1) + " " + " point ".join(m.group(2).split(".")),
                       protected)
    # Detach sentence-final ?/! from a trailing digit, which the number grammar would
    # otherwise consume and speak aloud. _capitalize_and_terminate re-attaches it.
    protected = re.sub(r"(?<=\d)\s*([?!])(?=\s|$)", r" \1", protected)

    raw = _dashes_to_commas(_UI_GLYPHS.sub(" ", _reformat_nontag_brackets(protected)))
    if not raw.strip():
        return _restore_tags(raw, tags)
    if not normalize:
        return _restore_tags(_capitalize_and_terminate(raw), tags)
    try:
        spoken = (_backend.normalize_text(_smart_quote_pairs(raw)) or "").strip()
    except Exception as exc:                # a normalizer problem must not cost a generation
        print(f"[normalize] {type(exc).__name__} on {raw[:60]!r}, speaking raw text: {exc}",
              flush=True)
        spoken = ""
    if not spoken:
        return _restore_tags(_capitalize_and_terminate(raw), tags)
    return _restore_tags(_capitalize_and_terminate(_dashes_to_commas(spoken)), tags)


# ── chunking ──────────────────────────────────────────────────────────────────
# 380 rather than the ~560 first tried: the model compresses its delivery as a chunk
# grows, so longer chunks read as rushed even when every word is correct.
CHUNK_CHARS = 380
# Undersized chunks are the unstable ones — the model over-generates on short prompts
# and holds a token instead of terminating — so fold anything below this into a
# neighbour rather than generating it alone.
CHUNK_MIN_CHARS = 130
CHUNK_GAP_S = 0.16

_SENT_END = re.compile(r'(?<=[.!?])["’”\')\]]*\s+')
_TERMINAL = '.!?,;:'


def split_for_synthesis(text, max_chars=CHUNK_CHARS, min_chars=CHUNK_MIN_CHARS):
    """Sentence-aware greedy packing.

    Returns a single chunk when the text already fits, so short and medium inputs
    take the unchunked path exactly as they would without this.
    """
    stripped = (text or "").strip()
    if not max_chars or len(stripped) <= max_chars:
        return [stripped] if stripped else []
    out, current = [], ""
    for sentence in [s for s in _SENT_END.split(stripped) if s and s.strip()]:
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            # A sentence longer than the budget: break at the last clause boundary
            # before the limit, falling back to a hard split so this cannot loop.
            cut = max(sentence.rfind(", ", 0, max_chars), sentence.rfind("; ", 0, max_chars),
                      sentence.rfind(" — ", 0, max_chars))
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            head, sentence = sentence[:cut].strip(), sentence[cut:].strip()
            if current:
                out.append(current)
                current = ""
            out.append(head)
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            out.append(current)
            current = sentence
    if current:
        out.append(current)
    return [_ensure_terminal_punct(c) for c in _merge_short(out, max_chars, min_chars)]


def _merge_short(parts, max_chars, min_chars):
    """Fold undersized chunks into a neighbour, preferring the smaller result.

    Merging only ever removes a boundary, never creates one mid-sentence. The budget
    stretches to 1.4x rather than refusing to merge, since a slightly long chunk is
    far safer than a very short one.
    """
    if not parts or min_chars <= 0:
        return parts
    out = list(parts)
    limit = int(max_chars * 1.4)
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, chunk in enumerate(out):
            if len(chunk) >= min_chars:
                continue
            prev_ok = i > 0 and len(out[i - 1]) + 1 + len(chunk) <= limit
            next_ok = i < len(out) - 1 and len(chunk) + 1 + len(out[i + 1]) <= limit
            if prev_ok and (not next_ok or len(out[i - 1]) <= len(out[i + 1])):
                out[i - 1] = out[i - 1].rstrip() + " " + chunk.lstrip()
                out.pop(i)
            elif next_ok:
                out[i] = chunk.rstrip() + " " + out[i + 1].lstrip()
                out.pop(i + 1)
            else:
                continue                # nothing it can merge into; leave it
            changed = True
            break
    return out


def _ensure_terminal_punct(chunk):
    """Give every chunk a closing mark.

    A chunk that ends mid-phrase gives the model no cue that it is done, which is
    when it holds a token and produces a tail artefact. A clause break gets a comma
    rather than a full stop, so the prosody stays continuation-like.
    """
    c = (chunk or "").rstrip()
    if not c:
        return c
    if c[-1] in _TERMINAL or c[-1] in '"’”\')]':
        return c
    return c + ","


# ── joining ───────────────────────────────────────────────────────────────────
def trim_silence(wav, sample_rate, thresh_db=-45.0, keep_s=0.02):
    """Strip the lead-in and trail-out silence a generation carries.

    Without this, a join stacks one segment's trailing silence onto the next one's
    leading silence plus the inserted pause — which is what makes chunking audible.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < int(sample_rate * 0.05):
        return wav
    hop = max(1, int(sample_rate * 0.01))
    frames = wav[:len(wav) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    loud = np.nonzero(rms > 10 ** (thresh_db / 20.0))[0]
    if not len(loud):
        return wav
    keep = int(sample_rate * keep_s)
    start = max(0, loud[0] * hop - keep)
    end = min(len(wav), (loud[-1] + 1) * hop + keep)
    return wav[start:end]


def join_chunks(wavs, sample_rate, gap_s=CHUNK_GAP_S):
    """Concatenate segments so the seam is inaudible.

    Trim each segment's own edge silence, insert one consistent sentence-length
    pause, and fade a few milliseconds at each edge — segments rarely begin or end
    on a zero crossing, and that discontinuity is heard as a click.
    """
    wavs = [w for w in wavs if w is not None and len(w)]
    if not wavs:
        return None
    if len(wavs) == 1:
        return np.asarray(wavs[0], dtype=np.float32)
    gap = np.zeros(int(sample_rate * gap_s), dtype=np.float32)
    fade = max(1, int(sample_rate * 0.008))
    out = []
    for i, wav in enumerate(wavs):
        wav = trim_silence(np.asarray(wav, dtype=np.float32), sample_rate).copy()
        if len(wav) > 2 * fade:
            wav[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        out.append(wav)
        if i != len(wavs) - 1:
            out.append(gap)
    return np.concatenate(out)

"""Shared by the runner script and the report: test names, statuses and WER.

Imported on every runner before anything is installed, including Python 3.9, so
it uses the standard library only and no newer syntax.
"""
import re

# What one platform job found. Whether a failure is new is decided in the report,
# by comparing with the latest run on main.
PASSED = "passed"
FAILED = "failed"
NO_RESULT = "no-result"       # the job died, timed out or was cancelled

# Each test is one README example, run in its own Python process.
# test name -> (column header, what it runs)
TESTS = {
    "speak": ("Speak", "`generate()` speaks the sample text"),
    "stream": ("Stream", "`generate_stream()`"),
    "speed": ("Speed", "`generate(speed=0.8)` gives longer audio"),
    "to_file": ("To file", "`generate_to_file()`"),
    "expression": ("Expression", "`[joyful]` and `<laugh>` tags with `preset=\"expressive\"`"),
    "clone": ("Clone", "voice cloning with `reference=` and `reference_text=`"),
    "clone_whisper": ("Clone (Whisper)", "voice cloning with `reference=` only; Whisper writes the transcript"),
    "emb4": ("emb4", "`weights=\"emb4\"`, the smaller weights"),
}


def test_keys(model):
    """'nano' (speaks the sample text), then 'nano:stream' and the like for each check."""
    return [model["key"]] + [f"{model['key']}:{c}" for c in model.get("checks", [])]


def test_name(key):
    return key.partition(":")[2] or "speak"


def wer_failed(row, fail_above):
    return fail_above is not None and row.get("wer") is not None and row["wer"] > fail_above


def test_ok(row, fail_above):
    """A test works when it ran without error and, where Whisper listened, it heard the text."""
    return row.get("status") == "pass" and not wer_failed(row, fail_above)


def why(row, fail_above):
    """One line on why a test does not work."""
    if row.get("status") == "pass" and wer_failed(row, fail_above):
        heard = (row.get("transcript") or "").strip()
        return f"Whisper heard “{heard[:100]}” (WER {row['wer']:.0%})"
    error = re.sub(r"\s+", " ", row.get("error") or row.get("status") or "failed").strip()
    return error[:200]


def install_ok(result):
    install = result.get("install") or {}
    return bool(install.get("ok") and install.get("import_ok"))


def classify(result):
    """(status, reasons) for one platform job's result.json."""
    install = result.get("install") or {}
    if not install:
        return NO_RESULT, ["the job did not record an install"]
    fail_above = result["spec"].get("asr", {}).get("fail_above")
    reasons = []
    if not install.get("ok"):
        reasons.append(f"install: {install.get('reason') or install.get('error') or 'failed'}")
    elif not install.get("import_ok"):
        reasons.append(f"import: {install.get('import_error') or 'failed'}")
    for row in result.get("tests", []):
        if not test_ok(row, fail_above):
            reasons.append(f"{row.get('title', row['key'])}: {why(row, fail_above)}")
    return (FAILED, reasons) if reasons else (PASSED, [])


# ── Word error rate ──────────────────────────────────────────────────────────────

def normalize_words(text):
    """Words for WER: case and punctuation do not count, KittenTTS == Kitten TTS."""
    t = text.lower()
    t = re.sub(r"\bt\.?\s?t\.?\s?s\b\.?", "tts", t)
    t = re.sub(r"\bkitten[\s-]*tts\b", "kitten tts", t)
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", t)


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer(reference, hypothesis):
    """(wer, edits, reference word count)."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    edits = edit_distance(ref, hyp)
    return (edits / len(ref) if ref else 0.0), edits, len(ref)

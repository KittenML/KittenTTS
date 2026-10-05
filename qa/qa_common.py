"""Shared by the runner script and the report: statuses and WER.

Imported on every runner before anything is installed, including Python 3.9, so
it uses the standard library only and no newer syntax.
"""
import re

# What one platform job found. Whether a failure is new is decided in the report,
# by comparing with the latest run on main.
PASSED = "passed"
FAILED = "failed"
NO_RESULT = "no-result"       # the job died, timed out or was cancelled

STATUS_LABEL = {PASSED: "Works", FAILED: "Does not work", NO_RESULT: "No result"}


def classify(result):
    """(status, reasons) for one platform job's result.json."""
    spec = result["spec"]
    install = result.get("install") or {}
    if not install:
        return NO_RESULT, ["the job did not record an install"]
    if not install.get("ok"):
        return FAILED, [f"install: {install.get('error') or 'failed'}"]
    reasons = []
    package = result.get("package") or {}
    if package.get("status") in ("crash", "timeout"):
        reasons.append(f"package checks: {package.get('error', package['status'])}")
    for chk in package.get("checks", []):
        if chk["status"] != "pass":
            reasons.append(f"{chk['name']}: {chk.get('error', chk['status'])}")
    for model in result.get("models", []):
        if model["status"] != "pass":
            reasons.append(f"{model['label']}: {model.get('error') or model['status']}")
    asr = result.get("asr") or {}
    if asr.get("status") in ("crash", "timeout"):
        # No WER means no check that the audio says the text; that must not pass quietly.
        reasons.append(f"WER transcription: {asr.get('error', asr['status'])}")
    fail_above = spec.get("asr", {}).get("fail_above")
    for row in asr.get("rows", []):
        if fail_above is not None and row.get("wer") is not None and row["wer"] > fail_above:
            reasons.append(f"{row['label']}: WER {row['wer']:.0%} is above {fail_above:.0%}")
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

"""Shared by the runner script and the report: statuses and WER.

Imported on every runner before anything is installed, including Python 3.9, so
it uses the standard library only and no newer syntax.
"""
import re

# A platform job ends in exactly one of these.
PASSED = "passed"
FAILED = "failed"
UNSUPPORTED = "unsupported"   # failed to install, as config.toml says it should
CHANGED = "changed"           # expected not to install, but now it does
NO_RESULT = "no-result"       # the job died, timed out or was cancelled

STATUS_LABEL = {
    PASSED: "Passed",
    FAILED: "Failed",
    UNSUPPORTED: "Unsupported (expected)",
    CHANGED: "Now installs",
    NO_RESULT: "No result",
}


def classify(result):
    """(status, reasons, failing) for one platform job's result.json.

    `failing` says whether this job should fail the run: a FAILED or NO_RESULT
    status on a gating target, or pip accepting a Python it must refuse.
    """
    spec = result["spec"]
    expect = spec.get("expect", "works")
    gating = spec.get("gating", True)
    install = result.get("install") or {}
    reasons = []

    if not install:
        return NO_RESULT, ["the job did not record an install"], gating

    if expect == "refused":
        if install.get("refused"):
            return UNSUPPORTED, [], False
        if install.get("ok"):
            return FAILED, ["pip installed it on a Python it should refuse"], True
        return FAILED, ["install failed, but not on Requires-Python"], gating

    if not install.get("ok"):
        if expect == "install-fails":
            return UNSUPPORTED, [], False
        return FAILED, ["install failed"], gating

    for name in ("package",):
        part = result.get(name) or {}
        for chk in part.get("checks", []):
            if chk["status"] != "pass":
                reasons.append(f"{chk['name']}: {chk.get('error', chk['status'])}")
    for model in result.get("models", []):
        if model["status"] != "pass":
            reasons.append(f"{model['label']}: {model.get('error') or model['status']}")
    asr = result.get("asr") or {}
    fail_above = spec.get("asr", {}).get("fail_above")
    for row in asr.get("rows", []):
        if fail_above is not None and row.get("wer") is not None and row["wer"] > fail_above:
            reasons.append(f"{row['label']}: WER {row['wer']:.0%} is above {fail_above:.0%}")

    if expect == "install-fails":
        # Good news, but config.toml is now out of date; flag without failing.
        return CHANGED, reasons, False
    if reasons:
        return FAILED, reasons, gating
    return PASSED, [], False


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

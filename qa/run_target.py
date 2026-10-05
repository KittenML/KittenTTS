"""One platform job: install kittenml, run every model, transcribe, write result.json.

    python qa/run_target.py --spec spec.json --out qa-out

The spec is one job from qa/plan.py. QA_SOURCE says what to install: "checkout"
(default, this repository) or a pip requirement such as "kittenml==0.9.3".

Everything before the install uses the standard library only. Each model and
the transcription run in their own process with a timeout, so a model that
hangs, segfaults or exit()s (espeak does) is reported, not fatal.
"""
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qa_common import STATUS_LABEL, classify, wer  # noqa: E402

EXPRESSION_TEXT = "[joyful] We actually won the grant <laugh> I can (((hardly))) believe it!"
CLONE_TEXT = "This is my own voice, cloned from a short recording."


# ── System info ──────────────────────────────────────────────────────────────────

def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def system_info():
    info = {"os": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "cpu": "", "ram_gb": None}
    try:
        if sys.platform == "linux":
            for line in _run(["lscpu"]).splitlines():
                if line.startswith("Model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            info["ram_gb"] = round(kb / 2**20, 1)
        elif sys.platform == "darwin":
            info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            info["ram_gb"] = round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1)
        elif sys.platform == "win32":
            import ctypes
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MemStatus()
            ms.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            info["ram_gb"] = round(ms.ullTotalPhys / 2**30, 1)
    except Exception as e:
        info["cpu"] = info["cpu"] or f"unknown ({type(e).__name__})"
    info["cpu"] = info["cpu"] or platform.processor() or "unknown"
    return info


def peak_rss_mb():
    """Peak resident memory of this process, in MiB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return round(c.PeakWorkingSetSize / 2**20)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 1024)


# ── Install ──────────────────────────────────────────────────────────────────────

def install(out):
    source = os.environ.get("QA_SOURCE", "checkout").strip() or "checkout"
    target = os.path.dirname(HERE) if source == "checkout" else source
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "pip"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "pip", "install", target],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace")
    log = proc.stdout
    with open(os.path.join(out, "install.log"), "w", encoding="utf-8") as f:
        f.write(log)
    refused = proc.returncode != 0 and (
        "requires a different Python" in log or "Requires-Python" in log)
    res = {"source": source, "ok": proc.returncode == 0, "refused": refused,
           "secs": round(time.time() - t0, 1)}
    if proc.returncode != 0:
        lines = log.strip().splitlines()
        errs = [l for l in lines if l.startswith("ERROR")]
        res["error"] = (errs[-1] if errs else lines[-1] if lines else "pip failed")[:300]
        res["log_tail"] = log[-3000:]
    else:
        res["versions"] = installed_versions()
    return res


def installed_versions():
    code = ("import json\nfrom importlib.metadata import version\nout={}\n"
            "for p in ['kittenml','torch','transformers','onnxruntime','numpy','phonemizer','espeakng_loader']:\n"
            "    try: out[p]=version(p)\n    except Exception: out[p]=None\nprint(json.dumps(out))")
    try:
        return json.loads(_run([sys.executable, "-c", code]) or "{}")
    except ValueError:
        return {}


# ── Child processes ──────────────────────────────────────────────────────────────

def run_child(kind, arg, spec_path, out, timeout_s):
    """Run one isolated part; returns its JSON, or a crash/timeout record."""
    part_out = os.path.join(out, "parts", f"{kind}-{arg}.json")
    log_path = os.path.join(out, "logs", f"{kind}-{arg}.log")
    # faulthandler prints the Python stack when native code crashes the process
    # (segfault, illegal instruction, Windows exceptions), which otherwise dies silently.
    cmd = [sys.executable, "-X", "faulthandler", os.path.abspath(__file__), "--child", kind, arg,
           "--spec", spec_path, "--out", out]
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = None
    secs = round(time.time() - t0, 1)
    with open(log_path, encoding="utf-8", errors="replace") as f:
        tail = f.read()[-3000:]
    print(f"  {kind} {arg}: {secs}s, exit {code}", flush=True)
    if os.path.exists(part_out):
        with open(part_out, encoding="utf-8") as f:
            return json.load(f)
    if code is None:
        return {"status": "timeout", "secs": secs, "error": f"timed out after {timeout_s // 60} min",
                "log_tail": tail}
    return {"status": "crash", "secs": secs, "error": f"process {exit_reason(code)} before reporting",
            "log_tail": tail}


# Exit codes worth naming: native crashes and kills, which leave no Python traceback.
WINDOWS_CODES = {0xC0000005: "access violation", 0xC000001D: "illegal CPU instruction",
                 0xC00000FD: "stack overflow", 0xC0000409: "stack buffer overrun",
                 0xC0000094: "integer divide by zero", 0xC0000374: "heap corruption"}
SIGNALS = {4: "illegal CPU instruction (SIGILL)", 6: "aborted (SIGABRT)", 7: "bus error (SIGBUS)",
           8: "floating point exception (SIGFPE)", 9: "killed (SIGKILL), often out of memory",
           11: "segmentation fault (SIGSEGV)"}


def exit_reason(code):
    """'exited with code 3221225501 (0xC000001D, illegal CPU instruction)' and the like."""
    if code is not None and code < 0 and -code in SIGNALS:
        return f"was killed by signal {-code}: {SIGNALS[-code]}"
    if code is not None and code > 128 and code - 128 in SIGNALS and sys.platform != "win32":
        return f"exited with code {code}: {SIGNALS[code - 128]}"
    unsigned = code & 0xFFFFFFFF if code is not None else None
    if unsigned in WINDOWS_CODES:
        return f"exited with code {code} (0x{unsigned:08X}, {WINDOWS_CODES[unsigned]})"
    return f"exited with code {code}"


def write_part(out, kind, arg, data):
    with open(os.path.join(out, "parts", f"{kind}-{arg}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


class Checks:
    """Records named checks; a failing check never stops the others."""

    def __init__(self):
        self.rows = []

    def run(self, name, fn):
        t0 = time.time()
        try:
            detail = fn() or {}
            self.rows.append({"name": name, "status": "pass", "secs": round(time.time() - t0, 2), **detail})
            print(f"PASS {name} {detail}", flush=True)
            return detail
        except Exception as e:
            self.rows.append({"name": name, "status": "fail", "secs": round(time.time() - t0, 2),
                              "error": f"{type(e).__name__}: {e}"[:400],
                              "trace": traceback.format_exc()[-2000:]})
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return None


def check_audio(audio, sr, text=None):
    import numpy as np
    a = np.asarray(audio)
    assert a.ndim == 1, f"expected mono 1-D audio, got shape {a.shape}"
    assert a.size, "no audio"
    assert np.isfinite(a).all(), "audio contains NaN or inf"
    dur = a.size / sr
    peak = float(np.abs(a).max())
    assert peak > 1e-3, f"audio is silent (peak {peak:.2g})"
    if text:
        cps = len(text) / dur
        assert 3 <= cps <= 40, f"{dur:.1f}s of audio for {len(text)} characters is implausible"
    return round(dur, 3)


def child_package(spec, out):
    checks = Checks()

    def imports():
        import kittenml
        from kittenml import KittenTTS  # noqa: F401
        return {"version": kittenml.__version__}

    def normalize():
        from kittenml import normalize_text
        got = normalize_text("I paid $3.50 for 2 apples.")
        assert "$" not in got and "three" in got.lower(), got
        return {"output": got}

    checks.run("import", imports)
    checks.run("normalize_text", normalize)
    write_part(out, "package", "all", {"checks": checks.rows})


def child_model(spec, key, out):
    import numpy as np
    import soundfile as sf
    from kittenml import KittenTTS

    m_spec = next(m for m in spec["models"] if m["key"] == key)
    text, voice = spec["text"], spec["voice"]
    res = {"key": key, "label": m_spec["label"], "repo": m_spec["repo"], "status": "pass"}
    os.makedirs(os.path.join(out, "audio"), exist_ok=True)
    wav = os.path.join(out, "audio", f"{key}.wav")
    checks = Checks()

    t0 = time.time()
    kwargs = {"weights": m_spec["weights"]} if m_spec.get("weights") else {}
    model = KittenTTS(m_spec["repo"], **kwargs)
    res["load_s"] = round(time.time() - t0, 2)
    sr = getattr(model, "sample_rate", 24000)
    res["sample_rate"] = sr
    res["voices"] = len(model.available_voices)
    assert voice in model.available_voices, f"voice {voice!r} missing from {model.available_voices}"

    times, audio = [], None
    for i in range(1 + int(m_spec.get("warm_runs", 0))):
        t0 = time.time()
        audio = model.generate(text, voice=voice)
        times.append(time.time() - t0)
        res["audio_s"] = check_audio(audio, sr, text)
        print(f"generate #{i + 1}: {times[-1]:.2f}s for {res['audio_s']}s of audio", flush=True)
    sf.write(wav, np.asarray(audio), sr)
    res["peak_rss_mb"] = peak_rss_mb()   # load + generate; the checks below can load more
    res["first_s"] = round(times[0], 3)
    res["warm_s"] = [round(t, 3) for t in times[1:]]
    res["wav"] = os.path.relpath(wav, out)
    res["asr_refs"] = [{"wav": res["wav"], "text": text, "label": m_spec["label"]}]

    def stream():
        t0 = time.time()
        first, chunks = None, []
        for c in model.generate_stream(text, voice=voice):
            first = first if first is not None else time.time() - t0
            chunks.append(np.asarray(c))
        assert chunks, "stream yielded nothing"
        return {"chunks": len(chunks), "first_chunk_s": round(first, 2),
                "audio_s": check_audio(np.concatenate(chunks), sr, text)}

    def speed():
        slow = check_audio(model.generate(text, voice=voice, speed=0.8), sr)
        assert slow > res["audio_s"] * 1.1, f"speed=0.8 gave {slow}s vs {res['audio_s']}s at 1.0"
        return {"audio_s": slow}

    def to_file():
        path = os.path.join(out, "audio", f"{key}-to-file.wav")
        model.generate_to_file(text, path, voice=voice)
        data, file_sr = sf.read(path)
        return {"audio_s": check_audio(data, file_sr)}

    def expression():
        a = model.generate(EXPRESSION_TEXT, voice="Kiki", preset="expressive")
        return {"audio_s": check_audio(a, sr)}

    def clone():
        path = os.path.join(out, "audio", f"{key}-clone.wav")
        a = model.generate(CLONE_TEXT, reference=wav, reference_text=text)
        sf.write(path, np.asarray(a), sr)
        res["asr_refs"].append({"wav": os.path.relpath(path, out), "text": CLONE_TEXT,
                                "label": f"{m_spec['label']} · clone"})
        return {"audio_s": check_audio(a, sr, CLONE_TEXT)}

    def clone_whisper():
        a = model.generate(CLONE_TEXT, reference=wav)
        return {"audio_s": check_audio(a, sr, CLONE_TEXT)}

    def emb4():
        # README: KittenTTS("KittenML/kitten-tts-2", weights="emb4"). Drop the main
        # model first so the two never sit in memory together.
        nonlocal model
        model = None
        gc.collect()
        t0 = time.time()
        small = KittenTTS(m_spec["repo"], weights="emb4")
        load_s = round(time.time() - t0, 2)
        return {"load_s": load_s, "audio_s": check_audio(small.generate(text, voice=voice), sr, text)}

    known = {"stream": stream, "speed": speed, "to_file": to_file, "expression": expression,
             "clone": clone, "clone_whisper": clone_whisper, "emb4": emb4}
    wanted = m_spec.get("checks", [])
    for name in [c for c in wanted if c != "emb4"] + [c for c in wanted if c == "emb4"]:   # emb4 drops the model
        checks.run(name, known[name])
    res["checks"] = checks.rows
    failed = [c["name"] for c in checks.rows if c["status"] != "pass"]
    if failed:
        res["status"] = "fail"
        res["error"] = "failed checks: " + ", ".join(failed)
    res["peak_rss_checks_mb"] = peak_rss_mb()
    write_part(out, "model", key, res)


def child_model_safe(spec, key, out):
    """child_model, but load/generate errors become a failed row, not a crash."""
    try:
        child_model(spec, key, out)
    except Exception as e:
        m_spec = next(m for m in spec["models"] if m["key"] == key)
        traceback.print_exc()
        write_part(out, "model", key, {
            "key": key, "label": m_spec["label"], "repo": m_spec["repo"], "status": "fail",
            "error": f"{type(e).__name__}: {e}"[:400], "trace": traceback.format_exc()[-2000:],
            "peak_rss_mb": peak_rss_mb()})


def child_asr(spec, out):
    asr = spec["asr"]
    with open(os.path.join(out, "parts", "asr-refs.json"), encoding="utf-8") as f:
        refs = json.load(f)
    try:
        import librosa
        from transformers import pipeline
    except ImportError as e:
        write_part(out, "asr", "all", {"status": "skipped", "reason": f"{e}", "rows": []})
        return
    t0 = time.time()
    pipe = pipeline("automatic-speech-recognition", model=asr["model"], device="cpu")
    rows = []
    for ref in refs:
        try:
            audio, _ = librosa.load(os.path.join(out, ref["wav"]), sr=16000)
            hyp = pipe(audio)["text"].strip()
            w, edits, n = wer(ref["text"], hyp)
            rows.append({**ref, "transcript": hyp, "wer": round(w, 4), "edits": edits, "words": n})
            print(f"{ref['label']}: WER {w:.1%} — {hyp}", flush=True)
        except Exception as e:
            rows.append({**ref, "wer": None, "error": f"{type(e).__name__}: {e}"[:300]})
    write_part(out, "asr", "all", {"status": "done", "model": asr["model"],
                                   "secs": round(time.time() - t0, 1), "rows": rows})


# ── Driver ───────────────────────────────────────────────────────────────────────

def drive(spec_path, out):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    for d in ("parts", "logs", "audio"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    t_start = time.time()
    result = {"spec": spec, "env": system_info()}
    print(json.dumps(result["env"]), flush=True)

    print("Installing...", flush=True)
    result["install"] = install(out)
    print(json.dumps({k: v for k, v in result["install"].items() if k != "log_tail"}), flush=True)

    if result["install"]["ok"]:
        result["package"] = run_child("package", "all", spec_path, out, 300)
        result["models"] = []
        refs = []
        for m in spec["models"]:
            row = run_child("model", m["key"], spec_path, out, int(m["timeout_minutes"] * 60))
            row.setdefault("key", m["key"])
            row.setdefault("label", m["label"])
            row.setdefault("repo", m["repo"])
            result["models"].append(row)
            refs += row.get("asr_refs", [])
        if spec["asr"].get("enabled") and refs:
            with open(os.path.join(out, "parts", "asr-refs.json"), "w", encoding="utf-8") as f:
                json.dump(refs, f)
            result["asr"] = run_child("asr", "all", spec_path, out, 20 * 60)

    result["secs"] = round(time.time() - t_start, 1)
    status, reasons, failing = classify(result)
    result["status"], result["reasons"], result["failing"] = status, reasons, failing
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"\n{spec['name']} · py{spec['python']}: {STATUS_LABEL[status]}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)
    return 1 if failing else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", nargs=2, metavar=("KIND", "ARG"))
    args = ap.parse_args()
    if not args.child:
        sys.exit(drive(args.spec, args.out))
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)
    kind, arg = args.child
    if kind == "package":
        child_package(spec, args.out)
    elif kind == "model":
        child_model_safe(spec, arg, args.out)
    elif kind == "asr":
        child_asr(spec, args.out)


if __name__ == "__main__":
    main()

"""One platform job: install kittenml, run each README example, transcribe, write result.json.

    python qa/run_target.py --spec spec.json --out qa-out

The spec is one job from qa/plan.py. QA_SOURCE says what to install: "checkout"
(default, this repository) or a pip requirement such as "kittenml==0.9.3".

Each test is one README example in its own Python process, stopped after
[limits] step_minutes. A test that hangs, segfaults or exit()s (espeak does) is
reported, not fatal, and one that crashes or stalls is run once more. Everything
before the install uses the standard library only.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qa_common import TESTS, classify, test_keys, test_name, wer  # noqa: E402

EXPRESSION_TEXT = "[joyful] We actually won the grant <laugh> I can (((hardly))) believe it!"
CLONE_TEXT = "This is my own voice, cloned from a short recording."


# ── System info ──────────────────────────────────────────────────────────────────

def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def system_info():
    """CPU, cores, RAM and the RAM free when the job starts."""
    info = {"os": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "cpu": "", "ram_gb": None, "ram_free_gb": None}
    try:
        if sys.platform == "linux":
            for line in _run(["lscpu"]).splitlines():
                if line.startswith("Model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
            with open("/proc/meminfo") as f:
                mem = {line.split(":")[0]: int(line.split()[1]) for line in f}
            info["ram_gb"] = round(mem["MemTotal"] / 2**20, 1)
            info["ram_free_gb"] = round(mem.get("MemAvailable", 0) / 2**20, 1)
        elif sys.platform == "darwin":
            info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            info["ram_gb"] = round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1)
            vm = _run(["vm_stat"])
            page = int((re.search(r"page size of (\d+)", vm) or [0, 4096])[1])
            pages = {k.strip(): int(v.strip(" .")) for k, v in re.findall(r"^(Pages [^:]+):\s+(\d+)", vm, re.M)}
            free = sum(pages.get(f"Pages {k}", 0) for k in ("free", "inactive", "speculative", "purgeable"))
            info["ram_free_gb"] = round(free * page / 2**30, 1)
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
            info["ram_free_gb"] = round(ms.ullAvailPhys / 2**30, 1)
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


def span(secs):
    secs = int(secs)
    return f"{secs // 60} min" if secs >= 60 else f"{secs} s"


# ── Install ──────────────────────────────────────────────────────────────────────

PY_REFUSED = re.compile(r"requires a different Python: \S+ not in '([^']+)'")
NO_BUILD = re.compile(r"(?:No matching distribution found for|"
                      r"Could not find a version that satisfies the requirement) ([^\s(]+)")


def pip(args, timeout_s):
    """(exit code or None on timeout, output)."""
    try:
        proc = subprocess.run([sys.executable, "-m", "pip"] + args, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", timeout=max(timeout_s, 1))
        return proc.returncode, proc.stdout
    except subprocess.TimeoutExpired as e:
        partial = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return None, partial + f"\nERROR: pip took longer than {span(timeout_s)}"


def install(out, limit_s):
    """pip install kittenml as the README says. If that fails only because a dependency
    has no build for this platform or Python, kittenml is installed without it, so the
    tests can show what works anyway; the install itself still counts as failed."""
    source = os.environ.get("QA_SOURCE", "checkout").strip() or "checkout"
    target = os.path.dirname(HERE) if source == "checkout" else source
    pip(["install", "-q", "-U", "pip"], limit_s)
    t0 = time.time()
    code, log = pip(["install", target], limit_s)
    res = {"source": source, "ok": code == 0, "secs": round(time.time() - t0, 1)}
    if code != 0:
        lines = log.strip().splitlines()
        errs = [line for line in lines if line.startswith("ERROR")]
        res["error"] = (errs[-1] if errs else lines[-1] if lines else "pip failed")[:300]
        res["log_tail"] = log[-3000:]
        refused, missing = PY_REFUSED.search(log), NO_BUILD.findall(log)
        if refused:
            res["reason"] = f"kittenml requires Python {refused.group(1)}"
        elif code is None:
            res["reason"] = f"pip install took longer than {span(limit_s)}"
        elif missing:
            res["reason"] = f"pip finds no {missing[0]} for this platform and Python"
            if source == "checkout":
                log += partial_install(res, target, limit_s)
        else:
            res["reason"] = res["error"]
    with open(os.path.join(out, "install.log"), "w", encoding="utf-8") as f:
        f.write(log)
    if res["ok"] or res.get("without") is not None:
        res["versions"] = installed_versions()
    return res


def partial_install(res, target, limit_s):
    """kittenml with every dependency that installs here; records the ones that do not."""
    t0 = time.time()
    code, log = pip(["install", "--no-deps", target], limit_s)
    if code != 0:
        return log
    reqs = _run([sys.executable, "-c", "from importlib.metadata import requires\n"
                 "print('\\n'.join(r for r in requires('kittenml') or [] if 'extra ==' not in r))"]).splitlines()
    without = []
    for req in reqs:
        left = limit_s - (time.time() - t0)
        if left < 20:
            without.append(req)
            continue
        code, more = pip(["install", req], min(left, 300))
        log += more
        if code != 0:
            without.append(req)
    res["without"] = [re.split(r"[<>=!~;\[ ]", r, 1)[0] for r in without]
    print(f"kittenml installed without: {', '.join(res['without'])}", flush=True)
    return log


def installed_versions():
    code = ("import json\nfrom importlib.metadata import version\nout={}\n"
            "for p in ['kittenml','torch','transformers','onnxruntime','numpy','phonemizer','espeakng_loader']:\n"
            "    try: out[p]=version(p)\n    except Exception: out[p]=None\nprint(json.dumps(out))")
    try:
        return json.loads(_run([sys.executable, "-c", code]) or "{}")
    except ValueError:
        return {}


# ── Child processes ──────────────────────────────────────────────────────────────

def part_path(out, kind, arg):
    return os.path.join(out, "parts", f"{kind}-{arg.replace(':', '__')}.json")


def log_path(out, kind, arg):
    return os.path.join(out, "logs", f"{kind}-{arg.replace(':', '__')}.log")


def write_part(out, kind, arg, data):
    with open(part_path(out, kind, arg), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def run_child(kind, arg, spec_path, out, timeout_s):
    """Run one part in its own process: (its JSON or None, exit code or None on timeout, log tail)."""
    part, log = part_path(out, kind, arg), log_path(out, kind, arg)
    if os.path.exists(part):
        os.remove(part)
    # faulthandler prints the Python stack when native code crashes the process
    # (segfault, illegal instruction, Windows exceptions), which otherwise dies silently.
    cmd = [sys.executable, "-X", "faulthandler", os.path.abspath(__file__), "--child", kind, arg,
           "--spec", spec_path, "--out", out, "--limit", str(int(timeout_s))]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        try:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            code = None
    with open(log, encoding="utf-8", errors="replace") as f:
        tail = f.read()[-3000:]
    print(f"  {kind} {arg}: {time.time() - t0:.0f} s, exit {code}", flush=True)
    data = None
    if os.path.exists(part):
        with open(part, encoding="utf-8") as f:
            data = json.load(f)
    return data, code, tail, round(time.time() - t0, 1)


# Exit codes worth naming: native crashes and kills, which leave no Python traceback.
WINDOWS_CODES = {0xC0000005: "access violation", 0xC000001D: "illegal CPU instruction",
                 0xC00000FD: "stack overflow", 0xC0000409: "stack buffer overrun",
                 0xC0000094: "integer divide by zero", 0xC0000374: "heap corruption"}
SIGNALS = {4: "illegal CPU instruction (SIGILL)", 6: "aborted (SIGABRT)", 7: "bus error (SIGBUS)",
           8: "floating point exception (SIGFPE)", 9: "killed (SIGKILL), often out of memory",
           11: "segmentation fault (SIGSEGV)"}


def crash_reason(code, log=""):
    """'crashed: illegal CPU instruction (0xC000001D)' and the like."""
    if code < 0 and -code in SIGNALS:
        return f"crashed: {SIGNALS[-code]}"
    if code > 128 and code - 128 in SIGNALS and sys.platform != "win32":
        return f"crashed: {SIGNALS[code - 128]}"
    unsigned = code & 0xFFFFFFFF
    if unsigned in WINDOWS_CODES:
        return f"crashed: {WINDOWS_CODES[unsigned]} (0x{unsigned:08X})"
    # Something called exit() (espeak-ng does): its last words say why.
    said = [line.strip() for line in log.splitlines() if line.strip() and "Warning" not in line]
    return f"exited with code {code}" + (f": {said[-1][:200]}" if said else "")


def run_test(key, spec, spec_path, out, timeout_s):
    data, code, tail, secs = run_child("test", key, spec_path, out, timeout_s)
    if data is None or code is None and data.get("status") != "pass":
        data = dict(data or {}, key=key, log_tail=tail)
        if code is None:
            data.update(status="timeout", error=f"took longer than {span(timeout_s)}")
        else:
            data.update(status="crash", error=crash_reason(code, tail))
    data["secs"] = secs
    return data


def plan_tests(spec):
    tests = []
    for m in spec["models"]:
        for key in test_keys(m):
            header = TESTS[test_name(key)][0]
            tests.append({"key": key, "model": m["key"], "test": test_name(key),
                          "title": f"{m['label']} · {header}"})
    return tests


# ── The tests, each in its own process ───────────────────────────────────────────

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
        assert 3 <= cps <= 40, f"{dur:.1f} s of audio for {len(text)} characters is implausible"
    return round(dur, 3)


def child_import(spec, out):
    res = {}
    try:
        import kittenml
        from kittenml import KittenTTS, normalize_text  # noqa: F401
        res["version"] = kittenml.__version__
        got = normalize_text("I paid $3.50 for 2 apples.")
        assert "$" not in got and "three" in got.lower(), f"normalize_text gave {got!r}"
        res["ok"] = True
    except Exception as e:
        traceback.print_exc()
        res.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
    write_part(out, "import", "all", res)


def child_test(spec, key, out, limit_s, res):
    import faulthandler
    # If the parent has to stop a stalled test, the log shows where every thread was.
    faulthandler.dump_traceback_later(max(limit_s - 20, 10), exit=False)
    import numpy as np
    import soundfile as sf
    from kittenml import KittenTTS

    model_key, name = key.partition(":")[0], test_name(key)
    m = next(x for x in spec["models"] if x["key"] == model_key)
    text, voice = spec["text"], spec["voice"]
    audio_dir = os.path.join(out, "audio")

    t0 = time.time()
    weights = "emb4" if name == "emb4" else m.get("weights")
    model = KittenTTS(m["repo"], **({"weights": weights} if weights else {}))
    res["load_s"] = round(time.time() - t0, 2)
    sr = getattr(model, "sample_rate", 24000)
    print(f"loaded {m['repo']} in {res['load_s']} s", flush=True)

    def timed(fn):
        t = time.time()
        value = fn()
        return value, round(time.time() - t, 3)

    def keep(audio, suffix, said):
        """Save the clip for listening and, with `said`, for Whisper to check."""
        path = os.path.join(audio_dir, f"{model_key}{suffix}.wav")
        sf.write(path, np.asarray(audio), sr)
        res["wav"] = os.path.relpath(path, out)
        if said:
            res["said"] = said

    if name == "speak":
        assert voice in model.available_voices, f"voice {voice!r} missing from {model.available_voices}"
        times = []
        for i in range(1 + int(m.get("warm_runs", 0))):
            audio, secs = timed(lambda: model.generate(text, voice=voice))
            times.append(secs)
            print(f"generate #{i + 1}: {secs:.2f} s", flush=True)
        res.update(first_s=times[0], warm_s=times[1:], gen_s=min(times[1:] or times))
        res["audio_s"] = check_audio(audio, sr, text)
        keep(audio, "", text)
    elif name == "stream":
        t, first, chunks = time.time(), None, []
        for c in model.generate_stream(text, voice=voice):
            first = first if first is not None else round(time.time() - t, 3)
            chunks.append(np.asarray(c))
        assert chunks, "generate_stream yielded nothing"
        res.update(gen_s=round(time.time() - t, 3), first_chunk_s=first, chunks=len(chunks))
        audio = np.concatenate(chunks)
        res["audio_s"] = check_audio(audio, sr, text)
        keep(audio, "-stream", text)
    elif name == "speed":
        normal = check_audio(model.generate(text, voice=voice), sr)
        audio, res["gen_s"] = timed(lambda: model.generate(text, voice=voice, speed=0.8))
        res["audio_s"] = check_audio(audio, sr, text)
        assert res["audio_s"] > normal * 1.1, f"speed=0.8 gave {res['audio_s']} s of audio, speed=1.0 {normal} s"
        keep(audio, "-speed", text)
    elif name == "to_file":
        path = os.path.join(audio_dir, f"{model_key}-to-file.wav")
        _, res["gen_s"] = timed(lambda: model.generate_to_file(text, path, voice=voice))
        data, file_sr = sf.read(path)
        res["audio_s"] = check_audio(data, file_sr, text)
        res.update(wav=os.path.relpath(path, out), said=text)
    elif name == "expression":
        audio, res["gen_s"] = timed(lambda: model.generate(EXPRESSION_TEXT, voice="Kiki", preset="expressive"))
        res["audio_s"] = check_audio(audio, sr)
        keep(audio, "-expression", None)
    elif name in ("clone", "clone_whisper"):
        reference = os.path.join(audio_dir, f"{model_key}.wav")
        if not os.path.exists(reference):
            raise RuntimeError("no reference clip to clone: the Speak test made none")
        kwargs = {"reference": reference}
        if name == "clone":
            kwargs["reference_text"] = text
        audio, res["gen_s"] = timed(lambda: model.generate(CLONE_TEXT, **kwargs))
        res["audio_s"] = check_audio(audio, sr, CLONE_TEXT)
        keep(audio, "-" + name.replace("_", "-"), CLONE_TEXT)
    elif name == "emb4":
        audio, res["gen_s"] = timed(lambda: model.generate(text, voice=voice))
        res["audio_s"] = check_audio(audio, sr, text)
        keep(audio, "-emb4", text)
    if res.get("gen_s") and res.get("audio_s"):
        res["rtf"] = round(res["gen_s"] / res["audio_s"], 3)
    res["status"] = "pass"


def child_test_safe(spec, key, out, limit_s):
    """child_test, but an exception becomes a failed test with its message."""
    res = {"key": key}
    try:
        child_test(spec, key, out, limit_s, res)
    except Exception as e:
        traceback.print_exc()
        res.update(status="fail", error=f"{type(e).__name__}: {e}"[:400], trace=traceback.format_exc()[-2000:])
    res["peak_rss_mb"] = peak_rss_mb()
    print(f"{key}: {res['status']} {res.get('error', '')}", flush=True)
    write_part(out, "test", key, res)


def child_asr(spec, out):
    """Whisper transcribes every clip; the part is rewritten after each, so a timeout keeps what finished."""
    asr = spec["asr"]
    with open(os.path.join(out, "parts", "asr-refs.json"), encoding="utf-8") as f:
        refs = json.load(f)
    res = {"status": "running", "model": asr["model"], "rows": []}
    try:
        import librosa
        from transformers import pipeline
    except ImportError as e:
        write_part(out, "asr", "all", {"status": "skipped", "error": f"cannot transcribe: {e}", "rows": []})
        return
    write_part(out, "asr", "all", res)
    pipe = pipeline("automatic-speech-recognition", model=asr["model"], device="cpu")
    for ref in refs:
        try:
            audio, _ = librosa.load(os.path.join(out, ref["wav"]), sr=16000)
            heard = pipe(audio)["text"].strip()
            w, edits, n = wer(ref["said"], heard)
            res["rows"].append({"key": ref["key"], "transcript": heard, "wer": round(w, 4), "edits": edits, "words": n})
            print(f"{ref['key']}: WER {w:.1%} — {heard}", flush=True)
        except Exception as e:
            res["rows"].append({"key": ref["key"], "wer": None, "error": f"{type(e).__name__}: {e}"[:300]})
        write_part(out, "asr", "all", res)
    res["status"] = "done"
    write_part(out, "asr", "all", res)


# ── Driver ───────────────────────────────────────────────────────────────────────

def drive(spec_path, out):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    for d in ("parts", "logs", "audio"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    t_start = time.time()
    limits = spec.get("limits", {})
    limit = int(limits.get("step_minutes", 10) * 60)
    # Stop starting tests before GitHub cancels the job, so result.json is always written.
    deadline = t_start + limits.get("job_minutes", 45) * 60 - 240
    reserve = limit // 2 if spec["asr"].get("enabled") else 0     # for the transcription at the end
    result = {"spec": spec, "env": system_info(), "tests": []}
    print(json.dumps(result["env"]), flush=True)

    print("Installing...", flush=True)
    install_res = install(out, limit)
    result["install"] = install_res
    print(json.dumps({k: v for k, v in install_res.items() if k != "log_tail"}), flush=True)

    if install_res["ok"] or install_res.get("without") is not None:
        data, code, tail, _ = run_child("import", "all", spec_path, out, limit)
        install_res["import_ok"] = bool(data and data.get("ok"))
        if not install_res["import_ok"]:
            install_res["import_error"] = (data or {}).get("error") or (
                f"took longer than {span(limit)}" if code is None else crash_reason(code, tail))
        for t in plan_tests(spec):
            left = deadline - time.time() - reserve
            if left < 60:
                result["tests"].append(dict(t, status="skipped", error="not run: the job ran out of time"))
                continue
            row = run_test(t["key"], spec, spec_path, out, min(limit, left))
            left = deadline - time.time() - reserve
            if row["status"] in ("crash", "timeout") and left >= 60:
                # Run a crash or a stall once more, in a fresh process. One that passes then is flaky.
                first_log = log_path(out, "test", t["key"])
                os.replace(first_log, first_log[:-4] + "-first.log")
                again = run_test(t["key"], spec, spec_path, out, min(limit, left))
                if again["status"] == "pass":
                    again["flaky"] = f"{row['error']} the first time; passed when run again"
                elif again.get("error") == row.get("error"):
                    again["error"] += ", twice"
                else:
                    again["error"] = f"{again.get('error')} (first run: {row.get('error')})"
                row = again
            result["tests"].append(dict(t, **row))

        refs = [{"key": r["key"], "wav": r["wav"], "said": r["said"]} for r in result["tests"]
                if r.get("status") == "pass" and r.get("said") and r.get("wav")]
        if spec["asr"].get("enabled") and refs:
            with open(os.path.join(out, "parts", "asr-refs.json"), "w", encoding="utf-8") as f:
                json.dump(refs, f)
            left = max(deadline - time.time(), 60)
            data, code, tail, secs = run_child("asr", "all", spec_path, out, min(limit, left))
            asr = data or {"rows": []}
            if asr.get("status") != "done" and asr.get("status") != "skipped":
                asr["error"] = f"took longer than {span(min(limit, left))}" if code is None else crash_reason(code, tail)
                asr["status"] = "timeout" if code is None else "crash"
                asr["log_tail"] = tail
            asr["secs"] = secs
            result["asr"] = {k: v for k, v in asr.items() if k != "rows"}
            heard = {row["key"]: row for row in asr.get("rows", [])}
            for r in result["tests"]:
                if r["key"] in heard:
                    r.update({k: v for k, v in heard[r["key"]].items() if k != "key"})

    result["secs"] = round(time.time() - t_start, 1)
    status, reasons = classify(result)
    result["status"], result["reasons"] = status, reasons
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"\n{spec['name']} · py{spec['python']}: {status}", flush=True)
    for r in reasons:
        print(f"  - {r}", flush=True)
    # The verdict (did this break something that works on main?) is the report job's.
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", nargs=2, metavar=("KIND", "ARG"))
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args()
    if not args.child:
        sys.exit(drive(args.spec, args.out))
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)
    kind, arg = args.child
    if kind == "import":
        child_import(spec, args.out)
    elif kind == "test":
        child_test_safe(spec, arg, args.out, args.limit)
    elif kind == "asr":
        child_asr(spec, args.out)


if __name__ == "__main__":
    main()

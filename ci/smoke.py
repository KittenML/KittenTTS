"""End-to-end smoke test: load every model family, synthesize, check the audio, save it.

    python ci/smoke.py --out smoke-out                  # the light suite (0.8 ONNX + KittenTTS 2 tiny)
    python ci/smoke.py --out smoke-out --suite heavy    # KittenTTS 2 (1.7B) on CPU
    python ci/smoke.py --out smoke-out --suite all
    python ci/smoke.py --list                           # print the cases

Each model is loaded in its own child process, so a crash inside a native library
(espeak, ONNX Runtime, torch) is reported against that model instead of taking the
whole run down. Writes:

    <out>/wav/<case>.wav       the audio, for a human to listen to (ci/report.py)
    <out>/results.json         per-case status, timings and audio statistics

Exits non-zero when any case fails. A case whose optional dependency is missing
(MLX off Apple silicon, the Japanese extra) is recorded as skipped, not failed.
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
from dataclasses import dataclass, field

SR = 24000

EN = "One day, a little girl named Lily found a needle in her room."
EN_NORM = "On 3/14/2025 at 9:30am, Dr. Smith paid $1,234.56 for 12 kg of apples, a 15% discount."
EN_LONG = ("One day, a little girl named Lily found a needle in her room. "
           "She knew it was difficult to play with it because it was sharp. "
           "Lily wanted to use the needle to sew a button on her shirt. "
           "She asked her mom for help, and her mom smiled and said yes. "
           "They sat together by the window, where the afternoon light was best, "
           "and her mom showed her how to hold the fabric steady with one hand.")
TINY_TEXTS = {
    "en": EN,
    "de": "Guten Morgen! Wie geht es dir heute? Das Wetter ist wirklich schön.",
    "es": "Buenos días. Hoy vamos a caminar por el parque y tomar un café.",
    "it": "Buongiorno a tutti. Oggi andiamo al mercato a comprare la frutta.",
    "pt": "Bom dia! Hoje o céu está limpo e o sol brilha sobre a cidade.",
    "ru": "Доброе утро! Сегодня мы пойдём в парк и будем гулять весь день.",
    "hi": "नमस्ते, आज मौसम बहुत अच्छा है और हम बाज़ार जा रहे हैं।",
    "fr": "Bonjour à tous. Aujourd'hui, nous allons visiter le musée du Louvre.",
    "da": "God morgen. I dag skal vi cykle en tur langs havnen.",
    "fi": "Hyvää huomenta. Tänään menemme kävelylle metsään.",
    "ja": "おはようございます。今日はとても良い天気ですね。",
    "ko": "안녕하세요. 오늘은 날씨가 정말 좋네요.",
    "nl": "Goedemorgen. Vandaag gaan we fietsen langs de grachten.",
    "sv": "God morgon. Idag ska vi promenera längs stranden.",
    "sw": "Habari za asubuhi. Leo tutaenda sokoni kununua matunda.",
    "tr": "Günaydın. Bugün hava çok güzel, parka yürüyüşe gidelim.",
}


@dataclass
class Case:
    id: str
    text: str
    gen: dict = field(default_factory=dict)      # generate() keyword arguments
    mode: str = "generate"                         # generate | stream | file
    note: str = ""
    requires: str = ""                             # optional import; skipped if missing
    clone_from: str = ""                           # case id whose wav is the cloning reference
    repeat: bool = False                           # generate twice, require identical audio


@dataclass
class Group:
    """One model load, then its cases."""
    id: str
    repo: str
    load: dict
    cases: list
    suite: str = "light"
    family: str = ""
    requires: str = ""


def tiny_cases(prefix, langs, voices=None):
    voices = voices or {}
    return [Case(f"{prefix}-{l}", TINY_TEXTS[l], dict(language=l, **({"voice": voices[l]} if l in voices else {})),
                 requires="pyopenjtalk" if l == "ja" else "") for l in langs]


K2_CPU = dict(device="cpu")

GROUPS = [
    # --- KittenTTS 0.8, ONNX ------------------------------------------------------------
    Group("onnx08-nano", "KittenML/kitten-tts-nano-0.8", {}, family="KittenTTS 0.8 (ONNX)", cases=[
        Case("onnx08-nano-bruno", EN, dict(voice="Bruno")),
        Case("onnx08-nano-luna-slow", EN, dict(voice="Luna", speed=0.8), note="speed=0.8"),
        Case("onnx08-nano-normalize", EN_NORM, dict(voice="Jasper", clean_text=True), note="clean_text=True"),
        Case("onnx08-nano-stream", EN_LONG, dict(voice="Kiki"), mode="stream", note="generate_stream, concatenated"),
        Case("onnx08-nano-tofile", EN, dict(voice="Hugo"), mode="file", note="generate_to_file"),
    ]),
    Group("onnx08-nano-int8", "KittenML/kitten-tts-nano-0.8-int8", {}, family="KittenTTS 0.8 (ONNX)", cases=[
        Case("onnx08-nano-int8-rosie", EN, dict(voice="Rosie")),
    ]),
    Group("onnx08-micro", "KittenML/kitten-tts-micro-0.8", {}, family="KittenTTS 0.8 (ONNX)", cases=[
        Case("onnx08-micro-leo", EN, dict(voice="Leo")),
    ]),
    Group("onnx08-mini", "KittenML/kitten-tts-mini-0.8", {"backend": "cpu"}, family="KittenTTS 0.8 (ONNX)", cases=[
        Case("onnx08-mini-bella", EN, dict(voice="Bella"), note="backend='cpu'"),
    ]),

    # --- KittenTTS 2 nano / pico / mini ---------------------------------------------------
    Group("tiny-nano-onnx", "KittenML/kitten-tts-2-nano", {}, family="KittenTTS 2 nano/pico/mini", cases=[
        *tiny_cases("tiny-nano-onnx", ["en", "de", "es", "it", "pt", "ru", "hi", "fr"]),
        Case("tiny-nano-onnx-voice-paul", TINY_TEXTS["de"], dict(voice="Paul"), note="voice sets the language"),
        Case("tiny-nano-onnx-fast", EN, dict(voice="Eleanor", speed=1.25), note="speed=1.25"),
        Case("tiny-nano-onnx-normalize", EN_NORM, dict(voice="Martha"), note="numbers, dates, currency"),
        Case("tiny-nano-onnx-stream", EN_LONG, dict(voice="Serena"), mode="stream"),
    ]),
    Group("tiny-nano-torch", "KittenML/kitten-tts-2-nano", {"backend": "torch", "device": "cpu"},
          family="KittenTTS 2 nano/pico/mini", cases=[
        *tiny_cases("tiny-nano-torch", ["en", "de"]),
    ]),
    Group("tiny-nano-int4", "KittenML/kitten-tts-2-nano", {"weights": "int4"}, family="KittenTTS 2 nano/pico/mini",
          cases=[
        *tiny_cases("tiny-nano-int4", ["en", "de", "ru", "hi"]),
        Case("tiny-nano-int4-stream", EN_LONG, dict(voice="Serena"), mode="stream", note="weights='int4'"),
    ]),
    Group("tiny-pico-onnx", "KittenML/kitten-tts-2-pico", {}, family="KittenTTS 2 nano/pico/mini", cases=[
        *tiny_cases("tiny-pico-onnx", ["en", "es", "hi"]),
    ]),
    Group("tiny-pico-int4", "KittenML/kitten-tts-2-pico", {"weights": "int4"}, family="KittenTTS 2 nano/pico/mini",
          cases=[
        *tiny_cases("tiny-pico-int4", ["en", "es", "hi", "fr"]),
    ]),
    Group("tiny-mini-onnx", "KittenML/kitten-tts-2-mini", {}, family="KittenTTS 2 nano/pico/mini", cases=[
        *tiny_cases("tiny-mini-onnx", ["en", "de", "fr", "ru", "da", "fi", "ja", "ko", "nl", "sv", "sw", "tr"]),
        Case("tiny-mini-onnx-stream", EN_LONG, dict(language="en"), mode="stream"),
    ]),
    Group("tiny-mini-torch", "KittenML/kitten-tts-2-mini", {"backend": "torch", "device": "cpu"},
          family="KittenTTS 2 nano/pico/mini", cases=[
        *tiny_cases("tiny-mini-torch", ["en", "it"]),
    ]),
    Group("tiny-mini-int4", "KittenML/kitten-tts-2-mini", {"weights": "int4"}, family="KittenTTS 2 nano/pico/mini",
          cases=[
        *tiny_cases("tiny-mini-int4", ["en", "de", "ru", "ja", "ko", "tr"]),
        Case("tiny-mini-int4-stream", EN_LONG, dict(language="en"), mode="stream", note="weights='int4'"),
    ]),

    # --- KittenTTS 2 (1.7B), PyTorch on CPU ---------------------------------------------------
    Group("k2-default", "KittenML/kitten-tts-2", K2_CPU, suite="heavy", family="KittenTTS 2", cases=[
        Case("k2-default-bruno", EN, dict(voice="Bruno", advanced={"seed": 1})),
        Case("k2-default-seeded", "Hello, world. This should sound the same every time.",
             dict(voice="Luna", advanced={"seed": 1234}), repeat=True, note="same seed twice, identical audio"),
        Case("k2-default-expressive", "[joyful] We actually won the grant <laugh> I can (((hardly))) believe it!",
             dict(voice="Kiki", preset="expressive", advanced={"seed": 1}), note="expression tags, expressive preset"),
        Case("k2-default-normalize", EN_NORM, dict(voice="Matthew", advanced={"seed": 1}), note="text normalization"),
        Case("k2-default-french", "Bonjour à tous. Aujourd'hui, nous allons visiter le musée.",
             dict(voice="French", advanced={"seed": 1})),
        Case("k2-default-stream", EN_LONG, dict(voice="Luna", advanced={"seed": 1}), mode="stream",
             note="two chunks, generate_stream"),
        Case("k2-default-clone", "This is a cloned voice, built from a short reference recording.",
             dict(advanced={"seed": 1}), clone_from="k2-default-stream",
             note="reference= the stream case's output, with reference_text"),
    ]),
    Group("k2-small", "KittenML/kitten-tts-2", dict(K2_CPU, weights="emb4", decoder="student_w4"), suite="heavy",
          family="KittenTTS 2", cases=[
        Case("k2-small-bruno", EN, dict(voice="Bruno", advanced={"seed": 1}), note="weights=emb4, decoder=student_w4"),
        Case("k2-small-german", "Guten Morgen! Wie geht es dir heute?", dict(voice="German", advanced={"seed": 1})),
    ]),
    Group("k2-mlx", "KittenML/kitten-tts-2", {"backend": "mlx"}, suite="heavy", family="KittenTTS 2",
          requires="mlx.core", cases=[
        Case("k2-mlx-bruno", EN, dict(voice="Bruno", advanced={"seed": 1}), note="backend='mlx'"),
    ]),
]


def check_audio(audio, text, mode):
    """Problems with a waveform that a human would hear as broken. Returns (stats, errors, warnings)."""
    import numpy as np
    a = np.asarray(audio)
    errors, warnings = [], []
    if a.ndim != 1:
        errors.append(f"expected a 1-D waveform, got shape {a.shape}")
        a = a.reshape(-1)
    if a.dtype != np.float32:
        warnings.append(f"dtype {a.dtype}, expected float32")
    n = a.size
    dur = n / SR
    finite = bool(np.isfinite(a).all()) if n else True
    if not n:
        errors.append("empty waveform")
    if not finite:
        errors.append(f"{int((~np.isfinite(a)).sum())} non-finite samples")
    a = np.nan_to_num(a.astype(np.float64))
    peak = float(np.abs(a).max()) if n else 0.0
    rms = float(np.sqrt(np.mean(a ** 2))) if n else 0.0
    # Fraction of 20 ms frames that are effectively silent.
    frame = SR // 50
    if n >= frame:
        frames = a[: n - n % frame].reshape(-1, frame)
        frame_rms = np.sqrt((frames ** 2).mean(axis=1))
        silent = float((frame_rms < 0.003).mean())
    else:
        silent = 1.0
    clipped = float((np.abs(a) >= 0.999).mean()) if n else 0.0
    # ~12-16 characters per second of speech for alphabetic scripts; CJK packs more per char.
    # Measured on the spoken form: "$1,234.56" is far longer said than written.
    if text.isascii():                               # the normalizer is English-only
        try:
            from kittenml import normalize_text
            text = normalize_text(text) or text
        except Exception:
            pass
    chars = len(text)
    cjk = sum(1 for c in text if ord(c) > 0x2E80)
    expected = (chars - cjk) / 14 + cjk / 5
    stats = dict(duration_s=round(dur, 3), expected_s=round(expected, 2), peak=round(peak, 4),
                 rms=round(rms, 4), silent_frac=round(silent, 3), clipped_frac=round(clipped, 5))
    if n:
        if rms < 0.005:
            errors.append(f"near-silent output (rms {rms:.4f})")
        if dur < 0.35 * expected:
            errors.append(f"too short: {dur:.2f}s for ~{expected:.1f}s of text")
        elif dur < 0.55 * expected:
            warnings.append(f"short: {dur:.2f}s for ~{expected:.1f}s of text")
        if dur > 3.5 * expected + 2:
            errors.append(f"too long: {dur:.2f}s for ~{expected:.1f}s of text (runaway generation?)")
        elif dur > 2.2 * expected + 1:
            warnings.append(f"long: {dur:.2f}s for ~{expected:.1f}s of text")
        if silent > 0.6:
            errors.append(f"{silent:.0%} of the clip is silence")
        elif silent > 0.4:
            warnings.append(f"{silent:.0%} of the clip is silence")
        if peak > 1.0:
            warnings.append(f"peak {peak:.3f} exceeds full scale")
        if clipped > 0.001:
            warnings.append(f"{clipped:.2%} of samples clipped")
    return stats, errors, warnings


def importable(name):
    if not name:
        return True
    import importlib
    try:
        importlib.import_module(name)
    except Exception:
        return False
    if name == "mlx.core":           # importable on a Mac without a usable Metal device
        import mlx.core as mx
        try:
            return bool(mx.metal.is_available())
        except Exception:
            return False
    return True


def run_group(group, out):
    """Child process: load one model, run its cases. Writes <out>/parts/<group>.json."""
    import numpy as np
    import soundfile as sf

    wav_dir = os.path.join(out, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(os.path.join(out, "parts"), exist_ok=True)
    part = os.path.join(out, "parts", f"{group.id}.json")
    results = []
    # Use every CPU: PyTorch defaults to one thread per physical core, which is a single
    # thread on a 2-vCPU runner and makes KittenTTS 2 take most of an hour.
    try:
        import torch
        torch.set_num_threads(max(torch.get_num_threads(), os.cpu_count() or 1))
    except Exception:
        pass

    def record(case, **kw):
        r = dict(id=case.id, group=group.id, family=group.family, repo=group.repo, load=group.load,
                 gen=case.gen, mode=case.mode, text=case.text, note=case.note)
        r.update(kw)
        results.append(r)
        # Written after every case, so a timeout or a native crash keeps what finished.
        with open(part + ".tmp", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        os.replace(part + ".tmp", part)
        status = r["status"].upper()
        extra = "; ".join(r.get("errors", []) + r.get("warnings", [])) or r.get("reason", "")
        print(f"  [{status:7}] {case.id:32} {r.get('stats', {}).get('duration_s', ''):>7} {extra}", flush=True)

    if not importable(group.requires):
        for c in group.cases:
            record(c, status="skipped", reason=f"{group.requires} not available")
        return results

    t0 = time.perf_counter()
    try:
        from kittenml import KittenTTS
        model = KittenTTS(group.repo, **group.load)
    except Exception as e:
        tb = traceback.format_exc()
        for c in group.cases:
            record(c, status="failed", errors=[f"model load failed: {type(e).__name__}: {e}"], traceback=tb)
        return results
    load_s = time.perf_counter() - t0
    sr = getattr(model, "sample_rate", SR)

    for c in group.cases:
        if not importable(c.requires):
            record(c, status="skipped", reason=f"{c.requires} not installed")
            continue
        path = os.path.join(wav_dir, f"{c.id}.wav")
        gen = dict(c.gen)
        errors, warnings = [], []
        try:
            if c.clone_from:
                ref = os.path.join(wav_dir, f"{c.clone_from}.wav")
                if not os.path.exists(ref):
                    record(c, status="skipped", reason=f"reference {c.clone_from} was not produced")
                    continue
                ref_case = next(x for x in group.cases if x.id == c.clone_from)
                gen.update(reference=ref, reference_text=ref_case.text)
            t = time.perf_counter()
            if c.mode == "stream":
                chunks = [np.asarray(x) for x in model.generate_stream(c.text, **gen)]
                if not chunks:
                    raise RuntimeError("generate_stream yielded nothing")
                audio = np.concatenate(chunks)
                extra = dict(chunks=len(chunks))
            elif c.mode == "file":
                tmp = path + ".tmp.wav"
                model.generate_to_file(c.text, tmp, **gen)
                audio, file_sr = sf.read(tmp, dtype="float32")
                os.remove(tmp)
                if file_sr != sr:
                    errors.append(f"file sample rate {file_sr}, expected {sr}")
                extra = {}
            else:
                audio = model.generate(c.text, **gen)
                extra = {}
            gen_s = time.perf_counter() - t
            if c.repeat:
                again = model.generate(c.text, **gen)
                if np.asarray(again).shape != np.asarray(audio).shape or not np.allclose(again, audio, atol=1e-4):
                    errors.append("same seed produced different audio")
            stats, e2, w2 = check_audio(audio, c.text, c.mode)
            errors += e2
            warnings += w2
            stats.update(gen_s=round(gen_s, 2), load_s=round(load_s, 2),
                         rtf=round(gen_s / max(stats["duration_s"], 1e-6), 3), **extra)
            sf.write(path, np.clip(np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1)), -1, 1), sr,
                     subtype="PCM_16")
            record(c, status="failed" if errors else ("warning" if warnings else "passed"),
                   stats=stats, errors=errors, warnings=warnings, wav=f"wav/{c.id}.wav")
        except Exception as e:
            record(c, status="failed", errors=[f"{type(e).__name__}: {e}"], traceback=traceback.format_exc())
    del model
    gc.collect()
    return results


def environment():
    info = dict(os=platform.system(), os_release=platform.release(), machine=platform.machine(),
                python=platform.python_version(), runner=os.environ.get("CI_LABEL", ""),
                cpus=os.cpu_count())
    for mod in ("kittenml", "torch", "onnxruntime", "numpy", "transformers", "phonemizer", "misaki",
                "espeakng_loader", "spacy", "mlx"):
        try:
            from importlib.metadata import version
            info[mod] = version(mod)
        except Exception:
            pass
    try:
        import kittenml
        info["kittenml_from"] = os.path.dirname(kittenml.__file__)   # installed wheel, not the checkout
    except Exception:
        pass
    try:
        import torch
        info["torch_threads"] = torch.get_num_threads()
        info["cuda"] = torch.cuda.is_available()
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="smoke-out")
    ap.add_argument("--suite", choices=["light", "heavy", "all"], default="light")
    ap.add_argument("--only", nargs="*", help="run only these group ids")
    ap.add_argument("--group-timeout", type=float, default=4500,
                    help="seconds before a model group is killed and its remaining cases failed")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--_group", help=argparse.SUPPRESS)
    args = ap.parse_args()

    by_id = {g.id: g for g in GROUPS}
    if args._group:                                  # child process
        run_group(by_id[args._group], args.out)
        return 0

    groups = [g for g in GROUPS if args.suite == "all" or g.suite == args.suite]
    if args.only:
        groups = [g for g in groups if g.id in args.only]
    if args.list:
        for g in groups:
            print(f"{g.id}  ({g.repo} {g.load})")
            for c in g.cases:
                print(f"    {c.id}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    env = environment()
    print(json.dumps(env, indent=1), flush=True)
    results = []
    for g in groups:
        print(f"== {g.id}: {g.repo} {g.load}", flush=True)
        part = os.path.join(args.out, "parts", f"{g.id}.json")
        if os.path.exists(part):
            os.remove(part)
        t = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--out", args.out, "--_group", g.id],
                                  timeout=args.group_timeout)
            code = f"exited with code {proc.returncode}"
        except subprocess.TimeoutExpired:
            code = f"was killed after the {args.group_timeout:.0f}s group timeout"
        if os.path.exists(part):
            with open(part, encoding="utf-8") as f:
                results += json.load(f)
        done = {r["id"] for r in results}
        missing = [c for c in g.cases if c.id not in done]
        if missing:
            # The child died before reaching these: a timeout, a native crash or a hard exit.
            msg = f"process {code} after {time.perf_counter() - t:.0f}s, before this case finished"
            print(f"  [FAILED ] {g.id}: {msg} ({len(missing)} cases)", flush=True)
            for c in missing:
                if True:
                    results.append(dict(id=c.id, group=g.id, family=g.family, repo=g.repo, load=g.load, gen=c.gen,
                                        mode=c.mode, text=c.text, note=c.note, status="failed", errors=[msg]))

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(env=env, suite=args.suite, results=results), f, ensure_ascii=False, indent=1)
    counts = {s: sum(r["status"] == s for r in results) for s in ("passed", "warning", "skipped", "failed")}
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items()), flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

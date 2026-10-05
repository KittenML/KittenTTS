"""Combine every platform job's result.json into the PR report.

    python qa/report.py RESULTS_DIR OUT_DIR [--plan plan.json] [--jobs jobs.json]
                        [--run-started ISO] [--baseline DIR]
    python qa/report.py --gate OUT_DIR/summary.json

The report says what works on which platform, CPU and Python, and what broke:
a test that works in the baseline run (the latest run on main) and does not
work here. Only that fails the run; something that does not work on main either
is listed as not supported yet.

Writes OUT_DIR/pr-comment.md (short), OUT_DIR/summary.md (the same plus every
job's numbers) and OUT_DIR/summary.json. jobs.json is the run's job list from
the GitHub API, for log links and job times. --gate exits 1 when something broke.
"""
import argparse
import datetime
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_common import FAILED, NO_RESULT, PASSED, STATUS_LABEL, classify  # noqa: E402

COMMENT_LIMIT = 60000
ICON = {PASSED: "✅", FAILED: "❌", NO_RESULT: "⏱️"}
SLOW = "🐢"
CHECK_TITLE = {
    "stream": "streaming (`generate_stream`)",
    "speed": "speed control (`speed=0.8`)",
    "to_file": "save to file (`generate_to_file`)",
    "expression": "expression tags (`[joyful]`, `<laugh>`, `preset`)",
    "clone": "voice cloning with a transcript",
    "clone_whisper": "voice cloning, Whisper writes the transcript",
    "emb4": 'smaller weights (`weights="emb4"`)',
}


# ── Loading ────────────────────────────────────────────────────────────────────────

def load_results(results_dir, plan):
    results = {}
    for path in glob.glob(os.path.join(results_dir, "**", "result.json"), recursive=True):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        results[r["spec"]["id"]] = r
    planned = plan.get("jobs", [])
    for spec in planned:
        if spec["id"] not in results:
            results[spec["id"]] = {"spec": spec, "env": {}, "missing": True}
    order = {s["id"]: i for i, s in enumerate(planned)}
    out = sorted(results.values(), key=lambda r: (order.get(r["spec"]["id"], 1e9), r["spec"]["id"]))
    for r in out:
        if r.get("missing"):
            r["status"], r["reasons"] = NO_RESULT, ["the job timed out, was cancelled or crashed before reporting"]
        else:
            r["status"], r["reasons"] = classify(r)
        r["outcomes"] = outcomes(r)
    return out


def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def attach_jobs(results, jobs):
    """Each result gets its job's log link and wall time from the GitHub API listing."""
    by_name = {j["name"]: j for j in jobs}
    for r in results:
        j = by_name.get(platform_name(r))
        if not j:
            continue
        r["job_url"] = j.get("html_url")
        start, end = parse_time(j.get("started_at")), parse_time(j.get("completed_at"))
        if start and end:
            r["job_secs"] = (end - start).total_seconds()


# ── Formatting helpers ─────────────────────────────────────────────────────────────

def cell(text):
    # <br> is the one tag kept: it puts a platform's CPU under its name.
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def fmt(v, digits=2, suffix=""):
    return "—" if v is None else f"{v:.{digits}f}{suffix}"


def pct(v):
    return "—" if v is None else f"{v:.1%}"


def table(header, rows, align=None):
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def minutes(secs):
    return f"{secs / 60:.0f} min" if secs >= 60 else f"{secs:.0f} s"


def py_key(v):
    return tuple(int(x) for x in v.split("."))


def py_list(versions, everyone):
    """'py3.9–3.15' for a run of consecutive versions (in `everyone`), else a list."""
    order = sorted(set(everyone), key=py_key)
    idx = sorted(order.index(v) for v in set(versions))
    runs, start = [], None
    for i, n in enumerate(idx):
        if start is None:
            start = n
        if i + 1 == len(idx) or idx[i + 1] != n + 1:
            runs.append(f"py{order[start]}" if start == n else f"py{order[start]}–{order[n]}")
            start = None
    return ", ".join(runs)


def describe(detail):
    """A check's numbers as words: {'chunks': 2, 'audio_s': 12.5} -> '2 chunks, 12.5 s of audio'."""
    words = {"chunks": "{} chunk(s)", "first_chunk_s": "first chunk after {:.2f} s",
             "audio_s": "{:.1f} s of audio", "load_s": "loaded in {:.1f} s"}
    text = ", ".join(words[k].format(v) if k in words else f"{k} {v}" for k, v in detail.items())
    return text.replace("1 chunk(s)", "1 chunk").replace(" chunk(s)", " chunks")


def platform_name(r):
    return f"{r['spec']['name']} · py{r['spec']['python']}"


def group_name(r):
    """The platform a job belongs to: 'macOS Apple Silicon · KittenTTS 2' -> 'macOS Apple Silicon'."""
    return r["spec"]["name"].split(" · ")[0]


def short_cpu(name):
    """'INTEL(R) XEON(R) PLATINUM 8573C' -> 'Intel Xeon Platinum 8573C'; drops '64-Core Processor'."""
    n = re.sub(r"\((R|TM)\)", "", name, flags=re.I)
    n = re.sub(r"\s+\d+-Core Processor|\s+CPU\s*@.*|\s+Processor$", "", n)
    words = {"INTEL": "Intel", "XEON": "Xeon", "PLATINUM": "Platinum", "GOLD": "Gold", "SILVER": "Silver"}
    return " ".join(words.get(w, w) for w in n.split())


def cpu_of(r):
    return short_cpu((r.get("env") or {}).get("cpu") or "")


def cpu_label(r):
    env = r.get("env") or {}
    if not env.get("cpu"):
        return r["spec"]["runner"]
    bits = ([f"{env['cpu_count']} cores"] if env.get("cpu_count") else []) + (
        [f"{round(env['ram_gb'])} GB"] if env.get("ram_gb") else [])
    return " · ".join([cpu_of(r)] + bits)


def cpus_of(rs):
    """'AMD EPYC 7763/9V45, Intel Xeon Platinum 8573C': the CPUs a platform's jobs drew."""
    families = {}
    for r in rs:
        cpu = cpu_of(r)
        if not cpu:
            continue
        head, _, model = cpu.rpartition(" ")
        if head and re.search(r"\d", model):
            families.setdefault(head, [])
            if model not in families[head]:
                families[head].append(model)
        else:
            families.setdefault(cpu, [])
    return ", ".join(f"{h} {'/'.join(sorted(ms))}" if ms else h for h, ms in families.items())


# ── Per-test outcomes ──────────────────────────────────────────────────────────────

def model_stats(m):
    audio = m.get("audio_s")
    warm = m.get("warm_s") or []
    runs = warm or ([m["first_s"]] if m.get("first_s") is not None else [])
    s = {"first": m.get("first_s"), "audio": audio, "best": min(runs) if runs else None,
         "p50": statistics.median(warm) if warm else None,
         "p95": sorted(warm)[min(len(warm) - 1, int(round(0.95 * (len(warm) - 1))))] if warm else None}
    s["rtf"] = s["best"] / audio if s["best"] and audio else None
    return s


def asr_rows(r):
    return (r.get("asr") or {}).get("rows", [])


def model_wer(r, m):
    for row in asr_rows(r):
        if row.get("wav") == m.get("wav"):
            return row
    return None


def wer_ok(r, wav):
    fail_above = r["spec"].get("asr", {}).get("fail_above")
    for row in asr_rows(r):
        if row.get("wav") == wav and row.get("wer") is not None and fail_above is not None:
            return row["wer"] <= fail_above
    return True


def model_status(r, m):
    """A model's status as shown in the details: a WER above asr.fail_above fails it too."""
    names = {"pass": "Passed", "fail": "Failed", "crash": "Crashed", "timeout": "Timed out", "skipped": "Not run"}
    if m.get("status") != "pass":
        return names.get(m.get("status"), str(m.get("status")))
    return "Passed" if wer_ok(r, m.get("wav")) else "Failed (WER)"


def outcomes(r):
    """{test key: True / False / "timeout"} for every test this job was asked to run.

    Keys: "install", "<model>" (it speaks the sample text) and "<model>:<check>".
    Nothing ran where kittenml did not install, so every test is False there.
    """
    out = {}
    tests = ["install"]
    for m in r["spec"].get("models", []):
        tests.append(m["key"])
        tests += [f"{m['key']}:{c}" for c in m.get("checks", [])]
    if r["status"] == NO_RESULT:
        return {}
    if not (r.get("install") or {}).get("ok"):
        return {t: False for t in tests}
    pkg = r.get("package") or {}
    out["install"] = pkg.get("status") not in ("crash", "timeout") and all(
        c["status"] == "pass" for c in pkg.get("checks", []))
    clone_wav = next((row["wav"] for row in asr_rows(r) if row.get("label", "").endswith("· clone")), None)
    for spec_m in r["spec"].get("models", []):
        m = next((x for x in r.get("models", []) if x.get("key") == spec_m["key"]), None)
        timed_out = bool(m) and m.get("status") == "timeout"
        if not m or m.get("first_s") is None:
            out[spec_m["key"]] = "timeout" if timed_out else False
        else:
            out[spec_m["key"]] = wer_ok(r, m.get("wav"))
        for c in spec_m.get("checks", []):
            row = next((x for x in (m or {}).get("checks", []) if x["name"] == c), None)
            if row and row["status"] in ("timeout", "skipped"):
                ok = "timeout"
            elif not row or row["status"] != "pass":
                ok = "timeout" if timed_out else False
            else:
                ok = wer_ok(r, clone_wav) if c == "clone" and clone_wav else True
            out[f"{spec_m['key']}:{c}"] = ok
    return out


# ── Comparing with the baseline run ────────────────────────────────────────────────

def compare(results, baseline):
    """Mark what broke and what started working since the baseline run.

    A test broke when it works in the baseline for the same platform and Python
    and does not work here, on a CPU the baseline also ran on. A CPU the baseline
    never drew cannot tell a regression from a CPU-specific problem, so a failure
    there is reported but does not fail the run.
    """
    base = {(group_name(b), b["spec"]["python"]): b for b in baseline}
    base_cpus = {}
    for b in baseline:
        base_cpus.setdefault(group_name(b), set()).add(cpu_of(b))
    for r in results:
        b = base.get((group_name(r), r["spec"]["python"]))
        r["broke"], r["fixed"], r["masked"], r["new_cpu"] = [], [], [], False
        if not b:
            continue
        cpu = cpu_of(r)
        # Installing does not depend on the CPU; what runs after it can.
        installed = (r.get("install") or {}).get("ok")
        r["new_cpu"] = bool(installed and cpu) and cpu not in base_cpus.get(group_name(r), set())
        if r["status"] == NO_RESULT:
            if b["outcomes"].get("install") is True:
                r["broke"] = ["install"]
            continue
        for key, ok in r["outcomes"].items():
            before = b["outcomes"].get(key)
            if ok is not True and before is True and not r["new_cpu"]:
                r["broke"].append(key)
            elif ok is not True and before is True:
                r["masked"].append(key)            # would have broken, but on a CPU the baseline never drew
            elif ok is True and before is not None and before is not True:
                r["fixed"].append(key)
    for r in results:
        for k in ("broke", "fixed", "masked"):
            r.setdefault(k, [])
        r["failing"] = bool(r["broke"])


def test_title(r, key):
    if key == "install":
        return "install and import"
    model, _, check = key.partition(":")
    label = next((m["label"] for m in r["spec"].get("models", []) if m["key"] == model), model)
    return f"{label}: {CHECK_TITLE.get(check, check)}" if check else f"{label}: speaks the sample text"


# ── Short report (the PR comment) ──────────────────────────────────────────────────

def groups_of(results):
    groups = {}
    for r in results:
        groups.setdefault(group_name(r), []).append(r)
    return groups


def headline(results, ctx):
    broke = [r for r in results if r["broke"]]
    works = sum(r["status"] == PASSED for r in results)
    if ctx.get("baseline"):
        against = f"[{ctx['baseline']['label']}]({ctx['baseline']['url']})" if ctx["baseline"].get("url") \
            else ctx["baseline"]["label"]
        n = sum(len(r["broke"]) for r in broke)
        verdict = (f"❌ {n} test{'s' if n != 1 else ''} broke compared with {against}" if broke else
                   f"✅ Nothing broke compared with {against}")
    else:
        verdict = "✅ Report only: there is no earlier run to compare with yet"
    versions = next(((r.get("install") or {}).get("versions") for r in results
                     if (r.get("install") or {}).get("versions")), {}) or {}
    source = "this PR" if ctx.get("pr") else "this commit"
    if ctx["source"] != "checkout":
        source = f"`{ctx['source']}`"
    bits = [f"{works} of {len(results)} platform and Python combinations work fully",
            f"kittenml {versions.get('kittenml', '?')} from {source}" + (f" ({ctx['sha'][:7]})" if ctx.get("sha") else "")]
    if ctx.get("run_url"):
        took = f" took {minutes(ctx['run_secs'])}" if ctx.get("run_secs") else ""
        bits.append(f"[run {ctx['run_id']}]({ctx['run_url']}){took}")
    return f"## {verdict}\n\n{' · '.join(bits)}"


def platform_cell(rs):
    """One Platforms cell: does kittenml work for this platform on this Python?"""
    if not rs:
        return ""
    if any(r["broke"] for r in rs):
        return "❌ new"
    if any(r["status"] == FAILED for r in rs):
        return "❌"
    if any(r["status"] == NO_RESULT for r in rs):
        return "⏱️"
    return "✅ new" if any(r["fixed"] for r in rs) else "✅"


def first_problem(r):
    """The one line that says why a job does not fully work."""
    reason = r["reasons"][0] if r["reasons"] else r["status"]
    return re.sub(r"\s+", " ", reason).strip()[:220]


def platforms_section(results, slow_minutes):
    groups = groups_of(results)
    pythons = sorted({r["spec"]["python"] for r in results}, key=py_key)
    rows, slow = [], []
    for name, rs in groups.items():
        ran = [r for r in rs if r.get("models")]
        times = [r.get("job_secs") or r.get("secs") for r in ran if r.get("job_secs") or r.get("secs")]
        if times:
            lo, hi = min(times), max(times)
            took = minutes(hi) if minutes(lo) == minutes(hi) else (
                f"{lo / 60:.0f}–{minutes(hi)}" if lo >= 60 else f"{minutes(lo)}–{minutes(hi)}")
            if hi > slow_minutes * 60:
                took = f"{SLOW} {took}"
                slowest = max(ran, key=lambda r: r.get("job_secs") or r.get("secs") or 0)
                slow.append(f"{platform_name(slowest)} took {minutes(hi)}")
        else:
            took = "—"
        rows.append([name, cpus_of(rs) or rs[0]["spec"]["runner"]]
                    + [platform_cell([r for r in rs if r["spec"]["python"] == py]) for py in pythons] + [took])
    md = "### Platforms\n\n" + table(["Platform", "CPUs"] + [f"py{p}" for p in pythons] + ["Job time"], rows,
                                     ["---", "---"] + [":---:"] * len(pythons) + ["---:"])
    notes = []
    # Why each ❌: one line per platform and reason, with the Python versions it covers.
    problems = {}
    for r in results:
        if r["status"] == FAILED and not r["broke"]:
            where = group_name(r) + (f" on {cpu_of(r)}" if (r.get("install") or {}).get("ok") and cpu_of(r) else "")
            problems.setdefault((where, first_problem(r)), []).append(r["spec"]["python"])
    if problems:
        notes.append("**Does not work** (" + ("on the baseline too, so it does not fail the run" if any(
            r.get("compared") for r in results) else "nothing to compare with yet") + "):\n\n"
            + "\n".join(f"- {where} {py_list(pys, pythons)}: {reason}" for (where, reason), pys in problems.items()))
    fixed = [r for r in results if r["fixed"] and not r["broke"]]
    if fixed:
        notes.append("**Works now, did not before:** " + "; ".join(
            f"{platform_name(r)} ({', '.join(test_title(r, k) for k in r['fixed'][:3])})" for r in fixed))
    unclear = [r for r in results if r["masked"]]
    if unclear:
        notes.append("**On a CPU the baseline never drew**, so not counted as broken: " + ", ".join(
            f"{platform_name(r)} on {cpu_of(r)}" for r in unclear))
    if slow:
        notes.append(f"{SLOW} slower than {slow_minutes} min: " + "; ".join(slow))
    return md + ("\n\n" + "\n\n".join(notes) if notes else "")


def is_tts2(m):
    return "kitten-tts-2" in (m.get("repo") or "")


def family_keys(results, tts2):
    """[(title, key)] for one model family's tests, in config order."""
    keys, seen = ([] if tts2 else [("Install and import", "install")]), set()
    for r in results:
        for m in r["spec"].get("models", []):
            if is_tts2(m) != tts2:
                continue
            for key, title in [(m["key"], f"{m['label']}: speaks the sample text")] + [
                    (f"{m['key']}:{c}", f"{m['label']}: {CHECK_TITLE.get(c, c)}") for c in m.get("checks", [])]:
                if key not in seen:
                    seen.add(key)
                    keys.append((title, key))
    return keys


def cpu_columns(results):
    """[(header, jobs)]: one column per platform and CPU the runners drew."""
    cols = {}
    for r in results:
        cols.setdefault((group_name(r), cpu_of(r) or r["spec"]["runner"]), []).append(r)
    return [(f"{group}<br>{cpu}", rs) for (group, cpu), rs in cols.items()]


def test_cell(rs, key):
    many = len({r["spec"]["python"] for r in rs}) > 1
    pys = lambda xs: (" " + ", ".join(f"py{p}" for p in sorted(set(xs), key=py_key))) if many else ""   # noqa: E731
    broke = [r["spec"]["python"] for r in rs if key in r["broke"]]
    bad = [r["spec"]["python"] for r in rs if r["outcomes"].get(key) is False]
    late = [r["spec"]["python"] for r in rs if r["outcomes"].get(key) == "timeout"]
    if broke:
        return "❌ new" + pys(broke)
    if bad:
        return "❌" + pys(bad)
    if late:
        return "⏱️" + pys(late)
    if any(r["status"] == NO_RESULT for r in rs):
        return "⏱️"
    return "✅" if any(key in r["outcomes"] for r in rs) else "—"


def speed_cell(rs, label):
    rtfs = [model_stats(m)["rtf"] for r in rs for m in r.get("models", []) if m["label"] == label and model_stats(m)["rtf"]]
    if not rtfs:
        return "—"
    v = statistics.median(rtfs)
    text = f"{v:.2f}" if v < 10 else f"{v:.0f}"
    return f"{SLOW} {text}" if v > 1 else text


def family_section(results, tts2, first):
    keys = family_keys(results, tts2)
    cols = cpu_columns(results)
    if not keys or not cols or (tts2 and not keys):
        return ""
    rows = [[title] + [test_cell(rs, key) for _, rs in cols] for title, key in keys]
    labels = []
    for r in results:
        for m in r["spec"].get("models", []):
            if is_tts2(m) == tts2 and m["label"] not in labels:
                labels.append(m["label"])
    speed = [[f"{label} RTF"] + [speed_cell(rs, label) for _, rs in cols] for label in labels]
    if tts2:
        peaks = []
        for _, rs in cols:
            mb = [m["peak_rss_mb"] for r in rs for m in r.get("models", []) if m.get("label") in labels and m.get("peak_rss_mb")]
            peaks.append(f"{max(mb) / 1024:.1f} GB" if mb else "—")
        speed.append(["Peak RAM"] + peaks)
    headers = [h for h, _ in cols]
    intro = ""
    if first:
        fail_above = results[0]["spec"].get("asr", {}).get("fail_above", 0)
        intro = ("The README's examples on every Python version, one column per platform and CPU the runners "
                 "drew. ✅ works · ❌ does not · **new** = changed since the baseline · ⏱️ took too long · "
                 f"— not run there. Whisper must hear the text (WER ≤ {fail_above:.0%}). RTF is generation time "
                 f"÷ audio length, best warm run, median across Python versions; {SLOW} is slower than realtime.\n\n")
    title = "### KittenTTS 2" if tts2 else "### KittenTTS 0.8 (ONNX models)"
    return (f"{title}\n\n{intro}" + table(["Test"] + headers, rows, ["---"] + [":---:"] * len(headers))
            + "\n\n" + table(["Speed"] + headers, speed, ["---"] + ["---:"] * len(headers)))


def log_for(r, key):
    """The error and log tail behind one failed test."""
    install = r.get("install") or {}
    if r["status"] == NO_RESULT:
        return r["reasons"][0], ""
    if not install.get("ok"):
        return f"install failed: {install.get('error', '')}", install.get("log_tail", "")
    if key == "install":
        pkg = r.get("package") or {}
        bad = [c for c in pkg.get("checks", []) if c["status"] != "pass"]
        return (pkg.get("error") or (bad[0].get("error") if bad else "package checks failed"),
                pkg.get("log_tail") or (bad[0].get("trace", "") if bad else ""))
    model, _, check = key.partition(":")
    m = next((x for x in r.get("models", []) if x.get("key") == model), {})
    if check:
        c = next((x for x in m.get("checks", []) if x["name"] == check), {})
        return c.get("error") or m.get("error") or "did not run", c.get("trace") or m.get("log_tail") or ""
    if m.get("first_s") is not None and not wer_ok(r, m.get("wav")):
        w = model_wer(r, m) or {}
        return f"WER {pct(w.get('wer'))}, Whisper heard “{(w.get('transcript') or '')[:120]}”", ""
    return m.get("error") or m.get("status", "failed"), m.get("trace") or m.get("log_tail") or ""


def failures_section(results, log_chars, only_broken):
    items = []
    for r in results:
        keys = r["broke"] if only_broken else [k for k, ok in r["outcomes"].items() if ok is not True] or (
            ["install"] if r["status"] == NO_RESULT else [])
        link = f" · [log]({r['job_url']})" if r.get("job_url") else ""
        for key in keys:
            error, log = log_for(r, key)
            items.append((f"**{platform_name(r)}** · {test_title(r, key)}: {cell(error)}{link}", log))
            if key == "install" and not (r.get("install") or {}).get("ok"):
                break                      # one line is enough when nothing installed
    if not items:
        return ""
    # Each log is a top-level block under its line: nested in a list item, an
    # unindented log line ends the item and the fence swallows the rest. Four
    # backticks so a log containing ``` cannot close it early.
    blocks = []
    for line, log in items:
        blocks.append(f"- ❌ {line}")
        if log.strip() and log_chars:
            blocks.append(f"<details><summary>Log</summary>\n\n````\n{log.strip()[-log_chars:]}\n````\n\n</details>")
    title = "### Broke since the baseline" if only_broken else "### Everything that does not work"
    return f"{title}\n\n" + "\n\n".join(blocks)


def footer(results, ctx):
    spec = results[0]["spec"] if results else {}
    texts = [r["spec"].get("text", "") for r in results]
    text = max(set(texts), key=texts.count) if texts else ""
    asr = spec.get("asr", {})
    about = [f"Sample text ({len(text)} characters, voice {spec.get('voice', '?')}): “{text}”"]
    if asr.get("enabled"):
        about.append(f"WER: `{asr.get('model')}` on the runner; case and punctuation do not count, "
                     f"KittenTTS == Kitten TTS; fails above {asr.get('fail_above', 0):.0%}.")
    limit = spec.get("limits", {}).get("step_minutes")
    if limit:
        about.append(f"Each install, model load, generation, test and transcription is stopped after {limit} min.")
    about.append("Compared with the latest finished run on the base branch (main), or this branch's previous run "
                 "when main has none. Only a test that works there and not here fails the run.")
    about.append("What runs is set in `qa/config.toml`.")
    where = f"the [run summary]({ctx['run_url']})" if ctx.get("run_url") else "the run summary"
    return (f"Every job's numbers — load and generation times, peak RAM, transcripts, logs and audio — are in {where}."
            "\n\n<details><summary>About this run</summary>\n\n" + "\n".join(f"- {a}" for a in about) + "\n\n</details>")


# ── Details (run summary only) ─────────────────────────────────────────────────────

def details_section(results):
    out = ["### Every job\n"]
    for r in results:
        if not r.get("models"):
            continue
        install = r.get("install") or {}
        v = install.get("versions") or {}
        meta = [f"Install {fmt(install.get('secs'), 0, ' s')}"] + [
            f"{k} {v[k]}" for k in ("kittenml", "torch", "onnxruntime") if v.get(k)]
        if r.get("job_secs") or r.get("secs"):
            meta.append(f"job {minutes(r.get('job_secs') or r['secs'])}")
        for m in r["models"]:
            extra = m.get("peak_rss_checks_mb") or 0
            if extra > (m.get("peak_rss_mb") or 0) * 1.2:
                meta.append(f"{m['label']} peaks at {extra / 1024:.1f} GB during its tests")
        if r.get("audio_url"):
            meta.append(f"[audio]({r['audio_url']})")
        if r.get("job_url"):
            meta.append(f"[log]({r['job_url']})")
        perf = []
        for m in r["models"]:
            s = model_stats(m)
            w = model_wer(r, m) or {}
            warm = ("—" if s["p50"] is None else fmt(s["p50"], 3) if len(m.get("warm_s") or []) == 1
                    else f"{fmt(s['p50'], 3)} / {fmt(s['p95'], 3)}")
            perf.append([m["label"], model_status(r, m), fmt(m.get("load_s")), fmt(s["first"], 3), warm,
                         fmt(s["rtf"], 3), fmt(s["audio"], 2),
                         f"{m['peak_rss_mb'] / 1024:.1f} GB" if m.get("peak_rss_mb") else "—", pct(w.get("wer"))])
        checks = []
        for m in r["models"]:
            for c in m.get("checks", []):
                extra = {k: v for k, v in c.items() if k not in ("name", "status", "secs", "trace", "error")}
                checks.append([m["label"], CHECK_TITLE.get(c["name"], c["name"]),
                               {"pass": "Passed", "timeout": "Timed out", "skipped": "Not run"}.get(c["status"], "Failed"),
                               fmt(c.get("secs"), 1), c.get("error") or describe(extra)])
        heard = [[row["label"], pct(row.get("wer")), (row.get("transcript") or row.get("error") or "")[:300]]
                 for row in asr_rows(r) if row.get("wer") != 0]
        exact = sum(row.get("wer") == 0 for row in asr_rows(r))
        asr_md = ""
        if heard:
            asr_md = "What Whisper heard, where it differs from the text:\n\n" + table(
                ["Audio", "WER", "Transcript"], heard, ["---", "---:", "---"])
        if exact:
            asr_md += ("\n\n" if asr_md else "") + (
                "The clip transcribes word for word." if exact == 1 else f"{exact} clips transcribe word for word.")
        out.append(
            f"<details><summary><b>{platform_name(r)}</b> — {ICON[r['status']]} {STATUS_LABEL[r['status']]}"
            f" · {cpu_label(r)}</summary>\n\n{' · '.join(meta)}\n\n"
            + table(["Model", "Status", "Load (s)", "First gen (s)", "Warm p50/p95 (s)", "Best RTF",
                     "Audio (s)", "Peak RAM", "WER"], perf,
                    ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"])
            + (f"\n\n{asr_md}" if asr_md else "")
            + ("\n\n" + table(["Model", "Test", "Status", "Time (s)", "Result"], checks,
                              ["---", "---", "---", "---:", "---"]) if checks else "")
            + "\n\n</details>\n")
    return "\n".join(out) if len(out) > 1 else ""


def build(results, ctx, slow_minutes):
    short = [headline(results, ctx), platforms_section(results, slow_minutes),
             family_section(results, tts2=False, first=True), family_section(results, tts2=True, first=False)]
    comment = "\n\n".join(p for p in short + [failures_section(results, 1500, True), footer(results, ctx)] if p)
    if len(comment) > COMMENT_LIMIT:
        comment = "\n\n".join(p for p in short + [failures_section(results, 0, True), footer(results, ctx)] if p)
    full = "\n\n".join(p for p in short + [failures_section(results, 3000, True),
                                           failures_section(results, 1500, False), details_section(results)] if p)
    return full, comment[:COMMENT_LIMIT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--plan")
    ap.add_argument("--jobs", help="the run's jobs from the GitHub API, for log links and job times")
    ap.add_argument("--run-started", help="the run's start time (ISO 8601), for its total time")
    ap.add_argument("--baseline", help="result.json files of the run to compare with, and its about.json")
    ap.add_argument("--gate")
    args = ap.parse_args()

    if args.gate:
        with open(args.gate, encoding="utf-8") as f:
            summary = json.load(f)
        for j in summary["jobs"]:
            for key in j["broke"]:
                print(f"BROKE {j['platform']}: {key}")
        sys.exit(1 if summary["failing"] else 0)

    plan = {}
    if args.plan and os.path.exists(args.plan):
        with open(args.plan, encoding="utf-8") as f:
            plan = json.load(f)
    results = load_results(args.results_dir, plan)
    server, repo, run_id = (os.environ.get(k, "") for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    baseline, about = [], None
    if args.baseline and os.path.exists(os.path.join(args.baseline, "about.json")):
        with open(os.path.join(args.baseline, "about.json"), encoding="utf-8") as f:
            about = json.load(f)
        about["url"] = f"{server}/{repo}/actions/runs/{about['run_id']}" if server and about.get("run_id") else ""
        baseline = load_results(args.baseline, {})
    compare(results, baseline)
    for r in results:
        r["compared"] = bool(baseline)
    if args.jobs and os.path.exists(args.jobs):
        with open(args.jobs, encoding="utf-8") as f:
            attach_jobs(results, json.load(f))
    started = parse_time(args.run_started)
    ctx = {"source": os.environ.get("QA_SOURCE", "checkout") or "checkout",
           "pr": os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
           "sha": os.environ.get("QA_SHA") or os.environ.get("GITHUB_SHA", ""), "run_id": run_id,
           "run_url": f"{server}/{repo}/actions/runs/{run_id}" if run_id else "",
           "run_secs": (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() if started else None,
           "baseline": about if baseline else None}
    full, comment = build(results, ctx, plan.get("report", {}).get("slow_job_minutes", 30))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(full)
    with open(os.path.join(args.out_dir, "pr-comment.md"), "w", encoding="utf-8") as f:
        f.write(comment)
    jobs = [{"platform": platform_name(r), "status": r["status"], "broke": r["broke"], "fixed": r["fixed"],
             "reasons": r.get("reasons", [])} for r in results]
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"failing": sum(bool(j["broke"]) for j in jobs), "jobs": jobs}, f, indent=1)
    print(f"{len(results)} platform jobs, {sum(bool(j['broke']) for j in jobs)} with something broken; "
          f"comment {len(comment)} chars, summary {len(full)} chars")


if __name__ == "__main__":
    main()

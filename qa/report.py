"""Combine every platform job's result.json into the PR report.

    python qa/report.py RESULTS_DIR OUT_DIR [--plan plan.json] [--jobs jobs.json] [--run-started ISO]
    python qa/report.py --gate OUT_DIR/summary.json

Writes OUT_DIR/pr-comment.md (short: platforms, tests, speed, failures),
OUT_DIR/summary.md (the same plus every job's numbers, for the run summary) and
OUT_DIR/summary.json. jobs.json is the run's job list from the GitHub API, for
log links and job times. --gate exits 1 when a gating platform failed or
produced no result.
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
from qa_common import (CHANGED, FAILED, NO_RESULT, PASSED, STATUS_LABEL,  # noqa: E402
                       UNSUPPORTED, classify, known_issue)

COMMENT_LIMIT = 60000
ICON = {PASSED: "✅", FAILED: "❌", UNSUPPORTED: "➖", CHANGED: "🆕", NO_RESULT: "⏱️"}
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
            r["failing"] = r["spec"].get("gating", True)
        else:
            r["status"], r["reasons"], r["failing"] = classify(r)
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
    # <br> is the one tag kept: it puts a CPU's cores and RAM under its name.
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


def cpu_label(r, specs=True):
    """The runner's CPU; with specs, its cores and RAM after it."""
    env = r.get("env") or {}
    if not env.get("cpu"):
        return r["spec"]["runner"]
    bits = []
    if env.get("cpu_count"):
        bits.append(f"{env['cpu_count']} cores")
    if env.get("ram_gb"):
        bits.append(f"{round(env['ram_gb'])} GB")
    label = short_cpu(env["cpu"])
    return f"{label} · {' · '.join(bits)}" if specs and bits else label


def cpus_of(rs):
    """'AMD EPYC 7763/9V45, Intel Xeon Platinum 8573C': the CPUs a platform's jobs drew."""
    families = {}
    for r in rs:
        cpu = short_cpu((r.get("env") or {}).get("cpu") or "")
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


# ── Per-model numbers ──────────────────────────────────────────────────────────────

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


MODEL_STATUS = {"pass": "Passed", "fail": "Failed", "crash": "Crashed", "timeout": "Timed out"}


def model_status(r, m):
    """A model's status as shown: a WER above asr.fail_above fails it too."""
    if m.get("status") != "pass":
        return MODEL_STATUS.get(m.get("status"), str(m.get("status")))
    return "Passed" if wer_ok(r, m.get("wav")) else "Failed (WER)"


# ── Short report (the PR comment) ──────────────────────────────────────────────────

def groups_of(results):
    groups = {}
    for r in results:
        groups.setdefault(group_name(r), []).append(r)
    return groups


def headline(results, ctx):
    failing = [r for r in results if r["failing"]]
    supported = [r for r in results if r["spec"].get("expect", "works") == "works" and r["spec"].get("gating", True)]
    if failing:
        verdict = f"❌ {len(failing)} of {len(supported)} supported platform job{'s' if len(supported) != 1 else ''} failed"
    else:
        verdict = f"✅ All {len(supported)} supported platform jobs passed"
    versions = next(((r.get("install") or {}).get("versions") for r in results
                     if (r.get("install") or {}).get("versions")), {}) or {}
    source = "this PR" if ctx["source"] == "checkout" else f"`{ctx['source']}`"
    if ctx["source"] == "checkout" and not ctx.get("pr"):
        source = "this commit"
    bits = [f"kittenml {versions.get('kittenml', '?')} installed from {source}"
            + (f" ({ctx['sha'][:7]})" if ctx.get("sha") else "")]
    if ctx.get("run_url"):
        took = f" took {minutes(ctx['run_secs'])}" if ctx.get("run_secs") else ""
        bits.append(f"[run {ctx['run_id']}]({ctx['run_url']}){took}")
    return f"## {verdict}\n\n{' · '.join(bits)}"


def known_failures(r):
    """[(model label, known issue)] for models that failed in a way the config lists as known."""
    out = []
    for m in r.get("models", []):
        k = known_issue(r, m.get("key"))
        if k and (m.get("status") != "pass" or not wer_ok(r, m.get("wav"))):
            out.append((m["label"], k))
    return out


def cell_for(rs):
    """One table cell for the jobs of one platform on one Python."""
    if not rs:
        return ""
    statuses = [r["status"] for r in rs]
    if any(r["failing"] for r in rs):
        return "⏱️" if NO_RESULT in statuses and FAILED not in statuses else "❌"
    if FAILED in statuses or any(known_failures(r) for r in rs):
        return "⚠️"
    if CHANGED in statuses:
        return "🆕"
    if all(s == UNSUPPORTED for s in statuses):
        return "➖"
    return "✅"


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
                    + [cell_for([r for r in rs if r["spec"]["python"] == py]) for py in pythons] + [took])
    md = "### Platforms\n\n" + table(["Platform", "CPUs"] + [f"py{p}" for p in pythons] + ["Job time"], rows,
                                     ["---", "---"] + [":---:"] * len(pythons) + ["---:"])
    notes = []
    unsupported = {}
    for r in results:
        if r["status"] == UNSUPPORTED:
            # 'macOS Intel' when the whole platform is unsupported, 'Linux x64 py3.9' when one Python is.
            whole = all(x["status"] == UNSUPPORTED for x in groups[group_name(r)])
            label = group_name(r) if whole else f"{group_name(r)} py{r['spec']['python']}"
            unsupported.setdefault(r["spec"].get("reason") or "not supported", []).append(label)
    if unsupported:
        notes.append("➖ not supported, as expected: " + "; ".join(
            f"{', '.join(names)} ({reason.rstrip('.')})" for reason, names in unsupported.items()))
    soft = [r for r in results if r["status"] == FAILED and not r["failing"]]
    if soft:
        notes.append("⚠️ failed, but does not fail the run: " + "; ".join(
            f"{platform_name(r)} ({(r['spec'].get('reason') or '').rstrip('.')})" for r in soft))
    known = [(r, label, k) for r in results for label, k in known_failures(r)]
    if known:
        notes.append("⚠️ known issue, does not fail the run: " + "; ".join(
            f"{platform_name(r)} on {short_cpu(r['env'].get('cpu', ''))} ({k['reason'].rstrip('.')})"
            for r, label, k in known))
    changed = [r for r in results if r["status"] == CHANGED]
    if changed:
        notes.append("🆕 installs now, though `qa/config.toml` expects it not to: "
                     + ", ".join(platform_name(r) for r in changed) + " — update the config")
    if slow:
        notes.append(f"{SLOW} slower than {slow_minutes} min: " + "; ".join(slow))
    return md + ("\n\n" + "  \n".join(notes) if notes else "")


def test_rows(results):
    """(title, model key, check) for every test any job was asked to run, in config order."""
    rows, seen = [("Install and import", None, None)], set()
    for r in results:
        for m in r["spec"].get("models", []):
            if (m["key"], None) not in seen:
                seen.add((m["key"], None))
                rows.append((f"{m['label']}: speaks the sample text", m["key"], None))
            for c in m.get("checks", []):
                if (m["key"], c) not in seen:
                    seen.add((m["key"], c))
                    rows.append((f"{m['label']}: {CHECK_TITLE.get(c, c)}", m["key"], c))
    # Keep each model's checks under its own generate row.
    order = {key: i for i, (_, key, _) in enumerate(rows) if key}
    return [rows[0]] + sorted(rows[1:], key=lambda t: (order[t[1]], t[2] is not None))


def test_passed(r, key, check):
    """True/False for one test in one job, None when the job was not asked to run it."""
    if r["spec"].get("expect", "works") != "works":
        return None                       # known-unsupported: the Platforms table covers it
    installed = (r.get("install") or {}).get("ok")
    if key is None:
        if not installed:
            return False if r["spec"].get("gating", True) else None
        pkg = r.get("package") or {}
        return pkg.get("status") not in ("crash", "timeout") and all(
            c["status"] == "pass" for c in pkg.get("checks", []))
    if not installed:
        return None                       # nothing ran; the install row already says why
    spec_m = next((m for m in r["spec"].get("models", []) if m["key"] == key), None)
    if not spec_m or (check and check not in spec_m.get("checks", [])):
        return None
    m = next((m for m in r.get("models", []) if m.get("key") == key), None)
    if not m:
        return False
    if check is None:
        return m.get("first_s") is not None and wer_ok(r, m.get("wav"))
    c = next((c for c in m.get("checks", []) if c["name"] == check), None)
    if not c or c["status"] != "pass":
        return False
    clone_wav = next((row["wav"] for row in asr_rows(r) if row.get("label", "").endswith("· clone")), None)
    return wer_ok(r, clone_wav) if check == "clone" and clone_wav else True


def tests_section(results):
    groups = {name: rs for name, rs in groups_of(results).items() if any(r.get("models") for r in rs)}
    if not groups:
        return ""
    rows = []
    for title, key, check in test_rows(results):
        row = [title]
        for rs in groups.values():
            outcomes = [(r, test_passed(r, key, check)) for r in rs if r["status"] != NO_RESULT]
            outcomes = [(r, ok) for r, ok in outcomes if ok is not None]
            missing = [r for r in rs if r["status"] == NO_RESULT and (
                key is None or any(m["key"] == key for m in r["spec"].get("models", [])))]
            is_known = lambda r: key is not None and known_issue(r, key) is not None   # noqa: E731
            hard = sorted({r["spec"]["python"] for r, ok in outcomes
                           if not ok and r["spec"].get("gating", True) and not is_known(r)}, key=py_key)
            soft = sorted({r["spec"]["python"] for r, ok in outcomes
                           if not ok and (not r["spec"].get("gating", True) or is_known(r))}, key=py_key)
            if hard:
                row.append("❌ " + ", ".join(f"py{p}" for p in hard))
            elif missing:
                row.append("⏱️")
            elif soft:
                row.append("⚠️ " + ", ".join(f"py{p}" for p in soft))
            elif outcomes:
                row.append("✅")
            else:
                row.append("—")
        rows.append(row)
    return ("### Tests\n\nThe README's examples, run against this code on every Python version listed above. "
            "A Whisper transcript must match the text (WER ≤ "
            f"{results[0]['spec'].get('asr', {}).get('fail_above', 0):.0%}). — = not run on that platform in this run "
            "(`qa/config.toml` says where each test runs).\n\n"
            + table(["Test"] + list(groups), rows, ["---"] + [":---:"] * len(groups)))


def speed_section(results):
    groups = {name: rs for name, rs in groups_of(results).items() if any(r.get("models") for r in rs)}
    labels = []
    for r in results:
        for m in r.get("models", []):
            if m["label"] not in labels:
                labels.append(m["label"])
    if not labels:
        return ""
    rows = []
    for name, rs in groups.items():
        row = [name]
        for label in labels:
            rtfs = [model_stats(m)["rtf"] for r in rs for m in r.get("models", [])
                    if m["label"] == label and model_stats(m)["rtf"]]
            if not rtfs:
                row.append("—")
                continue
            v = statistics.median(rtfs)
            text = f"{v:.2f}" if v < 10 else f"{v:.0f}"
            row.append(f"{SLOW} {text}" if v > 1 else text)
        rows.append(row)
    return ("### Speed\n\nReal-time factor: generation time ÷ audio length, best warm run, median across Python "
            f"versions. Below 1 is faster than realtime; {SLOW} is slower.\n\n"
            + table(["Platform"] + labels, rows, ["---"] + ["---:"] * len(labels)))


def failure_items(results, trace_chars):
    """(one-line summary, log tail) for everything that fails the run, plus flagged changes."""
    items = []
    for r in results:
        if not r["failing"]:
            continue
        name = f"**{platform_name(r)}**"
        link = f" · [log]({r['job_url']})" if r.get("job_url") else ""
        install = r.get("install") or {}
        if r["status"] == NO_RESULT:
            items.append((f"{name}: no result — {r['reasons'][0]}{link}", ""))
            continue
        if install and not install.get("ok"):
            items.append((f"{name}: install failed — `{cell(install.get('error', ''))}`{link}",
                          install.get("log_tail", "")))
            continue
        for part, label in (("package", "package checks"), ("asr", "WER transcription")):
            p = r.get(part) or {}
            if p.get("status") in ("crash", "timeout"):
                items.append((f"{name} · {label}: {cell(p.get('error', p['status']))}{link}", p.get("log_tail", "")))
        for c in (r.get("package") or {}).get("checks", []):
            if c["status"] != "pass":
                items.append((f"{name} · {c['name']}: {cell(c.get('error', ''))}{link}", c.get("trace", "")))
        for m in r.get("models", []):
            if known_issue(r, m.get("key")):
                continue
            bad = [c for c in m.get("checks", []) if c["status"] != "pass"]
            w = model_wer(r, m) or {}
            if m.get("status") != "pass" and not bad:
                items.append((f"{name} · {m['label']}: {cell(m.get('error', m.get('status')))}{link}",
                              m.get("trace") or m.get("log_tail") or ""))
            for c in bad:
                items.append((f"{name} · {m['label']} · {CHECK_TITLE.get(c['name'], c['name'])}: "
                              f"{cell(c.get('error', ''))}{link}", c.get("trace", "")))
            if m.get("status") == "pass" and not wer_ok(r, m.get("wav")):
                items.append((f"{name} · {m['label']}: WER {pct(w.get('wer'))} — Whisper heard "
                              f"“{cell((w.get('transcript') or '')[:120])}”{link}", ""))
    return [(s, t[-trace_chars:]) for s, t in items]


def failures_section(results, trace_chars):
    items = failure_items(results, trace_chars)
    if not items:
        return ""
    # Each log is a top-level block under its line: nested in a list item, an
    # unindented log line ends the item and the fence swallows the rest of the
    # comment. Four backticks so a log containing ``` cannot close it early.
    blocks = []
    for line, log in items:
        blocks.append(f"- ❌ {line}")
        if log.strip():
            blocks.append(f"<details><summary>Log</summary>\n\n````\n{log.strip()}\n````\n\n</details>")
    return "### Failures\n\n" + "\n\n".join(blocks)


def footer(results, ctx):
    spec = results[0]["spec"] if results else {}
    texts = [r["spec"].get("text", "") for r in results]
    text = max(set(texts), key=texts.count) if texts else ""
    asr = spec.get("asr", {})
    about = [f"Sample text ({len(text)} characters, voice {spec.get('voice', '?')}): “{text}”"]
    if asr.get("enabled"):
        about.append(f"WER: `{asr.get('model')}` on the runner; case and punctuation do not count, "
                     f"KittenTTS == Kitten TTS; flagged above {asr.get('warn_above', 0):.0%}, "
                     f"fails above {asr.get('fail_above', 0):.0%}.")
    about.append("What runs is set in `qa/config.toml`.")
    where = f"the [run summary]({ctx['run_url']})" if ctx.get("run_url") else "the run summary"
    return (f"Every job's numbers — load and generation times, peak RAM, transcripts, audio downloads — are in {where}."
            "\n\n<details><summary>About this run</summary>\n\n" + "\n".join(f"- {a}" for a in about) + "\n\n</details>")


# ── Details (run summary only) ─────────────────────────────────────────────────────

def details_section(results):
    out = ["### Every job\n"]
    for r in results:
        if r["status"] == UNSUPPORTED or not r.get("models"):
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
                meta.append(f"{m['label']} peaks at {extra / 1024:.1f} GB during its checks")
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
                               "Passed" if c["status"] == "pass" else "Failed",
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
    short = [headline(results, ctx), platforms_section(results, slow_minutes), tests_section(results),
             speed_section(results)]
    comment = "\n\n".join(p for p in short + [failures_section(results, 1500), footer(results, ctx)] if p)
    if len(comment) > COMMENT_LIMIT:
        comment = "\n\n".join(p for p in short + [failures_section(results, 0), footer(results, ctx)] if p)
    full = "\n\n".join(p for p in short + [failures_section(results, 3000), details_section(results)] if p)
    return full, comment[:COMMENT_LIMIT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--plan")
    ap.add_argument("--jobs", help="the run's jobs from the GitHub API, for log links and job times")
    ap.add_argument("--run-started", help="the run's start time (ISO 8601), for its total time")
    ap.add_argument("--gate")
    args = ap.parse_args()

    if args.gate:
        with open(args.gate, encoding="utf-8") as f:
            summary = json.load(f)
        for j in summary["jobs"]:
            if j["failing"]:
                print(f"FAILED {j['platform']}: {STATUS_LABEL[j['status']]} — {'; '.join(j['reasons'])}")
        sys.exit(1 if summary["failing"] else 0)

    plan = {}
    if args.plan and os.path.exists(args.plan):
        with open(args.plan, encoding="utf-8") as f:
            plan = json.load(f)
    results = load_results(args.results_dir, plan)
    if args.jobs and os.path.exists(args.jobs):
        with open(args.jobs, encoding="utf-8") as f:
            attach_jobs(results, json.load(f))
    server, repo, run_id = (os.environ.get(k, "") for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    started = parse_time(args.run_started)
    ctx = {"source": os.environ.get("QA_SOURCE", "checkout") or "checkout",
           "pr": os.environ.get("GITHUB_EVENT_NAME") == "pull_request",
           "sha": os.environ.get("QA_SHA") or os.environ.get("GITHUB_SHA", ""), "run_id": run_id,
           "run_url": f"{server}/{repo}/actions/runs/{run_id}" if run_id else "",
           "run_secs": (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() if started else None}
    full, comment = build(results, ctx, plan.get("report", {}).get("slow_job_minutes", 30))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(full)
    with open(os.path.join(args.out_dir, "pr-comment.md"), "w", encoding="utf-8") as f:
        f.write(comment)
    jobs = [{"platform": platform_name(r), "status": r["status"], "failing": r["failing"],
             "reasons": r.get("reasons", [])} for r in results]
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"failing": sum(j["failing"] for j in jobs), "jobs": jobs}, f, indent=1)
    print(f"{len(results)} platform jobs, {sum(j['failing'] for j in jobs)} failing; "
          f"comment {len(comment)} chars, summary {len(full)} chars")


if __name__ == "__main__":
    main()

"""Combine every platform job's result.json into the PR report.

    python qa/report.py RESULTS_DIR OUT_DIR [--plan plan.json] [--jobs jobs.json]
                        [--run-started ISO] [--baseline DIR]
    python qa/report.py --gate OUT_DIR/summary.json

Laid out like the React Native SDK's report: a summary, one status row per
platform, then what each model family does on each platform and CPU. A run
fails only when a test that works in the baseline run (the latest run on main)
stops working here. Everything else is reported, not failed.

Writes OUT_DIR/pr-comment.md, OUT_DIR/summary.md (the same plus every job's
numbers) and OUT_DIR/summary.json. jobs.json is the run's job list from the
GitHub API, for log links and job times. --gate exits 1 when something broke.
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
from qa_common import NO_RESULT, PASSED, TESTS, classify, install_ok, test_keys, test_name, test_ok, why  # noqa: E402

COMMENT_LIMIT = 60000
SLOW = "🐢"
TITLE = "# KittenTTS Python Platform Report"


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
            r["status"], r["reasons"] = NO_RESULT, ["GitHub did not run the job to the end (no runner, or cancelled)"]
        else:
            r["status"], r["reasons"] = classify(r)
        r["rows"] = {t["key"]: t for t in r.get("tests", [])}
        r["outcomes"] = outcomes(r)
    return out


def outcomes(r):
    """{test key: "pass" / "fail" / "timeout" / "skipped"}: "install", then every test that ran."""
    if r["status"] == NO_RESULT:
        return {}
    fail_above = r["spec"].get("asr", {}).get("fail_above")
    out = {"install": "pass" if install_ok(r) else "fail"}
    for row in r.get("tests", []):
        s = row.get("status")
        out[row["key"]] = s if s in ("timeout", "skipped") else "pass" if test_ok(row, fail_above) else "fail"
    return out


def planned_keys(r):
    return ["install"] + [k for m in r["spec"].get("models", []) for k in test_keys(m)]


def parse_time(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def attach_jobs(results, jobs):
    """Each result gets its job's log link and wall time from the GitHub API listing."""
    by_name = {j["name"]: j for j in jobs}
    for r in results:
        j = by_name.get(job_name(r))
        if not j:
            continue
        r["job_url"] = j.get("html_url")
        start, end = parse_time(j.get("started_at")), parse_time(j.get("completed_at"))
        if start and end:
            r["job_secs"] = (end - start).total_seconds()


# ── Formatting helpers ─────────────────────────────────────────────────────────────

def cell(text):
    # <br> is the one tag kept: it stacks several places in one cell.
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def table(header, rows, align=None):
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def minutes(secs):
    return f"{secs / 60:.0f} min" if secs >= 60 else f"{secs:.0f} s"


def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def pct(v):
    return "—" if v is None else f"{v:.0%}" if v in (0, 1) else f"{v:.1%}"


def rtf_text(v):
    if v is None:
        return "—"
    text = f"{v:.2f}" if v < 10 else f"{v:.0f}"
    return f"{SLOW} {text}" if v > 1 else text


def py_key(v):
    return tuple(int(x) for x in v.split("."))


def py_ranges(versions, everyone):
    """'3.10–3.14' for consecutive versions (in the order of `everyone`), else a list."""
    order = sorted(set(everyone) | set(versions), key=py_key)
    idx = sorted(order.index(v) for v in set(versions))
    runs, start = [], None
    for i, n in enumerate(idx):
        if start is None:
            start = n
        if i + 1 == len(idx) or idx[i + 1] != n + 1:
            runs.append(order[start] if start == n else f"{order[start]}–{order[n]}")
            start = None
    return ", ".join(runs)


def job_name(r):
    """The GitHub job's name, as the workflow sets it."""
    return f"{r['spec']['name']} · py{r['spec']['python']}"


def where(r):
    return f"{r['spec']['name']} · {r['spec']['python']}"


def short_cpu(name):
    """'INTEL(R) XEON(R) PLATINUM 8573C' -> 'Intel Xeon Platinum 8573C'; drops '64-Core Processor'."""
    n = re.sub(r"\((R|TM)\)", "", name or "", flags=re.I)
    n = re.sub(r"\s+\d+-Core Processor|\s+CPU\s*@.*|\s+Processor$", "", n)
    words = {"INTEL": "Intel", "XEON": "Xeon", "PLATINUM": "Platinum", "GOLD": "Gold", "SILVER": "Silver"}
    return " ".join(words.get(w, w) for w in n.split())


def cpu_of(r):
    return short_cpu((r.get("env") or {}).get("cpu"))


def cpus_text(rs):
    """'AMD EPYC 7763/9V74, Intel Xeon Platinum 8370C': the CPUs a platform's jobs drew."""
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


def platforms(results):
    groups = {}
    for r in results:
        groups.setdefault(r["spec"]["name"], []).append(r)
    return groups


def ran_tests(r):
    return bool(r.get("tests"))


def installed(r):
    """pip install kittenml worked, so the README's tests ran as users would run them."""
    return ran_tests(r) and not partial(r)


def partial(r):
    """kittenml was installed without some dependencies, since pip install kittenml fails here."""
    return (r.get("install") or {}).get("without") is not None


def title_of(r, key):
    if key == "install":
        return "Install"
    row = r["rows"].get(key) or {}
    if row.get("title"):
        return row["title"]
    m = next((m for m in r["spec"].get("models", []) if m["key"] == key.partition(":")[0]), {})
    return f"{m.get('label', key)} · {TESTS.get(test_name(key), (key,))[0]}"


def why_of(r, key):
    if r["status"] == NO_RESULT:
        return r["reasons"][0]
    if key == "install":
        install = r.get("install") or {}
        if not install.get("ok"):
            return install.get("reason") or install.get("error") or "pip install failed"
        return f"import kittenml: {install.get('import_error') or 'failed'}"
    return why(r["rows"].get(key) or {}, r["spec"].get("asr", {}).get("fail_above"))


# ── Comparing with the baseline run ────────────────────────────────────────────────

def compare(results, baseline, limit_s):
    """Mark what broke and what started working since the baseline run.

    A test broke when it works in the baseline for the same platform and Python
    and does not work here. Not counted, but listed:
    - a failure on a CPU the baseline never drew, which cannot tell a regression
      from a problem with that CPU (GitHub assigns runner CPUs at random);
    - a crash of a test that also crashed in the baseline and passed when run again;
    - a timeout on a platform where the baseline had timeouts too, or where the
      baseline took over half the limit for that test: a slow runner, not a slowdown.
    """
    base = {(b["spec"]["name"], b["spec"]["python"]): b for b in baseline}
    base_cpus, base_timeouts = {}, set()
    for b in baseline:
        if ran_tests(b):
            base_cpus.setdefault(b["spec"]["name"], set()).add(cpu_of(b))
        if "timeout" in b["outcomes"].values():
            base_timeouts.add(b["spec"]["name"])
    for r in results:
        r.update(broke=[], fixed=[], excused=[])
        b = base.get((r["spec"]["name"], r["spec"]["python"]))
        r["base"] = b
        if not b or b["status"] == NO_RESULT:
            continue
        if r["status"] == NO_RESULT:
            continue            # says nothing about the code: it is listed under Notes
        name, cpu = r["spec"]["name"], cpu_of(r)
        new_cpu = ran_tests(r) and cpu not in base_cpus.get(name, set())
        for key, now in r["outcomes"].items():
            before = b["outcomes"].get(key)
            if now == "skipped" or before in (None, "skipped"):
                continue
            if now == "pass":
                if before != "pass":
                    r["fixed"].append(key)
                continue
            if before != "pass":
                continue
            took = (b["rows"].get(key) or {}).get("secs")
            crash = ((r["rows"].get(key) or {}).get("error") or "").startswith("crashed:")
            if key != "install" and new_cpu:
                r["excused"].append((key, f"on {cpu}, which the baseline never drew"))
            elif crash and (b["rows"].get(key) or {}).get("flaky"):
                r["excused"].append((key, "crashed; it crashed in the baseline too, then passed when run again"))
            elif now == "timeout" and name in base_timeouts:
                r["excused"].append((key, "timed out; tests on this platform timed out in the baseline too"))
            elif now == "timeout" and took and took >= limit_s / 2:
                r["excused"].append((key, f"timed out; it took {minutes(took)} in the baseline too"))
            else:
                r["broke"].append(key)
    for r in results:
        for k in ("broke", "fixed", "excused"):
            r.setdefault(k, [])


# ── The PR comment ─────────────────────────────────────────────────────────────────

def against(ctx):
    b = ctx.get("baseline") or {}
    return f"[{b['label']}]({b['url']})" if b.get("url") else b.get("label", "")


def headline(results, ctx):
    broke = sum(len(r["broke"]) for r in results)
    if not ctx.get("baseline"):
        verdict = "✅ **Report only**: there is no earlier run to compare with yet."
    elif broke:
        verdict = f"❌ **{broke} test{'s' if broke != 1 else ''} broke** compared with {against(ctx)}."
    else:
        verdict = f"✅ **Nothing broke** compared with {against(ctx)}."
    return f"{TITLE}\n\n{verdict}"


def summary_section(results, ctx):
    groups = platforms(results)
    pythons = sorted({r["spec"]["python"] for r in results}, key=py_key)
    full = sum(r["status"] == PASSED for r in results)
    lost = sum(r["status"] == NO_RESULT for r in results)
    none = sum(not installed(r) for r in results) - lost
    versions = next(((r.get("install") or {}).get("versions") for r in results
                     if ((r.get("install") or {}).get("versions") or {}).get("kittenml")), None) or {}
    source = "this PR" if ctx.get("pr") else "this commit"
    if ctx["source"] != "checkout":
        source = f"`{ctx['source']}`"
    sha = f"`{ctx['sha'][:7]}`, " if ctx.get("sha") else ""
    spec = results[0]["spec"] if results else {}
    rows = [
        ["Commit", f"{sha}kittenml {versions.get('kittenml', '')} installed from {source}".replace("  ", " ")],
        ["Jobs", f"{len(results)}: {len(groups)} platform{'s' if len(groups) != 1 else ''} × "
                 f"Python {py_ranges(pythons, pythons)}"],
        ["Results", f"{full} pass every test · {len(results) - full - none - lost} pass some · {none} cannot install"
                    + (f" · {lost} no result" if lost else "")],
        ["Sample text", f"{len(spec.get('text', ''))} characters, voice {spec.get('voice', '?')}"],
    ]
    if ctx.get("baseline"):
        rows.append(["Compared with", against(ctx)])
    if ctx.get("run_url"):
        took = f", {minutes(ctx['run_secs'])}" if ctx.get("run_secs") else ""
        rows.append(["Run", f"[{ctx['run_id']}]({ctx['run_url']}){took}"])
    return "## Summary\n\n" + table(["Field", "Value"], rows)


def broke_section(results):
    rows = []
    for r in results:
        for key in r["broke"]:
            row = r["rows"].get(key) or {}
            before = (r["base"]["rows"].get(key) or {}).get("secs") if r.get("base") else None
            rows.append([r["spec"]["name"], r["spec"]["python"], cpu_of(r) or "—", title_of(r, key), why_of(r, key),
                         f"passed in {minutes(before)}" if before else "passed",
                         f"[log]({r['job_url']})" if r.get("job_url") else "—"])
            if row.get("flaky"):
                rows[-1][4] += f" ({row['flaky']})"
    if not rows:
        return ""
    return "## Broke in This PR\n\n" + table(
        ["Platform", "Python", "CPU", "Test", "What happened", "In the baseline", "Log"], rows)


def status_cell(r):
    if r["status"] == NO_RESULT:
        text = "no result"
    elif not installed(r):
        text = "❌ install"
    elif r["status"] == PASSED:
        text = "✅"
    else:
        keys = planned_keys(r)
        text = f"❌ {sum(r['outcomes'].get(k) == 'pass' for k in keys)}/{len(keys)}"
        skipped = sum(r["outcomes"].get(k) == "skipped" for k in keys)
        text += f", {skipped} not run" if skipped else ""
    if r["broke"]:
        text += " new"
    elif r["fixed"] and r["status"] == PASSED:
        text += " new"
    return text


def status_section(results, slow_minutes):
    groups = platforms(results)
    pythons = sorted({r["spec"]["python"] for r in results}, key=py_key)
    rows = []
    for name, rs in groups.items():
        times = [r.get("job_secs") or r.get("secs") for r in rs if installed(r) and (r.get("job_secs") or r.get("secs"))]
        took = "—"
        if times:
            lo, hi = min(times), max(times)
            took = minutes(hi) if minutes(lo) == minutes(hi) else f"{lo / 60:.0f}–{minutes(hi)}"
            if hi > slow_minutes * 60:
                took = f"{SLOW} {took}"
        by_py = {r["spec"]["python"]: r for r in rs}
        rows.append([name, cpus_text(rs) or rs[0]["spec"]["runner"]]
                    + [status_cell(by_py[p]) if p in by_py else "" for p in pythons] + [took])
    legend = ("✅ every test passes · ❌ 13/15: 13 of 15 tests pass · ❌ install: kittenml does not install, so "
              f"nothing ran · no result: GitHub did not run the job · **new**: changed in this PR · {SLOW} a job "
              f"over {slow_minutes} min")
    return ("## Platform Status\n\n"
            + table(["Platform", "CPUs"] + pythons + ["Runtime"], rows,
                    ["---", "---"] + [":---:"] * len(pythons) + ["---:"])
            + f"\n\n{legend}")


def problems_section(results):
    """One row per reason something does not work, with where it happens."""
    everyone = sorted({r["spec"]["python"] for r in results}, key=py_key)
    order = list(platforms(results))
    groups = {}          # (what model, why) -> {"tests": [...], "where": {platform: {python: cpu}}}
    for r in results:
        keys = [k for k, v in r["outcomes"].items() if v in ("fail", "timeout") and (k == "install" or installed(r))]
        for key in keys:
            title = "Install" if key == "install" else title_of(r, key)
            model, _, test = title.partition(" · ")
            g = groups.setdefault((model, why_of(r, key)), {"tests": [], "where": {}})
            if test and test not in g["tests"]:
                g["tests"].append(test)
            g["where"].setdefault(r["spec"]["name"], {})[r["spec"]["python"]] = cpu_of(r)
    if not groups:
        return ""
    tested_cpus = {}
    for r in results:
        if installed(r):
            tested_cpus.setdefault(r["spec"]["name"], set()).add(cpu_of(r))
    rows = []
    for (model, reason), g in groups.items():
        places = {}      # the same Python versions on several platforms share a line
        for name in sorted(g["where"], key=order.index):
            pys = g["where"][name]
            text = py_ranges(list(pys), everyone)
            text = "every Python" if set(pys) == set(everyone) and len(everyone) > 1 else text
            cpus = {c for c in pys.values() if c}
            if model != "Install" and cpus and cpus < tested_cpus.get(name, set()):
                text += " on " + ", ".join(sorted(cpus))
            places.setdefault(text, []).append(name)
        lines = []
        for text, names in places.items():
            lines.append(f"{'Every platform' if len(names) == len(order) > 1 else ', '.join(names)} · {text}")
        tests = sorted(g["tests"], key=lambda t: [h for h, _ in TESTS.values()].index(t)
                       if t in [h for h, _ in TESTS.values()] else 99)
        everything = next((len(test_keys(m)) for r in results for m in r["spec"].get("models", [])
                           if m.get("label") == model), None)
        what = model if not tests else f"{model}: every test" if len(tests) == everything and len(tests) > 1 \
            else f"{model}: {', '.join(tests)}"
        rows.append([what, reason, "<br>".join(lines)])
    rows.sort(key=lambda row: (row[0] != "Install", row[0]))
    return "## What Does Not Work\n\n" + table(["What", "Why", "Where"], rows)


def is_tts2(m):
    return "kitten-tts-2" in (m.get("repo") or "")


def family_models(results, tts2):
    models, seen = [], set()
    for r in results:
        for m in r["spec"].get("models", []):
            if is_tts2(m) == tts2 and m["key"] not in seen:
                seen.add(m["key"])
                models.append(m)
    return models


def family_section(results, tts2):
    models = family_models(results, tts2)
    if not models:
        return ""
    columns = [("Speak" if len(models) == 1 else m["label"], m["key"]) for m in models]   # (header, key)
    for m in models:
        for key in test_keys(m)[1:]:
            header = TESTS[test_name(key)][0]
            columns.append((header if sum(test_name(key) in x.get("checks", []) for x in models) == 1
                            else f"{m['label']} {header}", key))
    devices = {}
    for r in results:
        if installed(r):
            devices.setdefault((r["spec"]["name"], cpu_of(r) or r["spec"]["runner"]), []).append(r)
    if not devices:
        return ""
    keys = {key for _, key in columns}
    rows, changed = [], {}
    for (name, cpu), rs in devices.items():
        label = name
        settings = {k: v for r in rs for m in models for k, v in r["spec"].get("overrides", {}).get(m["key"], {}).items()}
        if settings:
            label += " ¹"
            changed[name] = settings
        cells = [test_cell(rs, key) for _, key in columns]
        tested = [row for r in rs for k, row in r["rows"].items() if k in keys]
        wers = [row["wer"] for row in tested if row.get("wer") is not None]
        wer = pct(statistics.mean(wers)) if wers else "—"
        rtfs = {}
        for m in models:
            vals = [r["rows"][m["key"]]["rtf"] for r in rs
                    if r["outcomes"].get(m["key"]) == "pass" and r["rows"][m["key"]].get("rtf")]
            if vals:
                rtfs[m["label"]] = statistics.median(vals)
        if tts2:
            # The Speak test's: what generate() needs. Cloning without a transcript loads Whisper on top.
            peaks = [r["rows"][m["key"]]["peak_rss_mb"] for r in rs for m in models
                     if r["outcomes"].get(m["key"]) == "pass" and r["rows"][m["key"]].get("peak_rss_mb")]
            extra = [rtf_text(next(iter(rtfs.values()), None)), f"{max(peaks) / 1024:.1f} GB" if peaks else "—", wer]
        else:
            extra = [rtf_text(statistics.mean(rtfs.values())) if rtfs else "—", wer]
        rows.append([f"{label}<br>{cpu}"] + cells + extra)
    metrics = ["RTF", "Peak RAM", "WER"] if tts2 else ["RTF", "WER"]
    headers = ["Platform"] + [h for h, _ in columns] + metrics
    align = ["---"] + [":---:"] * len(columns) + ["---:"] * len(metrics)
    title = "## KittenTTS 2" if tts2 else "## KittenTTS 0.8 (ONNX)"
    tests = "; ".join(f"**{h}** {TESTS[test_name(k)][1]}" for h, k in columns if test_name(k) != "speak")
    speak = "**Speak** `generate()` speaks the sample text" if len(models) == 1 else (
        f"**{', '.join(m['label'] for m in models)}** each speak the sample text with `generate()`")
    users = [m["label"] for m in models if m.get("checks")]
    if len(models) > 1 and len(users) == 1:
        tests = f"with {users[0]}: {tests}"
    notes = [f"{speak}; {tests}." if tests else f"{speak}.",
             f"RTF: generation time ÷ audio length, best run, median over Python versions"
             + ("" if tts2 else ", averaged over the models") + f"; {SLOW} slower than realtime."]
    if tts2:
        notes.append("Peak RAM: the Speak test's; the run summary has every test's.")
    for name, settings in changed.items():
        notes.append(f"¹ {name} runs these with {', '.join(f'`{k}={v!r}`' for k, v in settings.items())} "
                     "(`overrides` in qa/config.toml).")
    return f"{title}\n\n" + table(headers, rows, align) + "\n\n" + "\n".join(f"- {n}" for n in notes)


def without_section(results):
    """What works where pip install kittenml fails, with kittenml installed without what pip could not install."""
    rows = {}
    families = [(tts2, family_models(results, tts2)) for tts2 in (False, True)]
    for r in results:
        if not partial(r):
            continue
        cells = []
        for tts2, models in families:
            keys = [k for m in models for k in test_keys(m)]
            if keys:
                passed = sum(r["outcomes"].get(k) == "pass" for k in keys)
                cells.append(f"{'✅' if passed == len(keys) else '❌'} {passed}/{len(keys)}")
        key = (", ".join(r["install"]["without"]), tuple(cells))
        rows.setdefault(key, {}).setdefault(r["spec"]["name"], []).append(r["spec"]["python"])
    if not rows:
        return ""
    everyone = sorted({r["spec"]["python"] for r in results}, key=py_key)
    out = []
    for (without, cells), where in rows.items():
        by_pys = {}
        for name, pys in where.items():
            by_pys.setdefault(py_ranges(pys, everyone), []).append(name)
        out.append(["<br>".join(f"{', '.join(names)} · {pys}" for pys, names in by_pys.items()), without] + list(cells))
    headers = ["Where", "Installed without"] + [
        "KittenTTS 2" if tts2 else "KittenTTS 0.8" for tts2, models in families if models]
    return ("## Where pip install Fails\n\n`pip install kittenml` fails on these, so kittenml was installed without "
            "the packages pip cannot install there, to see which tests pass anyway.\n\n"
            + table(headers, out, ["---", "---"] + [":---:"] * (len(headers) - 2)))


def test_cell(rs, key):
    ran = [r for r in rs if r["outcomes"].get(key) not in (None, "skipped")]
    if not ran:
        return "—"
    bad = sorted({r["spec"]["python"] for r in ran if r["outcomes"][key] in ("fail", "timeout")}, key=py_key)
    every = {r["spec"]["python"] for r in ran}
    if not bad:
        return "✅ new" if any(key in r["fixed"] for r in ran) else "✅"
    text = "❌" if set(bad) == every else "❌ " + py_ranges(bad, every)
    return text + (" new" if any(key in r["broke"] for r in ran) else "")


def tests_text(r, keys):
    """'KittenTTS 2 Speak, Stream; Nano (fp32) Speak' for some of a job's tests."""
    by_model = {}
    for k in keys:
        model, _, test = title_of(r, k).partition(" · ")
        by_model.setdefault(model, []).append(test or model)
    return "; ".join(f"{model} {', '.join(tests)}" if tests != [model] else model for model, tests in by_model.items())


def notes_section(results, slow_minutes):
    notes = []
    fixed = [f"{where(r)}: {tests_text(r, r['fixed'])}" for r in results if r["fixed"]]
    if fixed:
        notes.append("**Works now**, did not in the baseline: " + " · ".join(fixed))
    flaky = [f"{where(r)}: {row['title'].replace(' · ', ' ')} ({row['flaky']})"
             for r in results for row in r.get("tests", []) if row.get("flaky")]
    if flaky:
        notes.append("**Flaky**, failed and then passed when run again: " + " · ".join(flaky))
    excused = []
    for r in results:
        reasons = {}
        for k, reason in r["excused"]:
            reasons.setdefault(reason, []).append(k)
        excused += [f"{where(r)}: {tests_text(r, keys)} {reason}" for reason, keys in reasons.items()]
    if excused:
        notes.append("**Not counted as broken**: " + " · ".join(excused))
    unrun = [f"{where(r)}: {tests_text(r, [k for k, v in r['outcomes'].items() if v == 'skipped'])}"
             for r in results if "skipped" in r["outcomes"].values()]
    if unrun:
        notes.append("**Not run**, the job was near its time limit: " + " · ".join(unrun))
    lost = [where(r) for r in results if r["status"] == NO_RESULT]
    if lost:
        notes.append(f"**No result**, GitHub did not run the job to the end (no runner, or cancelled), so it says "
                     f"nothing about this PR: {', '.join(lost)}")
    slow = [r for r in results if (r.get("job_secs") or r.get("secs") or 0) > slow_minutes * 60]
    if slow:
        notes.append(f"{SLOW} **Slow jobs**: " + "; ".join(
            f"{where(r)} took {minutes(r.get('job_secs') or r['secs'])}" for r in slow))
    return "## Notes\n\n" + "\n".join(f"- {n}" for n in notes) if notes else ""


def footer(results, ctx):
    spec = results[0]["spec"] if results else {}
    asr, limit = spec.get("asr", {}), spec.get("limits", {}).get("step_minutes", 10)
    install = spec.get("limits", {}).get("install_minutes")
    lines = [f"Each test runs one README example in its own Python process and is stopped after {limit} min"
             + (f" (the install after {install} min)" if install and install != limit else "")
             + "; one that crashes or stalls is run once more."]
    if asr.get("enabled"):
        lines.append(f"`{asr.get('model')}` must hear the spoken text (WER ≤ {asr.get('fail_above', 0):.0%}).")
    where_ = f"the [run summary]({ctx['run_url']})" if ctx.get("run_url") else "the run summary"
    lines.append(f"Every job's numbers, transcripts, logs and audio are in {where_}.")
    about = [f"Sample text: “{spec.get('text', '')}” (voice {spec.get('voice', '?')})",
             "WER ignores case and punctuation, and KittenTTS == Kitten TTS.",
             "Compared with the latest finished run on the base branch (main), or this branch's previous run when "
             "main has none. Only a test that works there and not here fails the run.",
             "What runs is set in `qa/config.toml`."]
    return (" ".join(lines) + "\n\n<details><summary>About this run</summary>\n\n"
            + "\n".join(f"- {a}" for a in about) + "\n\n</details>")


# ── Every job's numbers (run summary only) ─────────────────────────────────────────

def details_section(results):
    out = ["## Job Details"]
    for r in results:
        env, install = r.get("env") or {}, r.get("install") or {}
        if r["status"] == NO_RESULT or not install:
            continue
        v = install.get("versions") or {}
        meta = [f"{cpu_of(r)}, {env.get('cpu_count')} cores, {env.get('ram_gb')} GB RAM"
                + (f" ({env['ram_free_gb']} GB free)" if env.get("ram_free_gb") else "")]
        meta.append(f"install {minutes(install.get('secs') or 0)}" + "".join(
            f", {k} {v[k]}" for k in ("kittenml", "torch", "onnxruntime") if v.get(k)))
        if partial(r):
            meta.append(f"installed without {', '.join(install['without'])}")
        if r.get("job_secs") or r.get("secs"):
            meta.append(f"job {minutes(r.get('job_secs') or r['secs'])}")
        meta += [f"[audio]({r['audio_url']})"] if r.get("audio_url") else []
        meta += [f"[log]({r['job_url']})"] if r.get("job_url") else []
        body = " · ".join(meta)
        if not ran_tests(r):
            body += f"\n\nInstall: {why_of(r, 'install')}"
            if install.get("log_tail"):
                body += f"\n\n````\n{install['log_tail'].strip()[-1500:]}\n````"
        else:
            rows, logs = [], []
            fail_above = r["spec"].get("asr", {}).get("fail_above")
            for row in r["tests"]:
                ok = test_ok(row, fail_above)
                status = "Passed" if ok else {"timeout": "Timed out", "skipped": "Not run", "crash": "Crashed"}.get(
                    row.get("status"), "Failed")
                gen = row.get("gen_s")
                note = ("heard word for word" if row.get("wer") == 0 else row.get("transcript")) if ok \
                    else why(row, fail_above)
                if row.get("flaky"):
                    note = f"flaky: {row['flaky']}"
                rows.append([row["title"], status, fmt(row.get("load_s"), 1), fmt(gen, 2), fmt(row.get("rtf"), 3),
                             fmt(row.get("audio_s"), 1),
                             f"{row['peak_rss_mb'] / 1024:.1f} GB" if row.get("peak_rss_mb") else "—",
                             pct(row.get("wer")), (note or "")[:200]])
                tail = row.get("log_tail") or row.get("trace")
                if not ok and tail:
                    logs.append(f"<details><summary>{row['title']} log</summary>\n\n````\n{tail.strip()[-1500:]}\n"
                                "````\n\n</details>")
            body += "\n\n" + table(["Test", "Status", "Load (s)", "Generate (s)", "RTF", "Audio (s)", "Peak RAM",
                                    "WER", "Heard / why not"], rows,
                                   ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---"])
            if logs:
                body += "\n\n" + "\n\n".join(logs)
        icon = "✅" if r["status"] == PASSED else "❌"
        out.append(f"<details><summary>{icon} <b>{where(r)}</b> — {cpu_of(r) or r['spec']['runner']}</summary>"
                   f"\n\n{body}\n\n</details>")
    return "\n\n".join(out) if len(out) > 1 else ""


def build(results, ctx, slow_minutes):
    parts = [headline(results, ctx), summary_section(results, ctx), broke_section(results),
             status_section(results, slow_minutes), problems_section(results),
             family_section(results, tts2=False), family_section(results, tts2=True), without_section(results),
             notes_section(results, slow_minutes)]
    comment = "\n\n".join(p for p in parts + [footer(results, ctx)] if p)
    full = "\n\n".join(p for p in parts + [details_section(results), footer(results, ctx)] if p)
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
        if not any("tests" in b for b in baseline):
            print("The baseline run predates per-test results; nothing to compare with.")
            baseline = []
    limit = (results[0]["spec"].get("limits", {}).get("step_minutes", 10) if results else 10) * 60
    compare(results, baseline, limit)
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
    jobs = [{"platform": job_name(r), "status": r["status"], "broke": r["broke"], "fixed": r["fixed"],
             "reasons": r.get("reasons", [])} for r in results]
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"failing": sum(bool(j["broke"]) for j in jobs), "jobs": jobs}, f, indent=1)
    print(f"{len(results)} platform jobs, {sum(bool(j['broke']) for j in jobs)} with something broken; "
          f"comment {len(comment)} chars, summary {len(full)} chars")


if __name__ == "__main__":
    main()

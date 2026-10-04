"""Combine every platform job's result.json into the PR report.

    python qa/report.py RESULTS_DIR OUT_DIR [--plan plan.json]
    python qa/report.py --gate OUT_DIR/summary.json

Writes OUT_DIR/summary.md (job summary, everything), OUT_DIR/pr-comment.md (the
same, trimmed to fit a GitHub comment) and OUT_DIR/summary.json. --gate exits 1
when a gating platform failed or produced no result.
"""
import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_common import (CHANGED, FAILED, NO_RESULT, PASSED, STATUS_LABEL,  # noqa: E402
                       UNSUPPORTED, classify)

COMMENT_LIMIT = 60000
ICON = {PASSED: "✅", FAILED: "❌", UNSUPPORTED: "➖", CHANGED: "🆕", NO_RESULT: "⏱️"}


def load_results(results_dir, plan_path):
    results = {}
    for path in glob.glob(os.path.join(results_dir, "**", "result.json"), recursive=True):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        results[r["spec"]["id"]] = r
    planned = []
    if plan_path and os.path.exists(plan_path):
        with open(plan_path, encoding="utf-8") as f:
            planned = json.load(f)["jobs"]
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


# ── Formatting helpers ─────────────────────────────────────────────────────────────

def cell(text):
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


def describe(detail):
    """A check's numbers as words: {'chunks': 2, 'audio_s': 12.5} -> '2 chunks, 12.5 s of audio'."""
    words = {"chunks": "{} chunk(s)", "first_chunk_s": "first chunk after {:.2f} s",
             "audio_s": "{:.1f} s of audio"}
    text = ", ".join(words[k].format(v) if k in words else f"{k} {v}" for k, v in detail.items())
    return text.replace("1 chunk(s)", "1 chunk").replace(" chunk(s)", " chunks")


def platform_name(r):
    return f"{r['spec']['name']} · py{r['spec']['python']}"


def cpu_label(r):
    env = r.get("env") or {}
    if not env.get("cpu"):
        return r["spec"]["runner"]
    bits = [env["cpu"]]
    if env.get("cpu_count"):
        bits.append(f"{env['cpu_count']} cores")
    if env.get("ram_gb"):
        bits.append(f"{env['ram_gb']:g} GB")
    return " · ".join(bits)


# ── Per-model numbers ──────────────────────────────────────────────────────────────

def model_stats(m):
    audio = m.get("audio_s")
    warm = m.get("warm_s") or []
    runs = warm or ([m["first_s"]] if m.get("first_s") is not None else [])
    s = {"first": m.get("first_s"), "audio": audio, "best": min(runs) if runs else None,
         "p50": statistics.median(warm) if warm else None,
         "p95": sorted(warm)[min(len(warm) - 1, int(round(0.95 * (len(warm) - 1))))] if warm else None}
    s["rtf"] = s["best"] / audio if s["best"] and audio else None
    s["p50_rtf"] = s["p50"] / audio if s["p50"] and audio else None
    return s


def asr_rows(r):
    return (r.get("asr") or {}).get("rows", [])


def model_wer(r, m):
    for row in asr_rows(r):
        if row.get("wav") == m.get("wav"):
            return row
    return None


MODEL_STATUS = {"pass": "Passed", "fail": "Failed", "crash": "Crashed", "timeout": "Timed out"}


def model_status(r, m):
    """A model's status as shown: a WER above asr.fail_above fails it too."""
    if m.get("status") != "pass":
        return MODEL_STATUS.get(m.get("status"), str(m.get("status")))
    w = (model_wer(r, m) or {}).get("wer")
    fail_above = r["spec"].get("asr", {}).get("fail_above")
    if w is not None and fail_above is not None and w > fail_above:
        return "Failed (WER)"
    return "Passed"


def wer_counts(results, asr_cfg):
    warn, fail = asr_cfg.get("warn_above", 1), asr_cfg.get("fail_above", 1)
    c = {"passed": 0, "flagged": 0, "failed": 0, "skipped": 0}
    for r in results:
        for row in asr_rows(r):
            w = row.get("wer")
            if w is None:
                c["skipped"] += 1
            elif w > fail:
                c["failed"] += 1
            elif w > warn:
                c["flagged"] += 1
            else:
                c["passed"] += 1
        a = r.get("asr") or {}
        if a.get("status") in ("skipped", "crash", "timeout"):
            c["skipped"] += 1
    return c


# ── Sections ───────────────────────────────────────────────────────────────────────

def header(results, ctx):
    gating = [r for r in results if r["spec"].get("expect", "works") == "works" and r["spec"].get("gating", True)]
    failing = [r for r in results if r["failing"]]
    counts = {s: sum(r["status"] == s for r in results) for s in STATUS_LABEL}
    soft = sum(r["status"] == FAILED and not r["failing"] for r in results)
    # Targets can override the text; show the one most jobs spoke.
    texts = [r["spec"].get("text", "") for r in results]
    spec = dict(results[0]["spec"]) if results else {}
    spec["text"] = max(set(texts), key=texts.count) if texts else ""
    asr_cfg = spec.get("asr", {})
    verdict = (f"❌ **{len(failing)} platform job{'s' if len(failing) != 1 else ''} failed.**" if failing else
               f"✅ **All {len(gating)} supported platform jobs passed.**")
    versions = next(((r.get("install") or {}).get("versions") for r in results
                     if (r.get("install") or {}).get("versions")), {}) or {}
    source = ctx["source"]
    pkg = f"kittenml {versions.get('kittenml', '?')} — " + (
        "this checkout" if source == "checkout" else f"`{source}` from PyPI")
    rows = [
        ["Package", pkg],
        ["Sample text", spec.get("text", "")],
        ["Character length", len(spec.get("text", ""))],
        ["Voice", spec.get("voice", "")],
        ["Platform jobs", " / ".join(
            f"{counts[s] - (soft if s == FAILED else 0)} {STATUS_LABEL[s].lower()}" for s in STATUS_LABEL
            if counts[s] - (soft if s == FAILED else 0)) + (f" / {soft} failed (non-gating)" if soft else "")],
    ]
    if asr_cfg.get("enabled"):
        c = wer_counts(results, asr_cfg)
        rows.append([f"WER ({asr_cfg.get('model', '').split('/')[-1]})",
                     f"{c['passed']} passed, {c['flagged']} above {asr_cfg.get('warn_above', 0):.0%}, "
                     f"{c['failed']} above {asr_cfg.get('fail_above', 0):.0%}, {c['skipped']} skipped"])
    rows += [
        ["WER normalization", "Case and punctuation do not count; KittenTTS == Kitten TTS."],
        ["RTF", "Generation time ÷ audio length, best warm run. Below 1 is faster than realtime."],
        ["GitHub run", f"[{ctx['run_id']}]({ctx['run_url']})" if ctx["run_url"] else "local"],
        ["Commit", ctx["sha"] or "local"],
        ["Config", "`qa/config.toml`"],
    ]
    return "\n".join([f"# Kitten TTS Python Platform Report\n", verdict, "", "## Summary\n",
                      table(["Field", "Value"], rows)])


def status_section(results):
    rows = []
    for r in results:
        if r["status"] == UNSUPPORTED:
            continue
        models = r.get("models", [])
        ok = sum(model_status(r, m) == "Passed" for m in models)
        rtfs = [(model_stats(m)["rtf"], m["label"]) for m in models if model_stats(m)["rtf"]]
        worst = max(rtfs) if rtfs else None
        wers = [row["wer"] for row in asr_rows(r) if row.get("wer") is not None]
        rss = [m.get("peak_rss_mb") for m in models if m.get("peak_rss_mb")]
        install = r.get("install") or {}
        notes = "; ".join(r.get("reasons", []))
        if install and not install.get("ok"):
            notes = "; ".join(x for x in (install.get("error", "install failed"), r["spec"].get("reason")) if x)
        if r["status"] == CHANGED:
            notes = "config.toml expects this install to fail — update it. " + notes
        rows.append([
            platform_name(r), cpu_label(r), f"{ICON[r['status']]} {STATUS_LABEL[r['status']]}"
            + ("" if r["spec"].get("gating", True) else " (non-gating)"),
            fmt(install.get("secs"), 0, " s") if install else "—",
            f"{ok}/{len(models)}" if models else "—",
            f"{worst[0]:.2f} ({worst[1]})" if worst else "—",
            pct(statistics.mean(wers)) if wers else "—",
            f"{max(rss) / 1024:.1f} GB" if rss else "—",
            notes[:300],
        ])
    return "## Platform Status\n\n" + table(
        ["Platform", "CPU", "Status", "Install", "Models", "Worst RTF", "Avg WER", "Peak RAM", "Notes"], rows,
        ["---", "---", "---", "---:", "---:", "---", "---:", "---:", "---"])


def rtf_section(results):
    labels = []
    for r in results:
        for m in r.get("models", []):
            if m["label"] not in labels:
                labels.append(m["label"])
    rows = []
    for r in results:
        if not r.get("models"):
            continue
        by = {m["label"]: m for m in r["models"]}
        row = [platform_name(r), cpu_label(r)]
        for label in labels:
            m = by.get(label)
            if not m:
                row.append("")
            elif m.get("status") != "pass" and model_stats(m)["rtf"] is None:
                row.append("❌")
            else:
                row.append(fmt(model_stats(m)["rtf"], 3))
        rows.append(row)
    if not rows:
        return ""
    return ("## RTF by Platform\n\nBest warm real-time factor for each model; lower is faster.\n\n"
            + table(["Platform", "CPU"] + labels, rows, ["---", "---"] + ["---:"] * len(labels)))


def unsupported_section(results):
    rows = [[platform_name(r), r["spec"]["runner"],
             (r.get("install") or {}).get("error", ""), r["spec"].get("reason", "")]
            for r in results if r["status"] in (UNSUPPORTED,)]
    if not rows:
        return ""
    return ("## Not Supported (expected)\n\nThese fail to install, as `qa/config.toml` says they should. "
            "They do not fail the run.\n\n" + table(["Platform", "Runner", "pip says", "Why"], rows))


def details_section(results):
    out = ["## Platform Details\n"]
    for r in results:
        if r["status"] == UNSUPPORTED or not r.get("models"):
            continue
        install = r.get("install") or {}
        v = install.get("versions") or {}
        audio = r.get("audio_url")
        meta = [f"Install {fmt(install.get('secs'), 0, ' s')}"] + [
            f"{k} {v[k]}" for k in ("kittenml", "torch", "onnxruntime") if v.get(k)]
        if r.get("secs"):
            meta.append(f"job {r['secs'] / 60:.0f} min")
        for m in r["models"]:
            extra = m.get("peak_rss_checks_mb") or 0
            if extra > (m.get("peak_rss_mb") or 0) * 1.2:
                meta.append(f"{m['label']} peaks at {extra / 1024:.1f} GB during its checks")
        if audio:
            meta.append(f"[Audio files]({audio})")
        perf = []
        for m in r["models"]:
            s = model_stats(m)
            w = model_wer(r, m) or {}
            status = model_status(r, m)
            perf.append([m["label"], status, fmt(m.get("load_s")), fmt(s["first"], 3),
                         f"{fmt(s['p50'], 3)} / {fmt(s['p95'], 3)}" if s["p50"] is not None else "—",
                         fmt(s["rtf"], 3), fmt(s["audio"], 2),
                         f"{m['peak_rss_mb'] / 1024:.1f} GB" if m.get("peak_rss_mb") else "—",
                         pct(w.get("wer")), (w.get("transcript") or w.get("error") or "")[:160]])
        checks = []
        for m in r["models"]:
            for c in m.get("checks", []):
                extra = {k: v for k, v in c.items() if k not in ("name", "status", "secs", "trace", "error")}
                checks.append([m["label"], c["name"], "Passed" if c["status"] == "pass" else "Failed",
                               fmt(c.get("secs"), 1), c.get("error") or describe(extra)])
        for c in (r.get("package") or {}).get("checks", []):
            checks.append(["package", c["name"], "Passed" if c["status"] == "pass" else "Failed",
                           fmt(c.get("secs"), 1), c.get("error") or c.get("version") or c.get("output", "")])
        for row in asr_rows(r):
            if " · " in row.get("label", ""):
                checks.append([row["label"].split(" · ")[0], "clone WER", "—", "—",
                               f"{pct(row.get('wer'))} — {row.get('transcript', row.get('error', ''))[:160]}"])
        out.append(
            f"<details><summary><b>{platform_name(r)}</b> — {ICON[r['status']]} {STATUS_LABEL[r['status']]}"
            f" · {cpu_label(r)}</summary>\n\n{' · '.join(meta)}\n\n"
            + table(["Model", "Status", "Load (s)", "First gen (s)", "Warm p50/p95 (s)", "Best RTF",
                     "Audio (s)", "Peak RAM", "WER", "Transcript"], perf,
                    ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"])
            + ("\n\n" + table(["Model", "Check", "Status", "Time (s)", "Result"], checks,
                              ["---", "---", "---", "---:", "---"]) if checks else "")
            + "\n\n</details>\n")
    return "\n".join(out) if len(out) > 1 else ""


def failures_section(results, trace_chars):
    items = []
    for r in results:
        if r["status"] not in (FAILED, NO_RESULT, CHANGED) and not any(
                m.get("status") != "pass" for m in r.get("models", [])):
            continue
        name = platform_name(r) + ("" if r["spec"].get("gating", True) else " (non-gating)")
        install = r.get("install") or {}
        if r["status"] == NO_RESULT:
            items.append(f"<details><summary>{name}: no result</summary>\n\n{'; '.join(r['reasons'])}\n</details>")
            continue
        if install and not install.get("ok") and r["spec"].get("expect", "works") == "works":
            items.append(f"<details><summary>{name}: install failed — {cell(install.get('error', ''))}</summary>"
                         f"\n\n```\n{install.get('log_tail', '')[-trace_chars:]}\n```\n</details>")
        for part in ("package", "asr"):
            p = r.get(part) or {}
            if p.get("status") in ("crash", "timeout"):
                label = "package checks" if part == "package" else "WER transcription"
                items.append(f"<details><summary>{name} · {label}: {cell(p.get('error', p['status']))}</summary>"
                             f"\n\n```\n{p.get('log_tail', '')[-trace_chars:]}\n```\n</details>")
        for c in (r.get("package") or {}).get("checks", []):
            if c["status"] != "pass":
                items.append(f"<details><summary>{name} · {c['name']}: {cell(c.get('error', ''))}</summary>"
                             f"\n\n```\n{c.get('trace', '')[-trace_chars:]}\n```\n</details>")
        for m in r.get("models", []):
            if m.get("status") == "pass":
                continue
            trace = m.get("trace") or m.get("log_tail") or ""
            for c in m.get("checks", []):
                if c["status"] != "pass":
                    trace += f"\n--- {c['name']}: {c.get('error', '')}\n{c.get('trace', '')}"
            items.append(f"<details><summary>{name} · {m['label']}: {cell(m.get('error', m.get('status')))}"
                         f"</summary>\n\n```\n{trace[-trace_chars:]}\n```\n</details>")
        for reason in r.get("reasons", []):
            if "WER" in reason:
                items.append(f"- {name} · {reason}")
    return "## Failures\n\n" + "\n".join(items) if items else ""


def build(results, ctx):
    head = header(results, ctx)
    parts = [status_section(results), rtf_section(results), unsupported_section(results)]
    details = details_section(results)
    full = "\n\n".join(p for p in [head] + parts + [failures_section(results, 3000), details] if p)
    comment = full
    if len(comment) > COMMENT_LIMIT:
        comment = "\n\n".join(p for p in [head] + parts + [failures_section(results, 1500)] if p)
        comment += f"\n\n_Per-platform details are in the [run summary]({ctx['run_url']})._"
    if len(comment) > COMMENT_LIMIT:
        comment = "\n\n".join(p for p in [head] + parts + [failures_section(results, 200)] if p)
        comment = comment[:COMMENT_LIMIT] + f"\n\n_Truncated. Full report in the [run summary]({ctx['run_url']})._"
    return full, comment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--plan")
    ap.add_argument("--gate")
    args = ap.parse_args()

    if args.gate:
        with open(args.gate, encoding="utf-8") as f:
            summary = json.load(f)
        for j in summary["jobs"]:
            if j["failing"]:
                print(f"FAILED {j['platform']}: {STATUS_LABEL[j['status']]} — {'; '.join(j['reasons'])}")
        sys.exit(1 if summary["failing"] else 0)

    results = load_results(args.results_dir, args.plan)
    server, repo, run_id = (os.environ.get(k, "") for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    ctx = {"source": os.environ.get("QA_SOURCE", "checkout") or "checkout",
           "sha": os.environ.get("QA_SHA") or os.environ.get("GITHUB_SHA", ""), "run_id": run_id,
           "run_url": f"{server}/{repo}/actions/runs/{run_id}" if run_id else ""}
    full, comment = build(results, ctx)
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

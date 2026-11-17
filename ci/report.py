"""Merge smoke-test outputs from every CI job into one listening page.

    python ci/report.py --inputs downloaded-artifacts/ --out report/ [--summary $GITHUB_STEP_SUMMARY]

`--inputs` holds one directory per job, each with the results.json and wav/ that
ci/smoke.py wrote. Jobs with the same runner label (the light and heavy suites of one
OS / Python) are merged. Writes report/index.html, which plays the wavs from
report/audio/<label>/, and needs nothing else: unzip the artifact and open it.
"""

import argparse
import html
import json
import os
import re
import shutil
from collections import OrderedDict

STATUS_ORDER = {"failed": 0, "warning": 1, "passed": 2, "skipped": 3}


def load(inputs):
    runs = OrderedDict()
    for root, _dirs, files in sorted(os.walk(inputs)):
        if "results.json" not in files:
            continue
        with open(os.path.join(root, "results.json"), encoding="utf-8") as f:
            data = json.load(f)
        env = data.get("env", {})
        label = env.get("runner") or os.path.basename(root)
        run = runs.setdefault(label, dict(label=label, env=env, results=[], roots={}))
        run["env"].update({k: v for k, v in env.items() if v not in (None, "")})
        for r in data.get("results", []):
            run["results"].append(r)
            run["roots"][r["id"]] = root
    return runs


def label_key(label):
    os_rank = next((i for i, k in enumerate(("ubuntu", "linux", "macos", "windows")) if k in label.lower()), 9)
    nums = [int(x) for x in re.findall(r"\d+", label)]
    return (os_rank, label.split("-")[0], nums)


def to_flac(src, dst):
    """Lossless, so artefacts survive, at about half the size of the PCM wav; every browser plays it."""
    try:
        import soundfile as sf
        data, sr = sf.read(src, dtype="int16")
        sf.write(dst, data, sr, format="FLAC", subtype="PCM_16")
    except Exception:
        shutil.copyfile(src, os.path.splitext(dst)[0] + ".wav")
        return os.path.splitext(dst)[0] + ".wav"
    return dst


def counts(results):
    c = {s: 0 for s in STATUS_ORDER}
    for r in results:
        c[r["status"]] = c.get(r["status"], 0) + 1
    return c


def fmt_kwargs(d):
    return ", ".join(f"{k}={v!r}" for k, v in d.items()) if d else ""


def build(runs, out, title, run_url):
    os.makedirs(out, exist_ok=True)
    labels = sorted(runs, key=label_key)

    # Case catalogue, in the order the smoke script defines them.
    cases = OrderedDict()
    for label in labels:
        for r in runs[label]["results"]:
            cases.setdefault(r["id"], r)
    families = OrderedDict()
    for cid, r in cases.items():
        families.setdefault(r.get("family") or r.get("group"), []).append(cid)

    cell = {}           # (case, label) -> result with a copied wav path
    for label in labels:
        run = runs[label]
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        for r in run["results"]:
            r = dict(r)
            if r.get("wav"):
                src = os.path.join(run["roots"][r["id"]], r["wav"])
                if os.path.exists(src):
                    dst = os.path.join("audio", safe, os.path.splitext(os.path.basename(src))[0] + ".flac")
                    os.makedirs(os.path.join(out, "audio", safe), exist_ok=True)
                    written = to_flac(src, os.path.join(out, dst))
                    r["src"] = os.path.relpath(written, out).replace(os.sep, "/")
            cell[(r["id"], label)] = r

    total = counts([c for c in cell.values()])
    esc = html.escape

    def badge(status):
        return f'<span class="b b-{status}">{status}</span>'

    # --- overview matrix ---
    head = "".join(f'<th class="lab"><span>{esc(l)}</span></th>' for l in labels)
    rows = []
    for fam, ids in families.items():
        rows.append(f'<tr class="fam"><th colspan="{len(labels) + 1}">{esc(fam)}</th></tr>')
        for cid in ids:
            tds = []
            for l in labels:
                r = cell.get((cid, l))
                if r is None:
                    tds.append('<td class="m m-none" title="not run">·</td>')
                    continue
                tip = "; ".join(r.get("errors", []) + r.get("warnings", [])) or r.get("reason", "") or r["status"]
                tds.append(f'<td class="m m-{r["status"]}" title="{esc(l)}: {esc(tip)}">'
                           f'<a href="#{esc(cid)}">{ {"passed": "✓", "warning": "!", "failed": "✗", "skipped": "–"}[r["status"]] }</a></td>')
            rows.append(f'<tr data-case="{esc(cid)}"><th class="cid"><a href="#{esc(cid)}">{esc(cid)}</a></th>{"".join(tds)}</tr>')
    matrix = f'<div class="scroll"><table class="matrix"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

    # --- case cards ---
    cards = []
    for fam, ids in families.items():
        cards.append(f'<h2 class="famh">{esc(fam)}</h2>')
        for cid in ids:
            meta = cases[cid]
            worst = min((cell[(cid, l)]["status"] for l in labels if (cid, l) in cell), key=STATUS_ORDER.get)
            lines = []
            for l in labels:
                r = cell.get((cid, l))
                if r is None:
                    continue
                s = r.get("stats", {})
                stat = (f'{s["duration_s"]:.2f}s · rtf {s.get("rtf", "?")} · rms {s.get("rms", "?")} · '
                        f'peak {s.get("peak", "?")} · silence {s.get("silent_frac", 0):.0%}') if s else ""
                if s.get("chunks"):
                    stat += f' · {s["chunks"]} chunks'
                msgs = "".join(f'<div class="err">{esc(m)}</div>' for m in r.get("errors", []))
                msgs += "".join(f'<div class="warn">{esc(m)}</div>' for m in r.get("warnings", []))
                if r.get("reason"):
                    msgs += f'<div class="muted">{esc(r["reason"])}</div>'
                if r.get("traceback"):
                    msgs += f'<details><summary>traceback</summary><pre>{esc(r["traceback"])}</pre></details>'
                player = (f'<audio controls preload="none" src="{esc(r["src"])}"></audio>' if r.get("src") else "")
                key = f"{cid}|{l}"
                lines.append(
                    f'<div class="run run-{r["status"]}" data-status="{r["status"]}">'
                    f'<div class="rl"><b>{esc(l)}</b> {badge(r["status"])}<div class="muted small">{esc(stat)}</div>{msgs}</div>'
                    f'<div class="rp">{player}</div>'
                    f'<label class="flag" title="Mark this clip as sounding wrong"><input type="checkbox" data-key="{esc(key)}"> sounds off</label>'
                    f'</div>')
            cards.append(
                f'<section class="case" id="{esc(cid)}" data-worst="{worst}">'
                f'<header><h3>{esc(cid)} {badge(worst)}</h3>'
                f'<button class="playall" type="button">▶ play all in turn</button></header>'
                f'<p class="text">“{esc(meta["text"])}”</p>'
                f'<p class="muted small"><code>KittenTTS({esc(repr(meta["repo"]))}{", " if meta.get("load") else ""}{esc(fmt_kwargs(meta.get("load", {})))})</code>'
                f' → <code>{esc({"stream": "generate_stream", "file": "generate_to_file"}.get(meta["mode"], "generate"))}'
                f'({esc(fmt_kwargs(meta.get("gen", {})))})</code>{(" · " + esc(meta["note"])) if meta.get("note") else ""}</p>'
                f'{"".join(lines)}</section>')

    # --- environments ---
    keys = []
    for l in labels:
        for k in runs[l]["env"]:
            if k not in keys and k != "runner":
                keys.append(k)
    env_rows = "".join(
        f'<tr><th>{esc(l)}</th>' + "".join(f'<td>{esc(str(runs[l]["env"].get(k, "")))}</td>' for k in keys) + "</tr>"
        for l in labels)
    env_table = (f'<div class="scroll"><table class="env"><thead><tr><th></th>'
                 f'{"".join(f"<th>{esc(k)}</th>" for k in keys)}</tr></thead><tbody>{env_rows}</tbody></table></div>')

    run_link = f' · <a href="{esc(run_url)}">workflow run</a>' if run_url else ""
    page = TEMPLATE.format(
        title=esc(title), run_link=run_link, n_cfg=len(labels), n_case=len(cases),
        passed=total["passed"], warning=total["warning"], failed=total["failed"], skipped=total["skipped"],
        matrix=matrix, cards="".join(cards), env=env_table)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    return labels, cell, cases


def summary_md(runs, labels, cell, cases, run_url):
    lines = ["## Audio smoke test", ""]
    lines.append("| configuration | passed | warnings | failed | skipped |")
    lines.append("|---|---:|---:|---:|---:|")
    for l in labels:
        c = counts(runs[l]["results"])
        lines.append(f"| {l} | {c['passed']} | {c['warning']} | {'**%d**' % c['failed'] if c['failed'] else 0} | {c['skipped']} |")
    fails = [(cid, l, r) for (cid, l), r in cell.items() if r["status"] == "failed"]
    warns = [(cid, l, r) for (cid, l), r in cell.items() if r["status"] == "warning"]
    for name, items, key in (("Failures", fails, "errors"), ("Warnings", warns, "warnings")):
        if items:
            lines += ["", f"### {name}", ""]
            for cid, l, r in sorted(items, key=lambda x: (x[0], label_key(x[1]))):
                lines.append(f"- `{cid}` on **{l}**: " + "; ".join(r.get(key, []))[:400])
    lines += ["", "Download the **audio-report** artifact, unzip it and open `index.html` to listen."]
    if run_url:
        lines.append(f"Artifacts: {run_url}#artifacts")
    return "\n".join(lines) + "\n"


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#fbfbfa; --fg:#1c1c1a; --muted:#6b6b66; --card:#fff; --line:#e4e3df;
  --ok:#1f7a4d; --ok-bg:#e3f3ea; --warn:#8a5a00; --warn-bg:#fbefd5; --bad:#b42318; --bad-bg:#fde7e4;
  --skip:#6b6b66; --skip-bg:#efefec; --accent:#3b5bdb; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#161615; --fg:#ecebe7; --muted:#9c9b95;
  --card:#1f1f1d; --line:#33332f; --ok:#5fd39a; --ok-bg:#173826; --warn:#f2c46b; --warn-bg:#3a2d10;
  --bad:#ff8a7a; --bad-bg:#45191380; --skip:#9c9b95; --skip-bg:#2a2a27; --accent:#8ea2ff; }} }}
:root[data-theme="dark"] {{ --bg:#161615; --fg:#ecebe7; --muted:#9c9b95; --card:#1f1f1d; --line:#33332f; --ok:#5fd39a;
  --ok-bg:#173826; --warn:#f2c46b; --warn-bg:#3a2d10; --bad:#ff8a7a; --bad-bg:#45191380; --skip:#9c9b95;
  --skip-bg:#2a2a27; --accent:#8ea2ff; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif }}
main {{ max-width:1180px; margin:0 auto; padding:24px 16px 80px }}
a {{ color:var(--accent); text-decoration:none }}
h1 {{ font-size:26px; margin:0 0 4px }} h2.famh {{ margin:40px 0 12px; font-size:19px }}
.muted {{ color:var(--muted) }} .small {{ font-size:13px }}
.stats {{ display:flex; gap:10px; flex-wrap:wrap; margin:16px 0 }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 16px; min-width:110px }}
.stat b {{ display:block; font-size:22px; font-variant-numeric:tabular-nums }}
.toolbar {{ position:sticky; top:0; z-index:5; background:var(--bg); padding:10px 0; border-bottom:1px solid var(--line);
  display:flex; gap:8px; flex-wrap:wrap; align-items:center }}
button {{ font:inherit; font-size:13px; padding:5px 12px; border-radius:7px; border:1px solid var(--line);
  background:var(--card); color:var(--fg); cursor:pointer }}
button.on {{ border-color:var(--accent); color:var(--accent) }}
.scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--card) }}
table {{ border-collapse:collapse; font-size:13px }}
.matrix th, .matrix td {{ padding:3px 6px; border-bottom:1px solid var(--line) }}
.matrix th.cid {{ text-align:left; font-weight:500; white-space:nowrap; font-family:ui-monospace,monospace; font-size:12px }}
.matrix th.lab {{ height:120px; vertical-align:bottom; white-space:nowrap; font-weight:500 }}
.matrix th.lab span {{ writing-mode:vertical-rl; transform:rotate(180deg) }}
.matrix tr.fam th {{ text-align:left; background:var(--bg); padding-top:10px }}
.matrix td.m {{ text-align:center; width:30px; font-weight:700 }}
.matrix td.m a {{ color:inherit; display:block }}
.m-passed {{ color:var(--ok) }} .m-warning {{ background:var(--warn-bg); color:var(--warn) }}
.m-failed {{ background:var(--bad-bg); color:var(--bad) }} .m-skipped, .m-none {{ color:var(--skip) }}
.b {{ font-size:11px; font-weight:600; padding:1px 7px; border-radius:99px; vertical-align:middle; text-transform:uppercase; letter-spacing:.03em }}
.b-passed {{ background:var(--ok-bg); color:var(--ok) }} .b-warning {{ background:var(--warn-bg); color:var(--warn) }}
.b-failed {{ background:var(--bad-bg); color:var(--bad) }} .b-skipped {{ background:var(--skip-bg); color:var(--skip) }}
.case {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin:12px 0; scroll-margin-top:60px }}
.case header {{ display:flex; justify-content:space-between; gap:8px; align-items:center; flex-wrap:wrap }}
.case h3 {{ margin:0; font-size:15px; font-family:ui-monospace,monospace }}
.case .text {{ margin:8px 0 2px; font-size:16px }}
code {{ font-size:12px; word-break:break-word }}
.run {{ display:grid; grid-template-columns:minmax(200px,1fr) minmax(260px,340px) auto; gap:4px 16px; align-items:center;
  padding:8px 0; border-top:1px solid var(--line) }}
.run audio {{ width:100%; height:36px }}
.err {{ color:var(--bad); font-size:13px }} .warn {{ color:var(--warn); font-size:13px }}
.flag {{ font-size:12px; color:var(--muted); white-space:nowrap; cursor:pointer }}
.run.flagged {{ background:var(--warn-bg); margin:0 -16px; padding:8px 16px }}
pre {{ font-size:11px; overflow-x:auto; background:var(--bg); padding:8px; border-radius:6px }}
.env {{ font-size:12px }} .env th, .env td {{ padding:4px 10px; border-bottom:1px solid var(--line); white-space:nowrap; text-align:left }}
.hidden {{ display:none !important }}
@media (max-width:720px) {{ .run {{ grid-template-columns:1fr }} .matrix th.lab {{ height:90px }} }}
</style></head>
<body><main>
<h1>{title}</h1>
<div class="muted">{n_case} cases × {n_cfg} configurations{run_link}. Listen for robotic artefacts, clicks, wrong language,
truncation, repeated words or runaway babble — the automatic checks only catch silence, NaNs, clipping and implausible length.</div>
<div class="stats">
<div class="stat"><b style="color:var(--ok)">{passed}</b>passed</div>
<div class="stat"><b style="color:var(--warn)">{warning}</b>warnings</div>
<div class="stat"><b style="color:var(--bad)">{failed}</b>failed</div>
<div class="stat"><b style="color:var(--skip)">{skipped}</b>skipped</div>
<div class="stat"><b id="nflag">0</b>flagged by you</div>
</div>
<h2>Overview</h2>
{matrix}
<div class="toolbar">
<span class="muted small">Show:</span>
<button type="button" data-filter="all" class="on">all</button>
<button type="button" data-filter="problems">failed / warnings</button>
<button type="button" data-filter="flagged">flagged</button>
<span style="flex:1"></span>
<button type="button" id="copyflags">copy flagged list</button>
<button type="button" id="clearflags">clear flags</button>
</div>
{cards}
<h2>Environments</h2>
{env}
</main>
<script>
(function () {{
  var store = {{}};
  try {{ store = JSON.parse(localStorage.getItem("kitten-smoke-flags") || "{{}}"); }} catch (e) {{}}
  function save() {{ try {{ localStorage.setItem("kitten-smoke-flags", JSON.stringify(store)); }} catch (e) {{}} }}
  var boxes = document.querySelectorAll(".flag input");
  function refresh() {{
    var n = 0;
    boxes.forEach(function (b) {{ var on = !!store[b.dataset.key]; b.checked = on;
      b.closest(".run").classList.toggle("flagged", on); if (on) n++; }});
    document.getElementById("nflag").textContent = n;
  }}
  boxes.forEach(function (b) {{ b.addEventListener("change", function () {{
    if (b.checked) store[b.dataset.key] = 1; else delete store[b.dataset.key]; save(); refresh(); }}); }});
  refresh();
  document.getElementById("clearflags").onclick = function () {{ store = {{}}; save(); refresh(); }};
  document.getElementById("copyflags").onclick = function () {{
    var t = Object.keys(store).map(function (k) {{ return "- " + k.replace("|", " on "); }}).join("\\n") || "(nothing flagged)";
    if (navigator.clipboard) navigator.clipboard.writeText(t).catch(function () {{ prompt("Flagged clips", t); }});
    else prompt("Flagged clips", t);
  }};
  // Only one clip plays at a time.
  document.addEventListener("play", function (e) {{
    document.querySelectorAll("audio").forEach(function (a) {{ if (a !== e.target) a.pause(); }});
  }}, true);
  document.querySelectorAll(".playall").forEach(function (btn) {{
    btn.onclick = function () {{
      var list = Array.prototype.slice.call(btn.closest(".case").querySelectorAll("audio"));
      var i = 0;
      function next() {{ if (i >= list.length) return; var a = list[i++]; a.currentTime = 0;
        a.addEventListener("ended", next, {{ once: true }}); a.play(); }}
      next();
    }};
  }});
  var buttons = document.querySelectorAll("[data-filter]");
  buttons.forEach(function (b) {{ b.onclick = function () {{
    buttons.forEach(function (x) {{ x.classList.toggle("on", x === b); }});
    var f = b.dataset.filter;
    document.querySelectorAll(".case").forEach(function (c) {{
      var show = f === "all" || (f === "problems" && /failed|warning/.test(c.dataset.worst)) ||
        (f === "flagged" && c.querySelector(".run.flagged"));
      c.classList.toggle("hidden", !show);
    }});
  }}; }});
}})();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", default="report")
    ap.add_argument("--summary", help="also write a markdown summary here (e.g. $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--title", default="KittenTTS audio smoke test")
    args = ap.parse_args()

    runs = load(args.inputs)
    if not runs:
        raise SystemExit(f"no results.json under {args.inputs}")
    server, repo, run_id = (os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    run_url = f"{server}/{repo}/actions/runs/{run_id}" if server and repo and run_id else ""
    title = args.title
    if os.environ.get("GITHUB_SHA"):
        title += f" · {os.environ.get('GITHUB_REF_NAME', '')} @ {os.environ['GITHUB_SHA'][:7]}"
    labels, cell, cases = build(runs, args.out, title, run_url)
    md = summary_md(runs, labels, cell, cases, run_url)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(md)
    print(md)


if __name__ == "__main__":
    main()

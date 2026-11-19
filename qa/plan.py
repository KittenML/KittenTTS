"""Expand qa/config.toml into the GitHub Actions job matrix.

    python qa/plan.py [--config qa/config.toml] [--out plan.json]

Targets with `events` run only for those triggers ($GITHUB_EVENT_NAME).

Narrowing a run (all optional, comma-separated, from workflow_dispatch inputs):
    QA_TARGETS  keep targets whose name or runner contains one of these
    QA_PYTHONS  run only these Python versions
    QA_MODELS   run only these models

Writes `matrix=<json>` to $GITHUB_OUTPUT and the full plan to --out, which the
report uses to spot jobs that never produced a result.
"""
import argparse
import json
import os
import re
import sys
import tomllib

KNOWN_CHECKS = {"stream", "speed", "to_file", "expression", "clone", "clone_whisper", "emb4"}
TTS2_ONLY = {"expression", "clone", "clone_whisper", "emb4"}
EVENTS = {"pull_request", "push", "workflow_dispatch"}
MODEL_KEYS = {"repo", "label", "warm_runs", "checks", "weights"}
TARGET_KEYS = {"name", "runner", "pythons", "models", "overrides", "text", "timeout_minutes", "voice", "events"}


def csv_env(name):
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    errors = []
    models = cfg.get("models", {})
    for key, m in models.items():
        for k in set(m) - MODEL_KEYS:
            errors.append(f"models.{key}: unknown setting {k!r}")
        if "repo" not in m:
            errors.append(f"models.{key}: needs a repo")
        for c in m.get("checks", []):
            if c not in KNOWN_CHECKS:
                errors.append(f"models.{key}: unknown check {c!r} (known: {sorted(KNOWN_CHECKS)})")
            elif c in TTS2_ONLY and "kitten-tts-2" not in m.get("repo", ""):
                errors.append(f"models.{key}: {c!r} only works with KittenTTS 2")
            elif c == "speed" and "kitten-tts-2" in m.get("repo", ""):
                errors.append(f"models.{key}: KittenTTS 2 has no speed control")
    names = set()
    for t in cfg.get("target", []):
        where = f"target {t.get('name', '?')!r}"
        for k in set(t) - TARGET_KEYS:
            errors.append(f"{where}: unknown setting {k!r}")
        for k in ("name", "runner"):
            if k not in t:
                errors.append(f"{where}: needs {k!r}")
        if t.get("name") in names:
            errors.append(f"{where}: duplicate name")
        names.add(t.get("name"))
        for e in t.get("events", []):
            if e not in EVENTS:
                errors.append(f"{where}: unknown event {e!r} (known: {sorted(EVENTS)})")
        for m in t.get("models", []) + list(t.get("overrides", {})):
            if m not in models:
                errors.append(f"{where}: unknown model {m!r}")
        for m, o in t.get("overrides", {}).items():
            for k in set(o) - MODEL_KEYS:
                errors.append(f"{where}: overrides.{m} has unknown setting {k!r}")
    for k in set(cfg.get("limits", {})) - {"step_minutes", "install_minutes", "job_minutes"}:
        errors.append(f"limits: unknown setting {k!r}")
    if errors:
        sys.exit("qa/config.toml is invalid:\n  " + "\n  ".join(errors))
    return cfg


def expand(cfg):
    only_targets = [v.lower() for v in csv_env("QA_TARGETS")]
    only_pythons = csv_env("QA_PYTHONS")
    only_models = csv_env("QA_MODELS")
    event = os.environ.get("GITHUB_EVENT_NAME")   # unset locally: plan every target
    jobs = []
    for t in cfg["target"]:
        if event and t.get("events") and event not in t["events"]:
            continue
        if only_targets and not any(v in t["name"].lower() or v in t["runner"] for v in only_targets):
            continue
        models = [m for m in t.get("models") or list(cfg["models"]) if not only_models or m in only_models]
        if only_models and t.get("models") and not models:
            continue
        resolved = []
        for key in models:
            m = dict(cfg["models"][key])
            m.update(t.get("overrides", {}).get(key, {}))
            m["key"] = key
            m.setdefault("label", key)
            m.setdefault("warm_runs", 1)
            m.setdefault("checks", [])
            resolved.append(m)
        limits = {"step_minutes": 10, "job_minutes": 45, **cfg.get("limits", {})}
        timeout = t.get("timeout_minutes") or limits["job_minutes"]
        limits["job_minutes"] = timeout     # the runner stops starting tests before GitHub ends the job
        for py in t.get("pythons") or cfg.get("matrix", {}).get("pythons", []):
            if only_pythons and py not in only_pythons:
                continue
            spec = {
                "id": f"{slug(t['name'])}-py{py}",
                "name": t["name"],
                "runner": t["runner"],
                "python": py,
                "text": t.get("text", cfg["sample"]["text"]),
                "voice": t.get("voice", cfg["sample"]["voice"]),
                "models": resolved,
                "asr": cfg.get("asr", {"enabled": False}),
                "limits": limits,
                "overrides": {k: v for k, v in t.get("overrides", {}).items() if k in models},
            }
            jobs.append({"id": spec["id"], "name": t["name"], "runner": t["runner"],
                         "python": py, "timeout": timeout, "spec": json.dumps(spec)})
    if not jobs:
        sys.exit("No jobs left after filtering; check QA_TARGETS / QA_PYTHONS / QA_MODELS.")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qa/config.toml")
    ap.add_argument("--out", default="plan.json")
    args = ap.parse_args()
    cfg = load(args.config)
    jobs = expand(cfg)
    with open(args.out, "w") as f:
        json.dump({"report": cfg.get("report", {}), "jobs": [json.loads(j["spec"]) for j in jobs]}, f, indent=1)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write("matrix=" + json.dumps({"include": jobs}) + "\n")
    for j in jobs:
        print(f"{j['name']:40} {j['runner']:18} py{j['python']:5} {j['timeout']:>3} min")
    print(f"{len(jobs)} jobs")


if __name__ == "__main__":
    main()

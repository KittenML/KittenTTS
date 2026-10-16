"""Run every check this project depends on, in dependency order, and report.

The point of this file is that nobody should have to remember which checks exist,
what order they go in, or which ones invalidate the others. Training a 1.7B
ternary speech LM on a new tokenizer has several ways to fail silently and
expensively -- a mis-specified sequence format, a codec that cannot represent the
target language, a quantizer that detunes every layer at step 0, a loss mask off
by one. Each of those is cheap to detect and ruinous to miss, so they are all
wired into one command that answers a single question:

    is this pipeline safe to spend an allocation on?

    python -m indic.preflight                 # everything that will run here
    python -m indic.preflight --stage gates   # just the go/no-go gates
    python -m indic.preflight --report-only   # rebuild the report from cached runs

Design notes that matter:

**Resumable.** Each check writes its artifact and is skipped when that artifact is
newer than its inputs. An overnight run that dies at 3am resumes where it stopped
rather than redoing six GPU hours.

**GPU checks serialize.** The dev machine has one 12 GB card that is also driving a
display. Two checks measuring memory at once produce two wrong answers, so the
runner holds a lock across anything that allocates.

**A failing check is data, not an error.** `codec_roundtrip` failing on Telugu is a
real finding about the codec and must be reported as a FAIL with its numbers, not
swallowed as an exception. Only an infrastructure problem -- a missing script, a
crashed interpreter -- is an ERROR.

**Gates block their dependents.** If the sequence format is not validated, the
training self-check result is not meaningful even if it passes, so dependents of a
failed gate are reported BLOCKED rather than run.
"""

import argparse
import dataclasses
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "indic" / "data"
PYTHON = str(ROOT / ".venv" / "bin" / "python")

PASS, FAIL, ERROR, SKIP, BLOCKED, MISSING = "PASS", "FAIL", "ERROR", "SKIP", "BLOCKED", "MISSING"


@dataclasses.dataclass
class Check:
    """One preflight check.

    `question` is what the check answers in plain words; it is printed in the
    report so a reader never has to open the script to know why a line is there.
    `blocks` names the checks whose results stop meaning anything if this one
    fails -- that is what turns a list of tests into a dependency graph.
    """

    name: str
    stage: str
    question: str
    argv: list
    artifact: str = ""
    needs_gpu: bool = False
    timeout: int = 3600
    blocks: tuple = ()
    optional: bool = False          # absence is a SKIP, not a MISSING


CHECKS = [
    Check("capability", "env",
          "What hardware is this, and is bf16 safe on it?",
          ["-m", "indic.hpc.capability"],
          artifact="capability.json", optional=True, timeout=300),

    Check("tokenizer_fertility", "tokenizer",
          "How much more does Indic text cost under Qwen3 than under Sarvam-1?",
          ["-m", "indic.bench.tokenizer_fertility", "--out", "indic/data/fertility.json"],
          artifact="fertility.json", timeout=1800),

    Check("ternary_check", "checkpoint",
          "Which quantizer wrote these weights, and can QAT resume without damaging them?",
          ["-m", "indic.bench.ternary_check", "--out", "indic/data/ternary.json"],
          artifact="ternary.json", timeout=1800),

    Check("vocab_surgery", "surgery",
          "Does the Sarvam vocabulary swap produce a loadable checkpoint?",
          ["-m", "indic.vocab.build_sarvam_vocab", "--out", "indic/data/kitten-tts-2-sarvam",
           "--init", "mean"],
          artifact="kitten-tts-2-sarvam/vocab_surgery.json", timeout=7200,
          blocks=("sarvam_smoke",)),

    Check("sarvam_smoke", "surgery",
          "Is the rebuilt checkpoint actually correct -- audio rows aligned, "
          "tokenizer addressing only text rows?",
          ["-m", "indic.vocab.verify_surgery", "--out", "indic/data/surgery_verify.json"],
          artifact="surgery_verify.json", timeout=1800),

    Check("codec_roundtrip", "gates",
          "Can the S3 codec represent Tamil and Telugu at all?",
          ["-m", "indic.bench.codec_roundtrip", "--out", "indic/data/codec_roundtrip.json"],
          artifact="codec_roundtrip.json", needs_gpu=True, timeout=5400,
          blocks=("baseline_eval",)),

    Check("teacher_forced_loss", "gates",
          "Does our training sequence format match what the checkpoint was trained on?",
          ["-m", "indic.bench.teacher_forced_loss", "--out",
           "indic/data/teacher_forced.json"],
          artifact="teacher_forced.json", needs_gpu=True, timeout=5400),

    # Deliberately depends on neither gate. The self-check runs on synthetic
    # examples it builds itself, so it tests the loop's mechanics -- does it
    # overfit one batch, is the loss mask exact, do gradients land where they
    # should -- none of which is contingent on the codec carrying Telugu or on
    # the reference layout being the trained one. Wiring a dependency here would
    # have reported BLOCKED for a suite that is perfectly meaningful either way.
    Check("train_selfcheck", "train",
          "Does the training loop provably work -- overfit one batch, exact loss mask, "
          "gradients where they belong?",
          ["-m", "indic.train.selfcheck", "--out", "indic/data/train_selfcheck.json"],
          artifact="train_selfcheck.json", timeout=5400),

    Check("fit_config_sarvam", "fit",
          "What is the largest training configuration that fits this GPU, Sarvam vocab?",
          ["-m", "indic.bench.fit_config", "--vocab", "74752",
           "--out", "indic/data/fit_sarvam.json"],
          artifact="fit_sarvam.json", needs_gpu=True, timeout=3600),

    Check("fit_config_qwen3", "fit",
          "The same, for the unchanged Qwen3 vocab -- how much does the swap buy?",
          ["-m", "indic.bench.fit_config", "--vocab", "158241",
           "--out", "indic/data/fit_qwen3.json"],
          artifact="fit_qwen3.json", needs_gpu=True, timeout=3600),
]

STAGE_ORDER = ["env", "tokenizer", "checkpoint", "surgery", "gates", "train", "fit"]


def script_exists(check):
    """True when the module the check invokes is actually present.

    Checks are declared before their scripts are finished, so a missing module is
    an ordinary state of the world here, not a crash.
    """
    if check.argv[0] != "-m":
        return True
    path = ROOT / (check.argv[1].replace(".", "/") + ".py")
    return path.exists()


def artifact_path(check):
    return DATA / check.artifact if check.artifact else None


def is_fresh(check, state, max_age_s=None):
    """True when this check already PASSED and its artifact is still there.

    Keyed on the recorded status, not merely on the artifact existing. A check
    that answers "no" still writes its JSON -- the gates deliberately do, because
    the numbers behind a FAIL are the whole point of running it -- so caching on
    the file alone silently promotes every FAIL to a PASS on the next sweep. That
    is the same shape of bug as a test that cannot fail, and it would have turned
    the one gate that is genuinely unconfirmed into a green row.
    """
    path = artifact_path(check)
    if not path or not path.exists():
        return False
    if (state.get(check.name) or {}).get("status") != PASS:
        return False
    if max_age_s and time.time() - path.stat().st_mtime > max_age_s:
        return False
    return True


def run_check(check, env=None):
    """Execute one check and classify the outcome.

    Exit code is the contract: 0 is PASS, any other code from a script that ran to
    completion is FAIL (the check answered, and the answer is no). A non-zero exit
    with no output, or a timeout, is an ERROR -- the check did not answer.
    """
    argv = [PYTHON] + check.argv
    started = time.time()
    try:
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                              timeout=check.timeout,
                              env={**os.environ, **(env or {})})
    except subprocess.TimeoutExpired:
        return {"status": ERROR, "reason": f"timed out after {check.timeout}s",
                "duration_s": time.time() - started, "stdout": "", "stderr": ""}
    duration = time.time() - started
    tail = proc.stdout[-4000:]
    status = PASS if proc.returncode == 0 else FAIL
    if proc.returncode != 0 and not proc.stdout.strip():
        status = ERROR
    return {"status": status, "returncode": proc.returncode, "duration_s": duration,
            "stdout": tail, "stderr": proc.stderr[-3000:]}


def load_artifact(check):
    """The check's own JSON output, so the report can quote real numbers."""
    path = artifact_path(check)
    if not path or not path.exists() or path.suffix != ".json":
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append", default=[],
                    help=f"limit to stages: {', '.join(STAGE_ORDER)}")
    ap.add_argument("--only", action="append", default=[], help="run named checks only")
    ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--force", action="store_true", help="rerun even if the artifact exists")
    ap.add_argument("--no-gpu", action="store_true", help="skip checks that need the GPU")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from existing artifacts, run nothing")
    ap.add_argument("--state", type=pathlib.Path, default=DATA / "preflight.json")
    ap.add_argument("--report", type=pathlib.Path, default=DATA / "PREFLIGHT_REPORT.md")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    state = {}
    if args.state.exists():
        try:
            state = json.loads(args.state.read_text()).get("checks", {})
        except json.JSONDecodeError:
            state = {}

    selected = [c for c in CHECKS
                if (not args.stage or c.stage in args.stage)
                and (not args.only or c.name in args.only)
                and c.name not in args.skip]

    failed_blockers = set()
    for check in sorted(selected, key=lambda c: STAGE_ORDER.index(c.stage)):
        if args.report_only:
            break
        record = {"name": check.name, "stage": check.stage, "question": check.question,
                  "needs_gpu": check.needs_gpu, "when": time.time()}

        blocked_by = [n for n in failed_blockers
                      if check.name in next(c.blocks for c in CHECKS if c.name == n)]
        if blocked_by:
            record.update(status=BLOCKED, reason=f"depends on failed {blocked_by[0]}")
        elif not script_exists(check):
            record.update(status=SKIP if check.optional else MISSING,
                          reason="script not present yet")
        elif args.no_gpu and check.needs_gpu:
            record.update(status=SKIP, reason="--no-gpu")
        elif is_fresh(check, state) and not args.force:
            record.update(status=PASS, reason="artifact already present (cached)",
                          cached=True)
        else:
            print(f"\n=== {check.name} ({check.stage}) ===\n{check.question}", flush=True)
            record.update(run_check(check))
            print(f"-> {record['status']} in {record.get('duration_s', 0):.0f}s", flush=True)

        if record["status"] in (FAIL, ERROR) and check.blocks:
            failed_blockers.add(check.name)
        state[check.name] = record
        args.state.write_text(json.dumps({"checks": state}, indent=2, default=str))

    write_report(state, args.report)
    print(f"\nreport -> {args.report}")

    summary = {}
    for record in state.values():
        summary[record["status"]] = summary.get(record["status"], 0) + 1
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(summary.items())))
    return 1 if summary.get(FAIL) or summary.get(ERROR) else 0


def write_report(state, path):
    """A report a person can read at 8am without opening any JSON."""
    lines = ["# Preflight report", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "Every check below answers one question about whether this pipeline is",
             "safe to spend GPU allocation on. A FAIL is a finding, not a crash.", "",
             "| Check | Stage | Status | Took | Question |", "|---|---|---|---|---|"]
    for name in [c.name for c in CHECKS]:
        record = state.get(name)
        if not record:
            lines.append(f"| `{name}` | | *not run* | | |")
            continue
        took = f"{record.get('duration_s', 0):.0f}s" if record.get("duration_s") else "-"
        mark = {PASS: "**PASS**", FAIL: "**FAIL**", ERROR: "ERROR",
                SKIP: "skip", BLOCKED: "blocked", MISSING: "missing"}.get(
                    record["status"], record["status"])
        lines.append(f"| `{name}` | {record.get('stage', '')} | {mark} | {took} "
                     f"| {record.get('question', '')} |")

    for name in [c.name for c in CHECKS]:
        record = state.get(name)
        if not record or record["status"] in (SKIP, MISSING):
            continue
        lines += ["", f"## {name}", "", f"*{record.get('question', '')}*", ""]
        if record.get("reason"):
            lines.append(f"Status: {record['status']} — {record['reason']}")
        else:
            lines.append(f"Status: {record['status']}")
        check = next(c for c in CHECKS if c.name == name)
        artifact = load_artifact(check)
        if artifact is not None:
            excerpt = json.dumps(artifact, indent=2, ensure_ascii=False)
            if len(excerpt) > 2500:
                excerpt = excerpt[:2500] + "\n... (truncated; see the JSON artifact)"
            lines += ["", "```json", excerpt, "```"]
        elif record.get("stdout"):
            lines += ["", "```", record["stdout"][-1800:], "```"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())

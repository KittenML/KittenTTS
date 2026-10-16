"""Keep running preflight until it is green, or until the deadline.

`preflight.py` answers "is this pipeline safe to train on?" once. This runs it
repeatedly, unattended, and exists for one reason: the checks and the scripts they
call are being written in parallel, so a check that is MISSING at midnight becomes
runnable at one in the morning. A single pass would report it missing and stop; a
loop picks it up on the next sweep and the suite converges on its own.

It is deliberately a separate process from whoever started it. Nothing here needs
a session to stay open, so the work carries to morning whether or not anything is
still watching.

    python -m indic.overnight --deadline-hours 8

Three rules it follows:

**Never re-run a green check.** Preflight caches on its artifacts, so each sweep
only costs the checks that are still missing or failing.

**Never contend for the GPU.** If another process holds GPU memory -- a subagent
still measuring, a stray notebook -- GPU checks are deferred to the next sweep
rather than run alongside it, because two processes measuring one 12 GB card
produce two wrong answers and possibly two OOMs.

**A FAIL is terminal for that check, a MISSING is not.** A check that ran and
answered "no" will answer "no" again; re-running it wastes the night. A check
whose script did not exist yet deserves another look. So failures are retried at
most `--retry-failed` times, and the default is once -- enough to absorb a
transient download error, not enough to loop on a real finding.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "indic" / "data"
PYTHON = str(ROOT / ".venv" / "bin" / "python")

TERMINAL = {"PASS", "SKIP"}


def gpu_busy(threshold_mib=600):
    """True when something other than the display is holding GPU memory.

    The threshold exists because the desktop compositor itself occupies a couple
    of hundred MiB on this machine; treating that as "busy" would defer every GPU
    check forever.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0]) > threshold_mib
    except (subprocess.SubprocessError, ValueError, IndexError, FileNotFoundError):
        return False                     # cannot tell; let preflight try


def read_state(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("checks", {})
    except (json.JSONDecodeError, OSError):
        return {}


def outstanding(state):
    """Checks that are not yet settled, as {name: status}."""
    from indic.preflight import CHECKS

    out = {}
    for check in CHECKS:
        record = state.get(check.name)
        status = record["status"] if record else "not run"
        if status not in TERMINAL:
            out[check.name] = status
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline-hours", type=float, default=8.0)
    ap.add_argument("--sweep-interval", type=int, default=300,
                    help="seconds between sweeps when work remains")
    ap.add_argument("--retry-failed", type=int, default=1,
                    help="how many times a FAIL/ERROR check is retried before "
                         "being accepted as the answer")
    ap.add_argument("--log", type=pathlib.Path, default=DATA / "overnight.log")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    DATA.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.deadline_hours * 3600
    state_path = DATA / "preflight.json"
    attempts = {}
    sweep = 0

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(args.log, "a") as handle:
            handle.write(line + "\n")

    log(f"overnight start; deadline in {args.deadline_hours}h")

    while time.time() < deadline:
        sweep += 1
        state = read_state(state_path)
        todo = outstanding(state)

        # A check that already answered FAIL/ERROR more times than allowed is its
        # own answer; stop paying for it.
        retryable = {n: s for n, s in todo.items()
                     if s not in ("FAIL", "ERROR") or attempts.get(n, 0) < args.retry_failed}
        if not retryable:
            log(f"sweep {sweep}: nothing left to run; settled")
            break

        skip_gpu = gpu_busy()
        log(f"sweep {sweep}: {len(retryable)} outstanding "
            f"({', '.join(sorted(retryable))}){'; GPU busy, deferring GPU checks' if skip_gpu else ''}")

        argv = [PYTHON, "-m", "indic.preflight"]
        if skip_gpu:
            argv.append("--no-gpu")
        for name in retryable:
            argv += ["--only", name]
            if todo[name] in ("FAIL", "ERROR"):
                attempts[name] = attempts.get(name, 0) + 1
        # --force so a retried FAIL actually re-runs rather than reading its own
        # stale artifact back as a cached PASS.
        if any(s in ("FAIL", "ERROR") for s in retryable.values()):
            argv.append("--force")

        subprocess.run(argv, cwd=ROOT)

        remaining = outstanding(read_state(state_path))
        if not remaining:
            log("all checks settled")
            break
        time.sleep(args.sweep_interval)

    subprocess.run([PYTHON, "-m", "indic.preflight", "--report-only"], cwd=ROOT)
    final = read_state(state_path)
    counts = {}
    for record in final.values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    log("final: " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    log(f"report at {DATA / 'PREFLIGHT_REPORT.md'}")


if __name__ == "__main__":
    main()

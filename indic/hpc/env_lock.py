"""Record the exact environment a run happened in, and detect drift before trusting a comparison.

The arms of this experiment are produced weeks apart, on whichever nodes the
scheduler had free, by whoever was at a terminal. Between arm A and arm B a module
gets upgraded, a site-packages tree is rebuilt, someone rebases the branch, and the
numbers move for a reason that has nothing to do with the tokenizer under test.
None of that announces itself: `transformers` changing its default attention
implementation or its RoPE arithmetic is a silent few-percent change in generated
audio, and a dirty working tree is invisible in the output directory.

So every run writes a lock file next to its results, and every run checks itself
against the sweep's lock before spending a GPU-hour. What is recorded is what can
move the numbers:

    the repository       commit, branch, and whether the tree was dirty (with a
                         hash of the diff, so "dirty" is comparable rather than
                         merely alarming)
    the interpreter      version and executable
    the packages         every installed distribution; the ones in TRACKED are
                         compared at blocking severity
    the accelerator      torch build, driver, NCCL, and the device names and
                         compute capabilities the job actually saw
    the environment      the variables that redirect caches and collectives

The device list is recorded because it is not neutral. Two GPU generations run
different kernels, reduce in a different order, and therefore sample different
tokens from the same seed; a sweep whose arms ran on mixed hardware has a
confound in it even when every version matches.

    python -m indic.hpc.env_lock capture --out runs/sweep-01/env.json
    python -m indic.hpc.env_lock check  --lock runs/sweep-01/env.json
    python -m indic.hpc.env_lock diff   --lock arm-a/env.json --against arm-b/env.json
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Distributions whose version changes what the model produces or how it is scored.
# A difference in any of these is blocking, because comparing across it compares
# two pipelines rather than two tokenizers.
TRACKED = ("torch", "torchaudio", "transformers", "tokenizers", "safetensors",
           "huggingface-hub", "numpy", "scipy", "librosa", "soundfile",
           "s3tokenizer", "diffusers", "kitten-text-processing", "datasets",
           "sentencepiece", "flash-attn", "vllm", "onnxruntime", "einops")

# Variables that silently redirect where a run reads models from, how collectives
# are routed, and how many threads each task takes. A sweep in which some tasks
# had NCCL_P2P_DISABLE and others did not is not one sweep.
TRACKED_ENV = ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
               "HF_DATASETS_OFFLINE", "TMPDIR", "CUDA_VISIBLE_DEVICES",
               "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF",
               "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME",
               "TOKENIZERS_PARALLELISM", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")

SLURM_ENV = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
             "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_JOB_QOS",
             "SLURM_GPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE")


def _git(*args):
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def git_state():
    """Commit, branch and a fingerprint of any uncommitted changes.

    The diff is hashed rather than stored: the point is to tell two dirty trees
    apart, not to reconstruct either. A run from a dirty tree is recorded as such
    so that a result nobody can reproduce is at least labelled.
    """
    diff = _git("diff", "HEAD")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--always", "--dirty"),
        "dirty": bool(diff),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else "",
        "untracked_files": len(untracked.splitlines()) if untracked else 0,
    }


def packages():
    """Every installed distribution and its version.

    importlib.metadata rather than `pip freeze`: it reads the same metadata without
    starting pip, which on a compute node with no network and a read-only
    site-packages is both faster and less likely to fail.
    """
    from importlib.metadata import distributions

    found = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name:
            found[name] = dist.version
    return dict(sorted(found.items()))


def accelerator():
    """Torch's build, the driver under it, and the devices this process can see."""
    try:
        from indic.hpc.capability import probe
    except ImportError:
        return {"error": "indic.hpc.capability not importable"}
    try:
        capability = probe()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return {
        "torch": capability.torch_version,
        "torch_cuda_build": capability.torch_cuda_build,
        "driver": capability.driver_version,
        "cudnn": capability.cudnn_version,
        "nccl": capability.nccl_version,
        "arch_list": capability.arch_list,
        "flash_attn": capability.flash_attn_version if capability.flash_attn_installed else "",
        "devices": [{"name": d.name, "sm": d.sm, "vram_gb": round(d.vram_gb, 1)}
                    for d in capability.devices],
    }


def capture(label=""):
    """The full record. Safe on a login node and inside a job step."""
    return {
        "label": label,
        "host": platform.node(),
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation(),
                   "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "git": git_state(),
        "accelerator": accelerator(),
        "packages": packages(),
        "env": {k: os.environ[k] for k in TRACKED_ENV if k in os.environ},
        "slurm": {k: os.environ[k] for k in SLURM_ENV if k in os.environ},
    }


def compare(recorded, current):
    """Differences between two records, as (severity, field, recorded, current).

    Severity is "blocking" when the difference changes what the pipeline computes,
    and "warning" when it changes only how fast or on what it ran. The device list
    is a warning rather than blocking because a deliberate cross-hardware sweep is
    a legitimate thing to run — but it is reported every time, because an
    accidental one looks identical.
    """
    differences = []

    def check(severity, field, a, b):
        if a != b:
            differences.append((severity, field, a, b))

    check("blocking", "git.commit", recorded["git"]["commit"], current["git"]["commit"])
    check("blocking", "git.dirty", recorded["git"]["dirty"], current["git"]["dirty"])
    if recorded["git"]["dirty"] and current["git"]["dirty"]:
        check("blocking", "git.diff_sha256", recorded["git"]["diff_sha256"],
              current["git"]["diff_sha256"])
    check("blocking", "python.version", recorded["python"]["version"],
          current["python"]["version"])

    recorded_packages, current_packages = recorded["packages"], current["packages"]
    for name in TRACKED:
        check("blocking", f"package.{name}", recorded_packages.get(name),
              current_packages.get(name))
    untracked = sorted(
        {*recorded_packages, *current_packages}
        - {n.lower() for n in TRACKED})
    moved = [n for n in untracked if recorded_packages.get(n) != current_packages.get(n)]
    if moved:
        differences.append(("warning", f"{len(moved)} other package(s)",
                            ", ".join(f"{n}={recorded_packages.get(n)}" for n in moved[:4]),
                            ", ".join(f"{n}={current_packages.get(n)}" for n in moved[:4])))

    recorded_accel = recorded.get("accelerator", {})
    current_accel = current.get("accelerator", {})
    for field in ("torch_cuda_build", "driver", "nccl", "flash_attn", "arch_list"):
        check("warning", f"accelerator.{field}", recorded_accel.get(field),
              current_accel.get(field))
    recorded_devices = [d["sm"] + " " + d["name"] for d in recorded_accel.get("devices", [])]
    current_devices = [d["sm"] + " " + d["name"] for d in current_accel.get("devices", [])]
    check("warning", "accelerator.devices", recorded_devices, current_devices)

    for name in TRACKED_ENV:
        check("warning", f"env.{name}", recorded["env"].get(name),
              current["env"].get(name))
    return differences


def format_differences(differences):
    if not differences:
        return "environment matches the lock exactly"
    lines = []
    for severity, field, before, after in differences:
        marker = "X" if severity == "blocking" else "!"
        lines.append(f"  {marker} {field}")
        lines.append(f"      locked:  {before!r}")
        lines.append(f"      current: {after!r}")
    blocking = sum(1 for d in differences if d[0] == "blocking")
    lines.append(f"\n{blocking} blocking, {len(differences) - blocking} warning")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["capture", "check", "diff", "show"])
    ap.add_argument("--out", type=pathlib.Path, help="where to write the lock")
    ap.add_argument("--lock", type=pathlib.Path, help="the lock to check against")
    ap.add_argument("--against", type=pathlib.Path,
                    help="a second lock, for `diff` between two recorded runs")
    ap.add_argument("--label", default="", help="a name for this run, stored in the lock")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero on warnings as well as blocking differences")
    args = ap.parse_args()

    if args.action == "capture":
        record = capture(args.label)
        text = json.dumps(record, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            git = record["git"]
            print(f"wrote {args.out}")
            print(f"  commit {git['describe'] or git['commit'][:12] or 'unknown'}"
                  f"{' (DIRTY)' if git['dirty'] else ''}")
            devices = record["accelerator"].get("devices", [])
            print(f"  torch {record['accelerator'].get('torch')}, "
                  f"transformers {record['packages'].get('transformers')}")
            print(f"  devices: {', '.join(d['sm'] + ' ' + d['name'] for d in devices) or 'none'}")
        else:
            print(text)
        return 0

    if args.action == "show":
        if not args.lock:
            ap.error("show needs --lock")
        print(json.dumps(json.loads(args.lock.read_text()), indent=2))
        return 0

    if not args.lock:
        ap.error(f"{args.action} needs --lock")
    if not args.lock.is_file():
        print(f"no lock at {args.lock}. Write one first:\n"
              f"  python -m indic.hpc.env_lock capture --out {args.lock}",
              file=sys.stderr)
        return 1
    recorded = json.loads(args.lock.read_text())
    current = json.loads(args.against.read_text()) if args.against else capture()

    differences = compare(recorded, current)
    what = (f"{args.lock} vs {args.against}" if args.against
            else f"this environment vs {args.lock}")
    print(what)
    print(format_differences(differences))

    blocking = [d for d in differences if d[0] == "blocking"]
    if blocking:
        sys.stdout.flush()
        print("\nThese results are not comparable with the locked run. Either "
              "reproduce the locked environment, or record this run under its own "
              "lock and label the arm accordingly.", file=sys.stderr)
    return 1 if blocking or (args.strict and differences) else 0


if __name__ == "__main__":
    sys.exit(main())

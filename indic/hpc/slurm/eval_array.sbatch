#!/bin/bash
# The evaluation sweep as a Slurm array: one task per (model, language, test set, shard).
#
# The sweep is embarrassingly parallel -- nothing a task computes is needed by any
# other -- so it belongs in an array rather than one long job. An array also means
# a task that dies takes one cell of the grid with it instead of the whole run, and
# `scontrol requeue` brings that cell back without recomputing the other 299.
#
# NOTHING HERE IS SPECIFIC TO A CLUSTER. Every site-dependent value is a variable
# at the top with a comment saying how to find your site's value. Edit those, not
# the body.
#
#   sbatch --array=0-299%20 indic/hpc/slurm/eval_array.sbatch tasks.tsv
#
# `%20` caps concurrency: raise it to fill the partition, lower it when the shared
# filesystem is the bottleneck (300 tasks each loading a 1 GB checkpoint from NFS
# at the same second is a denial of service against your own cluster).

# ─── edit these ───────────────────────────────────────────────────────────────
#SBATCH --job-name=kindic-eval
#
# Partition/account/QOS: `sinfo -s` lists partitions, `sacctmgr show assoc user=$USER`
# your accounts. There is no portable default, so these are left commented out on
# purpose -- uncomment and fill them in, or pass them on the sbatch command line.
##SBATCH --partition=REPLACE_ME
##SBATCH --account=REPLACE_ME
##SBATCH --qos=REPLACE_ME
#
# GPU type. This is the line that matters on a heterogeneous cluster: without it
# the scheduler hands you whatever is free, and a V100 cannot run this checkpoint
# at all (see indic/hpc/capability.py -- fp16 overflows it and the card has no
# bf16). Your site's feature names are whatever its admins typed; list them with
#     sinfo -o "%20N %10c %10m %25f %15G"
# and put the Ampere-or-newer ones here. Common spellings are a80|a100|h100 or
# gpu_a100; some sites instead expose `--gres=gpu:a100:1`, in which case use that
# and delete the constraint.
##SBATCH --constraint=REPLACE_ME_WITH_AMPERE_OR_NEWER
#
# One GPU per task. --gpus-per-task needs Slurm >= 21.08; on older schedulers
# replace it with `--gres=gpu:1`.
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#
# Memory is host RAM, not VRAM. The checkpoint unpacks on the CPU before it moves
# to the device (weights.py -> ternary.py), so a task peaks around 8 GB of host
# memory for the 1.7B model plus whatever the ASR scorer needs. 32 GB is slack for
# the audio batches; lower it if your partition charges by memory.
#SBATCH --mem=32G
#
# Wall time per task, not for the whole array. A 300-utterance language/test-set
# shard at roughly real time is well inside two hours; measure one task before
# committing the sweep.
#SBATCH --time=02:00:00
#
# %A is the array job id, %a the task index. Separate files per task: a shared one
# interleaves 20 concurrent writers into something unreadable.
#SBATCH --output=logs/eval-%A_%a.out
#SBATCH --error=logs/eval-%A_%a.err
#
# Preempted or node-failed tasks come back automatically. Combined with the
# "already done" check below this is safe to leave on.
#SBATCH --requeue

set -euo pipefail

# ─── configuration, overridable from the environment ──────────────────────────
# Where the artifacts were staged from a login node:
#     python -m indic.hpc.stage_cache download --hf-home "$STAGED_HF_HOME" --asr
: "${STAGED_HF_HOME:?set STAGED_HF_HOME to the directory staged by indic.hpc.stage_cache}"
: "${RESULTS_DIR:=results}"
: "${ENV_LOCK:=$RESULTS_DIR/env.lock.json}"
: "${REPO_DIR:=$(pwd)}"
# How to enter the Python environment. Sites differ completely here -- `module
# load`, conda, a venv, a container -- so this is one hook rather than a guess.
: "${PYTHON:=python}"
: "${ACTIVATE:=}"

# Anything after the task file is passed straight through to the eval module,
# so one array can be submitted twice with different flags:
#     sbatch ... eval_array.sbatch tasks.tsv --text-fixes indic
TASK_FILE="${1:-${TASK_FILE:-tasks.tsv}}"
[ $# -gt 0 ] && shift

# ─── environment ──────────────────────────────────────────────────────────────
[ -n "$ACTIVATE" ] && source "$ACTIVATE"
cd "$REPO_DIR"

# Offline, because compute nodes have no route out. Exported before Python starts:
# transformers reads TRANSFORMERS_OFFLINE at import time and setting it later has
# no effect. See indic/hpc/stage_cache.py for what each one does.
eval "$("$PYTHON" -m indic.hpc.stage_cache env --hf-home "$STAGED_HF_HOME")"

# Node-local scratch. /tmp on a compute node is often a small RAM-backed tmpfs
# shared by every job on the node: filling it counts against this job's memory
# limit and gets it killed for OOM with no mention of /tmp anywhere. Slurm exposes
# per-job scratch under different names per site; adjust the first candidate.
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/kindic-${SLURM_ARRAY_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Compile caches default under $HOME, which on a shared filesystem means every
# concurrent array task JIT-compiling into the same directory. They race, and a
# corrupted cache entry fails a task for reasons that look like a model bug.
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export XDG_CACHE_HOME="$TMPDIR/xdg"

# One task owns one GPU, so intra-op threads should match the CPUs it was given,
# not the node's core count -- otherwise 20 tasks each spawn 128 threads.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false

# ─── preflight: fail in seconds, not in an hour ───────────────────────────────
# Each of these answers a question that otherwise surfaces as a confusing failure
# deep inside a model constructor.
echo "== preflight on $(hostname) =="
"$PYTHON" -m indic.hpc.capability --require-gpu
"$PYTHON" -m indic.hpc.stage_cache verify --hf-home "$STAGED_HF_HOME" --no-load-check

if [ -f "$ENV_LOCK" ]; then
  # Blocking drift (repo commit, torch, transformers) stops the task: a result
  # computed against a different pipeline is not comparable with the rest of the
  # sweep, and finding that out at analysis time is far more expensive.
  "$PYTHON" -m indic.hpc.env_lock check --lock "$ENV_LOCK"
else
  echo "no lock at $ENV_LOCK — recording this environment as the sweep's reference"
  "$PYTHON" -m indic.hpc.env_lock capture --out "$ENV_LOCK" --label "${SLURM_JOB_NAME:-eval}"
fi

# ─── the work item ────────────────────────────────────────────────────────────
# One line per task, tab-separated:
#     model <TAB> language <TAB> testset <TAB> shard <TAB> num_shards
# Build it however you like; the array index selects the line. For example, with
# every (model, language, test set) split into four shards:
#   N=4
#   for m in KittenML/kitten-tts-2 ai4bharat/IndicF5; do
#     for l in hi bn ta te mr; do
#       for t in A B C D E; do
#         for s in $(seq 0 $((N - 1))); do
#           printf '%s\t%s\t%s\t%s\t%s\n' "$m" "$l" "$t" "$s" "$N"
#         done
#       done
#     done
#   done > tasks.tsv
# then  sbatch --array=0-$(( $(wc -l < tasks.tsv) - 1 ))%20 ... tasks.tsv
[ -f "$TASK_FILE" ] || { echo "no task file at $TASK_FILE" >&2; exit 2; }
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(sed -n "$((INDEX + 1))p" "$TASK_FILE")
[ -n "$LINE" ] || { echo "array index $INDEX past the end of $TASK_FILE" >&2; exit 2; }
IFS=$'\t' read -r MODEL LANG TESTSET SHARD NUM_SHARDS <<<"$LINE"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

# A model name is a Hub repo id, so it carries a slash that would otherwise turn
# into a directory level and make the results tree inconsistent with the flat one
# score.py expects.
MODEL_DIR="${MODEL//\//__}"
# All shards of one (model, language, test set) share an output directory: each
# writes its own run.<shard>.json and meta.<shard>.jsonl, and score.py reads
# whichever shards are there. Giving each shard its own directory would hide the
# others from it.
OUT="$RESULTS_DIR/$MODEL_DIR/$LANG/$TESTSET"
mkdir -p "$OUT"

# Requeue-safe: a task writes its marker last, so a requeued or re-submitted array
# does not redo completed cells. Per shard, because the directory is shared.
if [ -f "$OUT/.complete.$SHARD" ]; then
  echo "already complete: $OUT shard $SHARD"
  exit 0
fi

echo "== task $INDEX: model=$MODEL lang=$LANG testset=$TESTSET shard=$SHARD/$NUM_SHARDS =="
"$PYTHON" -m indic.hpc.env_lock capture --out "$OUT/env.$SHARD.json" \
  --label "$MODEL/$LANG/$TESTSET/$SHARD"

# The synthesis half of the eval. `indic/eval/` is owned separately — if its
# flags move, change them here rather than renaming anything there.
: "${EVAL_MODULE:=indic.eval.run_eval}"
srun --kill-on-bad-exit=1 "$PYTHON" -m "$EVAL_MODULE" \
  --model "$MODEL" --lang "$LANG" --testset "$TESTSET" \
  --shard "$SHARD" --num-shards "$NUM_SHARDS" --out "$OUT" "$@"

touch "$OUT/.complete.$SHARD"
echo "== done: $OUT shard $SHARD =="

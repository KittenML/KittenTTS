# Running this on a cluster

This repository was built on one laptop with one GPU and a working network. A
Slurm cluster is none of those things: the nodes are not the same as each other,
the compute nodes have no route off the machine, and the arms of an experiment are
produced weeks apart by whichever hardware was free. Each file here closes one of
those gaps.

| File | What it is for |
|---|---|
| [`capability.py`](capability.py) | What the GPU in front of you can run, and which dtype is safe on it |
| [`stage_cache.py`](stage_cache.py) | Pre-fetch every Hugging Face artifact from a login node; verify it offline |
| [`env_lock.py`](env_lock.py) | Record the exact environment of a run; refuse to compare across drift |
| [`slurm/eval_array.sbatch`](slurm/eval_array.sbatch) | The evaluation sweep as a Slurm array |
| [`slurm/train.sbatch`](slurm/train.sbatch) | Single-node fine-tuning on one GPU or all of them |
| [`../requirements-hpc.txt`](../requirements-hpc.txt) | Pinned versions, with the torch build left to the cluster |

Every module has a `--help` and a module docstring that explains why it exists.
This page is the order to do things in and the things that are not obvious from
any one file.

## The short version

```bash
# once, on a login node that has a network
python -m pip install -r indic/requirements-hpc.txt
python -m indic.hpc.capability                                   # does the wheel fit the cards?
python -m indic.hpc.stage_cache download --hf-home /scratch/shared/hf --asr
python -m indic.hpc.stage_cache verify   --hf-home /scratch/shared/hf
python -m indic.hpc.env_lock capture --out results/env.lock.json --label sweep-01

# then, per sweep
export STAGED_HF_HOME=/scratch/shared/hf
sbatch --array=0-$(( $(wc -l < tasks.tsv) - 1 ))%20 \
       indic/hpc/slurm/eval_array.sbatch tasks.tsv
```

`tasks.tsv` is one work item per line, tab-separated:
`model`, `language`, `testset`, `shard`, `num_shards`. The array index picks the
line; `eval_array.sbatch` has a loop that writes one. All shards of a
(model, language, test set) share an output directory, because that is what
`indic/eval/score.py` reads.

## What a heterogeneous cluster breaks

**The wheel does not fit every card.** A torch wheel carries compiled kernels for
a fixed list of architectures. A card outside that list fails at the first kernel
launch — not at install, not at import — with `no kernel image is available for
execution on the device`. The cu128 wheel this work was developed against starts
at `sm_75`, so it cannot run on a V100 at all. `python -m indic.hpc.capability`
prints an **in build** column that answers this per card before anything is
scheduled, and exits nonzero when the answer is no.

**`torch.cuda.is_bf16_supported()` is not the question you meant.** It counts
software emulation by default, so it returns `True` on a V100, where bf16 runs
through fp32 with the tensor cores idle. A sweep that trusts it gets one arm that
is several times slower than the rest and no warning anywhere.
`select_dtype()` uses the compute capability instead.

**fp16 is not an escape hatch for this checkpoint.** The tempting move on
pre-Ampere hardware is fp16, which those cards do have tensor cores for. It does
not work here, and the reason is measurable rather than theoretical:

- The per-group ternary scales in `lm/model-ternary.safetensors` are *fine* in
  fp16. All 11,010,048 of them lie in `[2.689e-04, 0.785]`, fp16's smallest
  normal is `6.1e-05`, and every one of them converts bf16 → fp16 → fp32
  unchanged. The usual "bf16 has a much wider exponent range" worry does not bite
  at these magnitudes. `python -m indic.hpc.capability --check-scales <file>`
  re-measures this for any checkpoint, because a QAT or vocab-surgery output has
  its own distribution.
- The **activations** are what break. One prefill through the 28 layers peaks at
  **3,194,880** in the residual stream at layer 27, against an fp16 ceiling of
  65,504 — 49× over — and 33 of 424 module outputs exceed it. The forward pass
  saturates to `inf` and the logits come out NaN. Loss scaling is a training
  trick and does not apply to a forward pass.

So pre-Ampere has bf16-by-emulation (correct, slow) or fp32 (correct, slow, 6.9 GB
of weights) and nothing else. `select_dtype()` refuses to choose for you:

```python
select_dtype()                      # sm_80+: bfloat16.  pre-Ampere: raises
select_dtype(pre_ampere="fp32")     # the supported downgrade, explicitly asked for
select_dtype(pre_ampere="fp16")     # still raises, with the measurement
```

The better answer is usually `--constraint` in the sbatch file, steering the task
to a card that can run it.

**Mixed hardware is a confound, not just a speed difference.** Two GPU
generations run different kernels and reduce in a different order, so the same
seed samples different tokens and produces different audio. An arm whose tasks
landed on a mixture of cards has that mixture baked into its CER. `env_lock`
records the device list per task for exactly this reason; `capability` warns when
one process sees more than one generation.

**The driver may be older than the runtime.** Clusters pin driver versions per
node image, and one straggler node with an old driver fails every task that lands
on it. `capability` checks the driver against the minimum for the torch build's
CUDA major and names the number.

**Collectives need to be told about the fabric.** `NCCL_SOCKET_IFNAME` and
`NCCL_P2P_DISABLE` are commented in `train.sbatch` rather than set, because the
right values are site-specific and setting them blindly costs real bandwidth.
`NCCL_DEBUG=WARN` is on, so a failing transport says so instead of hanging.

## The offline workflow

Compute nodes usually cannot reach the internet. The failure mode is not a clean
error: `hf_hub_download` waits out its timeout and then raises from inside a model
constructor, per task, an hour in.

1. **Stage from a login node.** `stage_cache download` fetches the TTS checkpoint
   (file list taken from `kittenml/get_model.py` itself, so it cannot drift from
   what the model loads), the s3gen vocoder from its *separate* repository, the
   Sarvam-1 tokenizer, the FLEURS transcripts, and — behind flags — the ASR
   scorers, FLEURS audio, and baseline systems.
2. **It writes a manifest** of every file at the revision it was staged at.
3. **Verify under real offline conditions.** `stage_cache verify` re-runs itself
   with `HF_HUB_OFFLINE=1` set in the child process rather than trusting
   `local_files_only`, resolves every manifest entry, compares sizes, and then
   actually constructs the tokenizers. It lists exactly what is missing and exits
   nonzero.

### Environment variables compute nodes must have

`stage_cache env --hf-home <dir>` prints these; both sbatch templates `eval` its
output. Export them **before** Python starts — `transformers` reads
`TRANSFORMERS_OFFLINE` at import time and setting it afterwards does nothing.

| Variable | Why |
|---|---|
| `HF_HOME` | The staged tree. Repositories live in `$HF_HOME/hub`, which is why the tool takes `--hf-home` and appends `hub` itself — pass the two different things and the node looks in an empty directory and tries to download. |
| `HF_HUB_OFFLINE=1` | Resolve from the cache and fail immediately instead of waiting out a TCP timeout that will never succeed. |
| `TRANSFORMERS_OFFLINE=1` | The same switch for the paths in `transformers` that consult the Hub for their own reasons. Neither variable covers the other. |
| `HF_DATASETS_OFFLINE=1` | The same again for `datasets`, which the eval harness uses for FLEURS audio. |
| `HF_HUB_DISABLE_TELEMETRY=1` | Telemetry is an outbound request; on a node with no route it is a timeout per process. |

And the ones that are not about Hugging Face at all:

| Variable | Why |
|---|---|
| `TMPDIR` | `/tmp` on a compute node is often a small RAM-backed tmpfs shared by every job on the node. Filling it with extracted FLEURS audio counts against *your* job's memory limit and gets the job killed for OOM, with no mention of `/tmp` in the message. Point it at node-local scratch (`$SLURM_TMPDIR` where the site provides it). |
| `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `XDG_CACHE_HOME` | These default under `$HOME`, which on a shared filesystem means every concurrent array task JIT-compiling into the same directory. They race, and a corrupted cache entry fails a task in a way that looks like a model bug. One per task, on node-local scratch. |
| `OMP_NUM_THREADS` | Defaults to the node's core count, per task. Twenty array tasks on one node then spawn twenty times that. Set it from `SLURM_CPUS_PER_TASK`. |

## Adapting the sbatch templates

Everything site-specific is a variable at the top of each file, with a comment
saying how to find your site's value. Nothing is hardcoded to a cluster, and the
partition/account/constraint lines are left commented out rather than filled with
a plausible-looking guess.

What you have to supply:

- **Partition, account, QOS.** `sinfo -s` and `sacctmgr show assoc user=$USER`.
- **`--constraint`.** The feature names are whatever your admins typed. List them
  with `sinfo -o "%20N %10c %10m %25f %15G"` and pick the Ampere-or-newer ones.
  Some sites use `--gres=gpu:a100:1` instead; use that and delete the constraint.
- **`ACTIVATE`.** How to enter the Python environment — `module load`, conda, a
  venv, a container. One hook, because no two sites agree.
- **`EVAL_MODULE` / `TRAIN_MODULE`.** `eval_array.sbatch` calls
  `indic.eval.run_eval` with the flags it declares today
  (`--model --lang --testset --shard --num-shards --out`), and passes anything
  after the task file straight through, so `sbatch ... tasks.tsv --text-fixes
  indic` runs the A/B arm. The training entry point does not exist yet — see
  `docs/indic/training.md` for the plan.

Things to know before editing them:

- `--gpus-per-task` needs Slurm ≥ 21.08; older schedulers want `--gres=gpu:1`.
- `--mem` is host RAM, not VRAM. The checkpoint unpacks on the CPU before it moves
  to the device, so a task needs real host memory even though the model is small.
- `%A_%a` in the output paths gives one log file per array task. One shared file
  interleaves twenty concurrent writers into something unreadable.
- Each task writes a `.complete` marker last, so `--requeue` and re-submitting the
  whole array are both safe — finished cells are skipped.
- `train.sbatch` uses `torchrun --standalone`, not N srun tasks. N srun tasks that
  each try to be rank 0 is the classic way to hang in `init_process_group` until
  the wall clock runs out. `--nproc_per_node=1` is the same code path, so the
  pilot and the main run use one file.
- Multi-**node** training is deliberately not templated: it needs a rendezvous
  endpoint and a per-site answer about the interconnect, and the compute estimate
  in `docs/indic/training.md` fits inside one node.

## Version drift between arms

Arms are compared weeks apart. Between them, a module gets upgraded, a
site-packages tree is rebuilt, someone rebases the branch. `transformers`
changing an attention default is a silent few-percent change in generated audio.

Each sweep captures one lock file; each task captures its own and checks it
against the sweep's. Differences are split by severity: the repository commit,
the Python version and the packages in `TRACKED` are **blocking** — a result
computed against a different pipeline is not comparable, and the task stops rather
than producing a number someone will average in later. Driver, NCCL, device list
and environment variables are **warnings**, reported every time because a
deliberate cross-hardware sweep and an accidental one look identical.

A dirty working tree is recorded, with a hash of the diff so two dirty trees can
be told apart. It is not refused — uncommitted work is how pilots get run — but it
is never invisible.

```bash
python -m indic.hpc.env_lock diff --lock arm-a/env.json --against arm-b/env.json
```

# Platform QA

`.github/workflows/platform-qa.yml` installs this checkout of `kittenml` on GitHub-hosted
runners — Linux, Windows and macOS, x86_64 and ARM, Intel, AMD and Apple CPUs — on every
supported Python version. It runs each model and posts one report to the pull request.

It runs on pull requests and pushes to `main` that touch the package, its dependencies or
this folder. You can also start it from the Actions tab.

## What the report shows

- **Platform Status:** pass or fail for each platform and Python, with the runner's CPU,
  install time, the slowest model, average WER and peak memory.
- **RTF by Platform:** each model's best warm real-time factor on each CPU. RTF is
  generation time ÷ audio length, so below 1 is faster than realtime.
- **Not Supported (expected):** platforms that `config.toml` says cannot install, with
  pip's error.
- **Failures:** each failure with its traceback or install log.
- **Platform Details** (job summary, and the PR comment when it fits): per-model load
  time, first and warm generation times, RTF, peak RAM, the Whisper transcript, WER and
  every API check. There is a link to download that platform's audio.

The run fails when a supported platform fails, a job produces no result, or a model's WER
is above `asr.fail_above`.

## Changing what is tested

Edit [`config.toml`](config.toml). The workflow needs no changes.

| To… | Do this |
|---|---|
| Add or drop a Python version | Edit `pythons` on a `[[target]]` |
| Add a platform | Add a `[[target]]` with a [runner label](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners) |
| Mark a platform as known-unsupported | `expect = "install-fails"` and a `reason` |
| Report a platform without failing the run | `gating = false` |
| Add a model | Add a `[models.<key>]` and list the key in a target's `models` |
| Change one model on one platform | `overrides = { tts2 = { weights = "emb4", warm_runs = 0 } }` |
| Change the spoken text or voice | `[sample]`, or `text = "..."` on a target |
| Change the WER thresholds or ASR model | `[asr]` |

`python qa/plan.py` checks the config and prints the jobs it expands to. The Plan job runs
it too, along with `python -m unittest discover -s qa/tests`.

## Narrowing a manual run

**Actions → Platform QA → Run workflow** takes comma-separated filters: `targets` (matches
name or runner, e.g. `Linux, macos`), `pythons` (e.g. `3.12`) and `models` (e.g.
`nano,tts2`). `source` set to a pip requirement such as `kittenml==0.9.3` tests a PyPI
release instead of the branch.

## Running one platform locally

```bash
python qa/plan.py --out plan.json          # QA_TARGETS / QA_PYTHONS / QA_MODELS filter it
python -c "import json; json.dump(json.load(open('plan.json'))['jobs'][0], open('spec.json', 'w'))"
python qa/run_target.py --spec spec.json --out qa-out
python qa/report.py qa-out report && open report/summary.md
```

Keep the virtualenv path short: espeak-ng exits the whole process when its data path is
160 characters or longer.

## Pull requests from forks

GitHub gives fork pull requests a read-only token, so the report job cannot comment on
them. `platform-qa-comment.yml` runs after the workflow, in the base repository, and posts
the report. It only works once it is on the default branch.

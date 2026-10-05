# Platform QA

`.github/workflows/platform-qa.yml` installs this checkout of `kittenml` on GitHub-hosted
runners (Linux, Windows and macOS; x86_64 and ARM; Intel, AMD and Apple CPUs) on every
Python in `[matrix]`, runs the README's examples for each model, and posts one report to
the pull request.

It assumes nothing about what should work. It answers two questions:

- **What works where?** Every platform runs every Python version, and the report shows
  ✅ or ❌ for each, with pip's or the test's own error for every ❌. That is the list of
  platforms and versions to add support for.
- **Did this change break anything?** Each run is compared with the latest finished run
  on `main` (or this branch's previous run when `main` has none). The run fails only when
  a test that works there stops working here. Something that does not work on `main`
  either is listed, not failed; something that starts working is marked **new**.

A failure on a CPU the `main` run never drew is reported but not counted as broken, since
it cannot tell a regression from a CPU-specific problem (GitHub assigns runner CPUs at
random). The first run, with nothing to compare with, only reports.

It runs on pull requests and pushes to `main` that touch the package, its dependencies or
this folder. You can also start it from the Actions tab.

Every install, load, generation, test and transcription is stopped after
`[limits] step_minutes` (10 min) and reported as timed out, and a job after
`[limits] job_minutes`, so nothing runs for hours.

## What the report shows

The pull request gets one short comment for each commit:

- **Platforms:** one row per platform and one column per Python version, the CPUs the
  runners drew, and how long the jobs took (🐢 over `report.slow_job_minutes`). Under it,
  why each ❌ does not work.
- **KittenTTS 0.8 (ONNX models)** and **KittenTTS 2:** one row per README example and one
  column per platform and CPU, then each model's real-time factor (🐢 slower than realtime)
  and, for KittenTTS 2, peak RAM.
- **Broke since the baseline:** one line per broken test, with a link to the job's log and
  the log's tail.

The run summary has everything above plus every job's numbers: load time, first and warm
generation times, peak RAM, what Whisper heard, every test's result, every failure's log,
and an audio download.

## Changing what is tested

Edit [`config.toml`](config.toml). The workflow needs no changes.

| To… | Do this |
|---|---|
| Add or drop a Python version | `[matrix] pythons`, or `pythons` on one `[[target]]` |
| Add a platform | Add a `[[target]]` with a [runner label](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners) |
| Add a model | Add a `[models.<key>]`; every platform runs it (`models` on a target narrows that) |
| Change one model on one platform | `overrides = { tts2 = { weights = "emb4", warm_runs = 0 } }` |
| Run a platform only on PRs, or only on `main` | `events = ["pull_request"]` or `events = ["push", "workflow_dispatch"]` |
| Change when a job counts as slow | `[report] slow_job_minutes` |
| Change the time limits | `[limits] step_minutes`, `job_minutes`, or `timeout_minutes` on a target |
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

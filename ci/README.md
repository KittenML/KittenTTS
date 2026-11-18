# CI

`.github/workflows/ci.yml` runs on every push to `main`, every pull request, weekly (to catch
a dependency release breaking a clean install), and on demand from the Actions tab.

| Job | Matrix | What it does |
|---|---|---|
| `test` | Linux, Windows, macOS × Python 3.10-3.14 (3.14 allowed to fail) | builds and installs the wheel, runs `pytest`, then the light smoke suite (0.8 ONNX; KittenTTS 2 nano / pico / mini with int8, int4 and torch weights) |
| `kittentts2` | Linux, Windows, macOS × Python 3.12 | KittenTTS 2 (1.7B) on CPU: both weight packings, both decoders, presets, expression tags, streaming, cloning, a fixed seed; MLX on macOS if the runner has Metal |
| `listening report` | | merges every job's audio into one page |

## Listening to the results

Open the workflow run, download the **audio-report** artifact, unzip it and open `index.html`.
The top grid shows every case against every OS / Python; below it each case lists its clips
side by side with a "play all in turn" button, so platform differences stand out. Tick
"sounds off" on anything suspicious and use "copy flagged list" to paste them into an issue.

The automatic checks (`ci/smoke.py`, `check_audio`) fail a clip that is empty, non-finite,
near-silent, mostly silence, or implausibly short or long for its text, and warn on clipping.
They cannot hear a wrong accent, slurring or a dropped word: that is what the page is for.

## Running locally

```bash
pip install . pytest
python ci/smoke.py --list                      # the cases
python ci/smoke.py --out smoke-out             # light suite: 0.8 ONNX + KittenTTS 2 nano / pico / mini
python ci/smoke.py --out smoke-out --suite heavy --only k2-small
python ci/report.py --inputs smoke-out --out report && open report/index.html
```

Add a case by appending to `GROUPS` in `ci/smoke.py`: a group is one model load, its cases are
`generate` / `generate_stream` / `generate_to_file` calls on it.

Setting an `HF_TOKEN` repository secret is optional; it avoids anonymous Hugging Face rate limits.

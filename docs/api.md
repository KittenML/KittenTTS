# API Reference

The KittenTTS 2 API. For the ONNX models see [onnx-models.md](onnx-models.md).

# `KittenTTS(model_name, cache_dir=None, device=None, hf_token=None)`

Loads whichever model family the repository's `config.json` declares.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | -- | Hugging Face repo ID, or a path to a local repository directory |
| `cache_dir` | `str` | `None` | Local directory for caching downloaded files |
| `device` | `str` | `None` | `"cuda"` or `"cpu"`; auto-detected when omitted. CPU works, at roughly 2-3x slower than real time -- and on a many-core host `torch.set_num_threads(8)` is markedly faster than letting it use every core |
| `hf_token` | `str` | `None` | Token, only for downloading a private model repository |
| `decoder` | `str` | repo default | Which decoder to load; see [decoders.md](decoders.md) |
| `weights` | `str` | `"packed"` | Which weight packing to load: `"packed"` (lossless), `"emb4"` (smaller, quantised embedding), `"full"` (bf16) |
| `backend` | `str` | `None` | Default PyTorch inference, or `"vllm"` for optional NVIDIA GPU inference on Linux |
| `vllm_options` | `dict` | `None` | Engine memory and context settings; see [vLLM](vllm.md) |

# `model.generate(text, voice=None, reference=None, ...)`

Returns a float32 NumPy array at `model.sample_rate` (24 kHz). The speaker comes from exactly one
of `voice` or `reference`; giving both, or `reference_text` without `reference`, raises
`ValueError`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | -- | Text to speak; supports the expression controls above |
| `voice` | `str` | repo default | Built-in voice name |
| `reference` | `str` | `None` | Path to a recording to clone instead of using a built-in voice |
| `reference_text` | `str` | `None` | Reference transcript; auto-transcribed when omitted |
| `preset` | `str` | `"stable"` | Decoding preset: `"stable"` or `"expressive"` |
| `max_new_tokens` | `int` | `1000` | Per-chunk token budget; 25 tokens is about one second |
| `normalize` | `bool` | `True` | Expand written forms before synthesis |
| `use_reference_prompt` | `bool` | repo default | Include the reference clip's audio in the prompt |
| `temperature`, `top_k`, `top_p`, `min_p` | | preset | Override individual preset values |
| `advanced` | `dict` | `None` | Rarely-needed knobs; see [Advanced options](#advanced-options) |

# Advanced options

`temperature`, `top_k`, `top_p` and `min_p` are arguments in their own right. Everything else
tunable lives behind `advanced=`, a dict validated against `ADVANCED_OPTIONS` -- an unknown key
raises `ValueError` rather than being silently ignored.

```python
audio = m.generate("Hello.", voice="Bruno", advanced={"seed": 1234})   # reproducible
```

| Option | Default | What it does |
|---|---|---|
| `seed` | `None` | Seed the sampler; the same seed gives the same audio |
| `repetition_penalty` | `1.1` | Penalty on tokens already seen; higher suppresses loops harder |
| `repetition_window` | repo default | Limit that penalty to the last N generated tokens |
| `token_run_penalty` | `1.3` | Escalating penalty once one codec token repeats |
| `token_run_grace` | `10` | How many repeats to allow before that kicks in |
| `chunk_chars` | repo default | Target chunk size for long text |
| `chunk_min_chars` | repo default | Chunks below this are merged into a neighbour |
| `chunk_gap_s` | `0.16` | Pause inserted at a chunk join |
| `use_emotion` | `None` | Force expression conditioning on or off; `None` auto-detects from tags |

`model.advanced_options` returns these with the repository's defaults already filled in.

# `model.generate_stream(...)`

Same arguments; yields each chunk's audio as it is produced.

# `model.generate_to_file(text, output_path, **kwargs)`

Synthesize and write a wav. Returns the path written.

# `model.available_voices`

List of the built-in voice names accepted by `voice`.

# `model.sample_rate`

Output sample rate, 24000.

# Preflight report

Generated 2026-10-16 04:42:16

Every check below answers one question about whether this pipeline is
safe to spend GPU allocation on. A FAIL is a finding, not a crash.

| Check | Stage | Status | Took | Question |
|---|---|---|---|---|
| `capability` | env | **PASS** | 1s | What hardware is this, and is bf16 safe on it? |
| `tokenizer_fertility` | tokenizer | **PASS** | - | How much more does Indic text cost under Qwen3 than under Sarvam-1? |
| `ternary_check` | checkpoint | **PASS** | - | Which quantizer wrote these weights, and can QAT resume without damaging them? |
| `vocab_surgery` | surgery | **PASS** | - | Does the Sarvam vocabulary swap produce a loadable checkpoint? |
| `sarvam_smoke` | surgery | **PASS** | - | Is the rebuilt checkpoint actually correct -- audio rows aligned, tokenizer addressing only text rows? |
| `codec_roundtrip` | gates | **PASS** | - | Can the S3 codec represent Tamil and Telugu at all? |
| `teacher_forced_loss` | gates | **FAIL** | 510s | Does our training sequence format match what the checkpoint was trained on? |
| `train_selfcheck` | train | **PASS** | 7s | Does the training loop provably work -- overfit one batch, exact loss mask, gradients where they belong? |
| `fit_config_sarvam` | fit | **PASS** | 31s | What is the largest training configuration that fits this GPU, Sarvam vocab? |
| `fit_config_qwen3` | fit | **PASS** | 27s | The same, for the unchanged Qwen3 vocab -- how much does the swap buy? |

## capability

*What hardware is this, and is bf16 safe on it?*

Status: PASS

```
host           khushiyant-Vector-16-HX-AI-A2XWHG
python         3.12.3
torch          2.11.0+cu128   (built against CUDA 12.8)
driver         580.173.02
cudnn          91900
nccl           2.28.9   (available: True)
flash-attn     not installed
arch list      sm_75, sm_80, sm_86, sm_90, sm_100, sm_120

dev      name                              sm           VRAM  bf16     flash  in build 
cuda:0   NVIDIA GeForce RTX 5070 Ti Laptop sm_120      11.5G  native   yes    yes      

policy
  cuda:0   dtype bfloat16   attn sdpa

warnings
  ! cuda:0 sm_120 could use flash-attention-2 but the package is not installed; SDPA will be used instead.

blockers
  none

```

## tokenizer_fertility

*How much more does Indic text cost under Qwen3 than under Sarvam-1?*

Status: PASS — artifact already present (cached)

```json
{
  "kitten-qwen3": {
    "repo": "KittenML/kitten-tts-2",
    "vocab_size": 158237,
    "class": "Qwen2Tokenizer",
    "languages": {
      "hi": {
        "tokens_per_word": 4.761904761904762,
        "tokens_per_char": 0.9297738958934987,
        "tokens_per_sentence": 111.86440677966101,
        "p95_tokens_per_sentence": 187,
        "n_sentences": 118,
        "n_tokens": 13200,
        "n_words": 2772,
        "n_chars": 14197
      },
      "bn": {
        "tokens_per_word": 7.139180171591993,
        "tokens_per_char": 1.0658222443606347,
        "tokens_per_sentence": 126.9322033898305,
        "p95_tokens_per_sentence": 227,
        "n_sentences": 118,
        "n_tokens": 14978,
        "n_words": 2098,
        "n_chars": 14053
      },
      "ta": {
        "tokens_per_word": 10.129977876106194,
        "tokens_per_char": 1.0988780224395511,
        "tokens_per_sentence": 155.21186440677965,
        "p95_tokens_per_sentence": 250,
        "n_sentences": 118,
        "n_tokens": 18315,
        "n_words": 1808,
        "n_chars": 16667
      },
      "te": {
        "tokens_per_word": 11.415300546448087,
        "tokens_per_char": 1.4572724101848622,
        "tokens_per_sentence": 177.03389830508473,
        "p95_tokens_per_sentence": 313,
        "n_sentences": 118,
        "n_tokens": 20890,
        "n_words": 1830,
        "n_chars": 14335
      },
      "mr": {
        "tokens_per_word": 6.654509610645638,
        "tokens_per_char": 0.9579283433841788,
        "tokens_per_sentence": 114.42372881355932,
        "p95_tokens_per_sentence": 187,
        "n_sentences": 118,
        "n_tokens": 13502,
        "n_words": 2029,
        "n_chars": 14095
      },
      "en": {
        "tokens_per_word": 1.2636324602833835,
        "tokens_per_char": 0.2073119188503804,
        "tokens_per_sentence": 24.940677966101696,
        "p95_tokens_per_sentence": 48,
        "n_sentences": 118,
        "n_tokens": 2943,
        "n_words": 2329,
        "n_chars": 14196
      }
    },
    "roundtrip_exact": {
      "hi": 1.0,
      "bn": 1.0,
      "ta": 1.0,
      "te": 1.0,
      "mr": 1.0,
      "en": 1.0
    }
  },
  "sarvam-1": {
    "repo": "sarvamai/sarvam-1",
    "vocab_size": 68096,
    "class": "LlamaTokenizer",
    "languages": {
      "hi": {
        "tokens_per_word": 1.376984126984127,
        "tokens_per_char": 0.26885961822920335,
        "tokens_per_sentence": 32.347457627118644,
        "p95_tokens_per_sentence": 58,
        "n_sentences": 118,

... (truncated; see the JSON artifact)
```

## ternary_check

*Which quantizer wrote these weights, and can QAT resume without damaging them?*

Status: PASS — artifact already present (cached)

```json
{
  "checkpoint": "/home/khushiyant/.cache/huggingface/hub/models--KittenML--kitten-tts-2/snapshots/f2eee74f4467855ad58fe16f1fb7b7df473263fd/lm/model-ternary.safetensors",
  "tensors": {
    "model.layers.0.mlp.down_proj.weight": {
      "zero_fraction": 0.16319674253463745,
      "exactly_ternary": true,
      "implied_absmean_scale_equals_stored": true,
      "absmean_is_fixed_point": false,
      "absmean_mean_abs_drift": 0.025981122627854347,
      "absmax_is_fixed_point": true,
      "absmax_mean_abs_drift": 0.0
    },
    "model.layers.0.mlp.gate_proj.weight": {
      "zero_fraction": 0.1554548144340515,
      "exactly_ternary": true,
      "implied_absmean_scale_equals_stored": true,
      "absmean_is_fixed_point": false,
      "absmean_mean_abs_drift": 0.04262867569923401,
      "absmax_is_fixed_point": true,
      "absmax_mean_abs_drift": 0.0
    },
    "model.layers.0.mlp.up_proj.weight": {
      "zero_fraction": 0.15508180856704712,
      "exactly_ternary": true,
      "implied_absmean_scale_equals_stored": true,
      "absmean_is_fixed_point": false,
      "absmean_mean_abs_drift": 0.04360410198569298,
      "absmax_is_fixed_point": true,
      "absmax_mean_abs_drift": 0.0
    },
    "model.layers.0.self_attn.k_proj.weight": {
      "zero_fraction": 0.12381839752197266,
      "exactly_ternary": true,
      "implied_absmean_scale_equals_stored": true,
      "absmean_is_fixed_point": false,
      "absmean_mean_abs_drift": 0.03801047056913376,
      "absmax_is_fixed_point": true,
      "absmax_mean_abs_drift": 0.0
    },
    "model.layers.0.self_attn.o_proj.weight": {
      "zero_fraction": 0.13218069076538086,
      "exactly_ternary": true,
      "implied_absmean_scale_equals_stored": true,
      "absmean_is_fixed_point": false,
      "absmean_mean_abs_drift": 0.031213369220495224,
      "absmax_is_fixed_point": true,
      "absmax_mean_abs_drift": 0.0
    },
    "model.layers.0.self_attn.q_proj.weight": {
      "zero_fraction": 0.14191699028015137,
      "exactly_ternary": true,
      "implied_absmean_scale_equals_stored": true,
      "absmean_is_fixed_point": false,
      "absmean_mean_abs_drift": 0.0385209359228611,
      "absmax_is_fixed_point": true,
      "absmax_mean_abs_drift": 0.0
    },
    "model.layers.0.self_attn.v_proj.weight": {
      "zero_fraction": 0.11294794082641602,
      "exactly_ternary": true,
      "implied_absmean_scale_equals_stored": true,
      "absmean_is_fixed_point": false,
      "absmean_mean_abs_drift": 0.0328759
... (truncated; see the JSON artifact)
```

## vocab_surgery

*Does the Sarvam vocabulary swap produce a loadable checkpoint?*

Status: PASS — artifact already present (cached)

```json
{
  "embedding_std": 0.04807351902127266,
  "n_rows": 74668,
  "init": "mean",
  "mapped": 67834,
  "unmapped": 262,
  "mean_old_tokens_per_piece": 5.364802311525194,
  "n_rows_padded": 74752
}
```

## sarvam_smoke

*Is the rebuilt checkpoint actually correct -- audio rows aligned, tokenizer addressing only text rows?*

Status: PASS — artifact already present (cached)

```json
{
  "checkpoint": "/home/khushiyant/Developer/kindic/indic/data/kitten-tts-2-sarvam",
  "passed": 12,
  "total": 12,
  "results": [
    {
      "check": "config token_map parses",
      "ok": true,
      "detail": "audio_id_base 68,102"
    },
    {
      "check": "vocab_size matches embedding rows",
      "ok": true,
      "detail": "74,752 vs 74,752"
    },
    {
      "check": "audio range inside table",
      "ok": true,
      "detail": "audio ends at 74,663, table has 74,752"
    },
    {
      "check": "lm_head is tied (no separate head stored)",
      "ok": true,
      "detail": ""
    },
    {
      "check": "stale packed-weight keys dropped",
      "ok": true,
      "detail": ""
    },
    {
      "check": "audio rows bit-identical to source",
      "ok": true,
      "detail": "6,561 rows"
    },
    {
      "check": "audio alignment is unique (off-by-one would differ)",
      "ok": true,
      "detail": ""
    },
    {
      "check": "special and marker rows copied exactly",
      "ok": true,
      "detail": "11/11 rows match"
    },
    {
      "check": "text-row norms in the trained regime",
      "ok": true,
      "detail": "median norm 1.288 vs source 2.158 (ratio 0.60)"
    },
    {
      "check": "no NaN or Inf in the table",
      "ok": true,
      "detail": ""
    },
    {
      "check": "shipped tokenizer round-trips mixed script",
      "ok": true,
      "detail": ""
    },
    {
      "check": "tokenizer emits only text rows",
      "ok": true,
      "detail": "max id 67,546 < first special 68,096"
    }
  ]
}
```

## codec_roundtrip

*Can the S3 codec represent Tamil and Telugu at all?*

Status: PASS — artifact already present (cached)

```json
{
  "config": {
    "split": "dev",
    "clips": 16,
    "asr": "none",
    "asr_note": "disabled by --asr none",
    "device": "cuda",
    "asr_backends": {},
    "cer_excess": 0.05,
    "mcd_ratio": 1.3
  },
  "summary": {
    "hi": {
      "mcd_db": 20.10516083240509,
      "log_mel_l1": 0.6298519689589739,
      "duration_ratio": 1.0003540097712946,
      "cer_original": null,
      "cer_reconstruction": null,
      "cer_delta": null,
      "frame_shift": 0.0,
      "n_clips": 16,
      "n_failed": 0
    },
    "bn": {
      "mcd_db": 17.133074462413788,
      "log_mel_l1": 0.481284873560071,
      "duration_ratio": 1.0004472767222319,
      "cer_original": null,
      "cer_reconstruction": null,
      "cer_delta": null,
      "frame_shift": 0.0,
      "n_clips": 16,
      "n_failed": 0
    },
    "ta": {
      "mcd_db": 19.050249457359314,
      "log_mel_l1": 0.596911858767271,
      "duration_ratio": 1.0002749029599824,
      "cer_original": null,
      "cer_reconstruction": null,
      "cer_delta": null,
      "frame_shift": -0.0625,
      "n_clips": 16,
      "n_failed": 0
    },
    "te": {
      "mcd_db": 17.98294299840927,
      "log_mel_l1": 0.5230094622820616,
      "duration_ratio": 1.0004971697583427,
      "cer_original": null,
      "cer_reconstruction": null,
      "cer_delta": null,
      "frame_shift": -0.125,
      "n_clips": 16,
      "n_failed": 0
    },
    "mr": {
      "mcd_db": 18.70557552576065,
      "log_mel_l1": 0.6017210576683283,
      "duration_ratio": 1.00013024013024,
      "cer_original": null,
      "cer_reconstruction": null,
      "cer_delta": null,
      "frame_shift": -0.125,
      "n_clips": 16,
      "n_failed": 0
    },
    "en": {
      "mcd_db": 24.631019830703735,
      "log_mel_l1": 0.8500829115509987,
      "duration_ratio": 1.000334406159902,
      "cer_original": null,
      "cer_reconstruction": null,
      "cer_delta": null,
      "frame_shift": -0.3125,
      "n_clips": 16,
      "n_failed": 0
    }
  },
  "clips": {
    "hi": [
      {
        "id": "1510",
        "n_tokens": 420,
        "original_s": 16.8,
        "reconstruction_s": 16.92,
        "duration_ratio": 1.0,
        "token_seconds": 16.8,
        "wav": {
          "original": "/home/khushiyant/Developer/kindic/.cache/codec_roundtrip/hi/1510.orig.wav",
          "reconstruction": "/home/khushiyant/Developer/kindic/.cache/codec_roundtrip/hi/1510.recon.wav"
        },
        "log_mel_l1": 0.8315528631210327,
        "mcd_db": 22.2516269
... (truncated; see the JSON artifact)
```

## teacher_forced_loss

*Does our training sequence format match what the checkpoint was trained on?*

Status: FAIL

```json
{
  "config": {
    "langs": [
      "en",
      "hi",
      "ta"
    ],
    "examples": 8,
    "split": "dev",
    "device": "cpu",
    "dtype": "torch.bfloat16",
    "normalize": true,
    "margin": 0.2,
    "pool": 48,
    "max_seconds": 14.0,
    "min_reference_cosine": 0.5,
    "seed": 0
  },
  "controls": {
    "correct": {
      "loss": 4.326864570379257,
      "n": 8,
      "per_example": [
        3.97452974319458,
        4.862091541290283,
        4.440759658813477,
        4.543141841888428,
        3.530759572982788,
        4.6799635887146,
        3.8229832649230957,
        4.760687351226807
      ],
      "mean_length": 526.875
    },
    "shuffled_audio": {
      "loss": 8.85469925403595,
      "n": 8,
      "per_example": [
        9.016926765441895,
        9.101408004760742,
        8.938526153564453,
        8.924358367919922,
        8.54526138305664,
        8.703372955322266,
        8.720121383666992,
        8.887619018554688
      ],
      "mean_length": 526.875
    },
    "other_audio": {
      "loss": 5.547959685325623,
      "n": 8,
      "per_example": [
        6.410423755645752,
        5.758687973022461,
        5.689727306365967,
        4.869379043579102,
        5.247623443603516,
        4.928806304931641,
        5.947427749633789,
        5.531601905822754
      ],
      "mean_length": 526.875
    },
    "role_swapped": {
      "loss": 4.301310062408447,
      "n": 8,
      "per_example": [
        4.019262313842773,
        4.854508399963379,
        4.405793190002441,
        4.553348064422607,
        3.5194344520568848,
        4.4888386726379395,
        3.8286566734313965,
        4.740638732910156
      ],
      "mean_length": 525.875
    },
    "speaker_zeros_no_reference": {
      "loss": 4.599450200796127,
      "n": 8,
      "per_example": [
        4.344799041748047,
        5.0659098625183105,
        4.661413669586182,
        4.96528434753418,
        3.8569176197052,
        4.718021869659424,
        4.121724605560303,
        5.061530590057373
      ],
      "mean_length": 239.625
    },
    "speaker_noise": {
      "loss": 4.391365170478821,
      "n": 8,
      "per_example": [
        4.127115726470947,
        4.99133825302124,
        4.510808944702148,
        4.660894870758057,
        3.69781756401062,
        4.378410816192627,
        3.8686821460723877,
        4.895853042602539
      ],
      "mean_length": 526.875
    },
    "speaker_zeros": {
      "loss": 4.363468259572983,
      "n":
... (truncated; see the JSON artifact)
```

## train_selfcheck

*Does the training loop provably work -- overfit one batch, exact loss mask, gradients where they belong?*

Status: PASS

```json
{
  "device": "cpu",
  "full": false,
  "checkpoint": "/home/khushiyant/.cache/huggingface/hub/models--KittenML--kitten-tts-2/snapshots/f2eee74f4467855ad58fe16f1fb7b7df473263fd/lm",
  "results": [
    {
      "check": "overfit_one_batch",
      "passed": true,
      "skipped": false,
      "note": "both the dense and the ternary-STE run memorise the batch; a zero-LR control does not move at all",
      "dense_first_loss": 5.428891181945801,
      "dense_final_loss": 9.948809019988403e-05,
      "dense_steps": 250,
      "dense_threshold": 0.05,
      "dense_loss_ratio": 1.8325674040168534e-05,
      "dense_passed": true,
      "qat_absmax_first_loss": 5.493366241455078,
      "qat_absmax_final_loss": 0.00010235392983304337,
      "qat_absmax_steps": 250,
      "qat_absmax_threshold": 0.05,
      "qat_absmax_loss_ratio": 1.8632278521799767e-05,
      "qat_absmax_passed": true,
      "zero_lr_control_delta": 0.0,
      "zero_lr_control_flat": true,
      "seconds": 4.02
    },
    {
      "check": "loss_masking",
      "passed": true,
      "skipped": false,
      "note": "mask located two independent ways; padding bit-invariant; supervised positions provably do move the loss",
      "row_lengths": [
        35,
        61
      ],
      "pad_positions": 26,
      "mask_matches_token_scan": true,
      "rows_checked": 2,
      "baseline_loss": 5.324972152709961,
      "pad_scrambled_loss": 5.324972152709961,
      "pad_scrambled_delta": 0.0,
      "supervised_token_control_delta": 0.015634536743164062,
      "different_target_loss": 5.468105316162109,
      "pad_invariant": true,
      "supervised_control_moves_loss": true,
      "target_changes_loss": true,
      "seconds": 0.01
    },
    {
      "check": "gradient_flow",
      "passed": true,
      "skipped": false,
      "note": "measured after one optimizer step so LoRA's zero-initialised B matrix does not read as a dead parameter",
      "embed": {
        "trainable_tensors": 5,
        "frozen_tensors": 23,
        "trainable_without_grad": [],
        "frozen_with_grad": [],
        "spk_proj_grad_abs_sum": 22.461849212646484,
        "passed": true
      },
      "lora": {
        "trainable_tensors": 42,
        "frozen_tensors": 14,
        "trainable_without_grad": [],
        "frozen_with_grad": [],
        "spk_proj_grad_abs_sum": 22.640705108642578,
        "passed": true
      },
      "full": {
        "trainable_tensors": 28,
        "frozen_tensors": 0,
        "trainable_without_grad": [
... (truncated; see the JSON artifact)
```

## fit_config_sarvam

*What is the largest training configuration that fits this GPU, Sarvam vocab?*

Status: PASS

```json
{
  "device": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
  "total_gb": 11.49639892578125,
  "budget_gb": 9.99639892578125,
  "results": [
    {
      "peak_gb": 4.063892364501953,
      "reserved_gb": 4.23828125,
      "trainable": 154146816,
      "oom": false,
      "mode": "embed",
      "batch": 1,
      "seq": 512,
      "vocab": 74752,
      "tokens_per_step": 512
    },
    {
      "peak_gb": 4.621523857116699,
      "reserved_gb": 4.96484375,
      "trainable": 154146816,
      "oom": false,
      "mode": "embed",
      "batch": 2,
      "seq": 512,
      "vocab": 74752,
      "tokens_per_step": 1024
    },
    {
      "peak_gb": 5.736786842346191,
      "reserved_gb": 6.142578125,
      "trainable": 154146816,
      "oom": false,
      "mode": "embed",
      "batch": 4,
      "seq": 512,
      "vocab": 74752,
      "tokens_per_step": 2048
    },
    {
      "peak_gb": 7.967312812805176,
      "reserved_gb": 9.048828125,
      "trainable": 154146816,
      "oom": false,
      "mode": "embed",
      "batch": 8,
      "seq": 512,
      "vocab": 74752,
      "tokens_per_step": 4096
    },
    {
      "peak_gb": null,
      "reserved_gb": null,
      "trainable": null,
      "oom": true,
      "mode": "embed",
      "batch": 16,
      "seq": 512,
      "vocab": 74752,
      "tokens_per_step": 8192
    },
    {
      "peak_gb": 4.259731292724609,
      "reserved_gb": 4.5859375,
      "trainable": 189011968,
      "oom": false,
      "mode": "lora",
      "batch": 1,
      "seq": 512,
      "vocab": 74752,
      "tokens_per_step": 512
    },
    {
      "peak_gb": 4.8173627853393555,
      "reserved_gb": 5.3203125,
      "trainable": 189011968,
      "oom": false,
      "mode": "lora",
      "batch": 2,
      "seq": 512,
      "vocab": 74752,
      "tokens_per_step": 1024
    },
    {
      "peak_gb": 5.932625770568848,
      "reserved_gb": 6.537109375,
      "trainable": 189011968,
      "oom": false,
      "mode": "lora",
      "batch": 4,
      "seq": 512,
      "vocab": 74752,
      "tokens_per_step": 2048
    },
    {
      "peak_gb": 8.163151741027832,
      "reserved_gb": 9.515625,
      "trainable": 189011968,
      "oom": false,
      "mode": "lora",
      "batch": 8,
      "seq": 512,
      "vocab": 74752,
      "tokens_per_step": 4096
    },
    {
      "peak_gb": null,
      "reserved_gb": null,
      "trainable": null,
      "oom": true,
      "mode": "lora",
      "batch": 16,
      "seq": 512,
      "vocab": 74752,
      "tokens_per_step": 8
... (truncated; see the JSON artifact)
```

## fit_config_qwen3

*The same, for the unchanged Qwen3 vocab -- how much does the swap buy?*

Status: PASS

```json
{
  "device": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
  "total_gb": 11.49639892578125,
  "budget_gb": 9.99639892578125,
  "results": [
    {
      "peak_gb": 6.140074253082275,
      "reserved_gb": 7.083984375,
      "trainable": 325132288,
      "oom": false,
      "mode": "embed",
      "batch": 1,
      "seq": 768,
      "vocab": 158241,
      "tokens_per_step": 768
    },
    {
      "peak_gb": 7.814906120300293,
      "reserved_gb": 9.08984375,
      "trainable": 325132288,
      "oom": false,
      "mode": "embed",
      "batch": 2,
      "seq": 768,
      "vocab": 158241,
      "tokens_per_step": 1536
    },
    {
      "peak_gb": null,
      "reserved_gb": null,
      "trainable": null,
      "oom": true,
      "mode": "embed",
      "batch": 4,
      "seq": 768,
      "vocab": 158241,
      "tokens_per_step": 3072
    },
    {
      "peak_gb": null,
      "reserved_gb": null,
      "trainable": null,
      "oom": true,
      "mode": "embed",
      "batch": 8,
      "seq": 768,
      "vocab": 158241,
      "tokens_per_step": 6144
    },
    {
      "peak_gb": null,
      "reserved_gb": null,
      "trainable": null,
      "oom": true,
      "mode": "embed",
      "batch": 16,
      "seq": 768,
      "vocab": 158241,
      "tokens_per_step": 12288
    },
    {
      "peak_gb": 6.337744235992432,
      "reserved_gb": 7.455078125,
      "trainable": 359997440,
      "oom": false,
      "mode": "lora",
      "batch": 1,
      "seq": 768,
      "vocab": 158241,
      "tokens_per_step": 768
    },
    {
      "peak_gb": 8.01074504852295,
      "reserved_gb": 9.486328125,
      "trainable": 359997440,
      "oom": false,
      "mode": "lora",
      "batch": 2,
      "seq": 768,
      "vocab": 158241,
      "tokens_per_step": 1536
    },
    {
      "peak_gb": null,
      "reserved_gb": null,
      "trainable": null,
      "oom": true,
      "mode": "lora",
      "batch": 4,
      "seq": 768,
      "vocab": 158241,
      "tokens_per_step": 3072
    },
    {
      "peak_gb": null,
      "reserved_gb": null,
      "trainable": null,
      "oom": true,
      "mode": "lora",
      "batch": 8,
      "seq": 768,
      "vocab": 158241,
      "tokens_per_step": 6144
    },
    {
      "peak_gb": null,
      "reserved_gb": null,
      "trainable": null,
      "oom": true,
      "mode": "lora",
      "batch": 16,
      "seq": 768,
      "vocab": 158241,
      "tokens_per_step": 12288
    },
    {
      "peak_gb": null,
      "reserved_gb": null,
      "trainable": 
... (truncated; see the JSON artifact)
```

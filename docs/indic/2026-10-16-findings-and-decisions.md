# KittenTTS-indic: findings, decisions, and what to do next

**Session of 2026-10-16.** Everything below was measured on this machine unless
marked otherwise. Numbers carried over from the `indic-research` branch notes are
labelled as such, and several of them turned out to be wrong.

---

## 1. State of play, in one paragraph

The tokenizer question is **settled with measurements**, the vocabulary swap is
**built and verified**, the training loop is **proven on synthetic data**, and
**both go/no-go gates have been run**. There is still no training data on this
machine and no trained model — what exists is the thing that was actually
missing: a harness that tells you, before you spend an allocation, whether the
plan is sound.

**Preflight: 9 PASS, 1 FAIL.**

| Gate | Verdict |
|---|---|
| Can the S3 codec carry Indic at all? | **PASS** — every Indic language reconstructs *better* than the English control |
| Does our sequence format match the checkpoint's? | **PARTIAL** — target segment CONFIRMED (+4.7 nats); reference layout NOT confirmed |
| Does the training loop provably work? | **PASS** — 8/8, including overfit-one-batch to 1e-4 and an exactly-zero-delta loss mask |

The one FAIL is honest and narrow: it blocks *claiming* the reference layout is
right. It does not block training, because the half that would have wasted the
compute — the target segment, its markers, the token offset, the loss mask and
the label shift — is confirmed.

**The headline risk the project was carrying is gone.** The fear was that the S3
codec, trained without a single Indic language, could not represent Tamil or
Telugu, which would have capped the whole effort before the LM was touched. It
represents them *better* than English.

---

## 2. What the model actually is

Measured from the released checkpoint, not from documentation:

| | |
|---|---|
| Architecture | `Qwen3ForCausalLM`, 28 layers, d_model 2048, FFN 6144, 16 q-heads / 8 kv-heads |
| Text tokenizer | **Qwen3 BPE** (`Qwen2Tokenizer`), vocab 158,241 |
| Shared vocabulary | text `[0, 151675)` · 6 specials · **6561 S3 audio codec tokens @ 151,675** · 5 marker rows |
| Ternary params | **1,409,286,144** across 196 linear weights, per-128-group scale |
| Full-precision params | **325,256,192**, of which `embed_tokens` is **324.08M — 99.6%** |
| Audio | S3 codec at 25 Hz, s3gen vocoder, 24 kHz output |

The single most consequential structural fact: **`embed_tokens` is tied to
`lm_head` and is 99.6% of all full-precision parameters.** It is simultaneously
the thing a tokenizer swap replaces and the dominant term in the training memory
budget. Every decision below follows from that.

---

## 3. Findings

### 3.1 Tokenizer fertility — Sarvam-1 is 3.5–5.4× cheaper on Indic

Measured on 118 content-matched FLEURS sentences (the same FLORES sentences in
every language, so the comparison is attributable to the tokenizer rather than to
the corpus). NFC-normalized first.

| tokens/word | hi | bn | ta | te | mr | en |
|---|---|---|---|---|---|---|
| Qwen3 (current) | 4.76 | 7.14 | 10.13 | 11.42 | 6.65 | 1.26 |
| Sarvam-1 | 1.38 | 1.81 | 2.17 | 2.11 | 1.77 | 1.44 |
| ratio | 3.46× | 3.96× | 4.67× | **5.42×** | 3.76× | 0.88× |

The clearest way to read it: **Sarvam costs ~32 tokens per sentence in every
Indic language — essentially the same as its 28 for English. Qwen3 charges
112–177.** Telugu under Qwen3 costs 1.46 tokens *per character*.

The one genuine cost is English getting ~14% more expensive, which is why English
recordings must stay in every training phase.

> **Corrections to the research-branch notes.** Their per-language figures came
> from a single sentence each and are off by up to 2× (Hindi stated 6.0, measured
> 4.76; Bengali stated 5.1, measured 7.14). Their ordering of which language is
> worst also changes.

> **A non-finding, recorded so nobody rediscovers it.** An early run showed Qwen3
> losing 48% of Bengali on a round trip. That was an artifact: FLEURS ships mixed
> Unicode normalization forms and both tokenizers NFC-normalize internally. After
> NFC both are 100% lossless on all six languages. Byte-level BPE cannot lose
> characters; a result saying otherwise is a bug in the harness.

### 3.2 The ternary quantizer — the training rule is *not* recoverable

This gates QAT, because a fine-tune must end ternary again, and QAT's forward
quantizer must reproduce the released weights at step 0 or training begins from a
damaged model.

- Measured zero fraction across all 196 ternary tensors: **min 9.1%, mean 14.9%,
  max 17.7%.**
- Measured yardsticks on Gaussian weights at group 128: absmean (BitNet b1.58)
  gives **30.9%** zeros, absmax gives **83.5%**.
- **15% matches neither.** absmax is excluded outright (5–6× too sparse).
  absmean is *not confirmed* either — the checkpoint is about half as sparse as
  absmean predicts.

So **the original training rule cannot be identified from the checkpoint alone,
and nothing downstream should assume it was absmean.** A plausible benign
explanation is that converged straight-through training drives latent weights off
the threshold, leaving a bimodal rather than Gaussian shadow distribution — but
that is consistent with absmean, not evidence for it.

Group width is confirmed **exactly 128** (tensors are exactly ternary at group
32/64/128 and not at 256+, and scales vary group-to-group within every row).

### 3.3 QAT can resume with zero damage — two ways

- `Q(W) = clamp(round(W/absmax_group), -1, 1) · absmax_group` is **bit-exact** on
  the released weights (drift 0.0). QAT can resume with an absmax forward and
  step-0 loss identical to the released model's.
- Applying an **absmean** forward directly instead causes a pure per-group
  multiplicative shrink of every nonzero by that group's nonzero fraction
  (~0.83–0.86): mean absolute drift 0.024–0.047, **zero sign flips, zero new
  zeros**. That is the dangerous kind of wrong — not a corrupted model, a quietly
  detuned one across all 196 layers at once, which looks like a merely
  disappointing fine-tune.
- **absmean can also resume losslessly** if the shadow weight is initialised as
  `W / f` per group (`f` = that group's nonzero fraction). Verified bit-exact to
  fp32 round-off; observed `f` range 0.61–0.98, and the `clamp(-1,1)` covers it.

### 3.4 What fits on a 12 GB card — measured, not estimated

RTX 5070 Ti Laptop, 11.50 GB usable to torch. bf16 + gradient checkpointing +
bitsandbytes `AdamW8bit` (which **works on sm_120**). Budgeted on
`max_memory_reserved`, because that is what OOM fires on.

| Phase | Config | Reserved | Genuinely free |
|---|---|---|---|
| **A — embedding-only warm-up** | Sarvam vocab, batch 8 @ seq 512 | 9.05 GB | **2.45 GB** |
| **B — LoRA r=32 joint** | Sarvam vocab, batch 8 @ seq 512 | 9.52 GB | **1.98 GB** |
| Full fine-tune | any batch, either vocab | — | **does not fit** |

Peak memory is a **pure function of tokens/step** — how you split that between
batch and sequence length is irrelevant to the second decimal. Fitted on 12
points across seq 256/512/1024:

```
reserved_GB = 3.897 + 0.001357 x tokens_per_step     (max residual 0.162 GB)
```

Ceiling at this budget: **4,495 tokens/step.** So if audio targets lengthen to
seq 1024, halve the batch to 4 — no re-probing needed.

**What the vocabulary shrink actually buys** (matched seq 512, reserved basis):

| tokens/step | Sarvam | Qwen3 | saved |
|---|---|---|---|
| 512 | 4.59 | 6.27 | 1.68 GB |
| 2048 | 6.54 | 10.05 | 3.52 GB |
| 4096 | 9.52 | OOM | — |

At equal memory, Sarvam runs **4,096 tokens/step against Qwen3's 1,536**. Since
Qwen3 also needs ~1.5× more tokens for the same text, effective text throughput
is roughly **4×**. The saving decomposes into a fixed ~1.0 GB (83,489 fewer
embedding rows × weight + grad + 8-bit state) plus a per-token term from the
cross-entropy over the vocabulary — and it is the per-token term that decides
outcomes at large steps.

### 3.5 The vocabulary swap works

Built and verified, three arms, **12/12 checks each**:

```
[0, 68096)          Sarvam-1 pieces
68096 - 68101       speech_start, speech_end, text_start, text_end, START, STOP
68102 - 74662       6561 S3 audio codec tokens   <- audio_id_base
74663 - 74667       final_seg, reference_* markers
                    74,668 rows, padded to 74,752
```

158,241 → 74,752 rows removes **171M parameters** (342 MB at bf16) from the tied
table. Audio and marker rows are **bit-identical** to the source, and the
alignment is proven *unique* — a one-row shift does not also match, which is the
specific silent failure (a model that loads, runs, and emits noise) the check
exists to catch.

Each Sarvam piece covers **5.36 Qwen tokens on average**, so mean-of-subtokens
initialisation is a real signal rather than noise.

---

## 4. Decisions taken, and why

**D1 — Use the Sarvam tokenizer, and keep a Qwen3 control.**
A 3.5–5.4× fertility reduction, with Indic brought to parity with English, is too
large to leave on the table. But no speech-LM TTS model has swapped its text
tokenizer before, and no study links text fertility to TTS quality, so the Qwen3
fine-tune remains a control arm. The experiment is only interpretable with it.

**D2 — Added a third init arm, `mean_rescaled`, and made it the default.**
Mean-of-subtokens init produced rows at **0.60×** the norm of the trained table —
averaging ~5.4 roughly-independent vectors shrinks norm by about √k. Since
`embed_tokens` feeds layer 0 *and* is tied to `lm_head`, uniformly short rows mean
a uniformly weak text signal and systematically smaller logits. One global rescale
factor (**1.675**) restores the trained norm regime while preserving per-row norm
differences, which carry real information about how specific each piece is.
Per-row normalisation would erase that, so it was rejected.

**D3 — Budget GPU memory on `reserved`, never `allocated`.**
OOM fires on reserved, which runs 1.1–1.4 GB higher and whose gap *grows* with
step size. On the allocated basis a full fine-tune appeared to fit with 1.44 GB
free; on the reserved basis it is over budget at every batch size. This single
change is the difference between a plan that survives the night and one that dies
at 3am.

**D4 — Fix the danda bugs in `kittenml/kittentts2/text.py` before measuring
anything.** `।`/`॥` now terminate sentences, count as existing terminators, and
NFC + Indic-digit normalization runs ahead of the English normalizer. Without
this an Indic CER measures the chunker's hard 380-character mid-word cut, not the
model. Latin behaviour verified unchanged.

**D5 — Prove the training loop on synthetic data before acquiring any.**
A training loop is proven by making it overfit a single batch to near-zero loss,
by checking the loss mask is exact, and by checking gradients reach what they
should and nothing else. That costs minutes and no data. Real data would only
hide these bugs behind a plausible-looking loss curve.

**D6 — For QAT, resume with absmax (zero-damage), and treat absmean as the
long-horizon option via the `W/f` shadow rescale.** absmax is bit-exact today but
is a poor long-horizon quantizer: once the shadow leaves the ternary lattice, one
outlier in a 128-group inflates the scale and can zero the group. Fine for a
short fine-tune, risky for a long one.

**D7 — Measure on FLEURS, not FLORES.** Both FLORES repos are gated; FLEURS is
open, carries the *same* FLORES sentences, and ships the recordings the evaluation
harness needs anyway. One download serves both purposes.

---

## 3.6 Gate 1 — the codec carries Indic fine. **PASS.**

16 FLEURS clips per language, real recordings → S3 tokens → s3gen → waveform,
scored against the original. English rides along as the control because it is in
the codec's training distribution.

| | clips | MCD | MCD / en | log-mel | duration |
|---|---|---|---|---|---|
| hi | 16 | 20.11 | 0.82× | 0.630 | 1.00 |
| bn | 16 | 17.13 | **0.70×** | 0.481 | 1.00 |
| ta | 16 | 19.05 | 0.77× | 0.597 | 1.00 |
| te | 16 | 17.98 | 0.73× | 0.523 | 1.00 |
| mr | 16 | 18.71 | 0.76× | 0.602 | 1.00 |
| **en (control)** | 16 | 24.63 | 1.00× | 0.850 | 1.00 |

**Every Indic language reconstructs *better* than English**, and duration is
preserved exactly. The worry that CosyVoice's S3 tokenizer — trained without a
single Indic language — would not represent Tamil or Telugu is **not borne out**.
Keep the codec frozen and train the LM.

Caveat stated plainly: this run is spectral-only, because IndicConformer is gated
and this machine has no `HF_TOKEN`. MCD and duration catch a codec that mangles
phonetics; they cannot catch a reconstruction that is clean but says a different
word. Re-run with `--asr auto` once the token is in place. The 160 reconstruction
pairs are on disk under `.cache/codec_roundtrip/` — **listen to them** before
acting on any future FAIL.

## 3.7 Gate 2 — the target format is right, the reference layout is not proven

16 examples per language, reference speaker cosine 0.70 (so the references really
are the same speaker and there was something real to corrupt).

| variant | loss (nats) | ppl | gap |
|---|---|---|---|
| **correct format** | **4.110** | 60.9 | — |
| shuffled_audio | 8.796 | 6606.8 | **+4.686** |
| other_audio | 5.317 | 203.8 | **+1.207** |
| role_swapped | 4.094 | 60.0 | −0.016 |
| speaker_zeros_no_reference | 4.389 | 80.6 | +0.217 |
| no_reference | 4.172 | 64.8 | +0.062 |
| uniform over 6561 | 8.789 | 6561 | +4.679 |

**Target segment: CONFIRMED.** Scrambling what is said costs 4.7 nats — the
markers, the audio-token offset, the loss mask and the label shift are the ones
this checkpoint continues. *This is the half that would have wasted the compute if
wrong.*

**Reference and speaker: NOT CONFIRMED.** `role_swapped` is indistinguishable
from correct (−0.016 nats), and the entire reference prefix is worth only 0.062
nats. Two readings remain and this test cannot separate them: either our
reference layout is not the trained one, or the checkpoint is near-indifferent to
identity conditioning *under teacher forcing* — which is plausible, since each
audio token is largely predictable from its predecessors. Note too that
`role_swapped` is not a pure corruption: its layout is what a *second segment* of
a multi-segment sample looks like, and `final_seg` existing at all implies the
checkpoint saw those.

**What this blocks and what it does not.** It does not block training the target
segment, which is on solid ground. It blocks *claiming* the reference layout is
correct. The training plan already calls for reference dropout, so train with it
and settle the layout by generating from both and scoring speaker similarity —
teacher-forced loss cannot see this.

### The Indic baseline this establishes

| lang | loss | ppl | vs en | ref cos |
|---|---|---|---|---|
| en | 4.110 | 60.9 | — | 0.70 |
| hi | 4.224 | 68.3 | +0.114 | 0.74 |
| **ta** | **5.660** | **287.2** | **+1.550** | 0.74 |

Hindi is nearly at parity with English. **Tamil is 4.7× worse in perplexity** —
and Tamil is also where Qwen3 fertility is worst (10.13 tokens/word). That is the
number a fine-tune has to improve, and the first place to look for whether the
tokenizer swap did anything.

---

## 3.8 The training loop is proven — on synthetic data. **8/8 PASS.**

| check | result |
|---|---|
| **overfit one batch** | dense 5.4289 → **9.95e-5** in 250 steps; through the ternary STE 5.4934 → **1.02e-4**. Zero-LR control moves by **exactly 0**, so the comparison is not vacuous |
| **loss masking** | mask derived two independent ways agrees; ragged batch [35, 61], 26 pads scrambled → loss delta **exactly 0**; scrambling a supervised token moves it 0.0156 |
| **gradient flow** | embed 5 trainable / 23 frozen, lora 42/14, full 28/0 — zero trainable-without-grad, zero frozen-with-grad in all three |
| **QAT fixed point** | on real cached weights: absmax **bit-exact, drift 0**; absmean on the `W/f` shadow drifts 5.96e-8. **All 4 negative controls rejected** |
| **pack round trip** | bit-exact through `ternary.py` *and* through the real `kittenml...weights.load_lm_weights` |
| **determinism** | two 6-step runs byte-identical; a different seed differs |
| **checkpoint resume** | 8 steps straight vs 4 + save + fresh model at a different seed + load + 4 — identical step for step |
| **speaker conditioning** | position 0 equals `spk_proj`'s output with max abs difference **0** |

Re-run against the **real 28-layer checkpoint**: loss masking still pad-invariant
to exactly 0 in bf16, and position 0 matches `spk_proj` with max abs diff 0 while
sitting 3.26 away from `[START]`'s own embedding.

## 3.9 Four more errors in the research-branch training plan

1. **The shadow weight must be float32, not bf16.** `training.md` says bf16.
   Measured: a bf16 shadow misses the released weights by up to **1.07e-3** even
   with the correct rescale. It is now one of the suite's negative controls.
2. **The doc's QAT sketch is not a fixed point as written.** `s = Wg.abs().mean()`
   shrinks every surviving weight by its group's nonzero fraction. The `W/f`
   pre-division cancels it, and **`clamp(-1, 1)` is load-bearing, not
   defensive**: at f = 0.609, `round(1/f)` returns 2 and the unclamped error is
   0.277.
3. **`Wg = W.view(-1, group)` is only accidentally correct** here, because the
   row length happens to be a multiple of 128. Grouping along the wrong axis
   costs 0.236 drift — now an explicit negative control.
4. **"Replaces hidden state of START" is imprecise** — it replaces the *input
   embedding* at position 0. This matters: the `[START]` id is never embedded at
   all, so a missing injection changes the input sequence rather than merely
   weakening conditioning.

**Right padding is mandatory.** Left padding would put the speaker vector on a
pad while `[START]` sat at a per-row offset — silently.

### One check deliberately *not* written

The brief asked for "randomising tokens at non-target positions leaves the loss
unchanged". That is **false for any causal LM** — a non-target position before the
target span is context. Writing it would have produced either a failing check or,
worse, a check quietly weakened until it passed. The honest version — exact pad
invariance in a ragged batch, an independent derivation of the mask from the
token stream, and a supervised-token control — is what was implemented.

---

## 4b. Evaluation-methodology decisions

Four of these change what the numbers mean, not merely how they are computed.

**D8 — The vendored speaker embedder must not be the headline similarity metric.**
`kittenml/kittentts2/speaker_embedding.py` is the pyannote x-vector *whose 512-d
output is fed to the checkpoint's own `spk_proj` conditioning head*. Scoring
speaker similarity with it is partly circular: it asks "did the conditioning
vector survive generation", not "does this sound like the speaker" — and an Indic
fine-tune is precisely where a model could learn to satisfy that embedding
without matching the voice. The default is therefore `microsoft/wavlm-base-plus-sv`
(ungated, CPU, what the SECS literature uses); the vendored one is kept as a
documented diagnostic.

**D9 — The ASR floor is structural, not a reporting option.** Every Set A record
carries the sentence's own human recording, transcribed through the *same*
recognizer, decoder and normalizer, so `floor_cer` and `cer_above_floor` appear on
every row. A recognizer's own error rate on Telugu is several times its rate on
Hindi; a CER without its floor is not comparable across languages. When only part
of a set has floors, the headline CER is restricted to the floor subset —
subtracting a floor measured on 3 recordings from a CER measured on 7 is not a
difference of anything.

**D10 — Cloning references never come from the target sentence.** Set A reference
clips are drawn from a pool of FLEURS utterances whose ids are *excluded* from the
scored set. Handing a model the target sentence's own recording as its prompt
measures copying, not cloning.

**D11 — Set C reports a gap, never a level.** Every robustness item is emitted
twice — ASCII digits and native digits, scored against the same ASCII reference —
so the digit-vs-word inflation is identical on both halves and cancels. The
absolute Set C CER is *not a reportable quantity* until an Indic number-to-words
backend exists: IndicConformer writes "दो हज़ार चौबीस" where the reference holds
"2024".

---

## 4c. Precision on heterogeneous hardware — the expected worry was the wrong one

The assumption going in was that the per-group **bf16 scales** would be the fp16
hazard, since bf16 and fp16 have very different exponent ranges. Measured, that is
false, and the real hazard is elsewhere.

**The scales are safe.** All 11,010,048 per-group scales across the 196 ternary
tensors fall in [2.689e-04, 0.785]. fp16's smallest normal is 6.1e-5 — a 4.4×
margin below the smallest scale — and **zero scales change** under a
bf16→fp16→fp32 round trip. (Generally: bf16 has an 8-bit significand and fp16 has
11, so every bf16 value in [2⁻²⁴, 65504] converts exactly; only overflow and
underflow lose, and these scales sit squarely inside the exact band.)

**The activations are what kill fp16.** One prefill on a 288-token
reference-supervised prompt, hooks on all 424 modules:

```
model.layers.27                 3,194,880   <- 49x the fp16 ceiling (65,504)
model.layers.27.mlp.down_proj   3,162,112
model.layers.26                   643,072
33 of 424 module outputs exceed 65,504
```

Classic Qwen-family massive activations in the residual stream, growing
monotonically across layers. An fp16 forward saturates to `inf` and emits NaN
logits. Loss scaling is a training trick and does not help a forward pass.

**D12 — On pre-Ampere hardware there is no half-precision option at all.**
sm_80+ gets bf16. V100/T4 get **fp32 or nothing**: bf16 runs only by emulation
with the tensor cores idle, and fp16 is numerically impossible. `select_dtype()`
refuses by default rather than silently taking a 5×-slower arm.

A trap worth carrying: **`torch.cuda.is_bf16_supported()` returns True on a
V100**, because it defaults to `including_emulation=True`. Compute capability is
the only honest signal, and `kittenml/kittentts2/model.py` now warns when it is
handed bf16 on a pre-Ampere card — it previously took that path in silence.

---

## 4d. The single most dangerous thing we found

**`spk_proj` silently does nothing unless `input_ids` arrives as a keyword
argument.**

`kittenml/kittentts2/speaker.py:_inject_speaker` is a forward pre-hook that reads
`kwargs.get("input_ids")` and returns the arguments untouched when it is absent.
So:

```python
model(ids, labels=...)                  # positional  -> speaker NEVER injected
model(inputs_embeds=embeds, labels=...) # normal in a trainer -> NEVER injected
model(input_ids=ids, labels=...)        # works
```

There is no error, no warning, and no shape mismatch. The loss curve looks
completely normal. A training run written the obvious way would condition on
nothing at all and produce a model that ignores the voice — and the only symptom
would be disappointing speaker similarity at the end of the run, which is exactly
the kind of result people explain away as "needs more data".

Mitigations in place: the training code passes `input_ids=` by keyword with a
comment saying why, and the self-check asserts both that changing the speaker
embedding changes the loss *and* that passing ids positionally reproduces the
zero-speaker loss — turning a silent failure into a red check.

**Related trap, same area:** `text_end_id` (151672) exists in the token map and
`prompt.py` never emits it. Target text is closed by `<final_seg>`, the reference
by `reference_text_end_id`. It looks like an oversight and is not — do not "fix"
it. And `[START]` lives *inside* `build_reference_prefix`, so dropping the
reference is not plain concatenation; both paths must be built through
`prompt.py` rather than re-implemented.

---

## 5. Bugs found

Three were in code written during this session and would each have produced a
**confident wrong answer** rather than a crash.

1. **A fixed-point test that was an algebraic identity.** On already-ternary
   weights, `W/absmax ∈ {-1,0,1}` for *any* tensor however produced, and absmean
   can never be a fixed point when zeros exist. The test "confirmed" absmax was
   the training rule — a conclusion the sparsity measurement flatly contradicts.
   *Lesson applied throughout: a check that cannot fail is worse than no check.*
2. **An fp32 materialization transient masquerading as the training peak.**
   `to_empty(device).to(dtype)` built the whole model in fp32 before casting, so
   the build spike (6.68–8.27 GB) *was* the reported peak at small batch, and was
   causing phantom OOMs. The tell: identical memory at batch 1, 2 and 4 — a
   batch-independent floor, which no real training peak does.
3. **Optimizer outliving the model**, pinning weights so `empty_cache()` released
   nothing and each row's blocks leaked into the next row's high-water mark.
   Harmless on the allocated basis, which is why it only surfaced after D3.
4. **A substring collision**, `"k_proj" in "spk_proj.0.weight"`, which counted the
   dense speaker projection as a ternary linear and hid the second-largest dense
   tensor from the report.
5. **`spk_proj` was not in the forward graph** in the memory probe, so it took no
   gradient and no optimizer state.

Pre-existing in the repository, both now fixed:

6. **The danda handling** described in D4.
7. **`ReferenceTranscriber` hardcoded `language="en"`** (`speaker.py`). Cloning
   from a Hindi or Tamil clip without passing `reference_text` did not merely
   mislabel it — Whisper *translates and transliterates* under a forced language,
   so the prompt carried a romanised English rendering of speech the reference
   contains in Devanagari or Tamil. The transcript and the audio in the reference
   prefix then disagree, which is the one pair in that prompt that must match.
   Now detects by default, with `language=` to force it.

---

## 6. Honest gaps — what is *not* done

- **No training data.** Nothing downloaded, nothing preprocessed. The long pole
  is not the training run; it is ~900 h of audio → resample → S3 tokens → speaker
  embeddings → ASR quality filtering. IndicVoices-R and Rasa are **gated** and
  need accepted HF terms.
- **No trained model.** Not a single gradient step on real data.
- **The sequence format is unvalidated** until the teacher-forced loss gate runs.
  This is the highest-value remaining hour of work.
- **The codec gate is unresolved** until the round trip runs on ta/te.
- **Indic number-to-words is still English.** `_normalize_script` makes digits
  *visible* to the grammar; the expansion is still English words. Needs NeMo for
  Hindi and `indic-num2words` for the rest.
- **Compute estimate is arithmetic, not measurement:** ~400k utterances × ~600
  tokens ≈ 240M tokens/epoch; 6·N·D ≈ 2.4×10¹⁸ FLOPs/epoch ≈ 5.4 A100-hours at
  40% MFU; ×3 epochs ×~1.5 for QAT ≈ **25–40 A100-hours**. Compute is *not* the
  constraint. Data preparation and correctness are.

---

## 7. Licensing, before anything ships

- **`sarvamai/sarvam-1` is non-commercial.** Fine for the experiment, not for a
  product. If the Sarvam arm wins, retrain a SentencePiece tokenizer of the same
  size on our own text — every script takes `--tokenizer`, so the pipeline is
  unchanged.
- **KittenTTS 2 weights** are under the Stellon Labs Community License: fine-tunes
  and derivative releases are permitted, but require attribution, a NOTICE
  recording *what was changed*, "Powered by Stellon Labs" displayed, and the
  license terminates above $1M revenue or $1M cumulative funding. §4b forbids
  using the model or its outputs to train *other* models — so KittenTTS output
  must never enter the training mix.

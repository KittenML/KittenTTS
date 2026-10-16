"""Does our training sequence match the one the checkpoint was actually trained on?

This is the gate that has to pass before any GPU-hour is spent. There is no
training code in this repository and no description of the SFT mix, so the
training format has been *reconstructed* from the inference path: `prompt.py`
assembles a generation prompt, `speaker.py` overwrites position 0's embedding
with the projected speaker vector, and the target audio tokens are whatever the
model would have gone on to emit. If that reconstruction is wrong -- one marker
in the wrong place, the reference wrapped with the other checkpoint's role
tokens, the speaker written at the wrong position -- then fine-tuning optimises
a mis-specified objective. Nothing crashes. The loss curve still goes down. The
compute is simply wasted, and the only symptom is a model that is worse than the
one we started from.

The test is teacher-forced cross-entropy on the target audio tokens alone, run
on **English**, which the checkpoint already speaks. A low number there is
necessary. It is nowhere near sufficient: 6561 audio tokens at 25 Hz are highly
predictable from their own left context, so a model handed a corrupted prompt
still scores respectably by continuing the audio it can see. A low loss proves
the model is doing *something*, not that it is doing what we think.

So the number that matters is the **gap to a set of deliberately wrong formats**:

    shuffled target audio   destroys the acoustics, keeps the token histogram
    other utterance's audio destroys the text-audio correspondence
    swapped role tokens     proves this checkpoint wants the reference-role
                            layout and not the plain text_start/speech_start one
    zeroed speaker, no      proves spk_proj at position 0 is actually read. The
    reference               reference has to go first, or the control measures
                            nothing -- see BASELINE
    reference dropped,      diagnostic, not decisive -- see DIAGNOSTIC_CONTROLS
    marker probes

If the correct format does not beat the decisive controls by a clear margin, the
reconstruction is wrong and this script says so and exits non-zero.

Hindi and Tamil are then measured in the same correct format. That is not part
of the gate -- it is the baseline it exists to make trustworthy: how much worse
Indic is today, under today's tokenizer, before anything is trained.

    python -m indic.bench.teacher_forced_loss --examples 16 --device cuda \\
        --out indic/data/teacher_forced_loss.json
    python -m indic.bench.teacher_forced_loss --self-test   # shapes and masking only

`--self-test` builds a tiny randomly-initialised Qwen3 with the real token map
and checks the sequence assembly, the loss mask and the speaker injection without
touching the 3.5 GB checkpoint, so the mechanics can be validated on any machine.
"""

import argparse
import dataclasses
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.bench.codec_roundtrip import CACHE_DIR, fleurs_clips
from indic.langs import LANGUAGES
from kittenml.kittentts2.prompt import build_generation_prompt, build_reference_prefix
from kittenml.kittentts2.tokens import TokenMap

REPO_ID = "KittenML/kitten-tts-2"
# Only what this gate reads. Without the explicit list a snapshot pulls the
# ternary and emb4 variants and the C++ export as well -- 6 GB for weights that
# are not the bf16 ones the training plan starts from.
REPO_FILES = ["config.json", "lm/*.json", "lm/merges.txt", "lm/model.safetensors",
              "speaker/model.safetensors"]

# How each variant is built. `reference` chooses the prefix layout, `speaker` what
# goes into spk_proj, `audio` what the supervised tokens are, `marker` which
# segment markers the target gets. Data rather than a chain of string comparisons
# because the combinations matter: `speaker_zeros_no_reference` is the only control
# that isolates the speaker head, and it is a pairing of two of these knobs.
VARIANT_SPEC = {
    "correct": {},
    "shuffled_audio": {"audio": "shuffled"},
    "other_audio": {"audio": "other"},
    "role_swapped": {"reference": "plain_role"},
    "speaker_zeros_no_reference": {"reference": "dropped", "speaker": "zeros"},
    "speaker_noise": {"speaker": "noise"},
    "speaker_zeros": {"speaker": "zeros"},
    "no_reference": {"reference": "dropped"},
    "no_final_seg": {"marker": "no_final_seg"},
    "text_end_after_target": {"marker": "text_end"},
}
VARIANTS = tuple(VARIANT_SPEC)

# What each control is read against. Corrupting the speaker embedding while the
# reference prefix is still there measures almost nothing: the prefix carries two
# hundred audio tokens of the same voice, so the model has the speaker in context
# and does not need spk_proj. The honest test of the speaker head is to drop the
# reference first, leaving spk_proj as the only identity signal -- which is exactly
# the configuration `prompt.py` supports and `speaker.py` describes -- and compare
# against `no_reference` rather than against `correct`.
BASELINE = {"speaker_zeros_no_reference": "no_reference"}

# Controls the verdict turns on, split by which half of the format they test --
# because the two halves can be right and wrong independently, and a single
# verdict over both says much less than two verdicts over one each.
#
#   content:  does the target segment -- text, markers, audio tokens, the loss
#             mask -- line up with what the model expects to continue?
#   identity: is the reference wrapped the way this checkpoint was trained to
#             read it, and is the speaker at position 0 reaching the forward pass?
CONTENT_CONTROLS = ("shuffled_audio", "other_audio")
IDENTITY_CONTROLS = ("role_swapped", "speaker_zeros_no_reference")
DECISIVE_CONTROLS = CONTENT_CONTROLS + IDENTITY_CONTROLS
# Reported, never gating.
#   * `speaker_noise` / `speaker_zeros` are the same corruption as the decisive
#     one but with the reference left in, so they are redundant by construction --
#     see BASELINE. They stay in the table because a *large* gap here would be a
#     finding in itself.
#   * `no_reference` is the prompt `prompt.py` itself builds when cloning is off,
#     and the training plan calls for dropping the reference on some fraction of
#     examples; a checkpoint trained that way is supposed to cope without it.
#   * the `final_seg` / `text_end` probes answer an open question in
#     docs/indic/training.md -- nobody knows how those markers were used in
#     training -- so they have no predicted direction to gate on.
DIAGNOSTIC_CONTROLS = ("speaker_noise", "speaker_zeros", "no_reference",
                       "no_final_seg", "text_end_after_target")
REFERENCE_CAVEAT = (
    "no_reference is diagnostic: inference supports dropping the reference and the\n"
    "  training plan calls for doing so on some examples, so a small gap is expected.")

# Loss must beat a control by at least this many nats to count as decisive.
MARGIN = 0.2
# Below this mean cosine, the inferred reference pairs are not the same speaker
# (pyannote embeddings put same-speaker pairs well above it), and every control
# that works by corrupting the speaker or the reference is then uninformative:
# there was nothing useful in the reference to destroy.
MIN_REFERENCE_COSINE = 0.5


# ── sequence assembly ─────────────────────────────────────────────────────────
def assemble(token_map, reference, target, variant="correct", rng=None):
    """One training example, built exactly the way inference builds a prompt.

    Position by position, and where each piece comes from:

        0                 `token_map.start_id`, whose *input embedding* is
                          replaced by spk_proj(speaker embedding) at prefill --
                          speaker.py:_inject_speaker, not a token substitution
        1 .. r            the rest of `build_reference_prefix`: reference-role
                          wrapped transcript, then reference audio tokens
        r+1 .. p          the tail of `build_generation_prompt`: <text_start>,
                          target text, <final_seg>, <speech_start>
        p+1 .. end        target audio tokens at audio_id_base, then <speech_end>

    Only that last run is supervised. Everything before it is context the model is
    given, so a loss over it would be measuring text modelling and reference
    copying -- neither of which is what generation does, and both of which would
    swamp the audio term that is.

    Returns the ids, the labels (-100 outside the supervised run) and which
    speaker embedding the example wants.
    """
    spec = VARIANT_SPEC[variant]

    wrapping = token_map
    if spec.get("reference") == "plain_role":
        # Zeroing the reference ids flips `uses_reference_role_tokens`, so
        # build_reference_prefix takes its other branch and emits the layout a
        # non-reference-supervised checkpoint was trained on. Built through
        # prompt.py rather than written out here, so the control is the real
        # alternative layout and not our idea of it.
        wrapping = dataclasses.replace(
            token_map, reference_text_start_id=0, reference_text_end_id=0,
            reference_speech_start_id=0, reference_speech_end_id=0)
    segmenting = token_map
    if spec.get("marker") == "no_final_seg":
        segmenting = dataclasses.replace(token_map, final_seg_id=0)

    prefix = None
    if spec.get("reference") != "dropped":
        prefix = build_reference_prefix(wrapping, reference["text_ids"],
                                        reference["audio_tokens"])
    prompt = build_generation_prompt(segmenting, target["text_ids"],
                                     reference_prefix=prefix)
    if spec.get("marker") == "text_end":
        # Spliced rather than rebuilt: prompt.py never emits text_end_id, which is
        # exactly the open question -- the id exists in the token map and nothing
        # in the inference path uses it.
        prompt = prompt[:-2] + [token_map.text_end_id] + prompt[-2:]

    audio = list(target["audio_tokens"])
    if spec.get("audio") == "shuffled":
        audio = [int(t) for t in rng.permutation(audio)]
    elif spec.get("audio") == "other":
        audio = list(target["other_audio_tokens"])

    supervised = [token_map.audio_id_base + t for t in audio] + [token_map.speech_end_id]
    return {"input_ids": prompt + supervised,
            "labels": [-100] * len(prompt) + supervised,
            "speaker": spec.get("speaker", "real"),
            "n_prompt": len(prompt), "n_supervised": len(supervised)}


def speaker_vector(example, embedding, rng, dim):
    """The speaker input this example asks for, unit-norm like the encoder's."""
    import numpy as np

    if example["speaker"] == "zeros":
        return np.zeros(dim, dtype=np.float32)
    if example["speaker"] == "noise":
        noise = rng.normal(size=dim).astype(np.float32)
        return noise / np.linalg.norm(noise)
    return embedding


# ── loss ──────────────────────────────────────────────────────────────────────
def batch_loss(model, examples, embeddings, device, dtype):
    """Mean cross-entropy per example over its supervised tokens only.

    Right-padded, never left-padded: `speaker.py` writes the projected speaker
    over position 0, so a left-padded row would have the speaker land on a pad
    token and the example would silently run without any speaker conditioning at
    all -- which is one of the controls, not the correct format.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.zeros(len(examples), width, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    mask = torch.zeros(len(examples), width, dtype=torch.long)
    for i, example in enumerate(examples):
        n = len(example["input_ids"])
        ids[i, :n] = torch.tensor(example["input_ids"])
        labels[i, :n] = torch.tensor(example["labels"])
        mask[i, :n] = 1

    ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
    speaker = torch.as_tensor(np.stack(embeddings), dtype=dtype, device=device)

    model._pending_speaker_embedding = speaker
    try:
        with torch.no_grad():
            # `input_ids=` by keyword, and never `inputs_embeds`: the hook in
            # speaker.py reads `kwargs.get("input_ids")` and silently returns
            # unchanged when it is absent. Pass the ids positionally, or hand the
            # model embeddings you built yourself, and the speaker is never
            # injected -- the run looks normal and conditions on nothing. A
            # training loop has the same trap.
            logits = model(input_ids=ids, attention_mask=mask).logits
    finally:
        model._pending_speaker_embedding = None

    # Teacher forcing: position i's logits predict token i+1.
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    supervised = shifted_labels != -100
    # Only the supervised rows are upcast. The full (batch, length, 158241) tensor
    # in float32 is the memory bound of this script -- it is why --batch-size
    # defaults to 1 and why a CPU run of a long example is the slow case.
    flat = F.cross_entropy(shifted_logits[supervised].float(),
                           shifted_labels[supervised], reduction="none")
    counts = supervised.sum(dim=1)
    out, cursor = [], 0
    for n in counts.tolist():
        out.append(float(flat[cursor:cursor + n].mean()) if n else None)
        cursor += n
    return out


# ── data ──────────────────────────────────────────────────────────────────────
def pair_by_speaker(embeddings, targets):
    """For each target clip, the pool clip most likely to be the same speaker.

    FLEURS has no speaker id column -- a row is (sentence id, file, transcripts,
    samples, gender) -- so the pairing has to be inferred, and the speaker encoder
    the model itself conditions on is the natural thing to infer it with. The
    cosine of each chosen pair is reported alongside the losses: a run whose pairs
    average a low similarity has not found same-speaker references, and its
    absolute losses are inflated by a speaker mismatch that training would not
    have.

    A different utterance, never the clip itself: putting the target's own audio
    in the reference prefix makes the task verbatim copying and the loss collapses
    to near zero, which would look like a resounding pass and mean nothing.
    """
    import numpy as np

    matrix = np.stack([e / max(np.linalg.norm(e), 1e-9) for e in embeddings])
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)
    pairs = []
    for index in targets:
        partner = int(np.argmax(similarity[index]))
        pairs.append((partner, float(similarity[index, partner])))
    return pairs


def prepare_language(code, args, codec, encoder, tokenizer, normalize):
    """Everything the examples for one language need, computed once.

    Order matters for memory: the clips are read and encoded here, so that the
    codec and the speaker encoder can both be released before the 1.7B language
    model is loaded.
    """
    import numpy as np
    import soundfile as sf

    from kittenml.kittentts2 import text as text_utils

    pool = fleurs_clips([code], args.split, args.pool, args.cache_dir)[code]
    usable = [c for c in pool
              if sf.info(str(c["path"])).duration <= args.max_seconds]
    if len(usable) < args.examples + 1:
        raise SystemExit(
            f"{code}: only {len(usable)} of {len(pool)} clips are under "
            f"{args.max_seconds}s; raise --pool or --max-seconds")

    embeddings = [encoder(c["path"]) for c in usable]
    targets = list(range(min(args.examples, len(usable))))
    pairs = pair_by_speaker(embeddings, targets)
    # The "different utterance" control takes the next target's audio, so it is
    # real speech of the right length distribution rather than noise -- the control
    # has to isolate the text-audio correspondence, not test whether the model
    # notices garbage. With a single example there is no next target, so it comes
    # from the pool instead; what it must never be is the target's own audio.
    others = [targets[(slot + 1) % len(targets)] if len(targets) > 1
              else next(i for i in range(len(usable)) if i != target)
              for slot, target in enumerate(targets)]

    needed = sorted(set(targets) | {p for p, _ in pairs} | set(others))
    tokens = {i: codec.encode_reference(usable[i]["path"]) for i in needed}

    def text_ids(clip):
        spoken = text_utils.normalize_text(clip["text"]) if normalize else clip["text"]
        return tokenizer.encode(spoken.strip(), add_special_tokens=False)

    examples = []
    for target, (partner, cosine), other in zip(targets, pairs, others):
        examples.append({
            "id": usable[target]["id"],
            "reference": {"text_ids": text_ids(usable[partner]),
                          "audio_tokens": tokens[partner]},
            "target": {"text_ids": text_ids(usable[target]),
                       "audio_tokens": tokens[target],
                       "other_audio_tokens": tokens[other]},
            "embedding": np.asarray(embeddings[target], dtype=np.float32),
            "reference_cosine": cosine,
            "reference_id": usable[partner]["id"],
        })
    return examples


# ── model ─────────────────────────────────────────────────────────────────────
def load_model(repo_dir, device, dtype, spk_dim=512):
    """The bf16 checkpoint in `Qwen3ForCausalLM`, with `spk_proj` attached.

    from_pretrained rather than `model.py`'s from_config path: that path exists
    because the *packed* ternary weights have no file on disk for from_pretrained
    to find (model.py:208-216). Here the plain `lm/model.safetensors` is exactly
    what the training plan starts from, so the ordinary loader is both correct and
    one fewer thing to get subtly wrong.
    """
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    from kittenml.kittentts2.model import _quiet_unexpected_keys
    from kittenml.kittentts2.speaker import attach_speaker_head

    lm_dir = pathlib.Path(repo_dir) / "lm"
    # spk_proj.* travels in the same file as the LM weights and the architecture
    # knows nothing about it, so from_pretrained reports it as unexpected and
    # drops it; it is read out separately and handed to attach_speaker_head,
    # which raises if it is absent -- a checkpoint without it cannot be
    # speaker-conditioned and every number here would be measuring the wrong model.
    with safe_open(str(lm_dir / "model.safetensors"), framework="pt") as handle:
        state = {k: handle.get_tensor(k) for k in handle.keys()
                 if k.startswith("spk_proj.")}
    with _quiet_unexpected_keys():
        model = AutoModelForCausalLM.from_pretrained(str(lm_dir), dtype=dtype)
    model = model.to(device).eval()
    return attach_speaker_head(model, str(lm_dir), device, dtype, spk_dim=spk_dim,
                               state=state)


def tiny_model(token_map, device, dtype, hidden=64, spk_dim=512):
    """A randomly-initialised Qwen3 with the real vocabulary, for --self-test.

    The vocabulary has to be the real 158,241 rows or the token map's ids index
    out of range, which is the whole point: this checks that the assembly, the
    mask and the speaker hook line up with the *ids the checkpoint uses*, on a
    model small enough to run anywhere. It says nothing about any loss value.
    """
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, Qwen3Config

    from kittenml.kittentts2.speaker import attach_speaker_head

    model = AutoModelForCausalLM.from_config(Qwen3Config(
        vocab_size=token_map.audio_id_end + 5, hidden_size=hidden,
        intermediate_size=2 * hidden, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=hidden // 4, tie_word_embeddings=True))
    model = model.to(device=device, dtype=dtype).eval()
    head = nn.Sequential(nn.Linear(spk_dim, hidden), nn.LayerNorm(hidden))
    attach_speaker_head(model, "<synthetic>", device, dtype, spk_dim=spk_dim,
                        state={f"spk_proj.{k}": v for k, v in head.state_dict().items()})
    return model


# ── self-test ─────────────────────────────────────────────────────────────────
def self_test(args):
    """Assert the invariants a wrong number would hide, on a synthetic model.

    Every check here is about *structure*: where the speaker lands, which
    positions carry a label, that padding a batch does not move a row's loss.
    None of them can tell whether the format matches the checkpoint -- only the
    real controls on the real weights do that.
    """
    import numpy as np
    import torch

    token_map = _stub_token_map()
    rng = np.random.default_rng(args.seed)
    device, dtype = args.device, torch.float32

    reference = {"text_ids": [11, 12, 13], "audio_tokens": [5, 6, 7, 8]}
    target = {"text_ids": [21, 22], "audio_tokens": [1, 2, 3, 4, 5],
              "other_audio_tokens": [9, 9, 9]}

    correct = assemble(token_map, reference, target, "correct",
                       rng=np.random.default_rng(0))
    ids, labels = correct["input_ids"], correct["labels"]
    checks = []

    def check(name, condition, detail=""):
        checks.append((name, bool(condition), detail))

    check("position 0 is START", ids[0] == token_map.start_id, f"got {ids[0]}")
    expected_prefix = build_reference_prefix(token_map, reference["text_ids"],
                                             reference["audio_tokens"])
    check("reference prefix is prompt.py's",
          ids[:len(expected_prefix)] == expected_prefix)
    check("reference uses the role tokens",
          token_map.reference_text_start_id in ids)
    check("<final_seg> precedes <speech_start>",
          ids[correct["n_prompt"] - 2:correct["n_prompt"]]
          == [token_map.final_seg_id, token_map.speech_start_id])
    supervised = [i for i, v in enumerate(labels) if v != -100]
    check("supervised run is the target audio plus <speech_end>",
          supervised == list(range(correct["n_prompt"], len(ids)))
          and len(supervised) == len(target["audio_tokens"]) + 1,
          f"{len(supervised)} positions")
    check("supervised ids are audio ids then <speech_end>",
          [ids[i] - token_map.audio_id_base for i in supervised[:-1]]
          == target["audio_tokens"] and ids[supervised[-1]] == token_map.speech_end_id)
    check("nothing before <speech_start> is supervised",
          all(v == -100 for v in labels[:correct["n_prompt"]]))

    swapped = assemble(token_map, reference, target, "role_swapped",
                       rng=np.random.default_rng(0))
    check("role_swapped drops the reference-role ids",
          token_map.reference_text_start_id not in swapped["input_ids"]
          and token_map.text_start_id in swapped["input_ids"])
    dropped = assemble(token_map, reference, target, "no_reference",
                       rng=np.random.default_rng(0))
    check("no_reference starts [START] <text_start>",
          dropped["input_ids"][:2] == [token_map.start_id, token_map.text_start_id])
    isolated = assemble(token_map, reference, target, "speaker_zeros_no_reference",
                        rng=np.random.default_rng(0))
    check("the spk_proj control differs from its baseline only in the speaker",
          isolated["input_ids"] == dropped["input_ids"]
          and (isolated["speaker"], dropped["speaker"]) == ("zeros", "real"))
    check("every variant supervises the same number of tokens",
          len({assemble(token_map, reference, target, v,
                        rng=np.random.default_rng(0))["n_supervised"]
               for v in VARIANTS if v != "other_audio"}) == 1)

    model = tiny_model(token_map, device, dtype)
    embedding = rng.normal(size=512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    alone = batch_loss(model, [correct], [embedding], device, dtype)[0]
    # A short companion row forces padding on the other, which is the case that
    # breaks silently if the mask or the padding side is wrong.
    short = assemble(token_map, reference,
                     {**target, "audio_tokens": target["audio_tokens"][:2]}, "correct",
                     rng=np.random.default_rng(0))
    padded = batch_loss(model, [correct, short], [embedding, embedding], device, dtype)
    check("padding does not move a row's loss", abs(padded[0] - alone) < 1e-4,
          f"{alone:.6f} vs {padded[0]:.6f}")

    zeroed = batch_loss(model, [{**correct, "speaker": "zeros"}],
                        [speaker_vector({"speaker": "zeros"}, embedding, rng, 512)],
                        device, dtype)[0]
    check("the speaker embedding reaches the forward pass", abs(zeroed - alone) > 1e-6,
          f"real {alone:.6f} vs zeros {zeroed:.6f}")

    print("\n== Self-test (synthetic weights; structure only) ==")
    for name, ok, detail in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}"
              + (f"   [{detail}]" if detail else ""))
    failed = [name for name, ok, _ in checks if not ok]
    print("\n  " + ("all structural invariants hold. Losses, controls and the format\n"
                    "  verdict itself still require the real checkpoint."
                    if not failed else f"{len(failed)} invariant(s) broken: {failed}"))
    return 1 if failed else 0


def _stub_token_map():
    """The released checkpoint's token map, inlined so --self-test needs no network.

    `main` compares it against the real config.json on every full run and warns
    if they have drifted apart.
    """
    return TokenMap(audio_id_base=151675, num_audio_tokens=6561, speech_start_id=151669,
                    speech_end_id=151670, text_start_id=151671, text_end_id=151672,
                    start_id=151673, stop_id=151674, final_seg_id=158236,
                    reference_speech_start_id=158237, reference_speech_end_id=158238,
                    reference_text_start_id=158239, reference_text_end_id=158240)


# ── report ────────────────────────────────────────────────────────────────────
def report(control_table, by_language, token_map, args):
    """The control table, the per-language baseline, then the verdict."""
    uniform = math.log(token_map.num_audio_tokens)
    correct = control_table["correct"]["loss"]
    cosine = by_language[args.control_lang]["reference_cosine"]

    print(f"\n== Format check on {args.control_lang} "
          f"({control_table['correct']['n']} examples, "
          f"reference speaker cosine {cosine:.2f}) ==")
    print(f"{'variant':<28}{'loss':>8}{'ppl':>10}{'gap':>9}  measured against")
    print(f"{'correct format':<28}{correct:>8.3f}{math.exp(correct):>10.1f}{'-':>9}")
    for name in DECISIVE_CONTROLS + DIAGNOSTIC_CONTROLS:
        row = control_table.get(name)
        if not row:
            continue
        against = BASELINE.get(name, "correct")
        gap = row["loss"] - control_table[against]["loss"]
        kind = against if name in DECISIVE_CONTROLS else f"{against}  (diagnostic)"
        print(f"{name:<28}{row['loss']:>8.3f}{math.exp(row['loss']):>10.1f}"
              f"{gap:>+9.3f}  {kind}")
    print(f"{f'uniform over {token_map.num_audio_tokens}':<28}{uniform:>8.3f}"
          f"{token_map.num_audio_tokens:>10}{uniform - correct:>+9.3f}"
          "  no model at all")

    def unbeaten(group):
        return [name for name in group if name in control_table
                and control_table[name]["loss"]
                - control_table[BASELINE.get(name, "correct")]["loss"] < args.margin]

    content, identity = unbeaten(CONTENT_CONTROLS), unbeaten(IDENTITY_CONTROLS)
    failed = content + identity

    def listing(names):
        for name in names:
            against = BASELINE.get(name, "correct")
            print(f"    {name}: "
                  f"{control_table[name]['loss'] - control_table[against]['loss']:+.3f}"
                  f" nats over {against} (needs >= {args.margin})")

    print("\n== Verdict ==")
    if correct >= uniform:
        failed.append("absolute")
        print(f"  The correct format scores {correct:.3f} nats against a uniform\n"
              f"  baseline of {uniform:.3f}: the model is predicting nothing. The\n"
              "  sequence is wrong, or the weights did not load. Nothing below is\n"
              "  worth reading until that is fixed.")

    print("  Target segment " + ("CONFIRMED." if not content else "NOT CONFIRMED."))
    if not content:
        print("    Scrambling what is said costs whole nats, so the markers, the token\n"
              "    offset, the loss mask and the shift are the ones this checkpoint\n"
              "    continues. This is the half that would waste the compute if wrong.")
    else:
        listing(content)
        print("    The model is indifferent to what the target audio actually is. That\n"
              "    is not a subtle finding -- check the audio_id_base offset, the label\n"
              "    shift, and that spk_proj did not swallow a real token at position 0.")

    print("  Reference and speaker " + ("CONFIRMED." if not identity
                                        else "NOT CONFIRMED."))
    if not identity:
        print("    The reference-role wrapping and the position-0 speaker both change\n"
              "    the loss, so our layout for them is the trained one.")
    else:
        listing(identity)
        if cosine < args.min_reference_cosine:
            print(f"    The reference pairs average {cosine:.2f} cosine, below"
                  f" {args.min_reference_cosine}: they are not\n"
                  "    the same speaker, so the reference carries a voice that does not\n"
                  "    match the target and there was nothing useful to corrupt. Re-run\n"
                  "    with a larger --pool before reading anything into this.")
        else:
            print(f"    Pairing is sound ({cosine:.2f} cosine), so the references are\n"
                  "    the right speaker and there was something real to corrupt. Two\n"
                  "    readings remain and this test cannot separate them: either our\n"
                  "    reference layout is not the trained one, or the checkpoint is\n"
                  "    near-indifferent to identity conditioning under teacher forcing.\n"
                  "    The second is plausible -- each audio token is mostly\n"
                  "    predictable from the ones before it, and the `no_reference` row\n"
                  "    above says what the entire reference prefix is worth. Note also\n"
                  "    that role_swapped is not a pure corruption: its layout is what a\n"
                  "    *second segment* of a multi-segment sample looks like, which\n"
                  "    `final_seg` existing at all implies the checkpoint saw.")
        print("    What this blocks: claiming the reference layout is right. Training\n"
              "    the target segment is still on solid ground, and the training plan\n"
              "    already calls for reference dropout -- so train with it, and settle\n"
              "    the reference layout by generating from both and scoring speaker\n"
              "    similarity, which this metric cannot see.")
    print("  " + REFERENCE_CAVEAT)

    print("\n== Correct format by language ==")
    print(f"{'lang':<8}{'n':>4}{'loss':>9}{'ppl':>9}{'vs en':>9}{'ref cos':>9}"
          f"{'ref tok':>9}{'tgt tok':>9}")
    baseline = by_language.get(args.control_lang, {}).get("loss")
    for code, row in by_language.items():
        delta = f"{row['loss'] - baseline:+.3f}" if baseline is not None else "-"
        print(f"{code:<8}{row['n']:>4}{row['loss']:>9.3f}{math.exp(row['loss']):>9.1f}"
              f"{delta:>9}{row['reference_cosine']:>9.2f}"
              f"{row['reference_tokens']:>9.0f}{row['target_tokens']:>9.0f}")
    print("\n  The per-language rows are a baseline, not a gate: they say how much\n"
          "  worse Indic is under today's tokenizer and today's weights, which is\n"
          "  the number a fine-tune has to improve on.")
    worst = min(row["reference_cosine"] for row in by_language.values())
    if worst < args.min_reference_cosine:
        print("  Read them as an upper bound on the loss, not a clean measurement:\n"
              "  at least one language's references are a different speaker from its\n"
              "  target, and predicting one voice's audio while conditioned on another\n"
              "  costs nats that training would never have paid.")
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="en,hi,ta",
                    help="first one is the control the format check runs on")
    ap.add_argument("--examples", type=int, default=8, help="examples per language")
    ap.add_argument("--pool", type=int, default=0,
                    help="clips to draw references from (default: 48, or 6x --examples)")
    ap.add_argument("--min-reference-cosine", type=float, default=MIN_REFERENCE_COSINE,
                    help="below this mean pair cosine the speaker controls are "
                         "uninformative and the verdict says so")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=14.0,
                    help="skip clips longer than this, both as target and reference")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float32"],
                    help="auto = bfloat16, the checkpoint's own dtype")
    ap.add_argument("--no-normalize", action="store_true",
                    help="tokenize the raw transcript instead of text.normalize_text")
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help="nats by which the correct format must beat each control")
    ap.add_argument("--repo-dir", type=pathlib.Path,
                    help="local kitten-tts-2 checkout (default: download from the Hub)")
    ap.add_argument("--cache-dir", type=pathlib.Path, default=CACHE_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true",
                    help="validate assembly, masking and the speaker hook, no checkpoint")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    # A pool this much larger than the example count is not waste: the reference
    # has to be a *different utterance by the same speaker*, FLEURS does not say
    # who spoke what, and the nearest-neighbour search can only find a same-speaker
    # partner if one is in the pool. An embedding per pool clip is milliseconds.
    args.pool = args.pool or max(48, 6 * args.examples)

    if args.self_test:
        return self_test(args)

    import numpy as np
    import torch

    codes = [c.strip() for c in args.langs.split(",") if c.strip()]
    unknown = [c for c in codes if c not in LANGUAGES]
    if unknown:
        raise SystemExit(f"unknown language codes: {unknown}")
    args.control_lang = codes[0]
    dtype = torch.bfloat16 if args.dtype in ("auto", "bfloat16") else torch.float32

    repo_dir = args.repo_dir
    if repo_dir is None:
        from huggingface_hub import snapshot_download
        repo_dir = pathlib.Path(snapshot_download(REPO_ID, allow_patterns=REPO_FILES))

    from transformers import AutoTokenizer

    from kittenml.kittentts2.speaker import SpeakerEncoder
    from kittenml.kittentts2.vocoder import S3Codec

    config = json.loads((repo_dir / "config.json").read_text())
    token_map = TokenMap.from_config(config)
    spk_dim = config.get("speaker_embedding_dim", 512)
    tokenizer = AutoTokenizer.from_pretrained(str(repo_dir / "lm"))
    if token_map != _stub_token_map():
        print("note: the checkpoint's token map differs from the one --self-test "
              "inlines; update _stub_token_map()")

    print(f"encoding {args.examples} examples per language on {args.device} ...",
          flush=True)
    codec = S3Codec(device=args.device)
    encoder = SpeakerEncoder(repo_dir / "speaker" / "model.safetensors",
                             device=args.device)
    data = {code: prepare_language(code, args, codec, encoder, tokenizer,
                                   not args.no_normalize) for code in codes}
    # Released before the 1.7B model is read, so neither a CPU box nor the GPU
    # holds s3gen, the speaker encoder and the language model at the same time.
    del codec, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"loading {REPO_ID} lm on {args.device} as {dtype} ...", flush=True)
    model = load_model(repo_dir, args.device, dtype, spk_dim=spk_dim)

    def run(code, variant):
        rng = np.random.default_rng(args.seed)
        built, vectors = [], []
        for example in data[code]:
            built.append(assemble(token_map, example["reference"], example["target"],
                                  variant, rng=rng))
            vectors.append(speaker_vector(built[-1], example["embedding"], rng,
                                          example["embedding"].shape[0]))
        losses = []
        for start in range(0, len(built), args.batch_size):
            window = slice(start, start + args.batch_size)
            losses += batch_loss(model, built[window], vectors[window],
                                 args.device, dtype)
            print(f"  {code}/{variant} {len(losses)}/{len(built)}", end="\r", flush=True)
        return [l for l in losses if l is not None], built

    control_table = {}
    for variant in VARIANTS:
        losses, built = run(args.control_lang, variant)
        control_table[variant] = {"loss": sum(losses) / len(losses), "n": len(losses),
                                  "per_example": losses,
                                  "mean_length": sum(len(b["input_ids"]) for b in built)
                                  / len(built)}
        print(f"  {args.control_lang}/{variant}: {control_table[variant]['loss']:.3f}"
              + " " * 24)

    by_language = {}
    for code in codes:
        if code == args.control_lang:
            losses = control_table["correct"]["per_example"]
        else:
            losses, _ = run(code, "correct")
            print(f"  {code}/correct: {sum(losses) / len(losses):.3f}" + " " * 24)
        by_language[code] = {
            "loss": sum(losses) / len(losses), "n": len(losses), "per_example": losses,
            "reference_cosine": sum(e["reference_cosine"] for e in data[code])
                                / len(data[code]),
            "reference_tokens": sum(len(e["reference"]["audio_tokens"])
                                    for e in data[code]) / len(data[code]),
            "target_tokens": sum(len(e["target"]["audio_tokens"])
                                 for e in data[code]) / len(data[code])}

    failed = report(control_table, by_language, token_map, args)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"config": {"langs": codes, "examples": args.examples, "split": args.split,
                        "device": args.device, "dtype": str(dtype),
                        "normalize": not args.no_normalize, "margin": args.margin,
                        "pool": args.pool, "max_seconds": args.max_seconds,
                        "min_reference_cosine": args.min_reference_cosine,
                        "seed": args.seed},
             "controls": control_table, "languages": by_language,
             "failed_controls": failed}, indent=2, default=str))
        print(f"\nwrote {args.out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

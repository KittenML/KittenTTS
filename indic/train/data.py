"""The training example, built by the same code that builds a generation prompt.

A speech LM fine-tune fails silently when the training sequence differs from the
inference sequence -- the loss falls, the model learns the layout it was shown,
and generation is subtly wrong in a way no training metric reports. So the
prompt half of every example comes from `prompt.py` itself rather than from a
restatement of it here:

    pos 0    [START], whose *input embedding* is overwritten by spk_proj
    ...      <reference_text_start> ref transcript <reference_text_end>
             <reference_speech_start> ref audio <reference_speech_end>
    ...      [emotion control ids]
             <text_start> target text <final_seg> <speech_start>
    ---- everything above is -100 ----
             target audio tokens
             <speech_end>

Position 0 is worth stating plainly: the `[START]` id there is never embedded.
`speaker.py` replaces that position's input embedding with the projected speaker
vector, so the token is a placeholder holding a slot. An example whose speaker
embedding fails to reach the model therefore trains on a *different input
sequence*, not merely a less-conditioned one -- see the speaker check in
`selfcheck.py`.

Loss covers the target audio tokens and `<speech_end>` and nothing else. The
first supervised position is the first audio token, which the model predicts
from `<speech_start>` -- the same position inference generates from.

There is no corpus yet, so `SyntheticDataset` fabricates well-formed examples.
That is what makes the pipeline provable today: every check in `selfcheck.py`
runs against it.
"""

import numpy as np
import torch

from . import load_kittentts2

_prompt = load_kittentts2("prompt")
_tokens = load_kittentts2("tokens")

TokenMap = _tokens.TokenMap

# Positions the loss ignores. torch's cross_entropy default, and what
# `Qwen3ForCausalLM.forward` passes to it.
IGNORE_INDEX = -100


def build_example(token_map, target_text_ids, target_audio_tokens,
                  reference_text_ids=None, reference_audio_tokens=None,
                  emotion_ids=None):
    """One training example as (input_ids, labels), both plain lists.

    `reference_text_ids=None` drops the in-context reference, which inference
    also supports (`build_generation_prompt(reference_prefix=None)`). Training a
    fraction of examples that way is what keeps the no-reference path working;
    a run that never sees it will degrade there without ever being measured.

    The supervised span is computed from the prompt's length, not found by
    scanning the finished sequence for markers. That matters for the self-check:
    the mask and the check that verifies it are then two independent
    derivations, and an off-by-one in one of them does not hide in the other.
    """
    # Both branches go through prompt.py, which is not a style preference:
    # `[START]` lives *inside* `build_reference_prefix`, so dropping the
    # reference is not concatenating one less block -- `build_generation_prompt`
    # emits its own. Assembling either path by hand gets two [START]s or none.
    # Note also that `text_end_id` (151672) exists in the token map and is
    # emitted nowhere: the target text is closed by `<final_seg>` and the
    # reference by `reference_text_end_id`. That is the layout the checkpoint
    # was prompted with, not an oversight to correct.
    reference_prefix = None
    if reference_text_ids is not None and reference_audio_tokens is not None:
        reference_prefix = _prompt.build_reference_prefix(
            token_map, list(reference_text_ids), list(reference_audio_tokens))

    prompt_ids = _prompt.build_generation_prompt(
        token_map, list(target_text_ids), reference_prefix=reference_prefix,
        emotion_ids=list(emotion_ids) if emotion_ids else None)

    supervised = [token_map.audio_id_base + t for t in target_audio_tokens]
    supervised.append(token_map.speech_end_id)

    input_ids = [*prompt_ids, *supervised]
    labels = [IGNORE_INDEX] * len(prompt_ids) + supervised
    return input_ids, labels


def tiny_token_map(n_text=150, n_audio=64):
    """A `TokenMap` with the released layout at a size that fits a toy model.

    Same block order as the real checkpoint -- text, six named specials, the
    audio range, then the marker rows at the tail -- because `prompt.py` and the
    logit masking both read positions out of that layout. Shrinking it is the
    only way the self-check runs a real forward pass in seconds on a CPU.
    """
    specials = n_text
    audio_base = specials + 6
    markers = audio_base + n_audio
    return TokenMap(
        audio_id_base=audio_base, num_audio_tokens=n_audio,
        speech_start_id=specials, speech_end_id=specials + 1,
        text_start_id=specials + 2, text_end_id=specials + 3,
        start_id=specials + 4, stop_id=specials + 5,
        final_seg_id=markers, reference_speech_start_id=markers + 1,
        reference_speech_end_id=markers + 2, reference_text_start_id=markers + 3,
        reference_text_end_id=markers + 4)


class SyntheticDataset(torch.utils.data.Dataset):
    """Well-formed examples with no corpus behind them.

    The token ids are noise, so nothing here says anything about speech. What it
    does say is that the sequence *layout* is assembled, masked, padded, batched
    and differentiated correctly, which is the part that can be proved before any
    data exists -- and the part that is hardest to debug once real data hides it.

    Deterministic per index: example `i` is the same example on every epoch, in
    every process and after a restart, so a resume check can compare losses
    step for step.
    """

    def __init__(self, token_map, size=64, speaker_dim=512, seed=0,
                 text_len=(4, 12), audio_len=(8, 24), reference_probability=0.8,
                 n_speakers=4):
        self.token_map = token_map
        self.size = size
        self.speaker_dim = speaker_dim
        self.seed = seed
        self.text_len = text_len
        self.audio_len = audio_len
        self.reference_probability = reference_probability
        self.n_speakers = n_speakers

    def __len__(self):
        return self.size

    def speaker_embedding(self, speaker):
        """Unit-norm, as `SpeakerEncoder` returns and `spk_proj` was trained on.

        Drawn from the speaker index alone, so the same speaker is the same
        vector across examples -- otherwise "speaker conditioning changes the
        loss" would be measuring fresh noise rather than identity.
        """
        rng = np.random.default_rng([self.seed, 0xBEEF, speaker])
        raw = rng.standard_normal(self.speaker_dim).astype(np.float32)
        return torch.from_numpy(raw / np.linalg.norm(raw))

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        tm = self.token_map
        rng = np.random.default_rng([self.seed, index])
        # Text ids stay below the specials block: an id drawn from the audio
        # range would put audio tokens inside the text span and quietly make the
        # synthetic example malformed in exactly the way the suite is supposed
        # to catch.
        n_text = int(rng.integers(*self.text_len))
        n_audio = int(rng.integers(*self.audio_len))
        text_ids = rng.integers(0, tm.speech_start_id, size=n_text).tolist()
        audio = rng.integers(0, tm.num_audio_tokens, size=n_audio).tolist()

        reference = rng.random() < self.reference_probability
        reference_text = reference_audio = None
        if reference:
            reference_text = rng.integers(0, tm.speech_start_id,
                                          size=int(rng.integers(*self.text_len))).tolist()
            reference_audio = rng.integers(0, tm.num_audio_tokens,
                                           size=int(rng.integers(*self.audio_len))).tolist()

        input_ids, labels = build_example(tm, text_ids, audio,
                                          reference_text_ids=reference_text,
                                          reference_audio_tokens=reference_audio)
        speaker = int(rng.integers(0, self.n_speakers))
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "speaker_embedding": self.speaker_embedding(speaker),
                "speaker": speaker}


class ShardDataset(torch.utils.data.Dataset):
    """Precomputed real shards. Not implemented -- there is no corpus yet.

    The shape it will have is fixed by `build_example` above, and by step 0.4 of
    the training plan: S3 codec tokens, pyannote embeddings and text ids are all
    computed offline, so a shard row needs no audio library at train time. One
    row is:

        text_ids             target text, Qwen3 ids, add_special_tokens=False
        audio_tokens         target S3 codec ids, 0-based (not offset)
        reference_text_ids   a *different* utterance by the same speaker
        reference_audio      that utterance's codec ids, 0-based
        speaker_embedding    float32[512], unit norm, from the reference clip
        emotion_ids          optional, tokenized "{emo: 1}" control

    Deliberately a stub rather than a guess: writing a reader against an
    imagined schema is how the reader and the shard writer end up disagreeing.
    """

    def __init__(self, shards):
        raise NotImplementedError(
            "ShardDataset has no corpus to read yet. Build shards per step 0 of "
            f"docs/indic/training.md, then implement this against them: {shards}")


class Collator:
    """Pad a batch to a rectangle without disturbing the mask or position 0.

    **Right padding, and it is not interchangeable.** `speaker.py` writes the
    projected speaker vector over sequence position 0 unconditionally. Left
    padding would put a pad token at position 0, so the speaker would overwrite
    padding while `[START]` sat at some per-row offset -- a batch where speaker
    conditioning is attached to nothing, with no error raised. Right padding
    keeps `[START]` at index 0 for every row, which is also where inference puts
    it (`build_generation_prompt` emits `start_id` first in both branches).

    Training is teacher-forced, so right padding costs nothing: the pads sit
    after the last real token, causal attention never looks forward at them, and
    their labels are -100. That is the opposite of batched *generation*, which
    needs left padding to keep every row's last token at the same index -- the
    inference path sidesteps the question by running one sequence at a time.
    """

    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        width = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[row, :length] = item["input_ids"]
            labels[row, :length] = item["labels"]
            attention_mask[row, :length] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "speaker_embedding": torch.stack([item["speaker_embedding"] for item in batch]),
        }


def supervised_span(input_ids, token_map):
    """Where the loss *should* apply, found by reading the token stream.

    The second, independent derivation of the mask: `build_example` gets it from
    the prompt's length while this one locates `<speech_start>` and
    `<speech_end>` in the finished sequence. The self-check compares them, which
    only means something because neither is computed from the other.

    Returns (first, last) inclusive label indices, or None when the row has no
    target segment. `<speech_start>` is taken as the last occurrence: a
    checkpoint without reference-role tokens wraps its reference in the plain
    speech markers too (`prompt.build_reference_prefix`), so the first one
    belongs to the reference.
    """
    ids = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
    starts = [i for i, t in enumerate(ids) if t == token_map.speech_start_id]
    ends = [i for i, t in enumerate(ids) if t == token_map.speech_end_id]
    if not starts or not ends:
        return None
    start = starts[-1]
    end = next((i for i in ends if i > start), None)
    return None if end is None else (start + 1, end)

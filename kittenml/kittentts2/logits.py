"""Sampling pipeline for KittenTTS 2 decoding.

The whole pipeline is assembled explicitly as a `LogitsProcessorList` and handed
to `generate()` with neutral sampling kwargs, so transformers adds nothing of its
own and the ordering here is the ordering that runs.

Both custom processors are vectorised. The stock equivalents call
`input_ids[b].tolist()` on every decode step for every row — a GPU-to-Python copy
of the whole sequence followed by a Python loop — which costs O(batch x seq) of
interpreter time per token and stalls the GPU as the sequence grows. These
compute the same thing on-device.
"""

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    MinPLogitsWarper,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# The S3 codec's silence token, as an offset within the audio range. Runs of it are
# the model's most common stuck state, and it is the one token the repetition
# penalty must leave alone (see RepetitionPenaltyExempt).
SILENCE_TOKEN = 4299


class RepetitionPenaltyExempt(LogitsProcessor):
    """Repetition penalty that never penalises the sequence-terminating tokens.

    `<speech_end>`, `[STOP]` and the silence codec token are exempt: penalising
    them suppresses the trailing silence the model needs before it can close a
    segment, which turns a clean ending into an over-generation.

    window=None scans the whole sequence. window=N restricts the scan to the last
    N *generated* positions, so codec tokens from a long in-context reference
    cannot suppress valid target-audio tokens merely for having occurred in the
    prompt. Vectorising across the batch is safe because every caller left-pads
    to a common width first, making `prompt_length` uniform across rows.
    """

    def __init__(self, penalty, exempt_ids, window=None, prompt_length=0):
        self.penalty = float(penalty)
        self._exempt = sorted({int(i) for i in exempt_ids})
        self._exempt_t = None
        self.window = int(window) if window else None
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids, scores):
        hist = input_ids
        if self.window is not None:
            start = max(self.prompt_length, input_ids.shape[1] - self.window)
            hist = input_ids[:, start:]
        if hist.shape[1] == 0:
            return scores
        got = torch.gather(scores, 1, hist)
        got = torch.where(got < 0, got * self.penalty, got / self.penalty)
        out = scores.scatter(1, hist, got)
        if self._exempt:
            if self._exempt_t is None or self._exempt_t.device != scores.device:
                self._exempt_t = torch.tensor(self._exempt, device=scores.device, dtype=torch.long)
            out[:, self._exempt_t] = scores[:, self._exempt_t]
        return out


class TokenRunPenalty(LogitsProcessor):
    """Escalating penalty on a codec token repeated past `grace` times in a row.

    Catches the model getting stuck on one token — silence, room tone, or a held
    phoneme — which is the dominant long-generation failure. The trailing run
    length is a reversed cumulative product over the equality mask, so no Python
    rescan of the sequence is needed.
    """

    def __init__(self, lo, hi, grace=10, strength=1.3):
        self.lo, self.hi = int(lo), int(hi)
        self.grace, self.strength = int(grace), float(strength)

    def __call__(self, input_ids, scores):
        last = input_ids[:, -1]
        eq = (input_ids == last.unsqueeze(1)).to(torch.int8)
        run = eq.flip(1).cumprod(dim=1).sum(dim=1)
        hit = ((last >= self.lo) & (last < self.hi) & (run >= self.grace)).nonzero().flatten()
        if hit.numel():
            penalty = (run[hit] - self.grace + 1).to(scores.dtype) * self.strength
            scores[hit, last[hit]] -= penalty
        return scores


def build_logits_processors(token_map, temperature, top_k, top_p, min_p,
                            repetition_penalty=1.1, repetition_window=None,
                            token_run_penalty=1.3, token_run_grace=10,
                            prompt_length=0):
    """Assemble the decode-time processor list.

    Penalties come before the warpers so that truncation sees already-penalised
    scores; that ordering is part of what the decode presets were tuned against.
    """
    lo, hi = token_map.audio_id_base, token_map.audio_id_end
    silence_id = lo + SILENCE_TOKEN
    processors = LogitsProcessorList()

    if token_run_penalty:
        processors.append(TokenRunPenalty(lo, hi, token_run_grace, token_run_penalty))
    if repetition_penalty != 1.0:
        exempt = {token_map.speech_end_id, token_map.stop_id, silence_id}
        processors.append(RepetitionPenaltyExempt(
            repetition_penalty, exempt, window=repetition_window, prompt_length=prompt_length))
    if temperature != 1.0:
        processors.append(TemperatureLogitsWarper(float(temperature)))
    if top_k:
        processors.append(TopKLogitsWarper(int(top_k)))
    if top_p < 1.0:
        processors.append(TopPLogitsWarper(float(top_p)))
    if min_p > 0.0:
        processors.append(MinPLogitsWarper(float(min_p)))
    return processors


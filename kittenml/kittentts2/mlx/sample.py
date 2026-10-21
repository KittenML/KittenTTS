"""KittenTTS 2's decode-time sampling, in MLX.

A port of `logits.py` and of the transformers warpers it composes, in the same
order, so a generation on this backend is drawn from the same distribution the
decode presets were tuned against:

    token-run penalty -> repetition penalty -> temperature
                      -> top_k -> top_p -> min_p -> sample

Every step runs on the GPU over the whole vocabulary, as the Torch backend's does;
the history is kept as a device array so no token ever round-trips through Python
to be penalised.
"""

import mlx.core as mx

FILTER_VALUE = -float("inf")


class TokenRunPenalty:
    """Escalating penalty on a codec token repeated past `grace` times in a row.

    The trailing run length is a reversed cumulative product over the equality
    mask, exactly as in `logits.TokenRunPenalty`.
    """

    def __init__(self, lo, hi, grace=10, strength=1.3):
        self.lo, self.hi = int(lo), int(hi)
        self.grace, self.strength = int(grace), float(strength)

    def __call__(self, history, scores):
        last = history[:, -1]
        equal = (history == last[:, None]).astype(mx.int32)
        run = mx.cumprod(equal[:, ::-1], axis=1).sum(axis=1)
        hit = (last >= self.lo) & (last < self.hi) & (run >= self.grace)
        penalty = mx.where(hit, (run - self.grace + 1).astype(scores.dtype) * self.strength, 0)
        at_last = mx.arange(scores.shape[-1]) == last[:, None]
        return mx.where(at_last, scores - penalty[:, None], scores)


class RepetitionPenaltyExempt:
    """Repetition penalty that never penalises the sequence-terminating tokens.

    See `logits.RepetitionPenaltyExempt`: `<speech_end>` and `[STOP]` keep their
    original scores, and `window` restricts the scan to the last N generated
    positions.
    """

    def __init__(self, penalty, exempt_ids, window=None, prompt_length=0):
        self.penalty = float(penalty)
        self.exempt_ids = sorted({int(i) for i in exempt_ids})
        self.window = int(window) if window else None
        self.prompt_length = int(prompt_length)
        self._exempt = None

    def __call__(self, history, scores):
        if self.window is not None:
            history = history[:, max(self.prompt_length, history.shape[1] - self.window):]
        if history.shape[1] == 0:
            return scores
        got = mx.take_along_axis(scores, history, axis=-1)
        got = mx.where(got < 0, got * self.penalty, got / self.penalty)
        out = mx.put_along_axis(scores, history, got, axis=-1)
        if self.exempt_ids:
            if self._exempt is None:
                index = mx.array(self.exempt_ids)[None, :]
                self._exempt = mx.zeros((1, scores.shape[-1]), mx.bool_)
                self._exempt = mx.put_along_axis(self._exempt, index,
                                                 mx.ones((1, index.shape[1]), mx.bool_),
                                                 axis=-1)
            out = mx.where(self._exempt, scores, out)
        return out


def _temperature(temperature):
    return lambda history, scores: scores / temperature


def _top_k(top_k):
    """Keep the `top_k` highest scores, as `TopKLogitsWarper` does."""
    def apply(history, scores):
        kth = mx.sort(scores, axis=-1)[..., -min(top_k, scores.shape[-1]), None]
        return mx.where(scores < kth, FILTER_VALUE, scores)
    return apply


def _top_p(top_p):
    """Nucleus truncation, following `TopPLogitsWarper` exactly.

    Ascending order, dropping the tail whose cumulative probability stays at or
    below `1 - top_p`, and always keeping the single most likely token.
    """
    def apply(history, scores):
        order = mx.argsort(scores, axis=-1)
        ordered = mx.take_along_axis(scores, order, axis=-1)
        cumulative = mx.cumsum(mx.softmax(ordered, axis=-1), axis=-1)
        drop = cumulative <= (1.0 - top_p)
        # min_tokens_to_keep=1: the last column is the most likely token.
        drop[..., -1:] = False
        return mx.put_along_axis(scores, order, mx.where(drop, FILTER_VALUE, ordered),
                                 axis=-1)
    return apply


def _min_p(min_p):
    """Drop tokens below `min_p` of the most likely one, as `MinPLogitsWarper` does."""
    def apply(history, scores):
        probabilities = mx.softmax(scores, axis=-1)
        threshold = min_p * mx.max(probabilities, axis=-1, keepdims=True)
        drop = probabilities < threshold
        keep_best = mx.arange(scores.shape[-1]) == mx.argmax(scores, axis=-1)[:, None]
        return mx.where(drop & ~keep_best, FILTER_VALUE, scores)
    return apply


def build_logits_processors(token_map, temperature, top_k, top_p, min_p,
                            repetition_penalty=1.1, repetition_window=None,
                            token_run_penalty=1.3, token_run_grace=10,
                            prompt_length=0):
    """The MLX counterpart of `logits.build_logits_processors`, in the same order."""
    lo, hi = token_map.audio_id_base, token_map.audio_id_end
    processors = []

    if token_run_penalty:
        processors.append(TokenRunPenalty(lo, hi, token_run_grace, token_run_penalty))
    if repetition_penalty != 1.0:
        exempt = {token_map.speech_end_id, token_map.stop_id}
        processors.append(RepetitionPenaltyExempt(
            repetition_penalty, exempt, window=repetition_window,
            prompt_length=prompt_length))
    if temperature != 1.0:
        processors.append(_temperature(float(temperature)))
    if top_k:
        processors.append(_top_k(int(top_k)))
    if top_p < 1.0:
        processors.append(_top_p(float(top_p)))
    if min_p > 0.0:
        processors.append(_min_p(float(min_p)))
    return processors


def sample(history, logits, processors):
    """Apply every processor, then draw one token from the result."""
    for processor in processors:
        logits = processor(history, logits)
    return mx.random.categorical(logits, axis=-1)

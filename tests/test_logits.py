import pytest
import torch

from kittenml.kittentts2.logits import build_logits_processors
from kittenml.kittentts2.tokens import TokenMap

TOKEN_MAP = TokenMap(audio_id_base=100, num_audio_tokens=6561, speech_start_id=1,
                     speech_end_id=2, text_start_id=3, text_end_id=4, start_id=5,
                     stop_id=6)
SILENCE = TOKEN_MAP.audio_id_base + 4299


def test_repetition_penalty_covers_silence_but_not_terminators():
    """Silence runs are the main checkpoint's stuck state; only the stop tokens are exempt."""
    processors = build_logits_processors(TOKEN_MAP, temperature=1.0, top_k=0, top_p=1.0,
                                         min_p=0.0, repetition_penalty=1.1,
                                         repetition_window=50, token_run_penalty=0,
                                         prompt_length=1)
    history = torch.tensor([[TOKEN_MAP.start_id, SILENCE, TOKEN_MAP.speech_end_id,
                             TOKEN_MAP.stop_id, 150]])
    scores = torch.full((1, TOKEN_MAP.audio_id_end), 2.0)
    out = processors(history, scores.clone())
    assert out[0, SILENCE].item() == pytest.approx(2.0 / 1.1)
    assert out[0, 150].item() == pytest.approx(2.0 / 1.1)
    assert out[0, TOKEN_MAP.speech_end_id] == 2.0
    assert out[0, TOKEN_MAP.stop_id] == 2.0

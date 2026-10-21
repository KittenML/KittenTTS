"""MLX backend: exact weight conversion and sampler parity; no model download."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from kittenml import KittenTTS
from kittenml.kittentts2.ternary import pack_state_dict
from kittenml.kittentts2.tokens import TokenMap

mx = pytest.importorskip("mlx.core", reason="the MLX backend is an optional extra")

from kittenml.kittentts2.mlx import loader, sample  # noqa: E402

# The real codec vocabulary size, so codec offsets such as the silence token
# (4299) fall inside the audio range as they do in a real checkpoint.
TOKEN_MAP = TokenMap(audio_id_base=100, num_audio_tokens=6561, speech_start_id=1,
                     speech_end_id=2, text_start_id=3, text_end_id=4, start_id=5,
                     stop_id=6)


def _write_packed(tmp_path, state):
    lm = tmp_path / "lm"
    lm.mkdir(exist_ok=True)
    tensors, metadata = pack_state_dict(state)
    save_file(tensors, lm / "model-ternary.safetensors", metadata=metadata)
    return lm / "model-ternary.safetensors"


def test_ternary_weights_convert_to_mlx_4bit_exactly(tmp_path):
    """The whole point of the backend: 4 bits, and not one weight changed."""
    generator = torch.Generator().manual_seed(0)
    scales = torch.tensor([[0.125], [0.5], [2.0], [0.0625]])
    ternary = (torch.randint(-1, 2, (4, 256), generator=generator).reshape(4, 2, 128)
               * scales.unsqueeze(1)).reshape(4, 256).to(torch.bfloat16)
    dense = torch.tensor([0.3, -1.7], dtype=torch.bfloat16)
    path = _write_packed(tmp_path, {"model.layer.weight": ternary, "spk_proj.0.bias": dense})

    weights, quantised, group_size = loader.load_lm_weights(path, dtype=mx.bfloat16)
    assert group_size == 128
    codes, scale, bias = quantised["model.layer.weight"]
    # 4 bits per weight: eight of them in each uint32 word.
    assert codes.shape == (4, 256 // 8) and codes.dtype == mx.uint32
    rebuilt = mx.dequantize(codes, scale, bias, group_size=group_size, bits=4)
    assert np.array_equal(np.array(rebuilt.astype(mx.float32)), ternary.float().numpy())
    assert np.array_equal(np.array(weights["spk_proj.0.bias"].astype(mx.float32)),
                          dense.float().numpy())
    # And every code really is one of the three a ternary weight can take.
    assert set(np.unique(np.array(mx.dequantize(codes, mx.ones_like(scale),
                                                mx.zeros_like(bias), group_size=group_size,
                                                bits=4).astype(mx.int32)))) <= {0, 1, 2}


def test_unpacked_checkpoints_are_refused_with_a_usable_message(tmp_path):
    path = tmp_path / "plain.safetensors"
    save_file({"model.layer.weight": torch.zeros(2, 128, dtype=torch.bfloat16)}, path)
    with pytest.raises(ValueError, match="packed"):
        loader.load_lm_weights(path)


@pytest.mark.parametrize("settings", [
    {"temperature": 0.8, "top_k": 50, "top_p": 0.8, "min_p": 0.0},
    {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.05},
    {"temperature": 0.9, "top_k": 8, "top_p": 0.95, "min_p": 0.02},
])
def test_sampling_pipeline_matches_the_torch_one(settings):
    """Same distribution, not merely the same shape: the presets were tuned on it."""
    from kittenml.kittentts2 import logits as torch_logits

    rng = np.random.default_rng(0)
    vocabulary = TOKEN_MAP.audio_id_end + 5
    history = np.concatenate([
        rng.integers(0, 100, 12),                       # prompt-side text tokens
        rng.integers(TOKEN_MAP.audio_id_base, TOKEN_MAP.audio_id_end, 30),
        np.full(14, TOKEN_MAP.audio_id_base + 7),       # a run, for TokenRunPenalty
    ]).astype(np.int64)[None]
    scores = rng.normal(size=(1, vocabulary)).astype(np.float32) * 4

    common = dict(repetition_penalty=1.1, repetition_window=20, token_run_penalty=1.3,
                  token_run_grace=10, prompt_length=12, **settings)
    expected = torch_logits.build_logits_processors(TOKEN_MAP, **common)(
        torch.from_numpy(history), torch.from_numpy(scores.copy()))
    actual = mx.array(scores)
    for processor in sample.build_logits_processors(TOKEN_MAP, **common):
        actual = processor(mx.array(history), actual)

    np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=1e-5, atol=1e-5)


def test_sampling_is_reproducible_and_stays_inside_the_filtered_set():
    processors = sample.build_logits_processors(
        TOKEN_MAP, temperature=0.8, top_k=4, top_p=1.0, min_p=0.0,
        repetition_penalty=1.0, token_run_penalty=0)
    scores = mx.array(np.linspace(-5, 5, TOKEN_MAP.audio_id_end + 5,
                                  dtype=np.float32)[None])
    history = mx.array(np.array([[TOKEN_MAP.start_id]]))
    allowed = set(range(scores.shape[-1] - 4, scores.shape[-1]))

    mx.random.seed(11)
    drawn = [int(sample.sample(history, scores, processors).item()) for _ in range(20)]
    assert set(drawn) <= allowed
    mx.random.seed(11)
    assert [int(sample.sample(history, scores, processors).item()) for _ in range(20)] == drawn


def test_backend_selection_and_its_refusals(tmp_path):
    from kittenml.kittentts2.model import KittenTTS2, _resolve_backend

    config = {"token_map": asdict(TOKEN_MAP), "type": "KITTEN2"}
    with mock.patch("kittenml.kittentts2.mlx.backend.MLXBackend") as backend, \
         mock.patch("kittenml.kittentts2.model.S3Codec"), \
         mock.patch("kittenml.kittentts2.model.SpeakerEncoder"), \
         mock.patch("kittenml.kittentts2.model.ReferenceTranscriber"):
        model = KittenTTS2(tmp_path, config, backend="mlx", device="cpu")
    backend.assert_called_once_with(model, None)
    assert model.backend == "mlx"

    assert _resolve_backend("auto") == "mlx"          # this test only runs where mlx imports
    assert _resolve_backend(None) == "torch"
    with pytest.raises(ValueError, match="unknown backend"):
        _resolve_backend("metal")

    with pytest.raises(ValueError, match="mlx_options requires"):
        KittenTTS("KittenML/kitten-tts-2", mlx_options={"threads": 4})


def test_emb4_weights_are_refused_rather_than_silently_dequantised(tmp_path):
    from kittenml.kittentts2.mlx.backend import MLXBackend

    owner = SimpleNamespace(token_map=TOKEN_MAP, repo_dir=tmp_path, config={},
                            weights="emb4", device="cpu",
                            _weights_path=lambda name: tmp_path / name)
    with pytest.raises(ValueError, match="weights='packed'"):
        MLXBackend(owner)


def test_speaker_embedding_is_projected_once_per_reference():
    from kittenml.kittentts2.model import KittenTTS2

    owner = SimpleNamespace(backend="mlx", model=mock.Mock(), device="cpu",
                            dtype=torch.float32)
    embedding = np.arange(8, dtype=np.float32)
    result = KittenTTS2._speaker_embedding(owner, embedding)
    assert result is owner.model.speaker_embedding.return_value
    np.testing.assert_array_equal(owner.model.speaker_embedding.call_args[0][0], embedding)

    owner.backend = "torch"
    tensor = KittenTTS2._speaker_embedding(owner, embedding)
    assert tensor.shape == (1, 8) and tensor.dtype is torch.float32

"""Small checkpoints and sampler parity; no GPU or model download required."""

import json
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
from kittenml.kittentts2.weights import load_lm_weights


def test_default_backend_keeps_native_loading(tmp_path):
    from kittenml.kittentts2.model import KittenTTS2
    config = {"token_map": asdict(TokenMap(5, 10, 1, 2, 3, 4, 0, 15))}
    with mock.patch.object(KittenTTS2, "_load_language_model") as load, \
         mock.patch("kittenml.kittentts2.model.S3Codec"), \
         mock.patch("kittenml.kittentts2.model.SpeakerEncoder"), \
         mock.patch("kittenml.kittentts2.model.ReferenceTranscriber"):
        model = KittenTTS2(tmp_path, config, device="cpu")
    load.assert_called_once_with()
    assert model.backend == "torch" and model.device == "cpu"


def test_selected_packed_weights_are_exact_and_full_ignores_packed_files(tmp_path):
    lm = tmp_path / "lm"
    lm.mkdir()
    state = {"model.layer.weight": torch.arange(256).remainder(3).reshape(2, 128)
             .sub(1).to(torch.bfloat16) * 0.125,
             "spk_proj.0.bias": torch.tensor([0.7, -0.2], dtype=torch.bfloat16)}
    save_file(state, lm / "model.safetensors")
    tensors, metadata = pack_state_dict(state)
    save_file(tensors, lm / "model-ternary.safetensors", metadata=metadata)
    config = {"lm_packed": "lm/model-ternary.safetensors"}
    for variant in ("packed", "full"):
        result = load_lm_weights(tmp_path, config, variant)
        assert result.keys() == state.keys()
        assert all(torch.equal(result[name], tensor) for name, tensor in state.items())
    with pytest.raises(ValueError, match="Unknown weights"):
        load_lm_weights(tmp_path, config, "invalid")
    with pytest.raises(ValueError, match="no 'emb4'"):
        load_lm_weights(tmp_path, config, "emb4")


def test_native_full_loader_preserves_explicit_checkpoint_directory(tmp_path):
    from kittenml.kittentts2.model import KittenTTS2

    lm_dir = tmp_path / "custom-lm"
    lm_dir.mkdir()
    expected = {"model.weight": torch.tensor([0.25, -0.5], dtype=torch.bfloat16)}
    save_file(expected, lm_dir / "model.safetensors")
    owner = SimpleNamespace(repo_dir=tmp_path, config={}, weights="full")
    loaded = KittenTTS2._load_lm_weights(owner, lm_dir)
    assert loaded.keys() == expected.keys()
    torch.testing.assert_close(loaded["model.weight"], expected["model.weight"], rtol=0, atol=0)


def test_shared_speaker_head_matches_native_prefill_and_leaves_decode_unchanged():
    from kittenml.kittentts2.speaker import attach_speaker_head, load_speaker_head

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.embedding = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids=None, inputs_embeds=None, cache_position=None):
            return inputs_embeds

    projection = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.LayerNorm(4))
    state = {"spk_proj." + k: v for k, v in projection.state_dict().items()}
    model = attach_speaker_head(TinyLM(), "unused", "cpu", torch.float32,
                                 spk_dim=2, state=state)
    head = load_speaker_head(4, "unused", "cpu", torch.float32, spk_dim=2, state=state)
    speaker = torch.tensor([[0.5, -0.3]])
    model._pending_speaker_embedding = speaker
    ids = torch.tensor([[0, 3, 1]])
    original = model.embedding(ids)
    prefill = model(input_ids=ids, cache_position=torch.tensor([0]))
    torch.testing.assert_close(prefill[:, 0], head(speaker))
    torch.testing.assert_close(prefill[:, 1:], original[:, 1:])
    decode = model(input_ids=ids, cache_position=torch.tensor([3]))
    torch.testing.assert_close(decode, original)
    with pytest.raises(ValueError, match="no spk_proj"):
        load_speaker_head(4, "unused", "cpu", torch.float32, spk_dim=2, state={})


def test_emb4_unpacking_is_independent_of_vllm_default_dtype():
    from kittenml.kittentts2.tl2_emb4 import dequant_embedding
    tensors = {
        "emb4/tier_map": torch.zeros(2, dtype=torch.uint8),
        "emb4/t0_codes": torch.zeros(1, dtype=torch.uint8),
        "emb4/t0_centers": torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32),
        "emb4/t0_s": torch.ones(2, dtype=torch.float32),
        "emb4/t0_t": torch.ones(2, dtype=torch.float32),
    }
    metadata = {"V": "2", "D": "2", "Ks": "[2]"}
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        expected = dequant_embedding(tensors.__getitem__, metadata)
        torch.set_default_dtype(torch.bfloat16)
        actual = dequant_embedding(tensors.__getitem__, metadata)
    finally:
        torch.set_default_dtype(previous)
    assert actual.dtype == torch.bfloat16 and torch.equal(actual, expected)


def _emb4_tensors(vocab=37, dim=16, seed=0):
    """A three-tier emb4 table (8-bit and two 4-bit books) and its full-matrix decode.

    Small integer centres, power-of-two scales and dim=16 (Hadamard entries ±1/4)
    keep every product and sum exact, so the expected matrix does not depend on
    how a matrix multiply orders its additions.
    """
    import math
    generator = torch.Generator().manual_seed(seed)
    sizes = [256, 16, 16]
    tier_map = torch.randint(0, len(sizes), (vocab,), generator=generator).to(torch.uint8)
    tensors = {"emb4/tier_map": tier_map}
    hadamard = torch.tensor([[1.0]])
    for _ in range(int(math.log2(dim))):
        hadamard = torch.cat([torch.cat([hadamard, hadamard], 1),
                              torch.cat([hadamard, -hadamard], 1)], 0)
    hadamard /= math.sqrt(dim)
    expected = torch.empty(vocab, dim)
    for tier, size in enumerate(sizes):
        rows = torch.nonzero(tier_map == tier).flatten()
        codes = torch.randint(0, size, (rows.numel(), dim), generator=generator)
        flat = codes.flatten().to(torch.uint8)
        if size == 16:
            flat = flat[0::2] | (flat[1::2] << 4)
        centres = torch.randint(-8, 9, (size, dim), generator=generator).to(torch.float16)
        row_scale = 2.0 ** torch.randint(-3, 3, (rows.numel(),), generator=generator)
        col_scale = 2.0 ** torch.randint(-3, 3, (dim,), generator=generator)
        tensors.update({f"emb4/t{tier}_codes": flat, f"emb4/t{tier}_centers": centres,
                        f"emb4/t{tier}_s": row_scale.to(torch.float16),
                        f"emb4/t{tier}_t": col_scale.to(torch.float16)})
        rotated = torch.gather(centres.float().T, 1, codes.T).T
        expected[rows] = (rotated @ hadamard.T) * col_scale[None, :] * row_scale[:, None]
    metadata = {"V": str(vocab), "D": str(dim), "Ks": json.dumps(sizes)}
    return tensors, metadata, expected.to(torch.bfloat16)


@pytest.mark.parametrize("chunk, lookup_chunk", [(1, 1), (5, 2), (8, 3), (1 << 15, 1 << 12)])
def test_emb4_embedding_is_the_same_decoded_in_chunks(chunk, lookup_chunk):
    from kittenml.kittentts2.tl2_emb4 import dequant_embedding
    tensors, metadata, expected = _emb4_tensors()
    actual = dequant_embedding(tensors.__getitem__, metadata, chunk=chunk,
                               lookup_chunk=lookup_chunk)
    assert actual.dtype == torch.bfloat16 and torch.equal(actual, expected)


def _write_emb4_checkpoint(path):
    tensors, metadata, embedding = _emb4_tensors()
    # One TL2 block (32 output rows x 128 inputs): random trits, finite group scales.
    tl2 = torch.randint(0, 256, (1, 1, 928), dtype=torch.uint8)
    tl2[..., :64] = (torch.rand(32) + 0.5).to(torch.float16).view(torch.uint8)
    tensors = {**tensors,
               "tl2/model.layers.0.self_attn.o_proj.weight": tl2,
               "fp/model.norm.weight": torch.rand(16, dtype=torch.bfloat16),
               "fp/spk_proj.0.weight": torch.rand(16, 4, dtype=torch.bfloat16)}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata={"schema": "kitten_ternary_tl2_emb4_v1",
                                       "embedding_metadata": json.dumps(metadata)})
    return embedding


def test_emb4_reader_unpacks_only_the_tensors_asked_for(tmp_path):
    from kittenml.kittentts2 import tl2_emb4
    from kittenml.kittentts2.weights import iter_lm_weights

    config = {"lm_emb4": "lm/model-tl2-emb4.safetensors"}
    embedding = _write_emb4_checkpoint(tmp_path / config["lm_emb4"])
    full = load_lm_weights(tmp_path, config, "emb4")
    assert set(full) == {"model.layers.0.self_attn.o_proj.weight", "model.norm.weight",
                         "spk_proj.0.weight", "model.embed_tokens.weight"}
    assert full["model.layers.0.self_attn.o_proj.weight"].shape == (32, 128)
    assert torch.equal(full["model.embed_tokens.weight"], embedding)

    # The main process asks for the speaker head only (and the embedding): nothing
    # else may be unpacked.
    with mock.patch.object(tl2_emb4, "unpack_tl2", side_effect=AssertionError), \
         mock.patch.object(tl2_emb4, "dequant_embedding", side_effect=AssertionError):
        head = load_lm_weights(tmp_path, config, "emb4",
                               include=lambda name: name.startswith("spk_proj."))
    assert list(head) == ["spk_proj.0.weight"]
    assert torch.equal(head["spk_proj.0.weight"], full["spk_proj.0.weight"])

    # The engine side gets the same tensors one at a time, not a dict.
    streamed = iter_lm_weights(tmp_path, config, "emb4")
    assert not isinstance(streamed, dict)
    streamed = dict(streamed)
    assert set(streamed) == set(full)
    assert all(torch.equal(streamed[name], full[name]) for name in full)
    with pytest.raises(FileNotFoundError):
        iter_lm_weights(tmp_path, {"lm_emb4": "lm/missing.safetensors"}, "emb4")


def test_optional_backend_errors_before_model_download():
    resolved = (None, "example/repo", {"type": "KITTEN2"})
    with mock.patch("kittenml.get_model.resolve_repo", return_value=resolved), \
         mock.patch("kittenml.kittentts2.vllm.backend.importlib.util.find_spec", return_value=None), \
         mock.patch("kittenml.get_model.snapshot_download") as download:
        with pytest.raises(ImportError, match=r"kittenml\[vllm\]"):
            KittenTTS("example/repo", backend="vllm")
        download.assert_not_called()
    with mock.patch("kittenml.get_model.resolve_repo", return_value=(None, "legacy", {"type": "ONNX1"})):
        with pytest.raises(ValueError, match="KittenTTS 2"):
            KittenTTS("legacy", backend="vllm")


def test_vllm_options_require_backend():
    with mock.patch("kittenml.get_model.resolve_repo", return_value=(None, "model", {"type": "KITTEN2"})):
        with pytest.raises(ValueError, match="requires backend"):
            KittenTTS("model", vllm_options={})


def test_explicit_default_options_do_not_break_legacy_models():
    resolved = (None, "legacy", {"type": "ONNX1"})
    with mock.patch("kittenml.get_model.resolve_repo", return_value=resolved), \
         mock.patch("kittenml.kittentts_legacy.model.build_onnx_model") as build:
        model = KittenTTS("legacy", vllm_options=None)
    assert model.model is build.return_value
    build.assert_called_once_with(resolved[2], None, "legacy", None, None)


def test_environment_rejects_unsupported_devices_before_engine_start():
    from kittenml.kittentts2.vllm.backend import validate_environment
    with mock.patch("kittenml.kittentts2.vllm.backend.sys.platform", "win32"):
        with pytest.raises(RuntimeError, match="Linux"):
            validate_environment()
    with mock.patch("kittenml.kittentts2.vllm.backend.importlib.util.find_spec", return_value=object()), \
         mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA GPU"):
            validate_environment()
    with mock.patch("kittenml.kittentts2.vllm.backend.importlib.util.find_spec", return_value=object()), \
         mock.patch("torch.cuda.is_available", return_value=True):
        with pytest.raises(RuntimeError, match="CUDA GPU"):
            validate_environment("cpu")
        with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
            validate_environment("cuda:1")


@pytest.mark.parametrize("key, default, override", [
    ("VLLM_USE_FLASHINFER_SAMPLER", "0", "1"),
    ("VLLM_LOGGING_LEVEL", "WARNING", "INFO"),
    ("TQDM_DISABLE", "1", "0"),
])
def test_engine_environment_preserves_user_choice_and_restores_after_failure(
        monkeypatch, key, default, override):
    import os
    from kittenml.kittentts2.vllm.backend import _engine_environment
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError):
        with _engine_environment():
            assert os.environ[key] == default
            raise RuntimeError("engine failed")
    assert key not in os.environ
    monkeypatch.setenv(key, override)
    with _engine_environment():
        assert os.environ[key] == override
    assert os.environ[key] == override


def test_engine_receives_quiet_logging_config_without_changing_native_options(monkeypatch, tmp_path):
    pytest.importorskip("vllm")
    import os
    from kittenml.kittentts2.vllm.backend import VLLMBackend

    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    owner = SimpleNamespace(device="cuda", repo_dir=tmp_path, config={}, weights="packed",
                            token_map=None, _load_lm_weights=mock.Mock(return_value={
                                "model.embed_tokens.weight": torch.zeros(2, 4)}))

    def create_engine(**kwargs):
        assert kwargs["logging_config"].log_level == "WARNING"
        assert os.environ["VLLM_LOGGING_LEVEL"] == "WARNING"
        assert os.environ["TQDM_DISABLE"] == "1"
        assert kwargs["enable_prompt_embeds"] and not kwargs["enable_prefix_caching"]
        assert kwargs["model_loader_extra_config"]["weights"] == "packed"
        return mock.sentinel.engine

    with mock.patch("kittenml.kittentts2.vllm.backend.validate_environment"), \
         mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)), \
         mock.patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(
             model_type="qwen3", hidden_size=4)), \
         mock.patch("transformers.AutoTokenizer.from_pretrained"), \
         mock.patch("kittenml.kittentts2.speaker.load_speaker_head"), \
         mock.patch("vllm.LLM", side_effect=create_engine):
        backend = VLLMBackend(owner)
    assert backend.engine is mock.sentinel.engine
    assert "VLLM_LOGGING_LEVEL" not in os.environ


def test_vllm_loader_filters_speaker_weights_and_rejects_missing_parameters():
    pytest.importorskip("vllm")
    from kittenml.kittentts2.vllm.loader import KittenModelLoader
    loader = object.__new__(KittenModelLoader)
    loader.load_config = SimpleNamespace(model_loader_extra_config={
        "repo_dir": "unused", "config": {}, "weights": "packed"})

    class TinyModel:
        def load_weights(self, weights):
            self.loaded = dict(weights)
            return set(self.loaded)

        def named_parameters(self):
            return [("model.weight", None)]

        def named_modules(self):
            return []

    model = TinyModel()
    state = {"model.weight": torch.ones(2), "spk_proj.0.bias": torch.zeros(2)}
    with mock.patch("kittenml.kittentts2.vllm.loader.iter_lm_weights", return_value=state.items()):
        loader.load_weights(model, None)
    assert set(model.loaded) == {"model.weight"}
    with mock.patch("kittenml.kittentts2.vllm.loader.iter_lm_weights", return_value={}.items()):
        with pytest.raises(ValueError, match="missing from the checkpoint"):
            loader.load_weights(model, None)


@pytest.mark.parametrize("window", [None, 3])
def test_sampling_matches_native_when_requests_reorder_and_slots_are_reused(window):
    pytest.importorskip("vllm")
    from kittenml.kittentts2.logits import build_logits_processors
    from kittenml.kittentts2.vllm.logits import KittenLogitsProcessor

    tm = TokenMap(audio_id_base=5, num_audio_tokens=5000, speech_start_id=1,
                  speech_end_id=2, text_start_id=3, text_end_id=4, start_id=0, stop_id=5005)
    sampling = dict(temperature=0.8, top_k=60, top_p=0.9, min_p=0.05,
                    repetition_penalty=1.1, repetition_window=window,
                    token_run_penalty=1.3, token_run_grace=3)
    original = {0: [0, 3, 7, 1], 1: [0, 3, 8, 9, 1]}
    history = {0: original[0] + [7, 7, 7, 7],
               1: original[1] + [tm.audio_id_base + 4299] * 4}
    buf = torch.zeros(2, 16, dtype=torch.long)
    # Embedding prompts may have placeholder IDs inside vLLM. The processor must
    # restore the real prompt before applying full-history repetition penalties.
    for slot in history:
        buf[slot, len(original[slot]):len(history[slot])] = torch.tensor(history[slot][len(original[slot]):])
    states = SimpleNamespace(device=torch.device("cpu"), all_token_ids=SimpleNamespace(gpu=buf))
    processor = KittenLogitsProcessor(None, states)
    for slot in original:
        params = SimpleNamespace(extra_args={"kittenml": {"prompt": original[slot],
                                  "token_map": asdict(tm), "sampling": sampling}})
        assert processor.add_request(slot, params)
    ctx = SimpleNamespace(idx_mapping_np=np.array([1, 0]),
                          seq_lens_upper_bound_np=np.array([len(history[1]), len(history[0])]))
    scores = torch.randn(2, 5006, generator=torch.Generator().manual_seed(7))
    expected = scores.clone()
    for row, slot in enumerate([1, 0]):
        native = build_logits_processors(tm, **sampling, prompt_length=len(original[slot]))
        expected[row:row + 1] = native(torch.tensor([history[slot]]), expected[row:row + 1])
    torch.testing.assert_close(processor.apply(scores, ctx), expected)
    assert not processor.add_request(0, SimpleNamespace(extra_args=None))
    assert 0 not in processor.requests


def test_sampler_is_neutral_and_speaker_replaces_only_first_embedding():
    pytest.importorskip("vllm")
    from kittenml.kittentts2.vllm.backend import VLLMBackend

    backend = object.__new__(VLLMBackend)
    backend.max_model_len = 32
    backend.token_map = TokenMap(5, 10, 1, 2, 3, 4, 0, 15)
    backend.embedding = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    backend.spk_proj = torch.nn.Identity()
    settings = dict(temperature=0.7, top_k=5, top_p=0.8, min_p=0.02)
    advanced = dict(seed=42, repetition_penalty=1.2, repetition_window=3,
                    token_run_penalty=1.5, token_run_grace=4)
    request, params = backend.prepare_request([0, 3, 1], torch.ones(1, 4), settings, 8, advanced)
    torch.testing.assert_close(request["prompt_embeds"][0], torch.ones(4))
    torch.testing.assert_close(request["prompt_embeds"][1:], backend.embedding[[3, 1]])
    assert params.temperature == params.top_p == params.repetition_penalty == 1.0
    assert params.top_k == -1 and params.min_p == 0.0 and params.seed == 42
    assert params.stop_token_ids == [2, 15] and params.ignore_eos
    assert params.extra_args["kittenml"]["sampling"]["temperature"] == 0.7
    with pytest.raises(ValueError, match="token budget"):
        backend.prepare_request([0, 3, 1], torch.ones(1, 4), settings, 30, advanced)


# ── Exact W4A16 (vllm_options={"quantization": "w4a16"}) ───────────────────────

_PROJECTIONS = {"self_attn.q_proj": (256, 256), "self_attn.k_proj": (128, 256),
                "self_attn.v_proj": (128, 256), "self_attn.o_proj": (256, 256),
                "mlp.gate_proj": (384, 256), "mlp.up_proj": (384, 256),
                "mlp.down_proj": (256, 384)}


def _ternary(rows, cols, generator, low=1e-4):
    signs = torch.randint(-1, 2, (rows, cols), generator=generator)
    scale = (torch.rand(rows, cols // 128, generator=generator) * 0.5 + low).to(torch.bfloat16)
    scale[0, 0] = 0  # an all-zero group has a zero scale
    signs[0, :128] = 0
    return signs.to(torch.bfloat16) * scale.repeat_interleave(128, dim=1)


def _w4a16_checkpoint(tmp_path, layers=2, perturb=None, low=1e-4):
    """A small Qwen3-shaped LM written as both full bf16 and packed ternary."""
    generator = torch.Generator().manual_seed(0)
    state = {"model.embed_tokens.weight": torch.randn(32, 256, generator=generator).to(torch.bfloat16),
             "model.norm.weight": torch.rand(256, generator=generator).to(torch.bfloat16),
             "spk_proj.0.weight": torch.randn(256, 8, generator=generator).to(torch.bfloat16),
             "spk_proj.0.bias": torch.zeros(256, dtype=torch.bfloat16),
             "spk_proj.1.weight": torch.ones(256, dtype=torch.bfloat16),
             "spk_proj.1.bias": torch.zeros(256, dtype=torch.bfloat16)}
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        state[prefix + "input_layernorm.weight"] = torch.ones(256, dtype=torch.bfloat16)
        state[prefix + "self_attn.q_norm.weight"] = torch.ones(128, dtype=torch.bfloat16)
        for name, shape in _PROJECTIONS.items():
            state[prefix + name + ".weight"] = _ternary(*shape, generator, low)
    if perturb is not None:
        weight = state[perturb]
        weight[1, 5] = weight[1, :128].abs().max() / 2  # no longer ternary in that group
    lm = tmp_path / "lm"
    lm.mkdir()
    save_file(state, lm / "model.safetensors")
    tensors, metadata = pack_state_dict(state)
    save_file(tensors, lm / "model-ternary.safetensors", metadata=metadata)
    return state, {"lm_packed": "lm/model-ternary.safetensors"}


def _unpack_int4(packed, shape):
    """Inverse of w4a16.pack_int4: int32 words -> int8 values of `shape`."""
    from kittenml.kittentts2.vllm.w4a16 import NUM_BITS, OFFSET, PACK_FACTOR

    words = packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(PACK_FACTOR, dtype=torch.int64) * NUM_BITS
    codes = (words.unsqueeze(2) >> shifts) & ((1 << NUM_BITS) - 1)
    return (codes - OFFSET).reshape(packed.shape[0], -1)[:, :int(shape[1])].to(torch.int8)


def _reference_unpack():
    """compressed-tensors' own unpacker when installed (it ships with vLLM)."""
    try:
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    except ImportError:
        return _unpack_int4
    return lambda packed, shape: unpack_from_int32(packed, 4, torch.Size(shape))


@pytest.mark.parametrize("variant", ["packed", "full"])
def test_w4a16_dequantizes_bit_exactly_to_the_selected_weights(tmp_path, variant):
    from kittenml.kittentts2.vllm.w4a16 import bit_equal, is_projection, iter_w4a16

    state, config = _w4a16_checkpoint(tmp_path)
    expected = load_lm_weights(tmp_path, config, variant)
    produced = dict(iter_w4a16(tmp_path, config, variant))
    unpack = _reference_unpack()
    projections = [name for name in state if is_projection(name)]
    assert len(projections) == 14
    dense = {name for name in state if not is_projection(name) and not name.startswith("spk_proj.")}
    assert set(produced) == dense | {name.removesuffix(".weight") + suffix
                                     for name in projections
                                     for suffix in (".weight_packed", ".weight_scale",
                                                    ".weight_shape")}
    for name in dense:
        assert bit_equal(produced[name], expected[name])
    for name in projections:
        base = name.removesuffix(".weight")
        rows, cols = expected[name].shape
        packed = produced[base + ".weight_packed"]
        scale = produced[base + ".weight_scale"]
        shape = produced[base + ".weight_shape"]
        assert packed.dtype == torch.int32 and packed.shape == (rows, cols // 8)
        assert scale.dtype == torch.bfloat16 and scale.shape == (rows, cols // 128)
        assert shape.dtype == torch.int64 and shape.tolist() == [rows, cols]
        signs = unpack(packed, shape.tolist())
        assert set((signs.to(torch.int16) + 8).unique().tolist()) <= {7, 8, 9}
        weight = signs.to(torch.bfloat16) * scale.repeat_interleave(128, dim=1)
        assert bit_equal(weight, expected[name]), name


def test_w4a16_packed_and_full_sources_give_identical_tensors(tmp_path):
    from kittenml.kittentts2.vllm.w4a16 import bit_equal, iter_w4a16

    _, config = _w4a16_checkpoint(tmp_path)
    packed = dict(iter_w4a16(tmp_path, config, "packed"))
    full = dict(iter_w4a16(tmp_path, config, "full"))
    assert packed.keys() == full.keys()
    assert all(bit_equal(packed[k], full[k]) for k in packed)


def test_w4a16_int32_bit_order_matches_compressed_tensors():
    from kittenml.kittentts2.vllm.w4a16 import pack_int4

    # Codes 7,8,9,8,8,8,8,9: column 0 in bits 0-3 ... column 7 in bits 28-31.
    golden = pack_int4(torch.tensor([[-1, 0, 1, 0, 0, 0, 0, 1]], dtype=torch.int8))
    assert golden.dtype == torch.int32 and golden.tolist() == [[-1735882361]]  # 0x98888987
    assert pack_int4(torch.zeros(1, 8, dtype=torch.int8)).tolist() == [[0x88888888 - (1 << 32)]]
    values = torch.randint(-8, 8, (6, 256), generator=torch.Generator().manual_seed(1)).to(torch.int8)
    assert torch.equal(_unpack_int4(pack_int4(values), values.shape), values)
    with pytest.raises(ValueError, match="4 bits"):
        pack_int4(torch.full((1, 8), 8, dtype=torch.int8))
    helpers = pytest.importorskip("compressed_tensors.compressors.pack_quantized.helpers")
    assert torch.equal(pack_int4(values), helpers.pack_to_int32(values, 4))
    assert torch.equal(helpers.unpack_from_int32(pack_int4(values), 4, values.shape), values)


@pytest.mark.parametrize("variant", ["packed", "full"])
def test_w4a16_refuses_weights_that_are_not_exactly_ternary(tmp_path, variant):
    from kittenml.kittentts2.vllm.w4a16 import iter_w4a16

    _, config = _w4a16_checkpoint(tmp_path, perturb="model.layers.1.mlp.up_proj.weight")
    with pytest.raises(ValueError, match="model.layers.1.mlp.up_proj.weight is not exactly ternary"):
        list(iter_w4a16(tmp_path, config, variant))


def test_w4a16_refuses_scales_that_are_not_bfloat16(tmp_path):
    # Marlin reads the group scales in the activation dtype, which is bfloat16.
    from kittenml.kittentts2.vllm.w4a16 import iter_w4a16

    weight = torch.zeros(2, 128, dtype=torch.float32)
    weight[0, :2] = torch.tensor([0.5, -0.5])
    (tmp_path / "lm").mkdir()
    save_file({"model.layers.0.self_attn.o_proj.weight": weight}, tmp_path / "lm" / "model.safetensors")
    with pytest.raises(ValueError, match="o_proj.weight: group scales are torch.float32, not bfloat16"):
        list(iter_w4a16(tmp_path, {}, "full"))


def test_w4a16_accepts_emb4_and_packs_its_ternary_body_exactly(tmp_path):
    from kittenml.kittentts2.vllm import w4a16
    from kittenml.kittentts2.vllm.backend import check_options

    signs = torch.randint(-1, 2, (16, 256), dtype=torch.int8)
    scale = torch.rand(16, 2, dtype=torch.bfloat16) + 0.5
    weight = (signs.reshape(16, 2, 128).to(torch.float32)
              * scale.to(torch.float32).unsqueeze(-1)).reshape(16, 256).to(torch.bfloat16)
    embedding = torch.randn(8, 4, dtype=torch.bfloat16)
    state = {"model.layers.0.self_attn.o_proj.weight": weight,
             "model.embed_tokens.weight": embedding,
             "spk_proj.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}
    with mock.patch("kittenml.kittentts2.weights.iter_lm_weights",
                    return_value=iter(state.items())) as load:
        out = dict(w4a16.iter_w4a16(tmp_path, {"lm_emb4": "x"}, "emb4"))
    load.assert_called_once_with(tmp_path, {"lm_emb4": "x"}, "emb4")
    assert "spk_proj.weight" not in out
    assert torch.equal(out["model.embed_tokens.weight"], embedding)
    expected = dict(w4a16._from_dense("model.layers.0.self_attn.o_proj.weight", weight))
    for key, value in expected.items():
        assert torch.equal(out[key], value)

    check_options({"quantization": "w4a16"}, "emb4")
    with pytest.raises(ValueError, match="float16"):
        check_options({"dtype": "float16"}, "emb4")


def test_w4a16_targets_112_fused_modules_for_196_projections():
    # Qwen3-1.7B shape: 28 layers x 7 ternary projections. vLLM's own Qwen3 model
    # loads q/k/v into qkv_proj and gate/up into gate_up_proj (its
    # packed_modules_mapping), so 4 quantized modules per layer.
    pytest.importorskip("vllm")
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from kittenml.kittentts2.vllm.w4a16 import is_projection

    fused = {part: module for module, parts in Qwen3ForCausalLM.packed_modules_mapping.items()
             for part in parts}
    names = ["model.embed_tokens.weight", "model.norm.weight", "spk_proj.0.weight"]
    for layer in range(28):
        names += [f"model.layers.{layer}.{p}.weight" for p in _PROJECTIONS]
        names += [f"model.layers.{layer}.input_layernorm.weight",
                  f"model.layers.{layer}.self_attn.k_norm.weight"]
    projections = [name.removesuffix(".weight") for name in names if is_projection(name)]
    assert len(projections) == 196
    modules = {name.rsplit(".", 1)[0] + "." + fused.get(name.rsplit(".", 1)[1], name.rsplit(".", 1)[1])
               for name in projections}
    assert len(modules) == 112
    assert {m.rsplit(".", 1)[1] for m in modules} == {"qkv_proj", "o_proj", "gate_up_proj",
                                                      "down_proj"}


def test_vllm_options_validate_quantization_dtype_and_cache_size():
    from kittenml.kittentts2.vllm.backend import resolve_options

    engine, kitten = resolve_options(None)
    assert engine == dict(gpu_memory_utilization=0.5, max_model_len=4096, max_num_seqs=8,
                          enforce_eager=False)
    assert kitten == dict(quantization=None, dtype=None, kv_cache_memory_bytes=None)
    engine, kitten = resolve_options({"quantization": "w4a16", "dtype": "bfloat16",
                                      "kv_cache_memory_bytes": 1 << 30, "max_num_seqs": 4})
    assert engine["kv_cache_memory_bytes"] == 1 << 30 and engine["max_num_seqs"] == 4
    assert "quantization" not in engine and "dtype" not in engine
    assert kitten["quantization"] == "w4a16" and kitten["dtype"] == "bfloat16"
    for bad, match in [({"quantization": "w8a8"}, "quantization"),
                       ({"quantization": "W4A16"}, "quantization"),
                       ({"dtype": "fp16"}, "dtype"),
                       ({"dtype": "float32"}, "dtype"),
                       ({"kv_cache_memory_bytes": 0}, "kv_cache_memory_bytes"),
                       ({"kv_cache_memory_bytes": 1.5e9}, "kv_cache_memory_bytes"),
                       ({"kv_cache_memory_bytes": True}, "kv_cache_memory_bytes"),
                       ({"kv_cache_dtype": "fp8"}, "unknown vllm_options")]:
        with pytest.raises(ValueError, match=match):
            resolve_options(bad)


def test_float16_is_an_explicit_option():
    # float16 is the opt-in fallback for GPUs without bfloat16 (fp16.py); it is
    # never chosen unless asked for.
    from kittenml.kittentts2.vllm.backend import resolve_options

    for options in ({"dtype": "float16"}, {"dtype": "float16", "quantization": "w4a16"}):
        engine, kitten = resolve_options(options)
        assert kitten["dtype"] == "float16" and "dtype" not in engine
    assert resolve_options(None)[1]["dtype"] is None


def test_dtype_follows_hardware_and_gpus_without_bfloat16_need_the_float16_opt_in():
    from kittenml.kittentts2.vllm.backend import validate_environment

    with mock.patch("kittenml.kittentts2.vllm.backend.importlib.util.find_spec", return_value=object()), \
         mock.patch("torch.cuda.is_available", return_value=True):
        for capability in ((8, 0), (8, 6), (12, 0)):
            with mock.patch("torch.cuda.get_device_capability", return_value=capability):
                for dtype in (None, "bfloat16", "float16"):
                    validate_environment(dtype=dtype)
        with mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)):
            for dtype in (None, "bfloat16"):
                with pytest.raises(RuntimeError, match=r"SM75\) has no bfloat16.*opt in to the "
                                                       r"float16 fallback with vllm_options="
                                                       r"\{'dtype': 'float16'\}.*not the optimal"):
                    validate_environment(dtype=dtype)
            validate_environment(dtype="float16")
        for capability, name in (((7, 0), "SM70"), ((6, 1), "SM61")):
            with mock.patch("torch.cuda.get_device_capability", return_value=capability):
                for dtype in (None, "float16"):
                    with pytest.raises(RuntimeError, match=name + r"\) is not supported by the "
                                                           r"vLLM backend.*7.5 or newer"):
                        validate_environment(dtype=dtype)


def test_vllm_options_are_checked_before_model_download(monkeypatch):
    resolved = (None, "example/repo", {"type": "KITTEN2"})
    with mock.patch("kittenml.get_model.resolve_repo", return_value=resolved), \
         mock.patch("kittenml.kittentts2.vllm.backend.validate_environment"), \
         mock.patch("kittenml.get_model.snapshot_download") as download:
        for options, weights, match in [({"quantisation": "w4a16"}, None, "unknown vllm_options"),
                                        ({"dtype": "float16"}, "emb4", "float16.*emb4"),
                                        ({"dtype": "fp16"}, None, "'float16'")]:
            with pytest.raises(ValueError, match=match):
                KittenTTS("example/repo", backend="vllm", weights=weights, vllm_options=options)
        monkeypatch.setenv("VLLM_MARLIN_INPUT_DTYPE", "int8")
        with pytest.raises(RuntimeError, match="W4A8"):
            KittenTTS("example/repo", backend="vllm", vllm_options={"quantization": "w4a16"})
        download.assert_not_called()


def test_gpus_without_bfloat16_are_refused_before_model_download():
    resolved = (None, "example/repo", {"type": "KITTEN2"})
    with mock.patch("kittenml.get_model.resolve_repo", return_value=resolved), \
         mock.patch("kittenml.kittentts2.vllm.backend.importlib.util.find_spec", return_value=object()), \
         mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)), \
         mock.patch("kittenml.get_model.snapshot_download") as download:
        with pytest.raises(RuntimeError, match="SM75.*no bfloat16.*'dtype': 'float16'"):
            KittenTTS("example/repo", backend="vllm")
        download.assert_not_called()
        # With the opt-in, the checks pass and the download starts.
        download.side_effect = RuntimeError("download started")
        with pytest.raises(RuntimeError, match="download started"):
            KittenTTS("example/repo", backend="vllm", vllm_options={"dtype": "float16"})


def test_w4a16_refuses_marlin_environment_overrides(monkeypatch, tmp_path):
    from kittenml.kittentts2.vllm.backend import VLLMBackend, check_w4a16_environment

    check_w4a16_environment({})
    # vLLM reads only VLLM_MARLIN_USE_ATOMIC_ADD=1 as on; other kernels may be disabled.
    check_w4a16_environment({"VLLM_MARLIN_USE_ATOMIC_ADD": "0",
                             "VLLM_DISABLED_KERNELS": "ExllamaLinearKernel"})
    for key, value in (("VLLM_MARLIN_INPUT_DTYPE", "int8"), ("VLLM_MARLIN_INPUT_DTYPE", "fp8"),
                       ("VLLM_MARLIN_USE_ATOMIC_ADD", "1"),
                       ("VLLM_DISABLED_KERNELS", "ConchLinearKernel,MarlinLinearKernel")):
        with pytest.raises(RuntimeError, match=key):
            check_w4a16_environment({key: value})
    monkeypatch.setenv("VLLM_MARLIN_INPUT_DTYPE", "int8")
    owner = SimpleNamespace(device="cuda", repo_dir=tmp_path, config={}, weights="packed",
                            _load_lm_weights=mock.Mock())
    with mock.patch("kittenml.kittentts2.vllm.backend.validate_environment"):
        with pytest.raises(RuntimeError, match="W4A8"):
            VLLMBackend(owner, {"quantization": "w4a16"})
    owner._load_lm_weights.assert_not_called()


def test_w4a16_hf_overrides_payload():
    from kittenml.kittentts2.vllm.backend import quantization_args
    from kittenml.kittentts2.vllm.w4a16 import QUANTIZATION_CONFIG

    assert quantization_args(None) == {}
    args = quantization_args("w4a16")
    assert args["quantization"] == "compressed-tensors"
    payload = args["hf_overrides"]["quantization_config"]
    assert payload == {
        "quant_method": "compressed-tensors", "format": "pack-quantized",
        "config_groups": {"group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
                        "group_size": 128, "dynamic": False, "actorder": None},
            "input_activations": None}},
        "ignore": ["lm_head"]}
    payload["total_num_heads"] = 16  # vLLM writes into the dict it receives
    assert "total_num_heads" not in QUANTIZATION_CONFIG


def test_w4a16_payload_selects_vllm_uint4b8_weight_only_scheme():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig)
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
        WNA16_SUPPORTED_TYPES_MAP)
    from kittenml.kittentts2.vllm.w4a16 import OFFSET, quantization_config

    config = CompressedTensorsConfig.from_config(quantization_config())
    assert config.ignore == ["lm_head"] and config.quant_format == "pack-quantized"
    scheme = config.target_scheme_map["Linear"]
    assert scheme["input_activations"] is None
    assert CompressedTensorsConfig._is_wNa16_group_channel(scheme["weights"], None)
    weights = scheme["weights"]
    assert (weights.num_bits, weights.group_size, weights.symmetric) == (4, 128, True)
    assert WNA16_SUPPORTED_TYPES_MAP[4].bias == OFFSET == 8  # uint4b8: code = sign + 8


def test_engine_receives_w4a16_arguments_and_main_process_loads_only_conditioning(tmp_path):
    pytest.importorskip("vllm")
    from kittenml.kittentts2.vllm.backend import VLLMBackend

    import os
    seen = []

    def create_engine(**kwargs):
        seen.append(dict(kwargs, disabled_kernels=os.environ.get("VLLM_DISABLED_KERNELS")))

    def run(options):
        owner = SimpleNamespace(device="cuda", repo_dir=tmp_path, config={}, weights="packed",
                                token_map=None, _load_lm_weights=mock.Mock(return_value={
                                    "model.embed_tokens.weight": torch.zeros(2, 4)}))
        with mock.patch("kittenml.kittentts2.vllm.backend.validate_environment"), \
             mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)), \
             mock.patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(
                 model_type="qwen3", hidden_size=4)), \
             mock.patch("transformers.AutoTokenizer.from_pretrained"), \
             mock.patch("kittenml.kittentts2.speaker.load_speaker_head"), \
             mock.patch("vllm.LLM", side_effect=create_engine):
            VLLMBackend(owner, options)
        include = owner._load_lm_weights.call_args.kwargs["include"]
        assert include("model.embed_tokens.weight") and include("spk_proj.0.weight")
        assert not include("model.layers.0.self_attn.q_proj.weight")
        return owner, seen[-1]

    os.environ.pop("VLLM_DISABLED_KERNELS", None)
    owner, kwargs = run(None)
    assert {"quantization", "hf_overrides", "kv_cache_memory_bytes"}.isdisjoint(kwargs)
    assert set(kwargs["model_loader_extra_config"]) == {"repo_dir", "config", "weights"}
    assert kwargs["dtype"] is torch.bfloat16 and owner.dtype is torch.bfloat16
    assert kwargs["disabled_kernels"] is None
    owner, kwargs = run({"quantization": "w4a16"})
    # Machete, which vLLM prefers on Hopper, is disabled for the engine only.
    assert kwargs["disabled_kernels"] == "MacheteLinearKernel"
    assert "VLLM_DISABLED_KERNELS" not in os.environ
    owner, kwargs = run({"quantization": "w4a16", "kv_cache_memory_bytes": 1 << 30,
                         "dtype": "bfloat16"})
    assert kwargs["quantization"] == "compressed-tensors"
    # bfloat16 W4A16: only the quantization config, the head stays tied.
    assert set(kwargs["hf_overrides"]) == {"quantization_config"}
    assert kwargs["hf_overrides"]["quantization_config"]["ignore"] == ["lm_head"]
    assert kwargs["model_loader_extra_config"]["quantization"] == "w4a16"
    assert "float16_rescale" not in kwargs["model_loader_extra_config"]
    assert kwargs["kv_cache_memory_bytes"] == 1 << 30
    assert kwargs["dtype"] is torch.bfloat16 and owner.dtype is torch.bfloat16


def _linear_with_kernel(kind, act_type=torch.bfloat16):
    """A module carrying a compressed-tensors scheme whose kernel is `kind`."""
    kernel = object.__new__(kind)
    kernel.config = SimpleNamespace(act_type=act_type)
    module = torch.nn.Module()
    module.scheme = SimpleNamespace(kernel=kernel)
    return module


def test_vllm_loader_streams_w4a16_tensors_without_speaker_head():
    pytest.importorskip("vllm")
    from vllm.model_executor.kernels.linear import MarlinLinearKernel
    from kittenml.kittentts2.vllm.loader import KittenModelLoader

    loader = object.__new__(KittenModelLoader)
    loader.load_config = SimpleNamespace(model_loader_extra_config={
        "repo_dir": "unused", "config": {}, "weights": "packed", "quantization": "w4a16"})

    class TinyModel:
        def load_weights(self, weights):
            self.loaded = dict(weights)
            return set(self.loaded)

        def named_parameters(self):
            return [("model.layers.0.mlp.down_proj.weight_packed", None)]

        def named_modules(self):
            return [("model.layers.0.mlp.down_proj", _linear_with_kernel(MarlinLinearKernel))]

    model = TinyModel()
    tensors = [("model.layers.0.mlp.down_proj.weight_packed", torch.zeros(1, 1, dtype=torch.int32))]
    with mock.patch("kittenml.kittentts2.vllm.w4a16.iter_w4a16",
                    return_value=iter(tensors)) as convert:
        loader.load_weights(model, SimpleNamespace(dtype=torch.bfloat16))
    convert.assert_called_once_with("unused", {}, "packed")
    assert set(model.loaded) == {"model.layers.0.mlp.down_proj.weight_packed"}
    loader.load_config.model_loader_extra_config["quantization"] = "int8"
    with pytest.raises(ValueError, match="unknown KittenTTS quantization"):
        loader.load_weights(model, None)


def test_vllm_loader_allows_only_kv_cache_scales_to_be_missing():
    # compressed-tensors adds KV-cache scale parameters to every attention layer;
    # the checkpoint has none (the KV cache stays in the model dtype). On a GPU,
    # the strict missing-weights check rejected the real model because of them.
    pytest.importorskip("vllm")
    from vllm.model_executor.kernels.linear import MarlinLinearKernel
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from kittenml.kittentts2.vllm.loader import KittenModelLoader

    loader = object.__new__(KittenModelLoader)
    loader.load_config = SimpleNamespace(model_loader_extra_config={
        "repo_dir": "unused", "config": {}, "weights": "packed", "quantization": "w4a16"})
    attn = torch.nn.Module()
    attn.quant_method = object.__new__(BaseKVCacheMethod)
    attn.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
    attn.k_zero_point = torch.nn.Parameter(torch.tensor(0), requires_grad=False)
    model = torch.nn.Module()
    model.attn = attn
    model.proj = torch.nn.Linear(2, 2, bias=False)
    model.proj.scheme = _linear_with_kernel(MarlinLinearKernel).scheme
    model.load_weights = lambda weights: {name for name, _ in weights}
    with mock.patch("kittenml.kittentts2.vllm.w4a16.iter_w4a16",
                    return_value=iter([("proj.weight", torch.zeros(2, 2))])):
        loader.load_weights(model, SimpleNamespace(dtype=torch.bfloat16))
    with mock.patch("kittenml.kittentts2.vllm.w4a16.iter_w4a16", return_value=iter([])):
        with pytest.raises(ValueError, match=r"missing from the checkpoint: \['proj.weight'\]"):
            loader.load_weights(model, SimpleNamespace(dtype=torch.bfloat16))


@pytest.mark.parametrize("variant", ["packed", "full"])
def test_selected_tensor_loading_matches_full_load(tmp_path, variant):
    from kittenml.kittentts2.vllm.backend import _main_process_tensor

    _, config = _w4a16_checkpoint(tmp_path, layers=1)
    everything = load_lm_weights(tmp_path, config, variant)
    selected = load_lm_weights(tmp_path, config, variant, include=_main_process_tensor)
    assert set(selected) == {"model.embed_tokens.weight", "spk_proj.0.weight", "spk_proj.0.bias",
                             "spk_proj.1.weight", "spk_proj.1.bias"}
    assert all(torch.equal(selected[k], everything[k]) for k in selected)


def test_vllm_loader_refuses_w4a16_layers_on_any_kernel_but_marlin():
    # On Hopper vLLM would pick Machete, and with VLLM_DISABLED_KERNELS it can fall
    # back to Triton or others; exact output was verified on Marlin only.
    pytest.importorskip("vllm")
    from vllm.model_executor.kernels.linear import MarlinLinearKernel
    from vllm.model_executor.kernels.linear.mixed_precision.machete import MacheteLinearKernel
    from kittenml.kittentts2.vllm.loader import KittenModelLoader

    loader = object.__new__(KittenModelLoader)
    loader.load_config = SimpleNamespace(model_loader_extra_config={
        "repo_dir": "unused", "config": {}, "weights": "packed", "quantization": "w4a16"})
    model = torch.nn.Module()
    model.a = _linear_with_kernel(MarlinLinearKernel)
    model.b = _linear_with_kernel(MacheteLinearKernel)
    model.load_weights = mock.Mock(return_value=set())
    with mock.patch("kittenml.kittentts2.vllm.w4a16.iter_w4a16") as convert:
        with pytest.raises(RuntimeError, match=r"\['MacheteLinearKernel'\] for 1 linear layers"):
            loader.load_weights(model, SimpleNamespace(dtype=torch.bfloat16))
        with pytest.raises(RuntimeError, match="no quantized linear layers"):
            loader.load_weights(torch.nn.Module(), SimpleNamespace(dtype=torch.bfloat16))
    convert.assert_not_called()
    model.load_weights.assert_not_called()


def test_vllm_loader_refuses_w4a8_activations():
    # VLLM_MARLIN_INPUT_DTYPE keeps MarlinLinearKernel but sets its activation type
    # to int8/fp8 (W4A8). The loader runs in the engine process and checks the
    # activation type vLLM gave each layer.
    pytest.importorskip("vllm")
    from vllm.model_executor.kernels.linear import MarlinLinearKernel
    from kittenml.kittentts2.vllm.loader import KittenModelLoader

    loader = object.__new__(KittenModelLoader)
    loader.load_config = SimpleNamespace(model_loader_extra_config={
        "repo_dir": "unused", "config": {}, "weights": "packed", "quantization": "w4a16"})
    bf16 = SimpleNamespace(dtype=torch.bfloat16)
    model = torch.nn.Module()
    model.a = _linear_with_kernel(MarlinLinearKernel)
    model.b = _linear_with_kernel(MarlinLinearKernel, act_type=torch.int8)
    model.load_weights = mock.Mock(return_value=set())
    with mock.patch("kittenml.kittentts2.vllm.w4a16.iter_w4a16") as convert:
        with pytest.raises(RuntimeError, match=r"1 linear layers would quantize their "
                                               r"activations to \['torch.int8'\].*"
                                               r"VLLM_MARLIN_INPUT_DTYPE"):
            loader.load_weights(model, bf16)
        model.b = _linear_with_kernel(MarlinLinearKernel)
        with pytest.raises(RuntimeError, match="engine dtype is torch.float16"):
            loader.load_weights(model, SimpleNamespace(dtype=torch.float16))
    convert.assert_not_called()
    model.load_weights.assert_not_called()


def test_engine_sees_the_checked_environment_after_an_earlier_engine(monkeypatch, tmp_path):
    # In single-process mode vLLM freezes its view of the environment once an
    # engine has started (enable_envs_cache). A later W4A16 engine must see the
    # environment KittenTTS checked, not the frozen one.
    pytest.importorskip("vllm")
    import vllm.envs
    from kittenml.kittentts2.vllm.backend import VLLMBackend

    seen = []

    def create_engine(**kwargs):
        seen.append((vllm.envs.VLLM_MARLIN_INPUT_DTYPE, vllm.envs.VLLM_DISABLED_KERNELS))

    owner = SimpleNamespace(device="cuda", repo_dir=tmp_path, config={}, weights="packed",
                            token_map=None, _load_lm_weights=mock.Mock(return_value={
                                "model.embed_tokens.weight": torch.zeros(2, 4)}))
    monkeypatch.delenv("VLLM_DISABLED_KERNELS", raising=False)
    monkeypatch.setenv("VLLM_MARLIN_INPUT_DTYPE", "int8")
    try:
        vllm.envs.enable_envs_cache()  # as after a first engine in this process
        monkeypatch.delenv("VLLM_MARLIN_INPUT_DTYPE")
        assert vllm.envs.VLLM_MARLIN_INPUT_DTYPE == "int8"
        with mock.patch("kittenml.kittentts2.vllm.backend.validate_environment"), \
             mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)), \
             mock.patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(
                 model_type="qwen3", hidden_size=4)), \
             mock.patch("transformers.AutoTokenizer.from_pretrained"), \
             mock.patch("kittenml.kittentts2.speaker.load_speaker_head"), \
             mock.patch("vllm.LLM", side_effect=create_engine):
            VLLMBackend(owner, {"quantization": "w4a16"})
    finally:
        vllm.envs.disable_envs_cache()
    assert seen == [(None, ["MacheteLinearKernel"])]


def test_w4a16_exactness_checks_compare_bits_not_values(tmp_path):
    # torch.equal treats -0.0 and +0.0 as equal; the W4A16 checks must not.
    from kittenml.kittentts2.vllm.w4a16 import bit_equal, iter_w4a16

    zero = torch.zeros(3, dtype=torch.bfloat16)
    assert torch.equal(-zero, zero) and not bit_equal(-zero, zero)
    assert bit_equal(zero, zero.clone()) and not bit_equal(zero, zero.float())
    weight = torch.zeros(2, 128, dtype=torch.bfloat16)
    weight[0, :2] = torch.tensor([0.5, -0.5])
    weight[1, 3] = -0.0  # a ternary rebuild gives +0.0 here
    (tmp_path / "lm").mkdir()
    save_file({"model.layers.0.self_attn.o_proj.weight": weight},
              tmp_path / "lm" / "model.safetensors")
    with pytest.raises(ValueError, match="not exactly ternary"):
        list(iter_w4a16(tmp_path, {}, "full"))
    weight[1, 3] = 0.0
    save_file({"model.layers.0.self_attn.o_proj.weight": weight},
              tmp_path / "lm" / "model.safetensors")
    assert len(list(iter_w4a16(tmp_path, {}, "full"))) == 3


# ── Opt-in float16 (vllm_options={"dtype": "float16"}), fp16.py ─────────────────

def _qwen3(layers=2, tied=True):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(vocab_size=32, hidden_size=256, intermediate_size=384,
                         num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=128, rms_norm_eps=1e-6, tie_word_embeddings=tied,
                         max_position_embeddings=64)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).double().eval()


def _realistic_state(model, generator):
    """Random weights at the published model's scales: small embeddings (mean(x^2)
    around 1e-3, so an unscaled epsilon would matter) and a residual stream that grows."""
    state = {}
    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight":
            continue
        if name.endswith("norm.weight"):
            value = torch.rand(tensor.shape, generator=generator, dtype=torch.float64) + 0.5
        elif name == "model.embed_tokens.weight":
            value = torch.randn(tensor.shape, generator=generator, dtype=torch.float64) * 0.033
        else:
            value = torch.randn(tensor.shape, generator=generator, dtype=torch.float64) * 0.5
        state[name] = value
    return state


def test_float16_rescale_is_the_same_model_in_float64():
    # The rescale is exact: with the stream's inputs at 2^-9, every branch at 2^-9 and
    # the residual norms' epsilon at 2^-18, the logits equal the original model's.
    # Checked on transformers' Qwen3 in float64, with the plan's own functions.
    from kittenml.kittentts2.vllm.fp16 import (EMBEDDING, LM_HEAD, RESIDUAL_SHIFT,
                                                residual_norms, scale_norm_epsilons, shift_of)

    generator = torch.Generator().manual_seed(3)
    reference = _qwen3(tied=True)
    state = _realistic_state(reference, generator)
    reference.load_state_dict(state, strict=False)
    reference.tie_weights()
    rescaled = _qwen3(tied=False)
    rescaled.load_state_dict({**{name: tensor * 2.0 ** -shift_of(name)
                                 for name, tensor in state.items()},
                              LM_HEAD: state[EMBEDDING]}, strict=True)
    assert rescaled.lm_head.weight.data_ptr() != rescaled.model.embed_tokens.weight.data_ptr()
    assert scale_norm_epsilons(rescaled, layers=2, eps=1e-6) == 5
    assert [name for name, _ in residual_norms(rescaled)] == [
        "model.layers.0.input_layernorm", "model.layers.0.post_attention_layernorm",
        "model.layers.1.input_layernorm", "model.layers.1.post_attention_layernorm",
        "model.norm"]
    assert rescaled.model.layers[0].self_attn.q_norm.variance_epsilon == 1e-6

    ids = torch.tensor([[0, 5, 9, 2, 31, 7]])
    embeds = state[EMBEDDING][ids].clone()
    embeds[0, 0] = torch.randn(256, generator=generator, dtype=torch.float64)  # spk_proj slot
    with torch.no_grad():
        expected = reference(inputs_embeds=embeds).logits
        actual = rescaled(inputs_embeds=embeds * 2.0 ** -RESIDUAL_SHIFT).logits
        # transformers' RMSNorm computes in float32, so agreement is to float32 rounding.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        # Negative control: without the epsilon scale the first norm (embeddings at
        # 2^-9, mean(x^2) ~ 4e-9 << 1e-6) is visibly wrong.
        for _, module in residual_norms(rescaled):
            module.variance_epsilon = 1e-6
        wrong = rescaled(inputs_embeds=embeds * 2.0 ** -RESIDUAL_SHIFT).logits
    assert (wrong - expected).abs().max() > 1e-2


def test_float16_plan_shifts_each_branch_by_the_residual_scale():
    from kittenml.kittentts2.vllm import fp16

    p = "model.layers.3."
    shifts = {name: fp16.shift_of(p + name + ".weight") for name in _PROJECTIONS}
    assert shifts == {"self_attn.q_proj": 0, "self_attn.k_proj": 0, "self_attn.v_proj": 3,
                      "self_attn.o_proj": 6, "mlp.gate_proj": 0, "mlp.up_proj": 5,
                      "mlp.down_proj": 4}
    assert fp16.shift_of(p + "mlp.down_proj.weight_scale") == 4
    assert fp16.shift_of(p + "mlp.down_proj.weight_packed") == 0
    assert fp16.shift_of("model.embed_tokens.weight") == fp16.RESIDUAL_SHIFT == 9
    assert fp16.shift_of(p + "input_layernorm.weight") == 0
    assert fp16.EPS_SHIFT == 18
    assert fp16.hf_overrides() == {"tie_word_embeddings": False,
                                   "kittenml_float16_rescale": fp16.PLAN}
    assert fp16.PLAN == ("residual 2^-9; v_proj 2^-3, o_proj 2^-6, up_proj 2^-5, down_proj 2^-4; "
                         "residual-norm eps x 2^-18; untied unscaled lm_head")


def test_float16_plan_reaches_the_config_vllm_hashes_for_its_compile_cache():
    # vLLM's ModelConfig.compute_hash ignores hf_overrides itself but hashes
    # hf_config.to_json_string() after the overrides are applied with setattr.
    from transformers import Qwen3Config
    from kittenml.kittentts2.vllm.fp16 import PLAN, hf_overrides

    config = Qwen3Config(tie_word_embeddings=True)
    before = config.to_json_string()
    for key, value in hf_overrides().items():
        setattr(config, key, value)
    after = config.to_json_string()
    assert after != before and PLAN in after and config.tie_word_embeddings is False


@pytest.mark.parametrize("variant", ["packed", "full"])
def test_float16_rescale_dense_and_w4a16_give_the_same_exact_float16_weights(tmp_path, variant):
    from kittenml.kittentts2.vllm.fp16 import rescale_weights, scaled, shift_of
    from kittenml.kittentts2.vllm.w4a16 import bit_equal, is_projection, iter_w4a16

    # Group scales >= 2^-6, as in the published checkpoint: every shifted weight
    # stays a normal float16 number.
    state, config = _w4a16_checkpoint(tmp_path, low=2.0 ** -6)
    weights = load_lm_weights(tmp_path, config, variant)
    dense = dict(rescale_weights((n, t) for n, t in weights.items()
                                 if not n.startswith("spk_proj.")))
    original = dict(iter_w4a16(tmp_path, config, variant))
    quantized = dict(rescale_weights(original.items()))
    unpack = _reference_unpack()
    embedding = state["model.embed_tokens.weight"]
    for out in (dense, quantized):
        assert bit_equal(out["model.embed_tokens.weight"], scaled(embedding, 9))
        assert bit_equal(out["lm_head.weight"], embedding.to(torch.float16))  # untied, unscaled
        assert bit_equal(out["model.norm.weight"], state["model.norm.weight"])  # unchanged
    for name in (n for n in state if is_projection(n)):
        exact = state[name].to(torch.float32) * 2.0 ** -shift_of(name)
        assert dense[name].dtype == torch.float16
        assert torch.equal(dense[name].to(torch.float32), exact), name
        base = name.removesuffix(".weight")
        scale = quantized[base + ".weight_scale"]
        assert scale.dtype == torch.float16
        assert quantized[base + ".weight_packed"] is original[base + ".weight_packed"]
        assert torch.equal(scale.to(torch.float32), original[base + ".weight_scale"].float()
                           * 2.0 ** -shift_of(name))
        signs = unpack(quantized[base + ".weight_packed"], state[name].shape)
        rebuilt = signs.to(torch.float16) * scale.repeat_interleave(128, dim=1)
        assert bit_equal(rebuilt, dense[name]), name


def test_float16_rescale_refuses_weights_it_cannot_hold_exactly():
    from kittenml.kittentts2.vllm.fp16 import is_exact, rescale_weights

    weight = torch.full((2, 128), 0.25, dtype=torch.bfloat16)
    weight[1, 7] = 2.0 ** -20  # x 2^-6 = 2^-26: below float16's smallest subnormal
    pairs = [("model.layers.0.self_attn.o_proj.weight", weight)]
    with pytest.raises(ValueError, match=r"o_proj.weight x 2\^-6 is not exact in float16"):
        dict(rescale_weights(pairs))
    # The same value is exact without a shift (q_proj) ... and -0.0 must stay -0.0.
    assert len(dict(rescale_weights([("model.layers.0.self_attn.q_proj.weight", weight)]))) == 1
    zero = torch.tensor([-0.0], dtype=torch.bfloat16)
    assert is_exact(zero, 6) and is_exact(torch.tensor([65504.0]), 0)
    assert not is_exact(torch.tensor([65536.0]), 0)  # overflow is not exact
    with pytest.raises(ValueError, match="tied"):
        dict(rescale_weights([("lm_head.weight", weight)]))


def test_float16_prompt_embeddings_match_the_engine_scale():
    pytest.importorskip("vllm")
    from kittenml.kittentts2.vllm.backend import VLLMBackend
    from kittenml.kittentts2.vllm.fp16 import ScaledSpeakerHead, scale_embedding, scaled

    generator = torch.Generator().manual_seed(5)
    table = (torch.randn(16, 4, generator=generator) * 0.03).to(torch.bfloat16)
    head = torch.nn.Linear(4, 4).to(torch.bfloat16)
    backend = object.__new__(VLLMBackend)
    backend.max_model_len = 32
    backend.token_map = TokenMap(5, 10, 1, 2, 3, 4, 0, 15)
    backend.embedding = scale_embedding(table)
    backend.spk_proj = ScaledSpeakerHead(head)
    speaker = torch.randn(1, 4, generator=generator).to(torch.float16)
    settings = dict(temperature=0.7, top_k=5, top_p=0.8, min_p=0.02)
    advanced = dict(seed=1, repetition_penalty=1.2, repetition_window=3,
                    token_run_penalty=1.5, token_run_grace=4)
    request, _ = backend.prepare_request([0, 3, 1], speaker, settings, 8, advanced)
    embeds = request["prompt_embeds"]
    assert embeds.dtype == torch.float16
    assert torch.equal(embeds[1:], scaled(table[[3, 1]], 9))
    expected = scaled(torch.nn.functional.linear(speaker.float(), head.weight.float(),
                                                 head.bias.float()), 9).squeeze(0)
    assert torch.equal(embeds[0], expected)


def _norm(eps=1e-6):
    module = torch.nn.Module()
    module.variance_epsilon = eps
    return module


def _float16_engine_model(layers=2, tied=False):
    """Just enough of vLLM's Qwen3ForCausalLM for the loader's float16 checks."""
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(4, 2)
    model.model.layers = torch.nn.ModuleList()
    for _ in range(layers):
        layer = torch.nn.Module()
        layer.input_layernorm, layer.post_attention_layernorm = _norm(), _norm()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_norm, layer.self_attn.k_norm = _norm(), _norm()
        model.model.layers.append(layer)
    model.model.norm = _norm()
    model.lm_head = model.model.embed_tokens if tied else torch.nn.Embedding(4, 2)
    model.loaded = {}

    def load_weights(weights):
        model.loaded.update(weights)
        return set(dict(model.named_parameters()))
    model.load_weights = load_weights
    return model


def test_vllm_loader_float16_rescales_weights_and_residual_norm_epsilon_only():
    pytest.importorskip("vllm")
    from kittenml.kittentts2.vllm.loader import KittenModelLoader

    loader = object.__new__(KittenModelLoader)
    loader.load_config = SimpleNamespace(model_loader_extra_config={
        "repo_dir": "unused", "config": {}, "weights": "packed", "float16_rescale": True})
    hf_config = SimpleNamespace(num_hidden_layers=2, rms_norm_eps=1e-6, tie_word_embeddings=False)
    fp16_config = SimpleNamespace(dtype=torch.float16, hf_config=hf_config)
    embedding = torch.full((4, 2), 0.03, dtype=torch.bfloat16)
    state = {"model.embed_tokens.weight": embedding, "spk_proj.0.bias": torch.zeros(2),
             "model.layers.0.mlp.up_proj.weight": torch.full((2, 2), 0.5, dtype=torch.bfloat16)}
    model = _float16_engine_model()
    with mock.patch("kittenml.kittentts2.vllm.loader.iter_lm_weights", return_value=state.items()):
        loader.load_weights(model, fp16_config)
        assert set(model.loaded) == {"model.embed_tokens.weight", "lm_head.weight",
                                     "model.layers.0.mlp.up_proj.weight"}
        assert torch.equal(model.loaded["lm_head.weight"], embedding.to(torch.float16))
        assert model.loaded["model.layers.0.mlp.up_proj.weight"].unique().tolist() == [2.0 ** -6]
        for layer in model.model.layers:
            assert layer.input_layernorm.variance_epsilon == 1e-6 * 2.0 ** -18
            assert layer.post_attention_layernorm.variance_epsilon == 1e-6 * 2.0 ** -18
            assert layer.self_attn.q_norm.variance_epsilon == 1e-6  # reads a normed input
            assert layer.self_attn.k_norm.variance_epsilon == 1e-6
        assert model.model.norm.variance_epsilon == 1e-6 * 2.0 ** -18
        loader.load_weights(model, fp16_config)  # a second load does not scale twice
        assert model.model.norm.variance_epsilon == 1e-6 * 2.0 ** -18
        with pytest.raises(RuntimeError, match="engine dtype is torch.bfloat16, not float16"):
            loader.load_weights(_float16_engine_model(),
                                SimpleNamespace(dtype=torch.bfloat16, hf_config=hf_config))
        with pytest.raises(RuntimeError, match="tied to the scaled embedding"):
            loader.load_weights(_float16_engine_model(tied=True), fp16_config)
        with pytest.raises(RuntimeError, match="expected 5 residual-stream norms, found 7"):
            loader.load_weights(_float16_engine_model(layers=3), fp16_config)


def test_vllm_loader_bfloat16_paths_hand_over_the_checkpoint_tensors_untouched():
    # Without the float16 opt-in the loader passes the very same tensors and never
    # touches an epsilon, for the dense and the W4A16 path.
    pytest.importorskip("vllm")
    from vllm.model_executor.kernels.linear import MarlinLinearKernel
    from kittenml.kittentts2.vllm.loader import KittenModelLoader

    bf16 = SimpleNamespace(dtype=torch.bfloat16)
    for quantization in (None, "w4a16"):
        loader = object.__new__(KittenModelLoader)
        extra = {"repo_dir": "unused", "config": {}, "weights": "packed"}
        if quantization:
            extra["quantization"] = quantization
        loader.load_config = SimpleNamespace(model_loader_extra_config=extra)
        model = _float16_engine_model(tied=True)
        model.proj = _linear_with_kernel(MarlinLinearKernel)
        tensors = {"model.embed_tokens.weight": torch.full((4, 2), 0.03, dtype=torch.bfloat16),
                   "model.layers.0.mlp.up_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16)}
        with mock.patch("kittenml.kittentts2.vllm.loader.iter_lm_weights",
                        return_value=tensors.items()), \
             mock.patch("kittenml.kittentts2.vllm.w4a16.iter_w4a16",
                        return_value=iter(tensors.items())):
            loader.load_weights(model, bf16)
        assert set(model.loaded) == set(tensors)
        assert all(model.loaded[name] is tensors[name] for name in tensors)
        assert all(module.variance_epsilon == 1e-6 for name, module in model.named_modules()
                   if hasattr(module, "variance_epsilon"))


def test_engine_receives_the_float16_rescale_only_when_asked(tmp_path):
    pytest.importorskip("vllm")
    from kittenml.kittentts2.vllm.backend import VLLMBackend
    from kittenml.kittentts2.vllm.fp16 import PLAN, ScaledSpeakerHead, scale_embedding

    seen = []
    embedding = torch.full((2, 4), 0.03, dtype=torch.bfloat16)

    def run(options):
        owner = SimpleNamespace(device="cpu", repo_dir=tmp_path, config={}, weights="packed",
                                token_map=None, _load_lm_weights=mock.Mock(return_value={
                                    "model.embed_tokens.weight": embedding}))
        with mock.patch("kittenml.kittentts2.vllm.backend.validate_environment") as validate, \
             mock.patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(
                 model_type="qwen3", hidden_size=4)), \
             mock.patch("transformers.AutoTokenizer.from_pretrained"), \
             mock.patch("kittenml.kittentts2.speaker.load_speaker_head",
                        return_value=torch.nn.Linear(4, 4)) as head, \
             mock.patch("vllm.LLM", side_effect=lambda **kwargs: seen.append(kwargs)):
            backend = VLLMBackend(owner, options)
        validate.assert_called_once_with("cpu", (options or {}).get("dtype"))
        return owner, backend, head, seen[-1]

    for options in ({"dtype": "float16"}, {"dtype": "float16", "quantization": "w4a16"}):
        owner, backend, head, kwargs = run(options)
        assert kwargs["dtype"] is torch.float16 and owner.dtype is torch.float16
        assert kwargs["model_loader_extra_config"]["float16_rescale"] is True
        overrides = kwargs["hf_overrides"]
        assert overrides["tie_word_embeddings"] is False
        assert overrides["kittenml_float16_rescale"] == PLAN
        assert ("quantization_config" in overrides) == ("quantization" in options)
        assert torch.equal(backend.embedding, scale_embedding(embedding))
        assert isinstance(backend.spk_proj, ScaledSpeakerHead)
        assert head.call_args.args[3] is torch.float32  # the speaker head runs in float32
    for options in (None, {"dtype": "bfloat16"}, {"quantization": "w4a16"}):
        owner, backend, head, kwargs = run(options)
        assert kwargs["dtype"] is torch.bfloat16 and owner.dtype is torch.bfloat16
        assert "float16_rescale" not in kwargs["model_loader_extra_config"]
        assert "tie_word_embeddings" not in kwargs.get("hf_overrides", {})
        assert backend.embedding.dtype is torch.bfloat16 and torch.equal(backend.embedding,
                                                                         embedding)
        assert backend.spk_proj is head.return_value
        assert head.call_args.args[3] is torch.bfloat16

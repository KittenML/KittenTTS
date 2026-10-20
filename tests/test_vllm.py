"""Small checkpoints and sampler parity; no GPU or model download required."""

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
    with mock.patch("kittenml.kittentts2.vllm.loader.load_lm_weights", return_value=state):
        loader.load_weights(model, None)
    assert set(model.loaded) == {"model.weight"}
    with mock.patch("kittenml.kittentts2.vllm.loader.load_lm_weights", return_value={}):
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


def _ternary(rows, cols, generator):
    signs = torch.randint(-1, 2, (rows, cols), generator=generator)
    scale = (torch.rand(rows, cols // 128, generator=generator) * 0.5 + 1e-4).to(torch.bfloat16)
    scale[0, 0] = 0  # an all-zero group has a zero scale
    signs[0, :128] = 0
    return signs.to(torch.bfloat16) * scale.repeat_interleave(128, dim=1)


def _w4a16_checkpoint(tmp_path, layers=2, perturb=None):
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
            state[prefix + name + ".weight"] = _ternary(*shape, generator)
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
    with mock.patch("kittenml.kittentts2.weights.load_lm_weights", return_value=state) as load:
        out = dict(w4a16.iter_w4a16(tmp_path, {"lm_emb4": "x"}, "emb4"))
    load.assert_called_once_with(tmp_path, {"lm_emb4": "x"}, "emb4")
    assert "spk_proj.weight" not in out
    assert torch.equal(out["model.embed_tokens.weight"], embedding)
    expected = dict(w4a16._from_dense("model.layers.0.self_attn.o_proj.weight", weight))
    for key, value in expected.items():
        assert torch.equal(out[key], value)

    check_options({"quantization": "w4a16"}, "emb4")


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


def test_float16_is_refused_with_the_measured_reason():
    # float16 was measured to overflow at layer 10 with the current checkpoint
    # (with and without W4A16) and then produce no audio, so it is refused.
    from kittenml.kittentts2.vllm.backend import resolve_options

    for options in ({"dtype": "float16"}, {"dtype": "float16", "quantization": "w4a16"}):
        with pytest.raises(ValueError, match=r"'float16' is not supported: this model's residual "
                                             r"stream exceeds the float16 range \(measured overflow "
                                             r"from layer 10\).*compute capability < 8.0"):
            resolve_options(options)


def test_dtype_follows_hardware_and_gpus_without_bfloat16_are_refused():
    from kittenml.kittentts2.vllm.backend import validate_environment

    with mock.patch("kittenml.kittentts2.vllm.backend.importlib.util.find_spec", return_value=object()), \
         mock.patch("torch.cuda.is_available", return_value=True):
        for capability in ((8, 0), (8, 6), (12, 0)):
            with mock.patch("torch.cuda.get_device_capability", return_value=capability):
                validate_environment()
        for capability, name in (((7, 5), "SM75"), ((7, 0), "SM70")):
            with mock.patch("torch.cuda.get_device_capability", return_value=capability):
                with pytest.raises(RuntimeError, match=name + ".*no bfloat16.*float16 range"
                                                       ".*layer 10.*< 8.0"):
                    validate_environment()


def test_vllm_options_are_checked_before_model_download(monkeypatch):
    resolved = (None, "example/repo", {"type": "KITTEN2"})
    with mock.patch("kittenml.get_model.resolve_repo", return_value=resolved), \
         mock.patch("kittenml.kittentts2.vllm.backend.validate_environment"), \
         mock.patch("kittenml.get_model.snapshot_download") as download:
        for options, weights, match in [({"quantisation": "w4a16"}, None, "unknown vllm_options"),
                                        ({"dtype": "float16"}, None, "float16 range")]:
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
        with pytest.raises(RuntimeError, match="SM75.*no bfloat16"):
            KittenTTS("example/repo", backend="vllm")
        download.assert_not_called()


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
    assert "quantization" not in kwargs["model_loader_extra_config"]
    assert kwargs["dtype"] is torch.bfloat16 and owner.dtype is torch.bfloat16
    assert kwargs["disabled_kernels"] is None
    owner, kwargs = run({"quantization": "w4a16"})
    # Machete, which vLLM prefers on Hopper, is disabled for the engine only.
    assert kwargs["disabled_kernels"] == "MacheteLinearKernel"
    assert "VLLM_DISABLED_KERNELS" not in os.environ
    owner, kwargs = run({"quantization": "w4a16", "kv_cache_memory_bytes": 1 << 30,
                         "dtype": "bfloat16"})
    assert kwargs["quantization"] == "compressed-tensors"
    assert kwargs["hf_overrides"]["quantization_config"]["ignore"] == ["lm_head"]
    assert kwargs["model_loader_extra_config"]["quantization"] == "w4a16"
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

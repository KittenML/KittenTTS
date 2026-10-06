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

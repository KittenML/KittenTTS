"""AMD (ROCm) paths of the vLLM backend and the audio decoder; no GPU, model download or ROCm
build required."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from kittenml.kittentts2.vllm import rocm
from kittenml.kittentts2.vllm.w4a16 import GROUP_SIZE, pack_int4


def _amd(arch="gfx1100", built=("gfx90a", "gfx942", "gfx1100", "gfx1201")):
    """Patches that make this PyTorch look like a ROCm build on an AMD GPU."""
    return [mock.patch.object(torch.version, "hip", "7.2.53211"),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_properties",
                       return_value=SimpleNamespace(gcnArchName=f"{arch}:sramecc-:xnack-")),
            mock.patch("torch.cuda.get_arch_list", return_value=list(built)),
            mock.patch("torch.cuda.get_device_capability",
                       side_effect=AssertionError("NVIDIA capability checked on AMD"))]


def _apply(patches):
    for patch in patches:
        patch.start()


def test_amd_gpus_are_checked_by_architecture_not_nvidia_capability():
    from kittenml.kittentts2.vllm.backend import validate_environment
    find = mock.patch("kittenml.kittentts2.vllm.backend.importlib.util.find_spec",
                      return_value=object())
    for arch in ("gfx1100", "gfx1201"):
        with find, mock.patch("kittenml.kittentts2.vllm.backend.check_torchcodec") as torchcodec:
            _apply(_amd(arch))
            try:
                validate_environment()
                # vLLM imports torchcodec on AMD too, so a broken one is refused there as well.
                torchcodec.assert_called_once_with()
                with pytest.raises(RuntimeError, match="'float16' is not supported on AMD GPUs"):
                    validate_environment(dtype="float16")
            finally:
                mock.patch.stopall()
    with find:
        _apply(_amd("gfx1030"))
        try:
            with pytest.raises(RuntimeError, match=r"AMD GPU \(gfx1030\) is not supported.*"
                                                   r"gfx1100, gfx1201"):
                validate_environment()
        finally:
            mock.patch.stopall()


def test_w4a16_on_amd_disables_the_rdna3_kernel_and_keeps_the_hybrid_one():
    from kittenml.kittentts2.vllm.backend import (check_w4a16_environment,
                                                  w4a16_engine_environment)
    with mock.patch.object(torch.version, "hip", "7.2.53211"):
        assert w4a16_engine_environment({}) == {"VLLM_DISABLED_KERNELS": "RDNA3W4A16LinearKernel"}
        assert w4a16_engine_environment({"VLLM_DISABLED_KERNELS": "ExllamaLinearKernel"}) == {
            "VLLM_DISABLED_KERNELS": "ExllamaLinearKernel,RDNA3W4A16LinearKernel"}
        check_w4a16_environment({"VLLM_DISABLED_KERNELS": "MarlinLinearKernel"})
        with pytest.raises(RuntimeError, match="RDNAHybridW4A16LinearKernel"):
            check_w4a16_environment({"VLLM_DISABLED_KERNELS": "RDNAHybridW4A16LinearKernel"})
    # NVIDIA GPUs are unchanged.
    with mock.patch.object(torch.version, "hip", None):
        assert w4a16_engine_environment({}) == {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}
        check_w4a16_environment({"VLLM_DISABLED_KERNELS": "RDNAHybridW4A16LinearKernel"})


def _kernel(name):
    kernel = object.__new__(type(name, (), {}))
    kernel.config = SimpleNamespace(act_type=torch.bfloat16)
    module = torch.nn.Module()
    module.scheme = SimpleNamespace(kernel=kernel)
    return module


def test_loader_allows_only_the_hybrid_kernel_on_amd_and_asks_for_the_probe():
    pytest.importorskip("vllm")
    from kittenml.kittentts2.vllm.loader import _check_w4a16_kernels

    bf16 = SimpleNamespace(dtype=torch.bfloat16)
    model = torch.nn.Module()
    model.a = _kernel("RDNAHybridW4A16LinearKernel")
    with mock.patch.object(torch.version, "hip", "7.2.53211"):
        assert _check_w4a16_kernels(model, bf16) is True
        for other in ("RDNA3W4A16LinearKernel", "TritonW4A16LinearKernel", "MarlinLinearKernel"):
            model.b = _kernel(other)
            with pytest.raises(RuntimeError, match=rf"\['{other}'\] for 1 linear layers.*AMD"):
                _check_w4a16_kernels(model, bf16)
    with mock.patch.object(torch.version, "hip", None):
        model.b = _kernel("RDNAHybridW4A16LinearKernel")
        with pytest.raises(RuntimeError, match="only with MarlinLinearKernel"):
            _check_w4a16_kernels(model, bf16)


# ── The probe, with stand-in kernels on the CPU ──────────────────────────────────────────


def _layer(out_features, in_features, generator):
    signs = torch.randint(-1, 2, (out_features, in_features), generator=generator,
                          dtype=torch.int8)
    scale = (torch.rand(out_features, in_features // GROUP_SIZE, generator=generator) * 0.05
             + 0.01).to(torch.bfloat16)
    return pack_int4(signs), scale


def _model(compute, generator):
    """Two probed layers whose kernels compute y = compute(x, w) with w the exact weight."""
    model, references = torch.nn.Module(), {}
    modules = {}
    for layer, in_features in zip(rocm.PROBE_LAYERS, (2048, 6144)):
        packed, scale = _layer(64, in_features, generator)
        references[f"{layer}.weight_packed"], references[f"{layer}.weight_scale"] = packed, scale
        weight = rocm.dequantize(packed, scale)
        module = torch.nn.Module()
        module.register_parameter("w", torch.nn.Parameter(torch.zeros(1), requires_grad=False))
        kernel = object.__new__(type("RDNAHybridW4A16LinearKernel", (), {}))
        kernel.config = SimpleNamespace(act_type=torch.bfloat16)
        module.scheme = SimpleNamespace(kernel=kernel,
                                        apply_weights=lambda _, x, bias, w=weight: compute(x, w))
        modules[layer] = module
    model.named_modules = lambda *args, **kwargs: iter(modules.items())
    return model, references


def _float32(x, w):
    return (x.float() @ w.T).to(x.dtype)


def _bfloat16_split_k(x, w, shuffle=False):
    """Like vLLM's RDNA3 kernel: float32 sums over 256-wide slices, added into bfloat16
    (with `shuffle`, in a different order on every call, as atomic adds can be)."""
    out = torch.zeros(x.shape[0], w.shape[0], dtype=x.dtype)
    starts = list(range(0, x.shape[1], 256))
    if shuffle:
        starts = [starts[i] for i in torch.randperm(len(starts))]
    for start in starts:
        part = x[:, start:start + 256].float() @ w[:, start:start + 256].T
        out = (out.float() + part).to(x.dtype)
    return out


def test_probe_passes_a_float32_kernel_with_the_exact_weights():
    model, references = _model(_float32, torch.Generator().manual_seed(1))
    report = rocm.probe_w4a16(model, references, torch.bfloat16)
    assert set(report) == set(rocm.PROBE_LAYERS)
    assert all(r["worst_error_vs_allowed"] <= 1 for r in report.values())


@pytest.mark.parametrize("compute, message", [
    (_bfloat16_split_k, "does not sum in float32"),
    # Codes read in the wrong nibble order.
    (lambda x, w: _float32(x, w.reshape(w.shape[0], -1, 8).flip(-1).reshape(w.shape)),
     "does not reproduce the 4-bit weights"),
    # Activations rounded to 8 bits (W4A8-like).
    (lambda x, w: _float32(x.float().to(torch.float8_e4m3fn).to(x.dtype), w),
     "does not reproduce the 4-bit weights"),
    # A different summation order on every call.
    (lambda x, w: _bfloat16_split_k(x, w, shuffle=True), "different results for the same input"),
])
def test_probe_refuses_kernels_that_change_the_result(compute, message):
    model, references = _model(compute, torch.Generator().manual_seed(2))
    with pytest.raises(RuntimeError, match=message) as refused:
        rocm.probe_w4a16(model, references, torch.bfloat16)
    # Advice that also works when small-GPU fitting chose 4-bit by itself.
    assert "vllm_options={'quantization': None}" in str(refused.value)


def test_references_keep_only_the_probed_tensors_and_pass_everything_through():
    tensors = [(f"{layer}.{part}", torch.ones(2)) for layer in rocm.PROBE_LAYERS
               for part in ("weight_packed", "weight_scale", "weight_shape")]
    tensors.append(("model.layers.1.mlp.down_proj.weight_packed", torch.ones(2)))
    references = {}
    assert [n for n, _ in rocm.keep_references(iter(tensors), references)] == [n for n, _ in tensors]
    assert sorted(references) == sorted(f"{layer}.{part}" for layer in rocm.PROBE_LAYERS
                                        for part in ("weight_packed", "weight_scale"))


def test_decoder_convolutions_bypass_miopen_only_on_amd():
    from kittenml.kittentts2.vocoder import _convolutions

    before = torch.backends.cudnn.enabled
    with mock.patch.object(torch.version, "hip", None):
        with _convolutions():
            assert torch.backends.cudnn.enabled == before
    with mock.patch.object(torch.version, "hip", "7.2.53211"):
        with _convolutions():
            assert torch.backends.cudnn.enabled is False
    assert torch.backends.cudnn.enabled == before


def test_vocoder_source_mix_avoids_the_gemm_only_on_amd_gpus():
    from kittenml.kittentts2.vocoder import _SmallLinear, _mix_source_without_blas

    def s3gen():
        torch.manual_seed(0)
        source = SimpleNamespace(l_linear=torch.nn.Linear(9, 1))
        return SimpleNamespace(mel2wav=SimpleNamespace(m_source=source))

    for hip, device in ((None, "cuda"), ("7.2.53211", "cpu")):
        model = s3gen()
        linear = model.mel2wav.m_source.l_linear
        with mock.patch.object(torch.version, "hip", hip):
            _mix_source_without_blas(model, device)
        assert model.mel2wav.m_source.l_linear is linear  # NVIDIA and CPU: unchanged
    model = s3gen()
    linear = model.mel2wav.m_source.l_linear
    with mock.patch.object(torch.version, "hip", "7.2.53211"):
        _mix_source_without_blas(model, "cuda")
    mix = model.mel2wav.m_source.l_linear
    assert isinstance(mix, _SmallLinear)
    # Same parameters (same state dict keys and tensors), same result as the linear layer.
    assert [name for name, _ in mix.named_parameters()] == ["weight", "bias"]
    assert mix.weight is linear.weight and mix.bias is linear.bias
    x = torch.randn(1, 4096, 9, dtype=torch.float64)
    with torch.no_grad():
        expected = torch.nn.functional.linear(x, linear.weight.double(), linear.bias.double())
        mix.double()
        torch.testing.assert_close(mix(x), expected, rtol=0, atol=1e-12)


def test_auto_fit_chooses_4bit_on_radeon_rx_7000_and_9000_only():
    from kittenml.kittentts2.vllm.backend import VLLMBackend

    gib = 1 << 30
    lm = SimpleNamespace(hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
                         num_key_value_heads=8, head_dim=128, intermediate_size=6144,
                         vocab_size=158241, tie_word_embeddings=True)
    # gfx1250 is a gfx12 name, but vLLM treats it as a data-center GPU and its RDNA kernel refuses it.
    for arch, expected in (("gfx1100", "w4a16"), ("gfx1201", "w4a16"), ("gfx942", None),
                           ("gfx1250", None)):
        with mock.patch.object(torch.version, "hip", "7.2.53211"), \
             mock.patch("torch.cuda.get_device_properties",
                        return_value=SimpleNamespace(gcnArchName=f"{arch}:sramecc-:xnack-")), \
             mock.patch("kittenml.kittentts2.vllm.backend._free_gpu_memory",
                        return_value=(int(7.39 * gib), int(7.62 * gib))):
            fitted, report = VLLMBackend._fit({}, lm)
        assert report["fitted"] and fitted.get("quantization") == expected, arch


@pytest.mark.parametrize("compute, passes", [(_float32, True), (_bfloat16_split_k, False)])
def test_loader_keeps_the_probed_weights_and_checks_the_kernel_after_loading(compute, passes):
    # The wiring between load_weights (keeps CPU copies of the probed layers' 4-bit tensors)
    # and load_model (runs the probe on the repacked layers, then drops the copies).
    pytest.importorskip("vllm")
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from kittenml.kittentts2.vllm.loader import KittenModelLoader

    model, references = _model(compute, torch.Generator().manual_seed(3))
    weights = list(references.items()) + [("model.norm.weight", torch.ones(4))]
    def load_weights(tensors):
        for _ in tensors:
            pass
        return set(dict(model.named_parameters()))
    model.load_weights = load_weights
    loader = object.__new__(KittenModelLoader)
    loader.load_config = SimpleNamespace(model_loader_extra_config={
        "repo_dir": "unused", "config": {}, "weights": "packed", "quantization": "w4a16"})
    bf16 = SimpleNamespace(dtype=torch.bfloat16)

    def base_load_model(self, vllm_config, model_config, prefix=""):
        self.load_weights(model, model_config)
        return model

    probe = mock.Mock(wraps=rocm.probe_w4a16)
    with mock.patch.object(torch.version, "hip", "7.2.53211"), \
         mock.patch.object(BaseModelLoader, "load_model", base_load_model), \
         mock.patch("kittenml.kittentts2.vllm.w4a16.iter_w4a16", return_value=iter(weights)), \
         mock.patch.object(rocm, "probe_w4a16", probe):
        if passes:
            assert loader.load_model(None, bf16) is model
        else:
            with pytest.raises(RuntimeError, match="does not sum in float32"):
                loader.load_model(None, bf16)
    probe.assert_called_once()
    seen = probe.call_args.args[1]
    assert sorted(seen) == sorted(references)
    assert all(torch.equal(seen[name], references[name]) for name in references)
    assert loader._probe_references is None


def test_auto_fit_does_not_choose_4bit_on_amd_when_the_environment_refuses_it(monkeypatch):
    from kittenml.kittentts2.vllm.backend import VLLMBackend

    gib = 1 << 30
    lm = SimpleNamespace(hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
                         num_key_value_heads=8, head_dim=128, intermediate_size=6144,
                         vocab_size=158241, tie_word_embeddings=True)
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "RDNAHybridW4A16LinearKernel")
    with mock.patch.object(torch.version, "hip", "7.2.53211"), \
         mock.patch("torch.cuda.get_device_properties",
                    return_value=SimpleNamespace(gcnArchName="gfx1201")), \
         mock.patch("kittenml.kittentts2.vllm.backend._free_gpu_memory",
                    return_value=(int(7.39 * gib), int(7.62 * gib))):
        fitted, report = VLLMBackend._fit({}, lm)
    assert report["fitted"] and "quantization" not in fitted

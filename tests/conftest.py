"""Shared test setup."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _nvidia_platform_for_nvidia_tests(request, monkeypatch):
    """The vLLM tests describe NVIDIA GPUs (none of them use a GPU). On a ROCm build of PyTorch
    they would take the AMD paths, so make it look like a CUDA build for them;
    tests/test_vllm_rocm.py sets the platform itself."""
    if getattr(torch.version, "hip", None) and not request.module.__name__.endswith("test_vllm_rocm"):
        monkeypatch.setattr(torch.version, "hip", None)


@pytest.fixture(autouse=True)
def _no_real_gpu_memory(monkeypatch):
    """Unit tests never size vLLM from the real GPU of the machine they run on (on a GPU host
    the small-GPU fitting would otherwise read its free memory and size the tests' stub model
    configs). Tests that need free/total numbers patch _free_gpu_memory themselves."""
    def no_gpu():
        raise RuntimeError("unit tests do not read the real GPU's free memory")
    try:
        import kittenml.kittentts2.vllm.backend as backend
    except ImportError:
        return
    monkeypatch.setattr(backend, "_free_gpu_memory", no_gpu)

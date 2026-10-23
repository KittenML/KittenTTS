"""Memory on small GPUs: the stock flow is not loaded under the student decoder, and
Whisper moves to the CPU when the GPU is short of memory. No GPU or download needed."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch


def test_s3gen_without_its_flow_has_every_other_weight_and_keeps_its_dtype():
    from kittenml.kittentts2.s3gen import S3Gen

    with torch.device("meta"):
        full = S3Gen(meanflow=True)
        lean = S3Gen(meanflow=True, flow=False)
    assert lean.flow is None
    full_keys, lean_keys = set(full.state_dict()), set(lean.state_dict())
    assert lean_keys == {k for k in full_keys if not k.startswith("flow.")}
    assert any(k.startswith("flow.") for k in full_keys)
    assert lean.dtype == full.dtype == torch.float32
    flow_params = sum(p.numel() for p in full.flow.parameters())
    assert flow_params > 100_000_000   # the ~110M parameters the student path no longer holds


@pytest.mark.parametrize("student", [False, True])
def test_codec_loads_the_stock_flow_only_without_a_student(student):
    from kittenml.kittentts2 import vocoder

    checkpoint = {"flow.decoder.w": torch.zeros(1), "mel2wav.w": torch.zeros(1),
                  "tokenizer.w": torch.zeros(1)}
    with mock.patch.object(vocoder, "hf_hub_download", return_value="s3gen.safetensors"), \
         mock.patch.object(vocoder, "load_file", return_value=dict(checkpoint)), \
         mock.patch.object(vocoder, "S3Gen") as s3gen, \
         mock.patch.object(vocoder, "load_student_flow") as load_student:
        codec = vocoder.S3Codec(device="cpu", student_weights="student.pt" if student else None)
    s3gen.assert_called_once_with(meanflow=True, flow=not student)
    (state,), kwargs = s3gen.return_value.load_state_dict.call_args
    assert kwargs == {"strict": True}
    assert set(state) == (set(checkpoint) - {"flow.decoder.w"} if student else set(checkpoint))
    assert (codec.student is load_student.return_value) == student


def test_whisper_runs_on_the_gpu_with_room_and_on_the_cpu_without(caplog):
    from kittenml.kittentts2.speaker import WHISPER_GPU_BYTES, ReferenceTranscriber

    assert ReferenceTranscriber(device="cpu")._choose_device() == "cpu"
    gpu = ReferenceTranscriber(device="cuda")
    with mock.patch("torch.cuda.mem_get_info", return_value=(WHISPER_GPU_BYTES, 8 << 30)), \
         mock.patch("torch.cuda.memory_reserved", return_value=0), \
         mock.patch("torch.cuda.memory_allocated", return_value=0):
        assert gpu._choose_device() == "cuda"
    # Memory this process has cached but not used counts as free.
    with mock.patch("torch.cuda.mem_get_info", return_value=(WHISPER_GPU_BYTES - (1 << 30), 0)), \
         mock.patch("torch.cuda.memory_reserved", return_value=3 << 30), \
         mock.patch("torch.cuda.memory_allocated", return_value=2 << 30):
        assert gpu._choose_device() == "cuda"
    with mock.patch("torch.cuda.mem_get_info", return_value=(2 << 30, 8 << 30)), \
         mock.patch("torch.cuda.memory_reserved", return_value=0), \
         mock.patch("torch.cuda.memory_allocated", return_value=0), \
         caplog.at_level("WARNING"):
        assert gpu._choose_device() == "cpu"
    # The cost is reported when the model is loaded, since it depends on which model that is.
    assert caplog.text == ""


def test_whisper_pipeline_follows_the_chosen_device(caplog):
    from kittenml.kittentts2.speaker import (WHISPER_CPU_FALLBACK_MODEL, WHISPER_MODEL,
                                             ReferenceTranscriber)

    # (device asked for, device chosen) -> pipeline device, dtype and model. Only the
    # fallback from a full GPU uses the smaller model; the GPU path and an explicit
    # device="cpu" keep large-v3.
    for asked, chosen, index, dtype, model in (
            ("cuda", "cuda", 0, torch.float16, WHISPER_MODEL),
            ("cuda", "cpu", -1, torch.float32, WHISPER_CPU_FALLBACK_MODEL),
            ("cpu", "cpu", -1, torch.float32, WHISPER_MODEL)):
        transcriber = ReferenceTranscriber(device=asked)
        pipeline = mock.Mock()
        caplog.clear()
        with mock.patch.object(ReferenceTranscriber, "_choose_device", return_value=chosen), \
             mock.patch.dict("sys.modules", {"transformers": SimpleNamespace(pipeline=pipeline)}), \
             caplog.at_level("WARNING"):
            transcriber._load()
        kwargs = pipeline.call_args.kwargs
        assert kwargs["device"] == index and kwargs["dtype"] is dtype and kwargs["model"] == model
        assert transcriber.resolved_device == chosen and transcriber.resolved_model == model
        if asked == "cuda" and chosen == "cpu":
            # Only the fallback from a full GPU is labelled, with the turbo model's cost and the
            # note that the transcript (and so the cloned voice) can differ from a GPU run.
            for text in ("supported but not the optimal path", "large-v3 turbo",
                         "4 GB of host memory", "can differ", "reference_text"):
                assert text in caplog.text
        else:
            assert caplog.text == ""
    # A configured transcriber other than the default is kept on the CPU too.
    assert ReferenceTranscriber(device="cuda", model_name="my/whisper")._model_for("cpu") == \
        "my/whisper"


def test_whisper_fallback_model_missing_offline_uses_the_configured_one(caplog):
    from kittenml.kittentts2.speaker import WHISPER_MODEL, ReferenceTranscriber

    def pipeline(task, model, **kwargs):
        if model != WHISPER_MODEL:
            raise OSError("not in the cache and offline")
        return mock.sentinel.pipeline

    transcriber = ReferenceTranscriber(device="cuda")
    with mock.patch.object(ReferenceTranscriber, "_choose_device", return_value="cpu"), \
         mock.patch.dict("sys.modules", {"transformers": SimpleNamespace(pipeline=pipeline)}), \
         caplog.at_level("WARNING"):
        assert transcriber._load() is mock.sentinel.pipeline
    assert transcriber.resolved_model == WHISPER_MODEL and "not available" in caplog.text
    # The slower second fallback is labelled with its own cost.
    for text in ("supported but not the optimal path", "43 s", "8.5 GB",
                 "openai/whisper-large-v3-turbo", "reference_text"):
        assert text in caplog.text


def test_cloning_with_a_transcript_never_loads_whisper(tmp_path):
    from kittenml.kittentts2.model import KittenTTS2

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"")
    owner = SimpleNamespace(_reference_cache={}, transcriber=mock.Mock(), speaker_encoder=mock.Mock(),
                            codec=mock.Mock(), _build_reference=mock.Mock(return_value="ref"))
    with mock.patch("kittenml.kittentts2.model.resolve_reference_path", return_value=str(clip)):
        assert KittenTTS2._clone_reference(owner, clip, "What the clip says.") == "ref"
        owner.transcriber.assert_not_called()
        KittenTTS2._clone_reference(owner, clip, None)
        owner.transcriber.assert_called_once_with(str(clip))

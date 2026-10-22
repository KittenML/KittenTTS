"""Reference transcription and public API regressions; no downloaded weights."""

import copy
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from kittenml.kittentts2.model import KittenTTS2
from kittenml.kittentts2.speaker import ReferenceTranscriber


def test_whisper_detects_or_uses_language_and_transcribes():
    # Exercise real Transformers generation with a tiny random model. Both legacy
    # config locations force English/translation; detection is replaced with Russian.
    config = WhisperConfig(
        vocab_size=32, num_mel_bins=8, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=8,
        max_target_positions=16, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        decoder_start_token_id=1, suppress_tokens=[], begin_suppress_tokens=[],
        forced_decoder_ids=[[1, 3], [2, 6]],
    )
    whisper = WhisperForConditionalGeneration(config).eval()
    generation_config = whisper.generation_config
    generation_config.language = "en"
    generation_config.task = "translate"
    generation_config.lang_to_id = {"<|en|>": 3, "<|ru|>": 4}
    generation_config.task_to_id = {"transcribe": 5, "translate": 6}
    generation_config.no_timestamps_token_id = 7
    generation_config.is_multilingual = True
    generation_config.max_new_tokens = 1
    # The pipeline has its own generation config, which may differ from the model's.
    pipeline_config = copy.deepcopy(generation_config)
    pipeline_config.forced_decoder_ids = [[1, 3], [2, 6]]
    decoder_prompts = []

    def capture_prompt(module, args, kwargs):
        decoder_prompts.append(kwargs["decoder_input_ids"].clone())

    whisper.register_forward_pre_hook(capture_prompt, with_kwargs=True)

    def transcribe(wav, generate_kwargs):
        assert generate_kwargs["task"] == "transcribe"
        with torch.inference_mode():
            whisper.generate(torch.zeros(1, 8, 16), generation_config=pipeline_config, **generate_kwargs)
        return {"text": "  Привет  "}

    pipeline = mock.Mock(side_effect=transcribe, generation_config=pipeline_config, model=whisper)
    transcriber = ReferenceTranscriber(device="cpu")
    librosa = SimpleNamespace(load=mock.Mock(return_value=(np.zeros(160), 16000)))
    with mock.patch.dict("sys.modules", {"librosa": librosa}), \
         mock.patch("transformers.pipeline", return_value=pipeline), \
         mock.patch.object(whisper, "detect_language", return_value=torch.tensor([4])) as detect:
        transcriber._load()
        original = copy.deepcopy(pipeline_config.to_dict())
        assert pipeline_config.language is None
        assert pipeline_config.forced_decoder_ids is None
        for language, language_id in [(None, 4), ("ru", 4), ("en", 3), (None, 4)]:
            detect.reset_mock()
            assert transcriber("sample.wav", reference_language=language) == "Привет"
            assert decoder_prompts[-1][0, :4].tolist() == [1, language_id, 5, 7]
            assert detect.call_count == (1 if language is None else 0)
    assert pipeline_config.to_dict() == original
    assert whisper.config.forced_decoder_ids is None
    assert whisper.generation_config.language is None


@pytest.fixture
def model():
    # Public methods and reference caching run unchanged; expensive components don't.
    instance = object.__new__(KittenTTS2)
    instance.default_preset = "stable"
    instance._reference_cache = {}
    instance.transcriber = ReferenceTranscriber(device="cpu")
    instance.speaker_encoder = mock.Mock()
    instance.codec = SimpleNamespace(encode_reference=mock.Mock(), sample_rate=24000)
    instance._build_reference = mock.Mock(side_effect=lambda **kwargs: kwargs)
    instance._resolve_advanced = mock.Mock(return_value={"chunk_gap_s": 0.16})
    instance._synthesize = mock.Mock(return_value=[np.zeros(8, dtype=np.float32)])
    instance._join = mock.Mock(side_effect=lambda wavs, gap: wavs[0])
    return instance


def call_public(model, method, tmp_path, **kwargs):
    if method == "generate_stream":
        return list(model.generate_stream("Hello", **kwargs))
    if method == "generate_to_file":
        return model.generate_to_file("Hello", tmp_path / "output.wav", **kwargs)
    return model.generate("Hello", **kwargs)


@pytest.mark.parametrize("method", ["generate", "generate_stream", "generate_to_file"])
def test_public_methods_forward_language_and_task(model, method, tmp_path):
    reference = tmp_path / "sample.wav"
    reference.touch()
    librosa = SimpleNamespace(load=mock.Mock(return_value=(np.zeros(160), 16000)))
    with mock.patch.dict("sys.modules", {"librosa": librosa}), \
         mock.patch("transformers.pipeline") as factory:
        factory.return_value.return_value = {"text": "Reference speech"}
        call_public(model, method, tmp_path, reference=reference, reference_language="ru")
    factory.return_value.assert_called_once_with(
        mock.ANY, generate_kwargs={"language": "ru", "task": "transcribe"})
    assert model._synthesize.call_args.args[0]["transcript"] == "Reference speech"


def test_manual_transcript_never_loads_whisper(model, tmp_path):
    reference = tmp_path / "sample.wav"
    reference.touch()
    with mock.patch.object(ReferenceTranscriber, "__call__",
                           side_effect=AssertionError("Whisper called")) as transcribe, \
         mock.patch.object(model.transcriber, "_load", side_effect=AssertionError("Whisper loaded")):
        for method in ("generate", "generate_stream", "generate_to_file"):
            call_public(model, method, tmp_path, reference=reference,
                        reference_text="Ручная расшифровка", reference_language="ignored-language")
    transcribe.assert_not_called()
    assert model.transcriber._pipeline is None
    assert model._synthesize.call_args.args[0]["transcript"] == "Ручная расшифровка"
    model._build_reference.assert_called_once()


def test_reference_cache_separates_languages_and_reuses_each_mode(model, tmp_path):
    reference = tmp_path / "sample.wav"
    reference.touch()
    model.transcriber = mock.Mock(side_effect=["Auto", "Russian", "English"])
    for language, expected in [(None, "Auto"), ("ru", "Russian"), ("en", "English")] * 2:
        model.generate("Hello", reference=reference, reference_language=language)
        assert model._synthesize.call_args.args[0]["transcript"] == expected
    assert model.transcriber.call_count == 3
    assert model._build_reference.call_count == 3

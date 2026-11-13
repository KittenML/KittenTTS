"""KittenTTS 2 nano / pico / mini: voice resolution, tokens and chunk post-processing without a model,
and, when KITTEN_TTS2_TINY_REPO points at a local repository (e.g. a kitten-tts-2-nano checkout), synthesis."""

import json
import os

import numpy as np
import pytest

from kittenml.kittentts2_tiny.model import END_TOKEN, KittenTTS2Tiny

VOCAB = {"$": 0, ",": 3, "…": 10, "h": 50, "i": 51, " ": 16}


def make_repo(tmp_path, languages=("en", "de")):
    cfg = dict(type="KITTEN2_TINY", name="test", phonemizer="v1", vocab="vocab.json", default_language="en",
               languages={lang: dict(model_file=f"m_{lang}.onnx", voices=f"v_{lang}.npz") for lang in languages},
               voices={"en": [dict(key="alpha", name="Alpha"), dict(key="shared", name="Shared")],
                       "de": [dict(key="anna", name="Anna", description="Female 1"), dict(key="shared", name="Shared")]})
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "vocab.json").write_text(json.dumps(VOCAB))
    for lang in languages:
        keys = [v["key"] for v in cfg["voices"][lang]]
        np.savez(tmp_path / f"v_{lang}.npz", **{k: np.full((510, 256), i, np.float32) for i, k in enumerate(keys)})
    return KittenTTS2Tiny(str(tmp_path))


def test_voice_resolution(tmp_path):
    m = make_repo(tmp_path)
    assert m.resolve_voice() == ("en", "alpha")
    assert m.resolve_voice(language="de") == ("de", "anna")
    assert m.resolve_voice("Anna") == ("de", "anna")              # unique name decides the language
    assert m.resolve_voice("anna") == ("de", "anna")              # any case
    assert m.resolve_voice("Female 1") == ("de", "anna")          # so does a unique description
    assert m.resolve_voice("Shared") == ("en", "shared")          # shared name: the default language wins
    assert m.resolve_voice("shared", language="de") == ("de", "shared")
    with pytest.raises(ValueError):
        m.resolve_voice("Nobody")
    with pytest.raises(ValueError):
        m.resolve_voice(language="xx")
    assert m.available_voices == {"en": ["Alpha", "Shared"], "de": ["Anna", "Shared"]}


def test_tokens(tmp_path):
    m = make_repo(tmp_path)
    # "(" and ")" become ","; symbols outside the vocab are skipped; [0] + ... + [end] + [0]
    assert m._tokens("hi (h)x") == [0, 50, 51, 16, 3, 50, 3, END_TOKEN, 0]


class FakeSession:
    """600 samples per frame of ones, durations [3, 2, ..., 2, 4]."""

    def __init__(self):
        self.feeds = None

    def run(self, _, feeds):
        self.feeds = feeds
        T = feeds["input_ids"].shape[1]
        dur = np.array([3] + [2] * (T - 2) + [4], np.int64)
        return np.ones(600 * int(dur.sum()), np.float32), dur


def test_chunk_postprocessing(tmp_path):
    m = make_repo(tmp_path)
    fake = FakeSession()
    m._sessions["m_en.onnx"] = fake
    tok_count = len(m._tokens("hi"))
    w_first = m._chunk("en", "shared", "hi", 1.0, keep_start=True)
    w_later = m._chunk("en", "shared", "hi", 1.0, keep_start=False)
    frames = 3 + 2 * (tok_count - 2) + 4
    assert len(w_first) == 600 * frames - 3 * 600                 # end token frames cut (all but one)
    assert len(w_later) == 600 * frames - 3 * 600 - 2 * 600       # and the start pause (all but one)
    assert w_first[0] == 0 and w_first[-1] == 0 and w_first[len(w_first) // 2] == 1   # fades
    # style row = token count - 1 of the voice pack; shared is pack 1 (filled with 1.0)
    assert fake.feeds["style"].shape == (1, 256) and float(fake.feeds["style"][0, 0]) == 1.0
    assert fake.feeds["speed"].dtype == np.float32


def test_backends(tmp_path):
    m = make_repo(tmp_path)
    assert m.runtime == "onnx" and m.providers is None
    calls = []
    m = KittenTTS2Tiny(str(tmp_path), backend="torch")
    assert m.runtime == "torch"
    m._torch_model = lambda lang: (lambda tok, style, speed: (calls.append((lang, len(tok), style.shape)),
                                                              FakeSession().run(None, {"input_ids": np.zeros((1, len(tok)))}))[1])
    m._chunk("en", "shared", "hi", 1.0, keep_start=True)
    assert calls == [("en", len(m._tokens("hi")), (1, 256))]
    assert KittenTTS2Tiny(str(tmp_path), backend="cuda").providers[0] == "CUDAExecutionProvider"
    assert KittenTTS2Tiny(str(tmp_path), device="cpu").providers == ["CPUExecutionProvider"]
    with pytest.raises(ValueError):
        KittenTTS2Tiny(str(tmp_path), backend="tensorflow")
    with pytest.raises(ValueError):                              # no weights_file in this repository
        KittenTTS2Tiny(str(tmp_path), backend="torch")._torch_model("en")


@pytest.mark.skipif(not os.environ.get("KITTEN_TTS2_TINY_REPO"), reason="KITTEN_TTS2_TINY_REPO not set")
@pytest.mark.parametrize("backend", ["onnx", "torch"])
def test_synthesis_local_repo(backend):
    from kittenml import KittenTTS
    m = KittenTTS(os.environ["KITTEN_TTS2_TINY_REPO"], backend=backend)
    assert isinstance(m, KittenTTS2Tiny) and m.runtime == backend
    lang = m.languages[0]
    text = {"en": "Hello there, this is a test.", "de": "Hallo, das ist ein Test."}.get(lang, "Hello.")
    wav = m.generate(text, language=lang)
    assert wav.dtype == np.float32 and 0.5 * m.sample_rate < len(wav) < 10 * m.sample_rate
    assert np.isfinite(wav).all() and 0.01 < np.abs(wav).max() <= 1.0
    fast = m.generate(text, language=lang, speed=1.5)
    assert len(fast) < len(wav)
    long = m.generate(" ".join([text] * 12), language=lang)       # several chunks
    assert len(long) > 5 * len(wav)

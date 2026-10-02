from kittenml.preprocess import NormalizedSpan, NormalizedTextResult, normalize_text, normalize_text_result

try:                                    # the installed distribution is the source of truth
    from importlib.metadata import version as _version

    __version__ = _version("kittenml")
except Exception:                       # running from a source tree without an install
    __version__ = "0.0.0.dev0"
__author__ = "KittenML"
__description__ = "Text-to-speech with voice cloning and expression control, plus ultra-lightweight models that run on CPU"

__all__ = [
    "get_model",
    "KittenTTS",
    "KittenTTS2",
    "KittenTTSOnnx",
    "normalize_text",
    "normalize_text_result",
    "NormalizedSpan",
    "NormalizedTextResult",
]


def __getattr__(name):
    if name in {"get_model", "KittenTTS"}:
        from kittenml.get_model import KittenTTS, get_model

        return {"get_model": get_model, "KittenTTS": KittenTTS}[name]
    # Imported lazily: KittenTTS 2 pulls in torch and transformers, which an
    # ONNX-only install does not have.
    if name == "KittenTTS2":
        from kittenml.kittentts2 import KittenTTS2

        return KittenTTS2
    # Likewise the ONNX backend, which needs espeak and phonemizer.
    if name == "KittenTTSOnnx":
        from kittenml.kittentts_legacy import KittenTTSOnnx

        return KittenTTSOnnx
    raise AttributeError(f"module 'kittenml' has no attribute {name!r}")

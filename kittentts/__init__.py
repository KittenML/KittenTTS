from kittentts.preprocess import NormalizedSpan, NormalizedTextResult, normalize_text, normalize_text_result

__version__ = "0.8.2"
__author__ = "KittenML"
__description__ = "Ultra-lightweight text-to-speech models powered by native inference"

__all__ = [
    "get_model",
    "KittenTTS",
    "normalize_text",
    "normalize_text_result",
    "NormalizedSpan",
    "NormalizedTextResult",
]


def __getattr__(name):
    if name in {"get_model", "KittenTTS"}:
        from kittentts.get_model import KittenTTS, get_model

        return {"get_model": get_model, "KittenTTS": KittenTTS}[name]
    raise AttributeError(f"module 'kittentts' has no attribute {name!r}")

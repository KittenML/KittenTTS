from kittentts.preprocess import NormalizedSpan, NormalizedTextResult, normalize_text, normalize_text_result

__version__ = "0.1.0"
__author__ = "KittenML"
__description__ = "Ultra-lightweight text-to-speech model with just 15 million parameters"

__all__ = [
    "get_model",
    "KittenTTS",
    "load_from_local",
    "normalize_text",
    "normalize_text_result",
    "NormalizedSpan",
    "NormalizedTextResult",
]


def __getattr__(name):
    if name in {"get_model", "KittenTTS", "load_from_local"}:
        from kittentts.get_model import KittenTTS, get_model, load_from_local

        return {"get_model": get_model, "KittenTTS": KittenTTS, "load_from_local": load_from_local}[name]
    raise AttributeError(f"module 'kittentts' has no attribute {name!r}")

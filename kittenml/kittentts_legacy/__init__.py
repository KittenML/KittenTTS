"""The original ONNX KittenTTS models (0.8 and earlier).

Loaded through `kittenml.KittenTTS` when a repository's config.json declares
`"type": "ONNX1"` or `"ONNX2"`. See `model.KittenTTSOnnx` for the API.
"""

from .model import KittenTTSOnnx, build_onnx_model, download_from_huggingface

__all__ = [
    "KittenTTSOnnx",
    "build_onnx_model",
    "download_from_huggingface",
]


def __getattr__(name):
    # Imported lazily: the ONNX runtime pulls in espeak and phonemizer, which a
    # KittenTTS 2 install has no reason to carry.
    if name == "KittenTTS_1_Onnx":
        from .onnx_model import KittenTTS_1_Onnx

        return KittenTTS_1_Onnx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

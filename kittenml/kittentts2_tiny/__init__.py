"""KittenTTS 2 nano / pico / mini: small multilingual models, ONNX (int8 weights) or PyTorch (float32).

Loaded through `kittenml.KittenTTS` when a repository's config.json declares `"type": "KITTEN2_TINY"`.
See `model.KittenTTS2Tiny` for the API.
"""

from .model import KittenTTS2Tiny

__all__ = ["KittenTTS2Tiny"]

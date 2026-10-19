"""The optional MLX backend: KittenTTS 2's language model on Apple-silicon GPU.

Selected with `KittenTTS("KittenML/kitten-tts-2", backend="mlx")`. Nothing here
is imported unless that backend is chosen, so `mlx` stays an optional dependency.
"""

from .backend import MLXBackend, validate_environment

__all__ = ["MLXBackend", "validate_environment"]

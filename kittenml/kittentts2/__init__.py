"""KittenTTS 2 — a speech language model with in-context voice cloning.

Loaded through `kittenml.KittenTTS` when a repository's config.json declares
`"type": "KITTEN2"`. See `model.KittenTTS2` for the API.
"""

from .model import KittenTTS2
from .text import (EXPRESSION_TAGS, VOCAL_EVENT_TAGS, has_expression_tag,
                   warm_normalizer)
from .tokens import TokenMap

__all__ = [
    "KittenTTS2",
    "TokenMap",
    "has_expression_tag",
    "warm_normalizer",
    "EXPRESSION_TAGS",
    "VOCAL_EVENT_TAGS",
]

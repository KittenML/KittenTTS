__version__ = "0.1.0"
__author__ = "KittenML"
__description__ = "Ultra-lightweight text-to-speech model with just 15 million parameters"

# Lazy imports - only load heavy dependencies when actually needed
def get_model(*args, **kwargs):
    """Lazy import of get_model"""
    from .get_model import get_model as _get_model
    return _get_model(*args, **kwargs)

def KittenTTS(*args, **kwargs):
    """Lazy import of KittenTTS"""
    from .get_model import KittenTTS as _KittenTTS
    return _KittenTTS(*args, **kwargs)

__all__ = ["get_model", "KittenTTS"]

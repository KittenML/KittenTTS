from kittentts.get_model import get_model, KittenTTS
from kittentts.realtime_engine import RealtimeKittenTTS, StreamingKittenTTS
from kittentts.utils import AudioBuffer, VoiceManager, PerformanceProfiler

__version__ = "0.2.0"
__author__ = ["Humair Munir", "KittenML"]
__description__ = "Ultra-lightweight real-time text-to-speech model with optimized streaming capabilities"

__all__ = [
    "get_model", 
    "KittenTTS", 
    "RealtimeKittenTTS", 
    "StreamingKittenTTS",
    "AudioBuffer",
    "VoiceManager", 
    "PerformanceProfiler"
]

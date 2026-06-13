import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from .cpp_model import KittenTTS_1_Cpp
from .preprocess import normalize_text


DEFAULT_MODEL = "KittenML/kitten-tts-nano-0.1"
NATIVE_CONFIG_TYPES = {"CPP1", "NATIVE1", "MODEL_INFERENCE"}
NATIVE_WEIGHTS_REPO = "KittenML/meownn-models"
VOICE_NAMES = [
    "expr-voice-2-m",
    "expr-voice-2-f",
    "expr-voice-3-m",
    "expr-voice-3-f",
    "expr-voice-4-m",
    "expr-voice-4-f",
    "expr-voice-5-m",
    "expr-voice-5-f",
]
DEFAULT_VOICE_ALIASES = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}
BUILTIN_NATIVE_MODELS = {
    "kittenml/kitten-tts-nano-0.1": {
        "arch_file": "kitten_fp32_15m_arch.json",
        "weights_file": "kitten_fp32_15m.bin",
        "voice_dir": "voices_kitten_15m",
    },
    "kittenml/kitten-tts-nano-0.8": {
        "arch_file": "kitten_fp32_15m_arch.json",
        "weights_file": "kitten_fp32_15m.bin",
        "voice_dir": "voices_kitten_15m",
    },
    "kittenml/kitten-tts-nano-0.8-fp32": {
        "arch_file": "kitten_fp32_15m_arch.json",
        "weights_file": "kitten_fp32_15m.bin",
        "voice_dir": "voices_kitten_15m",
    },
    "kittenml/kitten-tts-nano-0.8-int8": {
        "arch_file": "kitten_int8_15m_arch.json",
        "weights_file": "kitten_int8_15m.bin",
        "voice_dir": "voices_kitten_15m",
    },
    "kittenml/kitten-tts-micro-0.8": {
        "arch_file": "kitten_fp32_40m_arch.json",
        "weights_file": "kitten_fp32_40m.bin",
        "voice_dir": "voices_kitten_40m",
    },
    "kittenml/kitten-tts-micro-0.8-int8": {
        "arch_file": "kitten_int8_40m_arch.json",
        "weights_file": "kitten_int8_40m.bin",
        "voice_dir": "voices_kitten_40m",
    },
    "kittenml/kitten-tts-mini-0.8": {
        "arch_file": "kitten_fp32_80m_arch.json",
        "weights_file": "kitten_fp32_80m.bin",
        "voice_dir": "voices_kitten_80m",
    },
    "kittenml/kitten-tts-mini-0.8-int8": {
        "arch_file": "kitten_int8_80m_arch.json",
        "weights_file": "kitten_int8_80m.bin",
        "voice_dir": "voices_kitten_80m",
    },
}


class KittenTTS:
    """Main KittenTTS class for text-to-speech synthesis."""
    
    def __init__(self, model_name=DEFAULT_MODEL, cache_dir=None, backend=None):
        """Initialize KittenTTS with a native model.
        
        Args:
            model_name: Hugging Face repository ID, local model directory, or a
                KittenML model name such as ``kitten-tts-mini-0.8``.
            cache_dir: Directory to cache downloaded files
            backend: Native engine backend. Defaults to ``cpu``; pass ``metal``
                on Apple Silicon when using a Metal-enabled kitten-inference
                wheel.
        """
        repo_id = _resolve_model_id(model_name)
        self.model = download_from_huggingface(repo_id=repo_id, cache_dir=cache_dir, backend=backend)
    
    def normalize_text(self, text, locale="en-US", return_spans=False):
        """Normalize text for TTS without generating audio."""
        return normalize_text(text, locale=locale, return_spans=return_spans)

    def generate(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        """Generate audio from text.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            
        Returns:
            Audio data as numpy array
        """
        return self.model.generate(text, voice=voice, speed=speed, clean_text=clean_text)

    def generate_stream(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        """Generate audio as a stream of chunks.

        Yields:
            numpy.ndarray: Audio data for each text chunk.
        """
        yield from self.model.generate_stream(text, voice=voice, speed=speed, clean_text=clean_text)

    def generate_to_file(
        self,
        text,
        output_path,
        voice="expr-voice-5-m",
        speed=1.0,
        sample_rate=24000,
        clean_text=True,
    ):
        """Generate audio from text and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
        """
        return self.model.generate_to_file(
            text,
            output_path,
            voice=voice,
            speed=speed,
            sample_rate=sample_rate,
            clean_text=clean_text,
        )
    
    @property
    def available_voices(self):
        """Get list of available voices."""
        return self.model.all_voice_names


def _resolve_local_file(root, filename):
    return filename if os.path.isabs(filename) else os.path.join(root, filename)


def _download_file(repo_id, filename, cache_dir):
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def _builtin_arch_path(filename):
    return str(Path(__file__).resolve().parent.joinpath("model_defs", filename))


def _builtin_native_spec(repo_id):
    return BUILTIN_NATIVE_MODELS.get(repo_id.lower())


def _resolve_model_id(model_name):
    if os.path.isdir(model_name):
        return model_name
    if "/" not in model_name:
        return f"KittenML/{model_name}"
    return model_name


def _config_file(config, primary, *fallbacks):
    for key in (primary,) + fallbacks:
        value = config.get(key)
        if value:
            return value
    raise ValueError(f"Native KittenTTS config is missing required field {primary!r}.")


def _load_local_model(repo_dir, backend=None):
    config_path = os.path.join(repo_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return _instantiate_native_model(
        config=config,
        get_file=lambda filename: _resolve_local_file(repo_dir, filename),
        backend=backend,
    )


def _load_builtin_native_model(repo_id, spec, cache_dir=None, backend=None):
    voice_paths = {
        voice: _download_file(
            NATIVE_WEIGHTS_REPO,
            f"{spec['voice_dir']}/{voice}.bin",
            cache_dir,
        )
        for voice in VOICE_NAMES
    }

    return KittenTTS_1_Cpp(
        arch_path=_builtin_arch_path(spec["arch_file"]),
        weights_path=_download_file(NATIVE_WEIGHTS_REPO, spec["weights_file"], cache_dir),
        voice_paths=voice_paths,
        voice_aliases=DEFAULT_VOICE_ALIASES,
        backend=backend,
        style_index="token_count",
        sample_rate=24000,
    )


def _instantiate_native_model(config, get_file, backend=None):
    config_type = config.get("type")
    if config_type not in NATIVE_CONFIG_TYPES:
        if config_type in {"ONNX1", "ONNX2"}:
            raise ValueError(
                "This KittenTTS build uses the native model_inference engine. "
                "Use a native model repo/config with type CPP1, arch_file, and weights_file."
            )
        raise ValueError(f"Unsupported model type: {config_type!r}.")

    arch_path = get_file(_config_file(config, "arch_file", "arch"))
    weights_path = get_file(_config_file(config, "weights_file", "weights"))

    voices_path = None
    if config.get("voices"):
        voices_path = get_file(config["voices"])

    voice_paths = {
        voice: get_file(filename)
        for voice, filename in (config.get("voice_files") or {}).items()
    }

    return KittenTTS_1_Cpp(
        arch_path=arch_path,
        weights_path=weights_path,
        voices_path=voices_path,
        voice_paths=voice_paths,
        speed_priors=config.get("speed_priors", {}),
        voice_aliases=config.get("voice_aliases", {}),
        backend=backend,
        style_index=config.get("style_index", "token_count"),
        sample_rate=config.get("sample_rate", 24000),
    )


def download_from_huggingface(repo_id=DEFAULT_MODEL, cache_dir=None, backend=None):
    """Download model files from Hugging Face repository.
    
    Args:
        repo_id: Hugging Face repository ID
        cache_dir: Directory to cache downloaded files
        
    Returns:
        KittenTTS_1_Cpp: Instantiated model ready for use
    """
    if os.path.isdir(repo_id):
        return _load_local_model(repo_id, backend=backend)

    builtin_spec = _builtin_native_spec(repo_id)
    if builtin_spec is not None:
        return _load_builtin_native_model(
            repo_id=repo_id,
            spec=builtin_spec,
            cache_dir=cache_dir,
            backend=backend,
        )

    # Download config file first
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        cache_dir=cache_dir
    )
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    return _instantiate_native_model(
        config=config,
        get_file=lambda filename: _download_file(repo_id, filename, cache_dir),
        backend=backend,
    )


def get_model(repo_id=DEFAULT_MODEL, cache_dir=None, backend=None):
    """Get a KittenTTS model (legacy function for backward compatibility)."""
    return KittenTTS(repo_id, cache_dir, backend=backend)

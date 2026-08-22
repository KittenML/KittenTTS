import json
import os
from pathlib import Path
from .preprocess import normalize_text


class KittenTTS:
    """Main KittenTTS class for text-to-speech synthesis."""
    
    def __init__(self, model_name="KittenML/kitten-tts-nano-0.8", cache_dir=None, backend=None):
        """Initialize KittenTTS with a model from Hugging Face.
        
        Args:
            model_name: Hugging Face repository ID, model name, or local model directory
            cache_dir: Directory to cache downloaded files
        """
        local_model_path = _local_model_path(model_name)
        if local_model_path is not None:
            self.model = load_from_local(local_model_path, backend=backend)
            return

        # Handle different model name formats
        if "/" not in model_name:
            # If just model name provided, assume it's from KittenML
            repo_id = f"KittenML/{model_name}"
        else:
            repo_id = model_name
            
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
        print(f"Generating audio for text: {text}")
        return self.model.generate(text, voice=voice, speed=speed, clean_text=clean_text)

    def generate_stream(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        """Generate audio as a stream of chunks.

        Yields:
            numpy.ndarray: Audio data for each text chunk.
        """
        yield from self.model.generate_stream(text, voice=voice, speed=speed, clean_text=clean_text)

    def generate_to_file(self, text, output_path, voice="expr-voice-5-m", speed=1.0, sample_rate=24000):
        """Generate audio from text and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
        """
        return self.model.generate_to_file(text, output_path, voice=voice, speed=speed, sample_rate=sample_rate)
    
    @property
    def available_voices(self):
        """Get list of available voices."""
        return self.model.all_voice_names


def download_from_huggingface(repo_id="KittenML/kitten-tts-nano-0.1", cache_dir=None, backend=None):
    """Download model files from Hugging Face repository.
    
    Args:
        repo_id: Hugging Face repository ID
        cache_dir: Directory to cache downloaded files
        
    Returns:
        KittenTTS_1_Onnx: Instantiated model ready for use
    """
    from huggingface_hub import hf_hub_download

    # Download config file first
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        cache_dir=cache_dir
    )
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    if config.get("type") not in ["ONNX1", "ONNX2"]:
        raise ValueError("Unsupported model type.")

    # Download model and voices files based on config
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=config["model_file"],
        cache_dir=cache_dir
    )
    
    voices_path = hf_hub_download(
        repo_id=repo_id,
        filename=config["voices"],
        cache_dir=cache_dir
    )
    
    # Instantiate and return model
    model = _create_onnx_model(model_path=model_path, voices_path=voices_path, speed_priors=config.get("speed_priors", {}) , voice_aliases=config.get("voice_aliases", {}), backend=backend)
    
    return model


def load_from_local(model_path, backend=None):
    """Load model files directly from a local model directory.

    The directory must contain config.json plus the model and voice files named
    by that config.
    """
    model_dir = Path(model_path).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_dir}")

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Local model config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("type") not in ["ONNX1", "ONNX2"]:
        raise ValueError("Unsupported model type in local config.")

    try:
        model_file = config["model_file"]
        voices_file = config["voices"]
    except KeyError as exc:
        raise ValueError(f"Missing required local model config key: {exc.args[0]}") from exc

    model_file_path = model_dir / model_file
    voices_path = model_dir / voices_file
    for path in [model_file_path, voices_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Local model file not found: {path}")

    return _create_onnx_model(
        model_path=str(model_file_path),
        voices_path=str(voices_path),
        speed_priors=config.get("speed_priors", {}),
        voice_aliases=config.get("voice_aliases", {}),
        backend=backend,
    )


def get_model(repo_id="KittenML/kitten-tts-nano-0.1", cache_dir=None, backend=None):
    """Get a KittenTTS model (legacy function for backward compatibility)."""
    return KittenTTS(repo_id, cache_dir, backend=backend)


def _local_model_path(model_name):
    if not isinstance(model_name, (str, os.PathLike)):
        return None
    path = Path(model_name).expanduser()
    return path if path.exists() else None


def _create_onnx_model(**kwargs):
    from .onnx_model import KittenTTS_1_Onnx

    return KittenTTS_1_Onnx(**kwargs)

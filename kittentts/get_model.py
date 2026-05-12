import json
import os
from importlib import metadata

from .analytics import AnalyticsClient, error_code, parse_model_name


class KittenTTS:
    """Main KittenTTS class for text-to-speech synthesis."""
    
    def __init__(self, model_name="KittenML/kitten-tts-nano-0.8", cache_dir=None, backend=None, analytics=True):
        """Initialize KittenTTS with a model from Hugging Face.
        
        Args:
            model_name: Hugging Face repository ID or model name
            cache_dir: Directory to cache downloaded files
            analytics: Set to False to disable anonymous generation analytics
        """
        # Handle different model name formats
        if "/" not in model_name:
            # If just model name provided, assume it's from KittenML
            repo_id = f"KittenML/{model_name}"
        else:
            repo_id = model_name
            
        self.model = download_from_huggingface(repo_id=repo_id, cache_dir=cache_dir, backend=backend)
        model_info = parse_model_name(repo_id)
        self.analytics = AnalyticsClient(
            sdk_version=_sdk_version(),
            selected_model=model_info["selected_model"],
            model_version=model_info["model_version"],
            asset_source=getattr(self.model, "analytics_asset_source", "runtime-download"),
            enabled=analytics,
        )
    
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
        try:
            audio = self.model.generate(text, voice=voice, speed=speed, clean_text=clean_text)
        except Exception as exc:
            self._track_generation(voice, generation="wav", sdk_error_code=error_code(exc))
            raise
        self._track_generation(voice, generation="wav")
        return audio

    def generate_stream(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        """Generate audio as a stream of chunks.

        Yields:
            numpy.ndarray: Audio data for each text chunk.
        """
        try:
            yield from self.model.generate_stream(text, voice=voice, speed=speed, clean_text=clean_text)
        except Exception as exc:
            self._track_generation(voice, generation="speak", sdk_error_code=error_code(exc))
            raise
        self._track_generation(voice, generation="speak")

    def generate_to_file(self, text, output_path, voice="expr-voice-5-m", speed=1.0, sample_rate=24000):
        """Generate audio from text and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
        """
        try:
            result = self.model.generate_to_file(text, output_path, voice=voice, speed=speed, sample_rate=sample_rate)
        except Exception as exc:
            self._track_generation(voice, generation="wav", sdk_error_code=error_code(exc))
            raise
        self._track_generation(voice, generation="wav")
        return result
    
    @property
    def available_voices(self):
        """Get list of available voices."""
        return self.model.all_voice_names

    def _track_generation(self, voice, generation, sdk_error_code=None):
        try:
            self.analytics.track_generation(
                selected_voice=voice,
                generation=generation,
                sdk_error_code=sdk_error_code,
            )
        except Exception:
            return


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
    config_was_cached = _is_cached(repo_id, "config.json", cache_dir)
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

    model_was_cached = _is_cached(repo_id, config["model_file"], cache_dir)
    voices_were_cached = _is_cached(repo_id, config["voices"], cache_dir)

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
    
    from .onnx_model import KittenTTS_1_Onnx

    # Instantiate and return model
    model = KittenTTS_1_Onnx(model_path=model_path, voices_path=voices_path, speed_priors=config.get("speed_priors", {}) , voice_aliases=config.get("voice_aliases", {}), backend=backend)
    model.analytics_asset_source = "cache" if config_was_cached and model_was_cached and voices_were_cached else "runtime-download"
    
    return model


def get_model(repo_id="KittenML/kitten-tts-nano-0.1", cache_dir=None, backend=None, analytics=True):
    """Get a KittenTTS model (legacy function for backward compatibility)."""
    return KittenTTS(repo_id, cache_dir, backend=backend, analytics=analytics)


def _is_cached(repo_id, filename, cache_dir):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    try:
        cached_path = try_to_load_from_cache(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    except Exception:
        return False
    return isinstance(cached_path, str) and os.path.exists(cached_path)


def _sdk_version():
    try:
        return metadata.version("kittentts")
    except metadata.PackageNotFoundError:
        return "unknown"

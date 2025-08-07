import json
import os
from huggingface_hub import hf_hub_download
from .onnx_model import KittenTTS_1_Onnx


class KittenTTS:
    """Main KittenTTS class for text-to-speech synthesis."""
    
    def __init__(self, model_name="KittenML/kitten-tts-nano-0.1", cache_dir=None):
        """Initialize KittenTTS with a model from Hugging Face.
        
        Args:
            model_name: Hugging Face repository ID or model name
            cache_dir: Directory to cache downloaded files
        """
        # Handle different model name formats
        if "/" not in model_name:
            # If just model name provided, assume it's from KittenML
            repo_id = f"KittenML/{model_name}"
        else:
            repo_id = model_name
            
        self.model = download_from_huggingface(repo_id=repo_id, cache_dir=cache_dir)
    
    def generate(self, text, voice="expr-voice-5-m", speed=1.0):
        """Generate audio from text.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            
        Returns:
            Audio data as numpy array
        """
        return self.model.generate(text, voice=voice, speed=speed)
    
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
        return self.model.available_voices


def download_from_huggingface(repo_id="KittenML/kitten-tts-nano-0.1", cache_dir=None):
    """Download model files from Hugging Face repository.
    
    Args:
        repo_id: Hugging Face repository ID
        cache_dir: Directory to cache downloaded files
        
    Returns:
        KittenTTS_1_Onnx: Instantiated model ready for use
    """
    # Download config file first
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        cache_dir=cache_dir
    )
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    if config.get("type") != "ONNX1":
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
    model = KittenTTS_1_Onnx(model_path=model_path, voices_path=voices_path)
    
    return model


def get_model(repo_id="KittenML/kitten-tts-nano-0.1", cache_dir=None):
    """Get a KittenTTS model (legacy function for backward compatibility)."""
    return KittenTTS(repo_id, cache_dir)

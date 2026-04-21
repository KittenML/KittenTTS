import json
import os
from typing import Dict, Any, Tuple
import numpy as np
from huggingface_hub import hf_hub_download
from .onnx_model import KittenTTS_1_Onnx


class KittenTTS:
    """Main KittenTTS class for text-to-speech synthesis."""
    
    def __init__(self, model_name="KittenML/kitten-tts-nano-0.8", cache_dir=None, backend=None):
        """Initialize KittenTTS with a model from Hugging Face.
        
        Args:
            model_name: Hugging Face repository ID or model name
            cache_dir: Directory to cache downloaded files
        """
        if "/" not in model_name:
            repo_id = f"KittenML/{model_name}"
        else:
            repo_id = model_name
            
        self.model = download_from_huggingface(repo_id=repo_id, cache_dir=cache_dir, backend=backend)
    
    def generate(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False) -> np.ndarray:
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

    def generate_with_subtitles(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate audio with subtitle information.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            clean_text: If true, preprocess text
            
        Returns:
            Tuple of (audio_data, subtitle_info)
            subtitle_info contains:
                - original: Original text
                - pinyin: Pinyin transcription (if Chinese)
                - processed: Processed text used for synthesis
                - language: Detected language ('chinese' or 'english')
                - segments: Detailed segment information
        """
        return self.model.generate_with_subtitles(text, voice=voice, speed=speed, clean_text=clean_text)

    def generate_stream(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        """Generate audio as a stream of chunks.

        Yields:
            numpy.ndarray: Audio data for each text chunk.
        """
        yield from self.model.generate_stream(text, voice=voice, speed=speed, clean_text=clean_text)

    def generate_to_file(self, text, output_path, voice="expr-voice-5-m", speed=1.0, sample_rate=24000) -> Dict[str, Any]:
        """Generate audio from text and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary containing subtitle information
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
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        cache_dir=cache_dir
    )
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    if config.get("type") not in ["ONNX1", "ONNX2"]:
        raise ValueError("Unsupported model type.")

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
    
    model = KittenTTS_1_Onnx(model_path=model_path, voices_path=voices_path, speed_priors=config.get("speed_priors", {}) , voice_aliases=config.get("voice_aliases", {}), backend=backend)
    
    return model


def get_model(repo_id="KittenML/kitten-tts-nano-0.1", cache_dir=None, backend=None):
    """Get a KittenTTS model (legacy function for backward compatibility)."""
    return KittenTTS(repo_id, cache_dir, backend=backend)

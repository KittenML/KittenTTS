"""The original ONNX KittenTTS models (0.8 and earlier).

These map phonemes to a waveform directly through a single ONNX graph: 15M-80M
parameters, 25-80 MB on disk, and fast on CPU with no GPU and no PyTorch. They
offer eight fixed voices and a speech-speed control, and no voice cloning or
expression tags — for those, see `kittenml.kittentts2`.

Reached through `kittenml.KittenTTS` when a repository's config.json declares
`"type": "ONNX1"` or `"ONNX2"`.
"""

import os

from huggingface_hub import hf_hub_download

from ..preprocess import normalize_text
from ..repo import ONNX_TYPES, resolve_repo


class KittenTTSOnnx:
    """An ONNX KittenTTS model.

    Constructed by `kittenml.KittenTTS`, which resolves the repository first and
    hands the result over through `_resolved`.
    """

    def __init__(self, model_name="KittenML/kitten-tts-nano-0.8", cache_dir=None,
                 backend=None, device=None, hf_token=None):
        """Initialize an ONNX KittenTTS model.

        Args:
            model_name: Hugging Face repository ID, bare model name, or local directory
            cache_dir: Directory to cache downloaded files
            backend: ONNX execution provider ("cpu", "cuda", "amd_gpu")
            device: Unused here; accepted so both backends share a signature
            hf_token: Unused here; accepted so both backends share a signature
        """
        local_dir, repo_id, config = getattr(self, "_resolved", None) or \
            resolve_repo(model_name, cache_dir)
        self.model = build_onnx_model(config, local_dir, repo_id, cache_dir, backend)

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


def build_onnx_model(config, local_dir, repo_id, cache_dir, backend):
    """Instantiate KittenTTS_1_Onnx from an already-resolved repository config."""
    # Imported here, not at module scope: the ONNX runtime pulls in espeak and
    # phonemizer, which a KittenTTS 2 install has no reason to carry.
    from .onnx_model import KittenTTS_1_Onnx

    def _fetch(filename):
        if local_dir is not None:
            return os.path.join(local_dir, filename)
        return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)

    return KittenTTS_1_Onnx(
        model_path=_fetch(config["model_file"]),
        voices_path=_fetch(config["voices"]),
        speed_priors=config.get("speed_priors", {}),
        voice_aliases=config.get("voice_aliases", {}),
        backend=backend,
    )


def download_from_huggingface(repo_id="KittenML/kitten-tts-nano-0.1", cache_dir=None, backend=None):
    """Download an ONNX model's files from Hugging Face and instantiate it.

    Returns:
        KittenTTS_1_Onnx: Instantiated model ready for use
    """
    local_dir, resolved_id, config = resolve_repo(repo_id, cache_dir)
    if config.get("type") not in ONNX_TYPES:
        raise ValueError("Unsupported model type.")
    return build_onnx_model(config, local_dir, resolved_id, cache_dir, backend)

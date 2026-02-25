import gc
import hashlib
import os
import signal
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import soundfile as sf
from huggingface_hub import hf_hub_download

from kittentts import KittenTTS


SAMPLE_RATE = 24000
MAX_TEXT_LENGTH = 5000
DOWNLOAD_RETRY_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY = 2
CLEANUP_INTERVAL = 60
AUDIO_CACHE_DIR = Path.home() / ".cache" / "kittentts" / "audio"

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

MSG_MODEL_ALREADY_LOADED = "Model already loaded on"
MSG_MODEL_LOADED = "Model loaded on"
MSG_DOWNLOAD_FAILED = "Download failed"
MSG_LOAD_FAILED = "Failed to load model"
MSG_UNEXPECTED_ERROR = "Unexpected error"
MSG_NO_MODEL_LOADED = "No model loaded"
MSG_INVALID_VOICE = "Invalid voice"
MSG_TEXT_REQUIRED = "Please enter some text to synthesize"
MSG_TEXT_TOO_LONG = "Text exceeds maximum length of"
MSG_AUDIO_GENERATED = "Audio generated successfully"
MSG_AUDIO_CACHED = "Audio generated successfully (cached)"
MSG_SAVE_FAILED = "Failed to save audio"
MSG_GENERATION_FAILED = "Generation failed"
MSG_GENERATION_SUCCESSFUL = "Generation successful!"
MSG_OUTPUT_CLEARED = "Output cleared"
MSG_ALL_UNLOADED = "All models unloaded"

MODEL_CONFIGS = {
    "KittenML/kitten-tts-mini-0.8": {
        "name": "Kitten TTS Mini (80M)",
        "model_file": "kitten_tts_mini_v0_8.onnx",
        "voices_file": "voices.npz",
    },
    "KittenML/kitten-tts-micro-0.8": {
        "name": "Kitten TTS Micro (40M)",
        "model_file": "kitten_tts_micro_v0_8.onnx",
        "voices_file": "voices.npz",
    },
    "KittenML/kitten-tts-nano-0.8-fp32": {
        "name": "Kitten TTS Nano FP32 (15M)",
        "model_file": "kitten_tts_nano_v0_8.onnx",
        "voices_file": "voices.npz",
    },
    "KittenML/kitten-tts-nano-0.8-int8": {
        "name": "Kitten TTS Nano INT8 (15M)",
        "model_file": "kitten_tts_nano_v0_8.onnx",
        "voices_file": "voices.npz",
    },
}

ALL_VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]

VOICE_DESCRIPTIONS = {
    "Bella": "Female (Voice 2)",
    "Jasper": "Male (Voice 2)",
    "Luna": "Female (Voice 3)",
    "Bruno": "Male (Voice 3)",
    "Rosie": "Female (Voice 4)",
    "Hugo": "Male (Voice 4)",
    "Kiki": "Female (Voice 5)",
    "Leo": "Male (Voice 5)",
}


class AudioCache:
    def __init__(self, ttl_hours: int = 24, max_size_mb: int = 500):
        self._cache_dir = AUDIO_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_hours * 3600
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _get_cache_key(self, text: str, voice: str, clean_text: bool) -> str:
        key_string = f"{text}|{voice}|{clean_text}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def get(self, text: str, voice: str, clean_text: bool) -> Optional[str]:
        cache_key = self._get_cache_key(text, voice, clean_text)
        file_path = self._cache_dir / f"{cache_key}.wav"
        with self._lock:
            if file_path.exists():
                file_age = time.time() - file_path.stat().st_mtime
                if file_age < self._ttl_seconds:
                    return str(file_path)
                try:
                    file_path.unlink()
                except Exception:
                    pass
        return None

    def set(self, text: str, voice: str, clean_text: bool, file_path: str) -> str:
        cache_key = self._get_cache_key(text, voice, clean_text)
        cached_path = self._cache_dir / f"{cache_key}.wav"
        try:
            with open(file_path, "rb") as src:
                with open(cached_path, "wb") as dst:
                    dst.write(src.read())
            return str(cached_path)
        except Exception:
            return file_path

    def _cleanup_loop(self) -> None:
        while not self._stop_flag.is_set():
            self._stop_flag.wait(CLEANUP_INTERVAL)
            if self._stop_flag.is_set():
                break
            self._cleanup()

    def _cleanup(self) -> None:
        with self._lock:
            try:
                total_size = 0
                files_with_age = []
                for file_path in self._cache_dir.glob("*.wav"):
                    file_size = file_path.stat().st_size
                    file_age = time.time() - file_path.stat().st_mtime
                    total_size += file_size
                    if file_age > self._ttl_seconds:
                        files_with_age.append((file_path, file_size))
                for file_path, _ in files_with_age:
                    try:
                        file_path.unlink()
                        total_size -= file_path.stat().st_size if file_path.exists() else 0
                    except Exception:
                        pass
                while total_size > self._max_size_bytes:
                    oldest_files = sorted(
                        self._cache_dir.glob("*.wav"),
                        key=lambda p: p.stat().st_mtime
                    )
                    if not oldest_files:
                        break
                    oldest = oldest_files[0]
                    file_size = oldest.stat().st_size
                    oldest.unlink()
                    total_size -= file_size
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_flag.set()


class ModelManager:
    def __init__(self):
        self.model: Optional[KittenTTS] = None
        self.current_model_name: Optional[str] = None
        self.current_device: str = "CPU"
        self.available_models = list(MODEL_CONFIGS.keys())
        self.download_dir = Path.home() / ".cache" / "kittentts" / "models"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.model_path: Optional[str] = None
        self._lock = threading.RLock()
        self._gpu_config_failed: bool = False
        self._temp_files: List[str] = []
        self._audio_cache = AudioCache()
        self._stop_flag = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_temp_files, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_temp_files(self) -> None:
        while not self._stop_flag.is_set():
            self._stop_flag.wait(CLEANUP_INTERVAL)
            if self._stop_flag.is_set():
                break
            with self._lock:
                files_to_remove = []
                for file_path in self._temp_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            files_to_remove.append(file_path)
                    except Exception:
                        pass
                for file_path in files_to_remove:
                    if file_path in self._temp_files:
                        self._temp_files.remove(file_path)

    def get_model_path(self, model_id: str) -> Path:
        return self.download_dir / model_id.replace("/", "_")

    def download_model(self, model_id: str) -> Tuple[bool, str]:
        model_path = self.get_model_path(model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        for attempt in range(DOWNLOAD_RETRY_ATTEMPTS):
            try:
                hf_hub_download(repo_id=model_id, filename="config.json", cache_dir=str(model_path))
                model_file = MODEL_CONFIGS[model_id]["model_file"]
                voices_file = MODEL_CONFIGS[model_id]["voices_file"]
                model_file_path = hf_hub_download(repo_id=model_id, filename=model_file, cache_dir=str(model_path))
                hf_hub_download(repo_id=model_id, filename=voices_file, cache_dir=str(model_path))
                with self._lock:
                    self.model_path = str(model_file_path)
                return True, "Downloaded"
            except Exception as e:
                if attempt < DOWNLOAD_RETRY_ATTEMPTS - 1:
                    time.sleep(DOWNLOAD_RETRY_DELAY * (attempt + 1))
                else:
                    return False, f"Failed after {DOWNLOAD_RETRY_ATTEMPTS} attempts: {str(e)}"
        return False, "Unknown error"

    def unload_current_model(self) -> None:
        with self._lock:
            self.model = None
            self.current_model_name = None
            self.current_device = "CPU"
            self.model_path = None
            self._gpu_config_failed = False
        gc.collect()

    def load_model(self, model_name: str, use_gpu: bool = False) -> Tuple[bool, str]:
        device = "GPU (CUDA)" if use_gpu else "CPU"
        with self._lock:
            if self.current_model_name == model_name and self.model is not None:
                if self.current_device != device:
                    self._perform_unload()
                else:
                    return True, f"{MSG_MODEL_ALREADY_LOADED} {device}"
            if self.model is not None:
                self._perform_unload()
        try:
            success, msg = self.download_model(model_name)
            if not success:
                return False, f"{MSG_DOWNLOAD_FAILED}: {msg}"
            model_path = self.get_model_path(model_name)
            model_path.mkdir(parents=True, exist_ok=True)
            try:
                new_model = KittenTTS(model_name=model_name, cache_dir=str(model_path))
                if use_gpu:
                    gpu_success = self._configure_gpu_session(new_model)
                    if not gpu_success:
                        device = "CPU (GPU config failed)"
                with self._lock:
                    self.model = new_model
                    self.current_model_name = model_name
                    self.current_device = device
                return True, f"{MSG_MODEL_LOADED} {device}"
            except Exception as e:
                with self._lock:
                    self.model = None
                    self.current_model_name = None
                    self.current_device = "CPU"
                return False, f"{MSG_LOAD_FAILED}: {str(e)}"
        except Exception as e:
            return False, f"{MSG_UNEXPECTED_ERROR}: {str(e)}"

    def _perform_unload(self) -> None:
        self.model = None
        self.current_model_name = None
        self.current_device = "CPU"
        self.model_path = None
        self._gpu_config_failed = False
        gc.collect()

    def _configure_gpu_session(self, model: KittenTTS) -> bool:
        try:
            import onnxruntime as ort
            if not hasattr(model, 'model') or not hasattr(model.model, 'session'):
                self._gpu_config_failed = True
                return False
            model_path = self.model_path
            if model_path is None:
                self._gpu_config_failed = True
                return False
            session_options = ort.SessionOptions()
            model.model.session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=[CUDA_PROVIDER, CPU_PROVIDER]
            )
            return True
        except Exception:
            self._gpu_config_failed = True
            return False

    def get_voices(self) -> List[str]:
        with self._lock:
            if self.model is None:
                return ALL_VOICES
            voices = self.model.available_voices
            return voices if voices else ALL_VOICES

    def validate_text_input(self, text: str) -> Tuple[bool, str]:
        if not text or not text.strip():
            return False, MSG_TEXT_REQUIRED
        if len(text) > MAX_TEXT_LENGTH:
            return False, f"{MSG_TEXT_TOO_LONG} {MAX_TEXT_LENGTH} characters"
        return True, ""

    def generate_speech(self, text: str, voice: str, clean_text: bool) -> Tuple[bool, Tuple[str, Optional[str]], bool]:
        is_valid, error_msg = self.validate_text_input(text)
        if not is_valid:
            return False, (error_msg, None), False
        cached_path = self._audio_cache.get(text, voice, clean_text)
        if cached_path:
            return True, (MSG_AUDIO_CACHED, cached_path), True
        with self._lock:
            if self.model is None:
                return False, (MSG_NO_MODEL_LOADED, None), False
            if voice not in ALL_VOICES:
                return False, (f"{MSG_INVALID_VOICE}: {voice}", None), False
        try:
            audio = self.model.generate(text=text, voice=voice, speed=1.0, clean_text=clean_text)
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            try:
                os.close(fd)
                os.chmod(output_path, stat.S_IRUSR | stat.S_IWUSR)
                sf.write(output_path, audio, SAMPLE_RATE)
                cached_path = self._audio_cache.set(text, voice, clean_text, output_path)
                with self._lock:
                    self._temp_files.append(output_path)
                return True, (MSG_AUDIO_GENERATED, cached_path if cached_path else output_path), False
            except Exception as e:
                try:
                    os.remove(output_path)
                except Exception:
                    pass
                return False, (f"{MSG_SAVE_FAILED}: {str(e)}", None), False
        except Exception as e:
            return False, (f"{MSG_GENERATION_FAILED}: {str(e)}", None), False

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        with self._lock:
            if model_name is None:
                return self.model is not None
            return self.current_model_name == model_name and self.model is not None

    def unload_all_models(self) -> str:
        self.unload_current_model()
        return MSG_ALL_UNLOADED

    def stop(self) -> None:
        self._stop_flag.set()
        self._audio_cache.stop()


model_manager = ModelManager()


def signal_handler(signum: int, frame: object) -> None:
    model_manager.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_model_choices() -> List[str]:
    return [f"{cfg['name']} ({model_id})" for model_id, cfg in MODEL_CONFIGS.items()]


def parse_model_choice(choice: str) -> str:
    for model_id, cfg in MODEL_CONFIGS.items():
        if f"{cfg['name']} ({model_id})" == choice:
            return model_id
    return choice


def parse_voice_choice(voice_choice: str) -> str:
    if " - " in voice_choice:
        voice = voice_choice.split(" - ")[0]
        if voice in ALL_VOICES:
            return voice
    if voice_choice in ALL_VOICES:
        return voice_choice
    return ALL_VOICES[0]


def load_model_fn(model_choice: str, use_gpu: bool):
    model_name = parse_model_choice(model_choice)
    success, msg = model_manager.load_model(model_name, use_gpu)
    if success:
        voices = model_manager.get_voices()
        voice_choices = [f"{v} - {VOICE_DESCRIPTIONS.get(v, '')}" for v in voices]
        return (
            msg,
            gr.Dropdown(choices=voice_choices, value=voice_choices[0] if voice_choices else None, interactive=True),
            gr.Button(interactive=True),
        )
    else:
        return f"Error loading model: {msg}", gr.Dropdown(choices=[], interactive=False), gr.Button(interactive=False)


def generate_speech_fn(
    text: str,
    model_choice: str,
    voice_choice: str,
    clean_text: bool,
    use_gpu: bool
):
    model_name = parse_model_choice(model_choice)
    voice = parse_voice_choice(voice_choice)

    if not model_manager.is_model_loaded(model_name):
        success, msg = model_manager.load_model(model_name, use_gpu=use_gpu)
        if not success:
            return None, f"Error: {msg}", gr.Button(interactive=True), gr.Markdown(visible=False)
    success, result, is_cached = model_manager.generate_speech(text, voice, clean_text)
    if success:
        _, file_path = result
        status_msg = MSG_GENERATION_SUCCESSFUL
        if is_cached:
            status_msg += " 📦"
        return file_path, status_msg, gr.Button(interactive=True), gr.Markdown(value="📦 Cached", visible=is_cached)
    else:
        error_msg, _ = result
        return None, f"Error: {error_msg}", gr.Button(interactive=True), gr.Markdown(visible=False)


def unload_all_fn(model_status: str):
    msg = model_manager.unload_all_models()
    return msg


def update_char_count(text: str) -> str:
    return f"**{len(text)}** / {MAX_TEXT_LENGTH} characters"


def clear_output() -> Tuple[None, str, gr.Button, gr.Markdown]:
    return None, MSG_OUTPUT_CLEARED, gr.Button(interactive=True), gr.Markdown(visible=False)


CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    font-weight: 600;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}
.gr-box {
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
.gr-input-label, .gr-radio-label {
    font-weight: 500;
}
.gr-form {
    background: #f9fafb;
}
.warning-text {
    color: #f59e0b;
    font-weight: 600;
}
"""

with gr.Blocks(title="KittenTTS") as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="font-size: 2.5em; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            KittenTTS
        </h1>
        <p style="color: #6b7280; font-size: 1.1em;">Ultra-lightweight Text-to-Speech Synthesis</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Device Settings")
            gpu_checkbox = gr.Checkbox(label="Use GPU (CUDA)", value=False, info="Enable GPU acceleration if CUDA is available")

            gr.Markdown("### Model Selection")
            model_dropdown = gr.Dropdown(choices=get_model_choices(), value=get_model_choices()[0], label="Select Model", info="Choose model size and quality level")
            load_model_btn = gr.Button("Load Model", variant="primary", size="lg")
            unload_model_btn = gr.Button("Unload All Models", variant="secondary", size="sm")
            model_status = gr.Textbox(label="Model Status", interactive=False, lines=3)

            gr.Markdown("### Voice Settings")
            voice_dropdown = gr.Dropdown(choices=[], label="Select Voice", info="Choose from 8 available voices", interactive=False)

            clean_text_checkbox = gr.Checkbox(label="Clean Text", value=True, info="Expand numbers, symbols, currencies, etc.")

            gr.Markdown("### Input Text")
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text here...",
                lines=5,
                max_lines=10,
            )
            char_count = gr.Markdown("**0** / 5000 characters")

            with gr.Row():
                generate_btn = gr.Button("Generate Speech", variant="primary", size="lg", interactive=False)
                clear_btn = gr.Button("Clear Output", variant="secondary", size="lg")

            cached_indicator = gr.Markdown(visible=False)

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            output_status = gr.Textbox(label="Status", interactive=False)
            audio_output = gr.Audio(label="Generated Audio", type="filepath")

    text_input.change(fn=update_char_count, inputs=[text_input], outputs=[char_count])

    clear_btn.click(fn=clear_output, outputs=[audio_output, output_status, generate_btn, cached_indicator])

    load_model_btn.click(fn=load_model_fn, inputs=[model_dropdown, gpu_checkbox], outputs=[model_status, voice_dropdown, generate_btn])
    unload_model_btn.click(fn=unload_all_fn, inputs=[model_status], outputs=[model_status])
    generate_btn.click(fn=generate_speech_fn, inputs=[text_input, model_dropdown, voice_dropdown, clean_text_checkbox, gpu_checkbox], outputs=[audio_output, output_status, generate_btn, cached_indicator])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        allowed_paths=[str(AUDIO_CACHE_DIR)],
    )

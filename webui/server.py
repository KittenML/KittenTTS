import io
import base64
import tempfile
import time
import os
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
CACHE_DIR = Path.home() / ".cache" / "kittentts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "kitten-tts-mini": "KittenML/kitten-tts-mini-0.8",
    "kitten-tts-micro": "KittenML/kitten-tts-micro-0.8",
    "kitten-tts-nano": "KittenML/kitten-tts-nano-0.8-fp32",
    "kitten-tts-nano-int8": "KittenML/kitten-tts-nano-0.8-int8",
}

VOICE_ALIASES = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}

VOICES = [
    {
        "id": "Bella",
        "name": "Bella",
        "gender": "female",
        "description": "Warm & gentle",
    },
    {
        "id": "Jasper",
        "name": "Jasper",
        "gender": "male",
        "description": "Clear & professional",
    },
    {"id": "Luna", "name": "Luna", "gender": "female", "description": "Soft & melodic"},
    {
        "id": "Bruno",
        "name": "Bruno",
        "gender": "male",
        "description": "Deep & resonant",
    },
    {
        "id": "Rosie",
        "name": "Rosie",
        "gender": "female",
        "description": "Bright & cheerful",
    },
    {
        "id": "Hugo",
        "name": "Hugo",
        "gender": "male",
        "description": "Confident & steady",
    },
    {
        "id": "Kiki",
        "name": "Kiki",
        "gender": "female",
        "description": "Playful & energetic",
    },
    {"id": "Leo", "name": "Leo", "gender": "male", "description": "Friendly & warm"},
]

MODEL_INFO = [
    {
        "id": "kitten-tts-nano",
        "name": "Nano (FP32)",
        "params": "15M",
        "size": "56MB",
        "description": "⭐ Best quality - Full 32-bit precision",
        "quality": "best",
        "precision": "FP32",
    },
    {
        "id": "kitten-tts-mini",
        "name": "Mini (INT8)",
        "params": "80M",
        "size": "80MB",
        "description": "Largest model, INT8 quantized",
        "quality": "good",
        "precision": "INT8",
    },
    {
        "id": "kitten-tts-micro",
        "name": "Micro (INT8)",
        "params": "40M",
        "size": "41MB",
        "description": "Balanced size, INT8 quantized",
        "quality": "good",
        "precision": "INT8",
    },
    {
        "id": "kitten-tts-nano-int8",
        "name": "Nano (INT8)",
        "params": "15M",
        "size": "19MB",
        "description": "Smallest, INT8 quantized",
        "quality": "basic",
        "precision": "INT8",
    },
]

# In-memory model cache
loaded_models: Dict[str, Any] = {}
model_load_times: Dict[str, float] = {}

# Stats tracking
class StatsTracker:
    def __init__(self):
        self.total_requests = 0
        self.total_generation_time = 0.0
        self.total_audio_duration = 0.0
        self.request_history: list = []
        self.max_history = 50
        
    def record_request(self, model_id: str, voice: str, text_length: int, 
                       generation_time: float, audio_duration: float,
                       load_time: float = 0.0, preprocessing_time: float = 0.0):
        self.total_requests += 1
        self.total_generation_time += generation_time
        self.total_audio_duration += audio_duration
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model_id,
            "voice": voice,
            "text_length": text_length,
            "generation_time": round(generation_time, 3),
            "audio_duration": round(audio_duration, 3),
            "load_time": round(load_time, 3),
            "preprocessing_time": round(preprocessing_time, 3),
            "rtf": round(audio_duration / generation_time, 3) if generation_time > 0 else 0,
        }
        
        self.request_history.insert(0, entry)
        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[:self.max_history]
    
    def get_stats(self):
        avg_gen_time = (self.total_generation_time / self.total_requests) if self.total_requests > 0 else 0
        avg_rtf = (self.total_audio_duration / self.total_generation_time) if self.total_generation_time > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "avg_generation_time": round(avg_gen_time, 3),
            "avg_rtf": round(avg_rtf, 3),
            "total_audio_generated": round(self.total_audio_duration, 2),
            "recent_requests": self.request_history[:10],
        }

stats_tracker = StatsTracker()


class GenerateRequest(BaseModel):
    text: str
    model: str = "kitten-tts-nano"  # Default to FP32 for best quality
    voice: str = "Bella"
    speed: float = 1.0


class GenerateResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration: float
    debug_info: Optional[Dict[str, Any]] = None


def get_cache_size():
    """Calculate total cache size in MB."""
    total_size = 0
    if CACHE_DIR.exists():
        for dirpath, dirnames, filenames in os.walk(CACHE_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)


def get_model(model_id: str):
    """Get or load a model with caching and timing."""
    if model_id not in MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    if model_id not in loaded_models:
        from kittentts import KittenTTS
        
        start_time = time.time()
        repo_id = MODELS[model_id]
        loaded_models[model_id] = KittenTTS(repo_id, cache_dir=str(CACHE_DIR))
        load_time = time.time() - start_time
        model_load_times[model_id] = load_time
        print(f"[Model Load] {model_id} loaded in {load_time:.2f}s")
    
    return loaded_models[model_id], model_load_times.get(model_id, 0.0)


def create_app() -> FastAPI:
    app = FastAPI(
        title="KittenTTS WebUI",
        description="A cute kitten-themed text-to-speech web interface",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        template_path = Path(__file__).parent / "templates" / "index.html"
        if template_path.exists():
            return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Template not found")

    @app.get("/api/models")
    async def get_models():
        return {"models": MODEL_INFO}

    @app.get("/api/voices")
    async def get_voices():
        return {"voices": VOICES}

    @app.get("/api/health")
    async def health_check():
        return {
            "status": "healthy", 
            "loaded_models": list(loaded_models.keys()),
            "cache_dir": str(CACHE_DIR),
            "cache_size_mb": get_cache_size(),
        }

    @app.get("/api/stats")
    async def get_stats():
        """Get detailed stats for debugging."""
        import sys
        
        # Try to get memory usage, fallback if psutil not available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = round(memory_info.rss / (1024 * 1024), 2)
        except:
            memory_mb = "N/A"
        
        return {
            "generation_stats": stats_tracker.get_stats(),
            "system": {
                "cache_directory": str(CACHE_DIR),
                "cache_size_mb": get_cache_size(),
                "loaded_models": list(loaded_models.keys()),
                "model_load_times": {k: round(v, 3) for k, v in model_load_times.items()},
                "python_version": sys.version.split()[0],
                "memory_usage_mb": memory_mb,
            },
            "available_models": list(MODELS.keys()),
            "available_voices": [v["id"] for v in VOICES],
        }

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate_audio(request: GenerateRequest):
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if request.speed < 0.25 or request.speed > 3.0:
            raise HTTPException(
                status_code=400, detail="Speed must be between 0.25 and 3.0"
            )

        try:
            # Load model with timing
            model_start = time.time()
            model, load_time = get_model(request.model)
            model_load_elapsed = time.time() - model_start
            
            voice_id = VOICE_ALIASES.get(request.voice, request.voice)
            
            # Generate audio with timing
            gen_start = time.time()
            audio = model.generate(
                text=request.text, voice=voice_id, speed=request.speed
            )
            generation_time = time.time() - gen_start
            
            if isinstance(audio, np.ndarray):
                audio_array = audio
            else:
                audio_array = np.array(audio)

            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()

            sample_rate = 24000
            duration = len(audio_array) / sample_rate

            # Ensure proper audio format for web playback
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize if needed to prevent clipping
            max_val = np.max(np.abs(audio_array))
            if max_val > 0.99:
                audio_array = audio_array * (0.99 / max_val)

            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format="WAV", subtype='PCM_16')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            # Record stats
            stats_tracker.record_request(
                model_id=request.model,
                voice=request.voice,
                text_length=len(request.text),
                generation_time=generation_time,
                audio_duration=duration,
                load_time=load_time if load_time > 0 else model_load_elapsed,
            )

            # Debug info for API response
            debug_info = {
                "model_load_time": round(load_time if load_time > 0 else model_load_elapsed, 3),
                "generation_time": round(generation_time, 3),
                "total_time": round(load_time + generation_time, 3),
                "real_time_factor": round(duration / generation_time, 3) if generation_time > 0 else 0,
                "audio_samples": len(audio_array),
                "sample_rate": sample_rate,
            }

            return GenerateResponse(
                audio_base64=audio_base64,
                sample_rate=sample_rate,
                duration=round(duration, 2),
                debug_info=debug_info,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/favicon.ico")
    async def favicon():
        return FileResponse(
            Path(__file__).parent / "static" / "favicon.svg", media_type="image/svg+xml"
        )

    return app


def run_server(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    app = create_app()
    print(f"\n🐱 KittenTTS WebUI starting at http://{host}:{port}")
    print(f"📁 Cache directory: {CACHE_DIR}")
    print("Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

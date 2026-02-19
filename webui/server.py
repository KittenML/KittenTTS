import io
import base64
import tempfile
from typing import Optional
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
        "id": "kitten-tts-mini",
        "name": "Mini",
        "params": "80M",
        "size": "80MB",
        "description": "Highest quality, larger model",
    },
    {
        "id": "kitten-tts-micro",
        "name": "Micro",
        "params": "40M",
        "size": "41MB",
        "description": "Balanced quality & speed",
    },
    {
        "id": "kitten-tts-nano",
        "name": "Nano",
        "params": "15M",
        "size": "56MB",
        "description": "Lightweight & fast",
    },
    {
        "id": "kitten-tts-nano-int8",
        "name": "Nano (INT8)",
        "params": "15M",
        "size": "19MB",
        "description": "Smallest, quantized",
    },
]

loaded_models = {}


class GenerateRequest(BaseModel):
    text: str
    model: str = "kitten-tts-mini"
    voice: str = "Bella"
    speed: float = 1.0


class GenerateResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration: float


def get_model(model_id: str):
    if model_id not in MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    if model_id not in loaded_models:
        from kittentts import KittenTTS

        repo_id = MODELS[model_id]
        loaded_models[model_id] = KittenTTS(repo_id)

    return loaded_models[model_id]


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
        return {"status": "healthy", "loaded_models": list(loaded_models.keys())}

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate_audio(request: GenerateRequest):
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if request.speed < 0.25 or request.speed > 3.0:
            raise HTTPException(
                status_code=400, detail="Speed must be between 0.25 and 3.0"
            )

        try:
            model = get_model(request.model)

            voice_id = VOICE_ALIASES.get(request.voice, request.voice)

            audio = model.generate(
                text=request.text, voice=voice_id, speed=request.speed
            )

            if isinstance(audio, np.ndarray):
                audio_array = audio
            else:
                audio_array = np.array(audio)

            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()

            sample_rate = 24000
            duration = len(audio_array) / sample_rate

            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format="WAV")
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            return GenerateResponse(
                audio_base64=audio_base64,
                sample_rate=sample_rate,
                duration=round(duration, 2),
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
    print("Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

"""A basic OpenAI-compatible speech API for KittenTTS 2.

Serves a subset of the hosted KittenML API from one local model:

    POST   /v1/audio/speech           speech as a complete file, streamed audio or SSE,
                                      optionally cloned from reference audio
    POST   /v1/audio/voices           save a cloned voice
    GET    /v1/voices                 list built-in and saved voices
    DELETE /v1/audio/voices/{id}      delete a saved voice
    GET    /health

Usage: python -m kittenml.kittentts2.docker.server [--host 0.0.0.0] [--port 8000].
KITTENML_BACKEND selects "vllm" (default) or "torch"; KITTENML_VLLM_OPTIONS takes
vllm_options as JSON; KITTENML_VOICES_DIR is where saved voices are kept. See
docs/docker.md.
"""
import argparse
import asyncio
import base64
import binascii
import concurrent.futures
import functools
import io
import json
import os
import pathlib
import re
import shutil
import struct
import tempfile
import time
import traceback
import uuid
from typing import Literal

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kittenml import KittenTTS

MODEL_IDS = ("kitten-tts-2-latest", "kitten-tts-2")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
REFERENCE_SECONDS = (2.0, 30.0)
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg")
VOICE_ID = re.compile(r"voice_[0-9a-f]{32}")
# response_format -> (soundfile format, subtype, media type)
FORMATS = {
    "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/pcm"),
}
# Formats that can be sent chunk by chunk; flac and opus are sent once complete.
PROGRESSIVE = ("mp3", "wav", "pcm")


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = MODEL_IDS[0]
    input: str = Field(min_length=1, max_length=40_000)
    voice: str | dict | None = None
    response_format: Literal["mp3", "wav", "flac", "opus", "pcm"] = "mp3"
    speed: float = 1.0
    stream: bool = True
    stream_format: Literal["audio", "sse"] = "audio"
    mode: Literal["stable", "expressive"] | None = None
    temperature: float | None = Field(None, ge=0.3, le=2.0)
    top_p: float | None = Field(None, ge=0.5, le=1.0)
    top_k: int | None = Field(None, ge=0, le=200)
    min_p: float | None = Field(None, ge=0.0, le=0.5)
    max_new_tokens: int | None = Field(None, ge=500, le=12_000)


class APIError(Exception):
    def __init__(self, status, message, code="invalid_request_error", param=None):
        super().__init__(message)
        self.status, self.message, self.code, self.param = status, message, code, param


def error_body(status, message, code, param=None):
    kind = "invalid_request_error" if status < 500 else "server_error"
    return {"error": {"message": message, "type": kind, "param": param, "code": code}}


class Engine:
    """Holds the model. Every model call runs on one worker thread, one at a time."""

    def __init__(self, backend, vllm_options, voices_dir):
        self.backend = backend
        self.voices_dir = pathlib.Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        options = {"vllm_options": vllm_options} if vllm_options else {}
        self.model = self.pool.submit(
            KittenTTS, "KittenML/kitten-tts-2",
            backend="vllm" if backend == "vllm" else None, **options).result()

    async def run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.pool, functools.partial(fn, *args, **kwargs))

    def forget(self, reference, transcript, directory=None):
        """Drop a cloned reference from the model's cache, and its files if temporary."""
        self.model._reference_cache.pop(("clone", str(reference), transcript), None)
        if directory:
            shutil.rmtree(directory, ignore_errors=True)


ENGINE: Engine = None
app = FastAPI(title="KittenTTS 2", docs_url=None, redoc_url=None)


@app.exception_handler(APIError)
async def _api_error(request, exc):
    return JSONResponse(error_body(exc.status, exc.message, exc.code, exc.param), exc.status)


@app.exception_handler(Exception)
async def _server_error(request, exc):
    traceback.print_exception(exc)
    return JSONResponse(error_body(500, "internal server error", "server_error"), 500)


def parse_request(fields):
    try:
        return SpeechRequest.model_validate(fields)
    except ValidationError as exc:
        first = exc.errors()[0]
        param = ".".join(str(p) for p in first["loc"]) or None
        raise APIError(400, f"{param}: {first['msg']}" if param else first["msg"],
                       param=param) from None


def save_reference(data, suffix, param):
    """Write uploaded reference audio to a temporary directory and check it."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise APIError(413, "reference audio is larger than 10 MiB", "request_too_large", param)
    suffix = suffix.lower() if suffix.lower() in AUDIO_SUFFIXES else ".wav"
    directory = tempfile.mkdtemp(prefix="kittenml-ref-")
    path = pathlib.Path(directory) / f"reference{suffix}"
    path.write_bytes(data)
    try:
        info = sf.info(str(path))
        seconds = info.frames / info.samplerate
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise APIError(400, "reference audio must be WAV, MP3, FLAC or OGG",
                       "invalid_audio", param) from None
    if not REFERENCE_SECONDS[0] <= seconds <= REFERENCE_SECONDS[1]:
        shutil.rmtree(directory, ignore_errors=True)
        raise APIError(400, "reference audio must be 2 to 30 seconds long",
                       "invalid_audio_duration", param)
    return path, directory


async def read_upload(upload, param):
    if upload is None or isinstance(upload, str):
        raise APIError(400, f"{param} must be an uploaded file", param=param)
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    return save_reference(data, pathlib.Path(upload.filename or "").suffix, param)


def load_saved_voice(voice_id):
    meta = ENGINE.voices_dir / voice_id / "voice.json"
    if not VOICE_ID.fullmatch(voice_id) or not meta.is_file():
        raise APIError(404, f"voice {voice_id!r} not found", "voice_not_found", "voice")
    return json.loads(meta.read_text())


def resolve_voice(voice):
    """Speaker arguments for generate(), and a temporary directory to remove afterwards."""
    if voice is None:
        return {}, None
    if isinstance(voice, dict) and set(voice) == {"id"}:
        voice = voice["id"]
    if isinstance(voice, str):
        if voice.startswith("voice_"):
            saved = load_saved_voice(voice)
            return {"reference": saved["reference"],
                    "reference_text": saved["reference_transcript"]}, None
        if voice not in ENGINE.model.available_voices:
            raise APIError(400, f"unknown voice {voice!r}; GET /v1/voices lists them",
                           param="voice")
        return {"voice": voice}, None
    inline = voice.get("reference_audio") if isinstance(voice, dict) else None
    if not isinstance(inline, dict) or not set(voice) <= {"reference_audio",
                                                            "reference_transcript"}:
        raise APIError(400, "voice must be a voice name, {\"id\": \"voice_...\"} or "
                            "{\"reference_audio\": {\"data\", \"format\"}}", param="voice")
    try:
        data = base64.b64decode(inline.get("data", ""), validate=True)
    except (binascii.Error, TypeError):
        raise APIError(400, "reference_audio.data must be Base64",
                       param="voice.reference_audio") from None
    path, directory = save_reference(data, "." + str(inline.get("format", "wav")),
                                     "voice.reference_audio")
    return {"reference": str(path),
            "reference_text": voice.get("reference_transcript") or None}, directory


def encode(audio, response_format, sample_rate):
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    if response_format == "pcm":
        return (audio * 32767).astype("<i2").tobytes()
    container, subtype, _ = FORMATS[response_format]
    # MP3 at a constant 96 kbps, so streamed pieces join into one readable stream.
    options = {"bitrate_mode": "CONSTANT", "compression_level": 0.4} \
        if container == "MP3" else {}
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=container, subtype=subtype, **options)
    return buffer.getvalue()


# Layer III bitrates for MPEG-2/2.5 and MPEG-1, and sample rates per MPEG version.
MP3_KBPS = ((0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
            (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320))
MP3_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


def without_mp3_tag(data):
    """Drop the encoder's leading Xing/Info frame. Its frame count covers one piece only,
    so decoders would stop after the first piece of a streamed response."""
    try:
        version = (data[1] >> 3) & 3  # 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5
        kbps = MP3_KBPS[version == 3][data[2] >> 4]
        rate = MP3_RATES[version][(data[2] >> 2) & 3]
        size = (144 if version == 3 else 72) * kbps * 1000 // rate + ((data[2] >> 1) & 1)
    except (IndexError, KeyError):
        return data
    frame = data[:size]
    return data[size:] if b"Xing" in frame or b"Info" in frame else data


def streaming_wav_header(sample_rate):
    # Unknown length: players read PCM until the connection closes.
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


def sse_event(payload):
    return f"data: {json.dumps(payload)}\n\n".encode()


@app.get("/health")
async def health():
    return {"status": "ok", "backend": ENGINE.backend, "sample_rate": ENGINE.model.sample_rate}


@app.post("/v1/audio/speech")
async def speech(request: Request):
    temporary = None
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        fields = {}
        for key, value in form.multi_items():
            if key in fields:
                raise APIError(400, f"duplicate field {key!r}", param=key)
            fields[key] = value
        if "voice" in fields:
            raise APIError(400, "multipart requests take reference_audio instead of voice",
                           param="voice")
        upload = fields.pop("reference_audio", None)
        if upload is None:
            raise APIError(400, "multipart speech requests need a reference_audio file; "
                                "send JSON for a built-in voice", param="reference_audio")
        transcript = fields.pop("reference_transcript", None) or None
        req = parse_request(fields)
        path, temporary = await read_upload(upload, "reference_audio")
        speaker = {"reference": str(path), "reference_text": transcript}
    else:
        try:
            body = await request.json()
        except ValueError:
            raise APIError(400, "send a JSON object, or multipart/form-data with "
                                "reference_audio") from None
        if not isinstance(body, dict):
            raise APIError(400, "the JSON body must be an object")
        req = parse_request(body)
        speaker, temporary = resolve_voice(req.voice)

    try:
        if req.model not in MODEL_IDS:
            raise APIError(400, f"model must be {MODEL_IDS[0]!r}", "model_not_found", "model")
        if req.speed != 1.0:
            raise APIError(400, "this server supports speed 1.0 only", param="speed")
        if not req.stream and req.stream_format == "sse":
            raise APIError(400, "\"stream\": false cannot be combined with "
                                "stream_format \"sse\"", param="stream_format")
    except APIError:
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    options = dict(speaker, preset=req.mode, temperature=req.temperature, top_p=req.top_p,
                   top_k=req.top_k, min_p=req.min_p)
    if req.max_new_tokens:
        options["max_new_tokens"] = req.max_new_tokens

    def cleanup():
        # Only one-request clones are forgotten; saved voices stay cached.
        if temporary:
            ENGINE.forget(options["reference"], options["reference_text"], temporary)

    rate = ENGINE.model.sample_rate
    media_type = FORMATS[req.response_format][2]
    progressive = req.stream and req.response_format in PROGRESSIVE

    if not progressive:
        # The complete file: generate() joins the chunks and smooths the seams.
        try:
            audio = await ENGINE.run(ENGINE.model.generate, req.input, **options)
        except ValueError as exc:
            raise APIError(400, str(exc)) from None
        finally:
            ENGINE.pool.submit(cleanup)
        data = encode(audio, req.response_format, rate)
        if req.stream_format != "sse":
            return Response(data, media_type=media_type)
        events = [sse_event({"type": "speech.audio.delta",
                             "audio": base64.b64encode(data).decode()}),
                  sse_event({"type": "speech.audio.done",
                             "usage": usage(len(audio) / rate, req.input)})]
        return StreamingResponse(iter(events), media_type="text/event-stream")

    # Streaming: one piece per sentence chunk, as generate_stream yields them. The first
    # chunk is produced before responding, so a failure still returns a proper error.
    chunks = ENGINE.model.generate_stream(req.input, **options)
    try:
        first = await ENGINE.run(next, chunks, None)
    except ValueError as exc:
        ENGINE.pool.submit(chunks.close)
        ENGINE.pool.submit(cleanup)
        raise APIError(400, str(exc)) from None
    except BaseException:
        ENGINE.pool.submit(chunks.close)
        ENGINE.pool.submit(cleanup)
        raise
    if first is None:
        ENGINE.pool.submit(cleanup)
        raise APIError(500, "the model produced no audio", "server_error")

    async def body():
        wav, seconds, index = first, 0.0, 0
        try:
            while wav is not None:
                if req.response_format == "wav":
                    # Each chunk as raw PCM after one streaming header.
                    piece = ((streaming_wav_header(rate) if index == 0 else b"")
                             + encode(wav, "pcm", rate))
                elif req.response_format == "mp3":
                    piece = without_mp3_tag(encode(wav, "mp3", rate))
                else:
                    piece = encode(wav, req.response_format, rate)
                seconds += len(wav) / rate
                index += 1
                if req.stream_format == "sse":
                    yield sse_event({"type": "speech.audio.delta",
                                     "audio": base64.b64encode(piece).decode()})
                else:
                    yield piece
                wav = await ENGINE.run(next, chunks, None)
            if req.stream_format == "sse":
                yield sse_event({"type": "speech.audio.done",
                                 "usage": usage(seconds, req.input)})
        finally:
            # Queued behind any chunk still being generated for this request.
            ENGINE.pool.submit(chunks.close)
            ENGINE.pool.submit(cleanup)

    media = "text/event-stream" if req.stream_format == "sse" else media_type
    return StreamingResponse(body(), media_type=media)


def usage(seconds, text):
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "audio_seconds": round(seconds, 2), "input_characters": len(text)}


@app.post("/v1/audio/voices")
async def create_voice(request: Request):
    form = await request.form()
    unknown = set(form.keys()) - {"name", "audio_sample", "reference_transcript"}
    if unknown:
        raise APIError(400, f"unknown field {sorted(unknown)[0]!r}", param=sorted(unknown)[0])
    name = form.get("name")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 256:
        raise APIError(400, "name must be 1 to 256 characters", param="name")
    transcript = form.get("reference_transcript")
    if transcript is not None and not isinstance(transcript, str):
        raise APIError(400, "reference_transcript must be text", param="reference_transcript")
    path, temporary = await read_upload(form.get("audio_sample"), "audio_sample")
    voice_id = f"voice_{uuid.uuid4().hex}"
    folder = ENGINE.voices_dir / voice_id
    folder.mkdir()
    reference = folder / f"reference{path.suffix}"
    shutil.move(str(path), reference)
    shutil.rmtree(temporary, ignore_errors=True)
    voice = {"id": voice_id, "object": "audio.voice", "name": name.strip(),
             "created_at": int(time.time())}
    (folder / "voice.json").write_text(json.dumps(
        {**voice, "reference": str(reference),
         "reference_transcript": (transcript or "").strip() or None}))
    return voice


@app.get("/v1/voices")
async def list_voices():
    built_in = [{"id": name, "object": "audio.voice", "name": name, "type": "built_in"}
                for name in ENGINE.model.available_voices]
    saved = []
    for meta in ENGINE.voices_dir.glob("voice_*/voice.json"):
        voice = json.loads(meta.read_text())
        saved.append({"id": voice["id"], "object": "audio.voice", "name": voice["name"],
                      "type": "custom", "created_at": voice["created_at"]})
    saved.sort(key=lambda v: v["created_at"], reverse=True)
    return {"object": "list", "data": built_in + saved}


@app.delete("/v1/audio/voices/{voice_id}")
async def delete_voice(voice_id: str):
    saved = load_saved_voice(voice_id)
    shutil.rmtree(ENGINE.voices_dir / voice_id, ignore_errors=True)
    ENGINE.pool.submit(ENGINE.forget, saved["reference"], saved["reference_transcript"])
    return {"id": voice_id, "object": "audio.voice", "deleted": True}


def main():
    global ENGINE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    backend = os.environ.get("KITTENML_BACKEND", "vllm")
    if backend not in ("vllm", "torch"):
        raise SystemExit("KITTENML_BACKEND must be 'vllm' or 'torch'")
    vllm_options = json.loads(os.environ.get("KITTENML_VLLM_OPTIONS") or "{}")
    voices_dir = os.environ.get("KITTENML_VOICES_DIR",
                                os.path.expanduser("~/.cache/kittenml/voices"))
    print(f"Loading KittenTTS 2 ({backend} backend)...", flush=True)
    ENGINE = Engine(backend, vllm_options, voices_dir)
    print(f"Ready: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

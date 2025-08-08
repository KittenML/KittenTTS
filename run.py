from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from kittentts import KittenTTS
import soundfile as sf
import io

app = FastAPI()




@app.get("/tts")
@app.post("/tts")
def tts(text: str, voice: str = "expr-voice-4-f" ,):
    # Reuse a single model instance across requests
    global _model
    if "_model" not in globals():
        _model = KittenTTS("KittenML/kitten-tts-nano-0.1")

    # Normalize inputs
    text = text.strip()
    voice = voice.strip()

    audio = _model.generate(text, voice=voice)

    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="tts.wav"'}
    )

# m = KittenTTS("KittenML/kitten-tts-nano-0.1")


# audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]



# Save the audio
# import soundfile as sf
# sf.write('output.wav', audio, 24000)


# /Users/saeedanwar/code/KittenTTS/venv/bin/python -m uvicorn run:app --host 127.0.0.1 --port 8000 --reload
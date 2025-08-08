import re
import io
import os
import numpy as np
import soundfile as sf
from flask import Flask, render_template_string, request, send_file, redirect, url_for
from kittentts import KittenTTS
import secrets

# Initialize the Flask application
app = Flask(__name__)

# --- Dynamic Debugging Setup ---
if os.environ.get('BUILD_ENV') == 'development':
    try:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Debugger listening on 0.0.0.0:5678. Waiting for client to attach...")
        debugpy.wait_for_client()
    except ImportError:
        print("debugpy not found. Is it installed?", file=sys.stderr)
else:
    print("Running in production mode. Debugger is disabled.")


# --- Model Loading and Configuration ---
print("Loading TTS model from KittenML/kitten-tts-nano-0.1...")
try:
    tts_model = KittenTTS("KittenML/kitten-tts-nano-0.1")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    tts_model = None

# Available voices for the KittenTTS model
AVAILABLE_VOICES = [
    'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
    'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
]

# A simple in-memory cache for audio data
audio_cache = {}

# --- Helper Functions ---
def split_text_into_chunks(text):
    """
    Splits a long string into a list of sentences, preserving punctuation.
    """
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    if not sentences:
        return [text]
    return sentences

def generate_audio_data(text_input, voice, rate):
    """
    Generates audio and returns it as an in-memory BytesIO object.
    """
    if not tts_model:
        raise RuntimeError("Model not loaded.")

    if not text_input:
        raise ValueError("No text provided.")

    text_chunks = split_text_into_chunks(text_input)
    all_audio = []

    for chunk in text_chunks:
        if chunk.strip():
            audio_chunk = tts_model.generate(chunk.strip(), voice=voice)
            all_audio.append(audio_chunk)

    if not all_audio:
        raise ValueError("No audio generated from the provided text.")
        
    final_audio = np.concatenate(all_audio)

    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, final_audio, rate, format='WAV')
    audio_buffer.seek(0)
    return audio_buffer

# --- Flask Routes and Logic ---
@app.route('/')
def index():
    """
    Serves the main page with a form and an optional audio player.
    """
    audio_url = request.args.get('audio_url', None)
    return render_template_string("""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>KittenTTS Web App</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; }
            textarea { width: 100%; height: 200px; padding: 10px; box-sizing: border-box; }
            select { margin-right: 10px; }
            button { padding: 10px 20px; }
            audio { width: 100%; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>KittenTTS Web App</h1>
        <p>Enter text below to generate a downloadable audio file.</p>
        <form action="/generate-and-play" method="post">
            <textarea name="text" placeholder="Enter your text here..."></textarea><br><br>
            <label for="voice-select">Choose a voice:</label>
            <select id="voice-select" name="voice">
                {% for voice in voices %}
                    <option value="{{ voice }}">{{ voice }}</option>
                {% endfor %}
            </select>
            <label for="rate-input">Sample Rate (Hz):</label>
            <input type="number" id="rate-input" name="rate" value="24000"><br><br>
            <button type="submit">Generate Audio</button>
        </form>
        {% if audio_url %}
            <hr>
            <h2>Generated Audio</h2>
            <audio controls autoplay>
                <source src="{{ audio_url }}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        {% endif %}
    </body>
    </html>
    """, voices=AVAILABLE_VOICES, audio_url=audio_url)

@app.route('/generate-and-play', methods=['POST'])
def generate_and_play():
    """
    Handles form submission, generates audio, caches it, and redirects to the index
    page with a link to the audio.
    """
    try:
        text_input = request.form.get('text')
        voice = request.form.get('voice')
        rate = int(request.form.get('rate', 24000))
        
        audio_buffer = generate_audio_data(text_input, voice, rate)
        
        # Store the audio in a simple in-memory cache
        file_id = secrets.token_hex(8)
        audio_cache[file_id] = audio_buffer

        return redirect(url_for('index', audio_url=url_for('play_audio', file_id=file_id)))

    except Exception as e:
        app.logger.error(f"Error during audio generation: {e}")
        return f"An error occurred during audio generation: {e}", 500

@app.route('/play-audio/<file_id>')
def play_audio(file_id):
    """
    Serves the audio file from the in-memory cache for playback.
    """
    audio_buffer = audio_cache.get(file_id)
    if not audio_buffer:
        return "Audio not found.", 404

    # The audio_buffer needs to be reset to the beginning to be re-read
    audio_buffer.seek(0)
    
    return send_file(
        audio_buffer,
        mimetype='audio/wav'
    )

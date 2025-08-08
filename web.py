import debugpy
import re
import io
import os
import numpy as np
import soundfile as sf
from flask import Flask, render_template_string, request, send_file
from kittentts import KittenTTS

# --- Debugging setup ---
# This line tells the debugger to listen on port 5678.
# It should be placed at the very top of your application script.
if os.environ.get('FLASK_ENV') == 'development':
    debugpy.listen(("0.0.0.0", 5678))
    print("Debugger listening on 0.0.0.0:5678. Waiting for client to attach...")
    debugpy.wait_for_client() # Blocks until a debugger connects.


# Initialize the Flask application
app = Flask(__name__)

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

# --- Helper Functions ---
def split_text_into_chunks(text):
    """
    Splits a long string into a list of sentences, preserving punctuation.
    """
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    if not sentences:
        return [text]
    return sentences

# --- Flask Routes and Logic ---
@app.route('/')
def index():
    """
    Serves the main page with a form to accept text input.
    The HTML is embedded directly in this string for simplicity.
    """
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
        </style>
    </head>
    <body>
        <h1>KittenTTS Web App</h1>
        <p>Enter text below to generate a downloadable audio file.</p>
        <form action="/generate-audio" method="post">
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
    </body>
    </html>
    """, voices=AVAILABLE_VOICES)

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    """
    Handles the form submission, generates the audio, and sends it for download.
    """
    if not tts_model:
        return "Model not loaded. Please check the server logs.", 500

    text_input = request.form.get('text')
    voice = request.form.get('voice')
    rate = int(request.form.get('rate', 24000))

    if not text_input:
        return "No text provided.", 400

    try:
        text_chunks = split_text_into_chunks(text_input)
        all_audio = []

        for chunk in text_chunks:
            if chunk.strip():
                audio_chunk = tts_model.generate(chunk.strip(), voice=voice)
                all_audio.append(audio_chunk)

        if not all_audio:
            return "No audio generated from the provided text.", 400

        final_audio = np.concatenate(all_audio)

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, final_audio, rate, format='WAV')
        audio_buffer.seek(0)

        return send_file(
            audio_buffer,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='output.wav'
        )

    except Exception as e:
        app.logger.error(f"Error during audio generation: {e}")
        return f"An error occurred during audio generation: {e}", 500

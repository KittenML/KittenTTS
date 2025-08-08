import re
import io
import os
import numpy as np
import soundfile as sf
from flask import Flask, render_template_string, request, send_file, Response
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
            audio { width: 100%; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>KittenTTS Web App</h1>
        <p>Enter text below to generate a downloadable audio file.</p>
        <form action="/stream-audio" method="post">
            <textarea name="text" placeholder="Enter your text here..."></textarea><br><br>
            <label for="voice-select">Choose a voice:</label>
            <select id="voice-select" name="voice">
                {% for voice in voices %}
                    <option value="{{ voice }}">{{ voice }}</option>
                {% endfor %}
            </select>
            <label for="rate-input">Sample Rate (Hz):</label>
            <input type="number" id="rate-input" name="rate" value="24000"><br><br>
            <button type="submit">Generate and Stream Audio</button>
        </form>
        <hr>
        <h2>Audio Player</h2>
        <audio id="audio-player" controls autoplay></audio>
    </body>
    <script>
        const form = document.querySelector('form');
        const audioPlayer = document.getElementById('audio-player');
        
        form.addEventListener('submit', (event) => {
            event.preventDefault();
            const formData = new FormData(form);
            const text = formData.get('text');
            const voice = formData.get('voice');
            const rate = formData.get('rate');
            
            const url = new URL(form.action, window.location.origin);
            url.searchParams.append('text', text);
            url.searchParams.append('voice', voice);
            url.searchParams.append('rate', rate);
            
            audioPlayer.src = url.href;
            audioPlayer.play();
        });
    </script>
    </html>
    """, voices=AVAILABLE_VOICES)

@app.route('/stream-audio')
def stream_audio():
    """
    Generates and streams the audio in chunks.
    This route uses a generator to send data to the browser as it's created.
    """
    if not tts_model:
        return "Model not loaded.", 500

    text_input = request.args.get('text')
    voice = request.args.get('voice')
    rate = int(request.args.get('rate', 24000))

    if not text_input:
        return "No text provided.", 400

    def generate_stream():
        """
        A generator that yields audio chunks.
        """
        try:
            # We must manually create the WAV header first
            header_buffer = io.BytesIO()
            sf.write(header_buffer, np.zeros(0, dtype=np.float32), rate, format='WAV')
            yield header_buffer.getvalue()

            text_chunks = split_text_into_chunks(text_input)
            
            for chunk in text_chunks:
                if chunk.strip():
                    audio_chunk = tts_model.generate(chunk.strip(), voice=voice)
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_chunk, rate, format='WAV')
                    # The audio data starts after the header, at position 44.
                    yield buffer.getvalue()[44:]
            
        except Exception as e:
            app.logger.error(f"Error during audio streaming: {e}")
            # You can handle errors more gracefully here
            pass

    return Response(generate_stream(), mimetype='audio/wav')

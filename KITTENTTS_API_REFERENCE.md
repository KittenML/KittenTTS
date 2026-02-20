# KittenTTS — API Reference

> **Service**: KittenTTS Ultra-Lightweight Text-to-Speech  
> **Base URL**: `http://localhost:7860`  
> **Protocol**: REST (JSON)  
> **CORS**: Enabled for all origins  
> **Model**: KittenTTS — 15M to 80M param neural TTS, 24kHz sample rate

---

## Quick Start

```javascript
// Simplest usage — generate audio from text
const response = await fetch('http://localhost:7860/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Hello world!',
    voice: 'Jasper',
    speed: 1.0,
    model: 'kitten-tts-nano'
  })
});
const result = await response.json();
const audioBlob = await fetch(`data:audio/wav;base64,${result.audio_base64}`).then(r => r.blob());
const audioUrl = URL.createObjectURL(audioBlob);
```

```python
# Python equivalent
import requests
import base64
import io
import soundfile as sf

response = requests.post('http://localhost:7860/api/generate', json={
    'text': 'Hello world!',
    'voice': 'Jasper',
    'speed': 1.0,
    'model': 'kitten-tts-nano'
})
result = response.json()

# Decode and save audio
audio_bytes = base64.b64decode(result['audio_base64'])
audio, sr = sf.read(io.BytesIO(audio_bytes))
sf.write('output.wav', audio, sr)
```

---

## Available Models

| Model ID | Name | Params | Size | Precision | Quality | Description |
|----------|------|--------|------|-----------|---------|-------------|
| `kitten-tts-nano` ⭐ | Nano (FP32) | 15M | 56MB | FP32 | **Best** | Full 32-bit precision, highest quality |
| `kitten-tts-mini` | Mini (INT8) | 80M | 80MB | INT8 | Good | Largest model, quantized |
| `kitten-tts-micro` | Micro (INT8) | 40M | 41MB | INT8 | Good | Balanced size/performance |
| `kitten-tts-nano-int8` | Nano (INT8) | 15M | 19MB | INT8 | Basic | Smallest footprint |

> **💡 Quality Tip:** The FP32 nano model (56MB) produces the best audio quality. Use `kitten-tts-nano` for optimal results.

---

## Available Voices

KittenTTS includes 8 expressive voices — 4 male and 4 female:

| Voice ID | Name | Gender | Description |
|----------|------|--------|-------------|
| `Bella` | Bella | Female | Warm & gentle |
| `Jasper` | Jasper | Male | Clear & professional |
| `Luna` | Luna | Female | Soft & melodic |
| `Bruno` | Bruno | Male | Deep & resonant |
| `Rosie` | Rosie | Female | Bright & cheerful |
| `Hugo` | Hugo | Male | Confident & steady |
| `Kiki` | Kiki | Female | Playful & energetic |
| `Leo` | Leo | Male | Friendly & warm |

### Recommended Voices

| Use Case | Recommended Voice | Notes |
|----------|-------------------|-------|
| **Professional/Narration** | `Jasper` | Clear, professional tone |
| **Warm/Conversational** | `Bella` | Gentle, welcoming |
| **Energetic/Cheerful** | `Kiki` | Playful, upbeat |
| **Deep/Authoritative** | `Bruno` | Resonant, commanding |

---

## Endpoints

### `GET /api/health`

Health check and system info.

**Response:**
```json
{
  "status": "healthy",
  "loaded_models": ["kitten-tts-nano"],
  "cache_dir": "/home/user/.cache/kittentts",
  "cache_size_mb": 56.2
}
```

---

### `GET /api/models`

List available models.

**Response:**
```json
{
  "models": [
    {
      "id": "kitten-tts-nano",
      "name": "Nano (FP32)",
      "params": "15M",
      "size": "56MB",
      "description": "⭐ Best quality - Full 32-bit precision",
      "quality": "best",
      "precision": "FP32"
    }
  ]
}
```

---

### `GET /api/voices`

List available voices.

**Response:**
```json
{
  "voices": [
    {
      "id": "Bella",
      "name": "Bella",
      "gender": "female",
      "description": "Warm & gentle"
    },
    {
      "id": "Jasper",
      "name": "Jasper",
      "gender": "male",
      "description": "Clear & professional"
    }
  ]
}
```

---

### `GET /api/stats`

Get detailed generation statistics.

**Response:**
```json
{
  "generation_stats": {
    "total_requests": 42,
    "avg_generation_time": 0.156,
    "avg_rtf": 12.5,
    "total_audio_generated": 125.4,
    "recent_requests": []
  },
  "system": {
    "cache_directory": "/home/user/.cache/kittentts",
    "cache_size_mb": 56.2,
    "loaded_models": ["kitten-tts-nano"],
    "model_load_times": {"kitten-tts-nano": 2.34},
    "python_version": "3.12.0",
    "memory_usage_mb": 245.6
  },
  "available_models": ["kitten-tts-nano", "kitten-tts-mini", "kitten-tts-micro", "kitten-tts-nano-int8"],
  "available_voices": ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
}
```

---

### `POST /api/generate`

Synthesize speech from text. Returns JSON with base64-encoded WAV audio.

**Request Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to speak |
| `voice` | string | `Bella` | Voice ID |
| `speed` | float | `1.0` | Speech rate (0.25–3.0) |
| `model` | string | `kitten-tts-nano` | Model ID |

**Response:**
```json
{
  "audio_base64": "UklGRi4AAABXQVZFZm10...",
  "sample_rate": 24000,
  "duration": 2.45,
  "debug_info": {
    "model_load_time": 2.34,
    "generation_time": 0.156,
    "total_time": 2.496,
    "real_time_factor": 15.7,
    "audio_samples": 58800,
    "sample_rate": 24000
  }
}
```

**Example:**
```javascript
const res = await fetch('http://localhost:7860/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Hello world! This is a test.',
    voice: 'Jasper',
    speed: 1.0,
    model: 'kitten-tts-nano'
  })
});
const result = await res.json();
console.log(`Generated ${result.duration}s of audio`);
```

---

## Streaming API (for LLM Integration)

The streaming API enables real-time text-to-speech for conversational AI applications. Audio generation starts as soon as complete sentences are detected.

### `POST /api/stream/start`

Start a new streaming TTS session.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `kitten-tts-nano` | Model ID |
| `voice` | string | `Bella` | Voice ID |
| `speed` | float | `1.0` | Speech rate (0.25–3.0) |

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "created"
}
```

**Example:**
```javascript
const res = await fetch('http://localhost:7860/api/stream/start?voice=Jasper&speed=1.0', {
  method: 'POST'
});
const { session_id } = await res.json();
```

---

### `POST /api/stream/chunk`

Add text to a streaming session and receive audio for complete sentences.

**Query Parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | *required* | Session ID from `/api/stream/start` |

**Request Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | `""` | Text chunk to add |
| `flush` | boolean | `false` | Set `true` on final chunk to flush remaining text |

**Response:**
```json
{
  "audio_chunks": [
    "UklGRi4AAABXQVZFZm10...",
    "UklGRi4AAABXQVZFZm10..."
  ],
  "sample_rate": 24000,
  "buffered_text": " remaining text",
  "status": "streaming"
}
```

**Status Values:**
- `streaming` — Session is active, more chunks expected
- `flushed` — Session was flushed, no more buffered text

**Example:**
```javascript
// Stream text from an LLM
const tokens = ["Hello", " there", "! How", " are", " you", " today", "?"];

for (const token of tokens) {
  const res = await fetch(`http://localhost:7860/api/stream/chunk?session_id=${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: token, flush: false })
  });
  const result = await res.json();
  
  // Play each audio chunk immediately
  for (const audioBase64 of result.audio_chunks) {
    const audioBlob = await fetch(`data:audio/wav;base64,${audioBase64}`).then(r => r.blob());
    playAudio(audioBlob);
  }
}

// Flush remaining text
const finalRes = await fetch(`http://localhost:7860/api/stream/chunk?session_id=${sessionId}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: "", flush: true })
});
```

---

### `DELETE /api/stream/end/{session_id}`

End a streaming session and release resources.

**Path Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session ID to terminate |

**Response:**
```json
{
  "status": "ended",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Example:**
```javascript
await fetch(`http://localhost:7860/api/stream/end/${sessionId}`, {
  method: 'DELETE'
});
```

---

## Complete Streaming Example

### JavaScript (Browser)

```javascript
class KittenTTSStreamer {
  constructor(baseUrl = 'http://localhost:7860') {
    this.baseUrl = baseUrl;
    this.sessionId = null;
  }

  async start(voice = 'Jasper', speed = 1.0, model = 'kitten-tts-nano') {
    const res = await fetch(
      `${this.baseUrl}/api/stream/start?voice=${voice}&speed=${speed}&model=${model}`,
      { method: 'POST' }
    );
    const data = await res.json();
    this.sessionId = data.session_id;
    return this.sessionId;
  }

  async addText(text, flush = false) {
    if (!this.sessionId) throw new Error('Session not started');
    
    const res = await fetch(
      `${this.baseUrl}/api/stream/chunk?session_id=${this.sessionId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, flush })
      }
    );
    return await res.json();
  }

  async end() {
    if (!this.sessionId) return;
    await fetch(`${this.baseUrl}/api/stream/end/${this.sessionId}`, {
      method: 'DELETE'
    });
    this.sessionId = null;
  }
}

// Usage with Web Audio API for immediate playback
const audioCtx = new AudioContext();
let nextStartTime = 0;

async function playChunk(audioBase64) {
  const response = await fetch(`data:audio/wav;base64,${audioBase64}`);
  const arrayBuffer = await response.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  
  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);
  
  const startTime = Math.max(audioCtx.currentTime, nextStartTime);
  source.start(startTime);
  nextStartTime = startTime + audioBuffer.duration;
}

// Main streaming loop
async function streamFromLLM(llmStream) {
  const streamer = new KittenTTSStreamer();
  await streamer.start('Jasper');
  
  for await (const token of llmStream) {
    const result = await streamer.addText(token);
    for (const chunk of result.audio_chunks) {
      await playChunk(chunk);
    }
  }
  
  // Flush remaining text
  const final = await streamer.addText('', true);
  for (const chunk of final.audio_chunks) {
    await playChunk(chunk);
  }
  
  await streamer.end();
}
```

### Python

```python
import requests
import base64
import io
import soundfile as sf
import sounddevice as sd

class KittenTTSStreamer:
    def __init__(self, base_url='http://localhost:7860'):
        self.base_url = base_url
        self.session_id = None
    
    def start(self, voice='Jasper', speed=1.0, model='kitten-tts-nano'):
        res = requests.post(
            f'{self.base_url}/api/stream/start',
            params={'voice': voice, 'speed': speed, 'model': model}
        )
        self.session_id = res.json()['session_id']
        return self.session_id
    
    def add_text(self, text, flush=False):
        if not self.session_id:
            raise RuntimeError('Session not started')
        
        res = requests.post(
            f'{self.base_url}/api/stream/chunk',
            params={'session_id': self.session_id},
            json={'text': text, 'flush': flush}
        )
        return res.json()
    
    def end(self):
        if self.session_id:
            requests.delete(f'{self.base_url}/api/stream/end/{self.session_id}')
            self.session_id = None

def play_audio_chunk(audio_base64, sample_rate=24000):
    """Play audio chunk immediately using sounddevice."""
    audio_bytes = base64.b64decode(audio_base64)
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    sd.play(audio, sr)
    sd.wait()

# Usage example
def stream_from_llm(llm_generator):
    streamer = KittenTTSStreamer()
    streamer.start(voice='Jasper', speed=1.0)
    
    try:
        for token in llm_generator:
            result = streamer.add_text(token)
            for chunk in result['audio_chunks']:
                play_audio_chunk(chunk)
        
        # Flush remaining text
        final = streamer.add_text('', flush=True)
        for chunk in final['audio_chunks']:
            play_audio_chunk(chunk)
    finally:
        streamer.end()

# Simulate LLM stream
def mock_llm_stream():
    tokens = ["Hello", " there", "! How", " are", " you", " today", "?"]
    for token in tokens:
        yield token

if __name__ == '__main__':
    stream_from_llm(mock_llm_stream())
```

---

## Python Library API

For direct Python usage without the web server:

### Basic Usage

```python
from kittentts import KittenTTS
import soundfile as sf

# Initialize model (downloads from HuggingFace if needed)
model = KittenTTS("KittenML/kitten-tts-nano-0.8-fp32")

# Generate audio
audio = model.generate(
    text="Hello world! This is a test.",
    voice="Jasper",
    speed=1.0
)

# Save to file
sf.write('output.wav', audio, 24000)

# Or use the convenience method
model.generate_to_file(
    text="Hello world!",
    output_path='output.wav',
    voice="Jasper",
    speed=1.0
)

# List available voices
print(model.available_voices)
# ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', ...]
```

### Streaming API

```python
from kittentts import KittenTTS
import soundfile as sf

model = KittenTTS("KittenML/kitten-tts-nano-0.8-fp32")

# Create a streaming instance
streamer = model.create_streamer(voice="Jasper", speed=1.0)

# Simulate LLM stream
llm_tokens = ["Hello", " there", "! How", " are", " you", " today", "?"]

for token in llm_tokens:
    # add_text() yields audio chunks when complete sentences are detected
    for audio_chunk in streamer.add_text(token):
        sf.write("chunk.wav", audio_chunk, 24000)
        # Or play immediately for real-time output

# Flush remaining buffered text
for audio_chunk in streamer.flush():
    sf.write("final_chunk.wav", audio_chunk, 24000)

# Check buffered text at any time
print(streamer.buffered_text)  # Shows text waiting for sentence completion

# Reset buffer without generating (optional)
streamer.reset()
```

---

## Error Handling

All endpoints return appropriate HTTP status codes:

| Status | Description |
|--------|-------------|
| `200` | Success |
| `400` | Bad request (invalid parameters, empty text, speed out of range) |
| `404` | Not found (invalid session ID) |
| `500` | Server error (model loading failed, inference error) |

**Error Response Format:**
```json
{
  "detail": "Error description here"
}
```

**Example Error Handling:**
```javascript
const res = await fetch('http://localhost:7860/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: '', voice: 'Jasper' })
});

if (!res.ok) {
  const error = await res.json();
  console.error(`Error ${res.status}: ${error.detail}`);
}
```

---

## Performance Notes

- **Real-Time Factor (RTF)**: Typically 10-20x real-time on modern CPUs
- **First Request Latency**: ~2-3 seconds (model loading)
- **Subsequent Requests**: ~100-300ms for typical sentences
- **Streaming Latency**: Audio available within ~100-300ms of sentence completion
- **Memory Usage**: ~200-300MB depending on model

---

## Docker Deployment

```bash
# Build image
docker build -t kittentts-webui .

# Run container
docker run -d -p 7860:7860 -v ~/.cache/huggingface:/root/.cache/huggingface kittentts-webui

# Access at http://localhost:7860
```

---

## License

Apache License 2.0

# Kitten TTS 😻

> **🎉 ANNOUNCEMENT:** KittenTTS V1.0 is now available to try in our [Discord server](https://discord.com/invite/VJ86W4SURW)! Public model launch coming in ~10 days. ⏳


Kitten TTS is an open-source realistic text-to-speech model with just 15 million parameters, designed for lightweight deployment and high-quality voice synthesis.

*Currently in developer preview*

[Join our discord](https://discord.com/invite/VJ86W4SURW) 

[For custom support - fill this form ](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)

Email the creators with any questions : info@stellonlabs.com

## ✨ Features

- **Ultra-lightweight**: Model size less than 25MB
- **CPU-optimized**: Runs without GPU on any device
- **High-quality voices**: Several premium voice options available
- **Fast inference**: Optimized for real-time speech synthesis
- **Command-line interface**: Easy-to-use CLI with pipeline support

## 🚀 Quick Start

### Installation

```
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage

#### Python API
```python
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.2")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f')

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

#### Command Line Interface (CLI)

<details>
<summary>CLI Usage Instructions</summary>

##### Installation

```bash
# Clone the repository
git clone https://github.com/KittenML/KittenTTS.git
cd KittenTTS

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

##### Basic Usage

```bash
./kitten-tts "Hello world"                           # Speak text
./kitten-tts "Hello world" --output hello.wav       # Save to file
echo "Hello world" | ./kitten-tts                   # Read from stdin
./kitten-tts --list-voices                          # List available voices
```

##### Advanced Options

```bash
# With specific voice and fade-out
./kitten-tts "Hello world" --voice expr-voice-2-f --fade-out 0.3

# Adjust speech speed
./kitten-tts "Hello world" --speed 1.5

# Different audio formats
./kitten-tts "Hello world" --output audio.flac --format flac

# Pipeline usage with files
cat text_file.txt | ./kitten-tts --output speech.wav
```

##### CLI Features

- **Text input** via arguments or stdin (pipeline support)
- **8 different voices** (expr-voice-2/m/f through expr-voice-5/m/f)
- **Speed control** with `--speed` option (1.0 = normal)
- **Audio fade-out** with `--fade-out` option (default: 0.2s, use 0 to disable)
- **Multiple formats** (WAV, FLAC, OGG)
- **Cross-platform audio playback** (macOS, Linux, Windows)

##### Available Voices

- `expr-voice-2-m` / `expr-voice-2-f`
- `expr-voice-3-m` / `expr-voice-3-f`
- `expr-voice-4-m` / `expr-voice-4-f`
- `expr-voice-5-m` / `expr-voice-5-f`

</details>





## 💻 System Requirements

Works literally everywhere

## Checklist 

- [x] Release a preview model
- [x] CLI support
- [ ] Release the fully trained model weights
- [ ] Release mobile SDK 
- [ ] Release web version 

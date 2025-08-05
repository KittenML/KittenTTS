# Kitten TTS 😻

Kitten TTS is an open-source realistic text-to-speech model with just 15 million parameters, designed for lightweight deployment and high-quality voice synthesis.

*Currently in developer preview*

[Join our discord](https://discord.gg/upcyF5s6)

> **Note**: This is a personal fork of [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS) with additional features including a Gradio web interface for easy testing.


## ✨ Features

- **Ultra-lightweight**: Model size less than 25MB
- **CPU-optimized**: Runs without GPU on any device
- **High-quality voices**: Several premium voice options available
- **Fast inference**: Optimized for real-time speech synthesis



## 🚀 Quick Start

### Installation

```
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```



 ### Basic Usage 

```
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.1")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)

```

## 🌐 Web Interface

We've added a simple Gradio webapp for easy testing and experimentation with KittenTTS!

### Features
- 🎭 **Voice Selection**: Choose from 8 available voices (male/female variants)
- ⚡ **Speed Control**: Adjust speech speed from 0.5x to 2.0x
- 📝 **Easy Text Input**: Multi-line text input with example texts
- 🔊 **Audio Output**: High-quality 24kHz audio generation
- 💡 **Example Texts**: Pre-loaded examples to get started quickly

### Running the Web Interface

1. **Clone and setup**:
   ```bash
   git clone https://github.com/akashjss/KittenTTS.git
   cd KittenTTS
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   pip install gradio
   ```

3. **Launch the webapp**:
   ```bash
   python gradio_app.py
   ```

4. **Open in browser**: Navigate to http://localhost:7860

The webapp provides an intuitive interface where you can:
- Type or paste text to synthesize
- Select from 8 different voice options
- Adjust speech speed with a slider
- Generate and download audio files
- Try example texts to get started

## 💻 System Requirements

Works literally everywhere



## Checklist 

- [x] Release a preview model
- [ ] Release the fully trained model weights
- [ ] Release mobile SDK 
- [x] Release web version (Gradio webapp added) 


# KittenTTS 😻 - Enhanced for Windows

> **Note:** This is an enhanced fork of [KittenTTS](https://github.com/KittenML/KittenTTS) specifically optimized for Windows compatibility and improved user experience.

KittenTTS is an open-source realistic text-to-speech model with just 15 million parameters, designed for lightweight deployment and high-quality voice synthesis.

## 🚀 **What's Enhanced:**

- ✅ **Windows Compatibility** - Fixed dependency issues on Windows
- ✅ **Better Error Handling** - Comprehensive error messages and solutions
- ✅ **Organized Output** - Audio files saved in dedicated `output/` folder
- ✅ **Enhanced Examples** - Working examples with proper documentation
- ✅ **Improved Dependencies** - Fixed version conflicts and compatibility

## ✨ Features

- **Ultra-lightweight**: Model size less than 25MB
- **CPU-optimized**: Runs without GPU on any device
- **High-quality voices**: Several premium voice options available
- **Fast inference**: Optimized for real-time speech synthesis
- **Windows-friendly**: Specifically tested and optimized for Windows

## 🎵 **Audio Examples**

Listen to the generated audio samples:

### 🔊 **Sample Audio Files:**

**Male Voice:**
- [🎤 Male Voice](output/male_voice.wav) - "Welcome to the future of text-to-speech!"

**Female Voice:**
- [🎤 Female Voice](output/female_voice.wav) - "Welcome to the future of text-to-speech!"

### 🎧 **How to Listen:**
1. **Direct Download** - Click the links above to download and play
2. **Generate Your Own** - Run the examples to create custom audio
3. **Multiple Voices** - Try different voice options

### 📊 **Audio Quality:**
- **Sample Rate:** 24000 Hz
- **Format:** WAV (16-bit PCM)
- **Quality:** High-quality professional synthesis
- **Duration:** ~10-15 seconds per sample

---


## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from kittentts import KittenTTS
import numpy as np

# Load model
model = KittenTTS("KittenML/kitten-tts-nano-0.1")

# Generate audio
text = "Welcome to the future of text-to-speech! KittenTTS is absolutely incredible - it's fast, lightweight, and produces crystal clear audio quality. This revolutionary AI model is changing the game with just 15 million parameters. Amazing technology!"
audio = model.generate(text, voice='expr-voice-5-m')

# Save audio
audio_normalized = np.int16(audio * 32767)

# Create output folder
import os
os.makedirs('output', exist_ok=True)

with open('output/output.wav', 'wb') as f:
    # WAV header and data writing code...
    pass

print("✅ Audio file created: output/output.wav")
```

### Available Voices

- `expr-voice-2-m` - Male voice 2
- `expr-voice-2-f` - Female voice 2  
- `expr-voice-3-m` - Male voice 3
- `expr-voice-3-f` - Female voice 3
- `expr-voice-4-m` - Male voice 4
- `expr-voice-4-f` - Female voice 4
- `expr-voice-5-m` - Male voice 5
- `expr-voice-5-f` - Female voice 5

## 📁 Files

- `final_example.py` - Complete working example with multiple voices
- `simple_example.py` - Basic usage example
- `output/` - Generated audio files folder

## 🎯 Examples

### Different voices
```python
for voice in ['expr-voice-5-m', 'expr-voice-5-f']:
    audio = model.generate(text, voice=voice)
    # Save audio...
```

### Speed control
```python
# Normal speed
audio_normal = model.generate(text, voice='expr-voice-5-m', speed=1.0)

# Faster
audio_fast = model.generate(text, voice='expr-voice-5-m', speed=1.5)

# Slower
audio_slow = model.generate(text, voice='expr-voice-5-m', speed=0.7)
```

## 💻 System Requirements

Works literally everywhere - just needs Python 3.8+ and the required dependencies.

## 📊 Audio Specifications

- **Sample rate**: 24000 Hz
- **Format**: WAV (16-bit PCM)
- **Quality**: High-quality professional voice synthesis

## 🔧 Troubleshooting

If you encounter issues:

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Check Python version:** Python 3.8+ required
3. **Run examples:** `python simple_example.py`

## 📞 Credits

This enhanced version is based on the original [KittenTTS](https://github.com/KittenML/KittenTTS) by KittenML.

**Original Repository:** https://github.com/KittenML/KittenTTS  
**License:** Apache 2.0

---

**Note**: This project is currently in developer preview. Some features may change.


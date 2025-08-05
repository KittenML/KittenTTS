#!/usr/bin/env python3
"""
Simple Gradio webapp to test KittenTTS
"""

import gradio as gr
import tempfile
import os
from kittentts import KittenTTS


def initialize_model():
    """Initialize the KittenTTS model."""
    print("Loading KittenTTS model...")
    try:
        model = KittenTTS("KittenML/kitten-tts-nano-0.1")
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def generate_speech(text, voice, speed):
    """Generate speech from text using KittenTTS."""
    if not text.strip():
        return None, "Please enter some text to synthesize."

    try:
        # Generate audio
        audio_data = model.generate(text, voice=voice, speed=speed)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Save audio data to file using soundfile
            import soundfile as sf
            sf.write(tmp_file.name, audio_data, 24000)

            return tmp_file.name, f"✅ Generated speech for: '{text[:50]}{'...' if len(text) > 50 else ''}'"

    except Exception as e:
        return None, f"❌ Error generating speech: {str(e)}"


# Available voices from KittenTTS
AVAILABLE_VOICES = [
    'expr-voice-2-m', 'expr-voice-2-f',
    'expr-voice-3-m', 'expr-voice-3-f',
    'expr-voice-4-m', 'expr-voice-4-f',
    'expr-voice-5-m', 'expr-voice-5-f'
]

# Initialize model
print("🐱 Initializing KittenTTS...")
model = initialize_model()

if model is None:
    print("❌ Failed to initialize model. Please check your installation.")
    exit(1)

# Create Gradio interface
with gr.Blocks(title="🐱 KittenTTS Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐱 KittenTTS Demo

        Generate high-quality speech from text using KittenTTS - an ultra-lightweight TTS model!

        **Features:**
        - 🚀 Fast inference (CPU-optimized)
        - 🎙️ Multiple voice options
        - ⚡ Adjustable speech speed
        - 🔊 High-quality 24kHz audio output
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            # Input controls
            text_input = gr.Textbox(
                label="📝 Text to Synthesize",
                placeholder="Enter the text you want to convert to speech...",
                lines=3,
                max_lines=5
            )

            with gr.Row():
                voice_select = gr.Dropdown(
                    choices=AVAILABLE_VOICES,
                    value="expr-voice-5-f",  # Default to female voice
                    label="🎭 Voice Selection",
                    info="Choose from available voice models"
                )

                speed_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="⚡ Speed",
                    info="Adjust speech speed (1.0 = normal)"
                )

            generate_btn = gr.Button(
                "🎵 Generate Speech",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            # Output
            audio_output = gr.Audio(
                label="🔊 Generated Audio",
                type="filepath"
            )

            status_text = gr.Textbox(
                label="📊 Status",
                interactive=False,
                lines=2
            )

    # Example texts
    gr.Markdown("### 💡 Try these examples:")
    example_texts = [
        "Hello! This is KittenTTS, a lightweight text-to-speech model.",
        "The quick brown fox jumps over the lazy dog.",
        "Welcome to the future of AI-powered speech synthesis!",
        "KittenTTS works entirely on CPU without requiring a GPU."
    ]

    examples = gr.Examples(
        examples=[[text, "expr-voice-5-f", 1.0] for text in example_texts],
        inputs=[text_input, voice_select, speed_slider],
        cache_examples=False
    )

    # Event handlers
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, voice_select, speed_slider],
        outputs=[audio_output, status_text]
    )

    # Footer
    gr.Markdown(
        """
        ---

        **About KittenTTS:**
        - 🏋️ Ultra-lightweight: <25MB model size
        - 💻 CPU-optimized: No GPU required
        - ⚡ Fast inference: Real-time speech synthesis
        - 🎯 High quality: 24kHz audio output

        *Powered by [KittenML](https://github.com/KittenML/KittenTTS)*
        """
    )


if __name__ == "__main__":
    print("🚀 Starting Gradio webapp...")
    print("📱 The webapp will be available at: http://localhost:7860")
    print("🔄 Use Ctrl+C to stop the server")

    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True
    )
#!/usr/bin/env python3
"""
Kitten TTS CLI - Text-to-Speech Command Line Tool

Usage:
    python kittentts_cli.py "Hello world"                           # Speak text
    python kittentts_cli.py "Hello world" --voice expr-voice-2-f    # Use specific voice
    python kittentts_cli.py "Hello world" --output output.wav       # Save to file
    python kittentts_cli.py --list-voices                          # List available voices
    python kittentts_cli.py --help                                 # Show help
"""

import argparse
import sys
import os
import numpy as np
import soundfile as sf
import tempfile

# Add the current directory to Python path so we can import kittentts
# We need to add the parent directory since we're inside kittentts/cli.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Default fade out duration in seconds
DEFAULT_FADE_OUT = 0.2


# Lazy import - only load KittenTTS when actually needed (not for help)
def get_kittentts():
    try:
        # Import directly from get_model to avoid package-level imports
        from kittentts.get_model import KittenTTS
        return KittenTTS
    except ImportError:
        print("Error: KittenTTS not found. Please install it with:")
        print(
            "pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl")
        sys.exit(1)


def apply_fade_out(audio_data, sample_rate=24000, fade_duration=DEFAULT_FADE_OUT):
    """Apply exponential fade out to audio data.

    Args:
        audio_data: NumPy array of audio samples
        sample_rate: Audio sample rate (default: 24000)
        fade_duration: Fade out duration in seconds (default: {DEFAULT_FADE_OUT}s)

    Returns:
        Audio data with fade out applied
    """
    if len(audio_data) == 0:
        return audio_data

    fade_samples = int(fade_duration * sample_rate)
    if fade_samples >= len(audio_data):
        fade_samples = len(audio_data) // 2  # Limit fade to half of audio if very short

    # Create exponential fade curve
    fade_curve = np.linspace(1, 0, fade_samples) ** 2  # Quadratic fade for smoother curve

    # Apply fade to the end of audio
    audio_with_fade = audio_data.copy()
    audio_with_fade[-fade_samples:] *= fade_curve

    return audio_with_fade


def list_voices(model):
    """List all available voices."""
    print("Available voices:")
    for voice in model.available_voices:
        print(f"  - {voice}")


def play_audio_simple(audio_data, sample_rate=24000):
    """Direct audio streaming without temporary files."""
    try:
        # Try to import sounddevice for direct audio streaming
        import sounddevice as sd
        import numpy as np

        # Convert audio data to proper format if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Play audio directly
        sd.play(audio_data, sample_rate)
        sd.wait()  # Wait for playback to complete

    except ImportError:
        # Fallback to temp file method if sounddevice not available
        print("sounddevice not available, falling back to temp file method...")
        play_audio_with_tempfile(audio_data, sample_rate)
    except Exception as e:
        # Try alternative streaming method or fallback
        print(f"Direct streaming failed: {e}")
        play_audio_with_tempfile(audio_data, sample_rate)


def play_audio_with_tempfile(audio_data, sample_rate=24000):
    """Fallback method using temporary file in system temp directory."""
    temp_file = None
    try:
        # Create temp file in system temp directory
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_file = tmp.name
        sf.write(temp_file, audio_data, sample_rate)

        # Try different system audio players based on OS
        import subprocess
        import platform

        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", temp_file], check=True)
        elif system == "Linux":
            # Try common Linux audio players
            for player in ["aplay", "paplay", "mpg123", "mplayer"]:
                try:
                    subprocess.run([player, temp_file], check=True)
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            else:
                print(f"Audio saved to {temp_file} (no suitable audio player found)")
        elif system == "Windows":
            subprocess.run(["start", temp_file], shell=True, check=True)
        else:
            print(f"Audio saved to {temp_file} (unsupported OS for direct playback)")

        # Clean up temp file
        try:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass

    except Exception as e:
        print(f"Error playing audio: {e}")
        if temp_file and os.path.exists(temp_file):
            print(f"Audio saved to {temp_file}")
        else:
            print("Audio could not be saved - temp file creation failed")


def main():
    parser = argparse.ArgumentParser(
        description="Kitten TTS - Ultra-lightweight text-to-speech synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello world"                           # Speak text
  %(prog)s "Hello world" --voice expr-voice-2-f    # Use specific voice
  %(prog)s "Hello world" --output output.wav       # Save to file
  %(prog)s "Hello world" --speed 1.2               # Faster speech
  %(prog)s "Hello world" --fade-out 0.1            # 0.1s fade out
  echo "Hello world" | %(prog)s                    # Read from stdin
  %(prog)s --list-voices                          # List available voices
        """
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Text to synthesize into speech (if not provided, reads from stdin)"
    )

    parser.add_argument(
        "--model",
        default="KittenML/kitten-tts-nano-0.2",
        help="Model name or path (default: KittenML/kitten-tts-nano-0.2)"
    )

    parser.add_argument(
        "--voice",
        default="expr-voice-2-m",
        help="Voice to use (default: expr-voice-2-m)"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed (1.0 = normal, higher = faster, lower = slower)"
    )

    parser.add_argument(
        "--fade-out",
        type=float,
        default=DEFAULT_FADE_OUT,
        help=f"Fade out duration in seconds (default: {DEFAULT_FADE_OUT}, use 0 to disable)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file path (saves as WAV). If not specified, plays through speakers."
    )

    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit"
    )

    parser.add_argument(
        "--format",
        choices=["wav", "flac", "ogg"],
        default="wav",
        help="Audio format for output file (default: wav)"
    )

    args = parser.parse_args()

    # Handle --list-voices
    if args.list_voices:
        try:
            KittenTTS = get_kittentts()
            model = KittenTTS(args.model)
            list_voices(model)
            return 0
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            return 1

    # Get text from command line or stdin
    if args.text:
        text = args.text
    else:
        # Read from stdin
        try:
            if sys.stdin.isatty():
                # No pipe, interactive mode
                parser.print_help()
                print("\nError: Text input is required (provide as argument or pipe from stdin)", file=sys.stderr)
                return 1
            else:
                # Pipe detected, read from stdin
                text = sys.stdin.read().strip()
                if not text:
                    print("\nError: No text received from stdin", file=sys.stderr)
                    return 1
        except Exception as e:
            print(f"Error reading from stdin: {e}", file=sys.stderr)
            return 1

    try:
        # Initialize the model
        print(f"Loading model: {args.model}...")
        KittenTTS = get_kittentts()
        model = KittenTTS(args.model)

        # Validate voice
        if args.voice not in model.available_voices:
            print(f"Error: Voice '{args.voice}' not available.", file=sys.stderr)
            print(f"Available voices: {', '.join(model.available_voices)}")
            return 1

        # Add dots at the end to prevent cutoff (simple fix)
        if not text.endswith('...'):
            text = text + '...'
            print(f"Added dots to prevent audio cutoff")

        # Generate audio
        print(f"Generating speech using voice: {args.voice}...")
        audio = model.generate(text, voice=args.voice, speed=args.speed)

        # Apply fade out if specified
        if args.fade_out > 0:
            print(f"Applying {args.fade_out}s fade out...")
            audio = apply_fade_out(audio, sample_rate=24000, fade_duration=args.fade_out)

        if args.output:
            # Save to file
            print(f"Saving audio to: {args.output}")
            sf.write(args.output, audio, 24000)
            print("Done!")
        else:
            # Play through speakers
            print("Playing audio...")
            play_audio_simple(audio)
            print("Done!")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Optimized entry point for KittenTTS with fast help and lazy imports
"""

import argparse
import sys

def show_help():
    """Show help message without importing heavy dependencies"""
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
        default=0.2,
        help="Fade out duration in seconds (default: 0.2, use 0 to disable)"
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

    parser.print_help()

def main():
    """Optimized main entry point - fast help, full functionality when needed"""
    # Check if user just wants help
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']):
        show_help()
        return 0

    # For any other operation, run the full CLI
    from .cli_process import main as cli_main
    return cli_main()

if __name__ == "__main__":
    sys.exit(main())
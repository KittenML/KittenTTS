import argparse
import sys
import soundfile as sf
from kittentts import KittenTTS

# Available voices for the KittenTTS model
AVAILABLE_VOICES = [
    'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
    'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
]

def main():
    """
    Main function to parse command-line arguments and generate audio.
    """
    parser = argparse.ArgumentParser(
        description="Generate audio from text using KittenTTS.",
        epilog="Example: python generate_audio.py --output hello.wav --voice expr-voice-2-f 'Hello, world!'"
    )

    # Define mutually exclusive group for input (text or file)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        'text',
        nargs='?', # Makes it optional and captures positional argument
        help="Text to convert to speech (e.g., 'Hello, world!')."
    )
    group.add_argument(
        '--file', '-f',
        type=argparse.FileType('r', encoding='utf-8'),
        help="Path to a text file to read from."
    )

    # Arguments for output and voice
    parser.add_argument(
        '--output', '-o',
        default='output.wav',
        help="Path to the output WAV file. Defaults to 'output.wav'."
    )
    parser.add_argument(
        '--voice', '-v',
        default='expr-voice-2-f',
        choices=AVAILABLE_VOICES,
        help=f"The voice to use. Available voices: {AVAILABLE_VOICES}. Defaults to 'expr-voice-2-f'."
    )

    args = parser.parse_args()

    # Get the input text
    input_text = ""
    if args.file:
        input_text = args.file.read()
        args.file.close()
    elif args.text:
        input_text = args.text

    if not input_text:
        print("Error: No text input provided.", file=sys.stderr)
        sys.exit(1)

    print("Loading TTS model...")
    try:
        # Load the model
        m = KittenTTS("KittenML/kitten-tts-nano-0.1")

        print(f"Generating audio with voice: '{args.voice}'...")
        # Generate the audio
        audio = m.generate(input_text, voice=args.voice)

        # Save the audio file
        sf.write(args.output, audio, 24000)
        print(f"Audio saved to '{args.output}'")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

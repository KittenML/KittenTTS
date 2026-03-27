import argparse
from kittentts import KittenTTS
import os

def main():
    """
    Command-line interface to generate audio using KittenTTS.
    """
    parser = argparse.ArgumentParser(
        description="Generate speech from text using the KittenTTS library.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="The text to be synthesized."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output WAV file. E.g., /app/output/speech.wav"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="expr-voice-5-m",
        help="The voice to use for synthesis. Default: 'expr-voice-5-m'"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier. Default: 1.0"
    )

    args = parser.parse_args()

    print("Initializing KittenTTS model...")
    tts = KittenTTS()

    print(f"Available voices: {tts.available_voices}")
    print(f"Generating audio for text: '{args.text}'")

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the audio file
    tts.generate_to_file(
        text=args.text,
        output_path=args.output,
        voice=args.voice,
        speed=args.speed
    )

    print(f"Success! Audio saved to {args.output}")

if __name__ == "__main__":
    main()

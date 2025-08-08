import argparse
import sys
import soundfile as sf
import re
import numpy as np
from kittentts import KittenTTS

# Available voices for the KittenTTS model
AVAILABLE_VOICES = [
    'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
    'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
]

# A simple function to split text into chunks based on sentences.
# This helps avoid the model's text length limit.
def split_text_into_chunks(text):
    """
    Splits a long string into a list of sentences, preserving punctuation.
    """
    # Split by common sentence-ending punctuation, including the punctuation
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    if not sentences:
        return [text]
    return sentences

def main():
    """
    Main function to parse command-line arguments and generate audio.
    """
    parser = argparse.ArgumentParser(
        description="Generate audio from text using KittenTTS.",
        epilog="Example: python app.py --output hello.wav --voice expr-voice-2-f 'Hello, world!'"
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

    # New argument for the output sample rate
    parser.add_argument(
        '--rate', '-r',
        type=int,
        default=24000,
        help="The output sample rate in Hz. Defaults to 24000."
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
        m = KittenTTS("KittenML/kitten-tts-nano-0.1")

        # Split the text into manageable chunks (sentences)
        text_chunks = split_text_into_chunks(input_text)

        all_audio = []
        print(f"Generating audio in {len(text_chunks)} chunks with voice: '{args.voice}'...")

        for i, chunk in enumerate(text_chunks):
            # Skip empty chunks
            if not chunk.strip():
                continue

            print(f"  - Processing chunk {i+1}/{len(text_chunks)}...")
            audio_chunk = m.generate(chunk.strip(), voice=args.voice)
            all_audio.append(audio_chunk)

        # Concatenate all audio chunks into a single NumPy array
        final_audio = np.concatenate(all_audio)

        # Save the final audio file using the specified sample rate
        sf.write(args.output, final_audio, args.rate)
        print(f"Audio saved to '{args.output}' at {args.rate} Hz.")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

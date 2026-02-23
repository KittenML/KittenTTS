import argparse
import sys
from kittentts import KittenTTS

def main():
    parser = argparse.ArgumentParser(description="KittenTTS: High-quality, ultra-lightweight Text-to-Speech")
    parser.add_argument("text", nargs="?", help="The text to synthesize into speech")
    parser.add_argument("--voice", default="Jasper", help="Voice to use (e.g., Bella, Jasper, Luna). Default: Jasper")
    parser.add_argument("--model", default="KittenML/kitten-tts-nano-0.1", help="Model ID to use. Default: KittenML/kitten-tts-nano-0.1")
    parser.add_argument("--output", default="output.wav", help="Output audio file path. Default: output.wav")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (1.0 = normal). Default: 1.0")
    parser.add_argument("--list-voices", action="store_true", help="List available voices for the model and exit")
    
    args = parser.parse_args()
    
    if not args.list_voices and not args.text:
        parser.error("The 'text' argument is required unless --list-voices is specified.")

    # Initialize model
    print(f"Loading model: {args.model}...")
    model = KittenTTS(model_name=args.model)
    
    if model.model is None:
        sys.exit(1)
        
    if args.list_voices:
        print("\n=== Available Voices ===")
        print("Aliases:")
        for alias in model.model.voice_aliases.keys():
            print(f"  - {alias}")
        print("\nNative Voices:")
        for voice in model.available_voices:
            if voice not in model.model.voice_aliases.values():
                print(f"  - {voice}")
        sys.exit(0)

        
    print(f"Generating audio with voice '{args.voice}'...")
    try:
        model.generate_to_file(args.text, args.output, voice=args.voice, speed=args.speed)
    except ValueError as e:
        sys.exit(1)

if __name__ == "__main__":
    main()

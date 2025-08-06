import argparse
import datetime
import io
import sys

voices = [
    "expr-voice-2-m",
    "expr-voice-2-f",
    "expr-voice-3-m",
    "expr-voice-3-f",
    "expr-voice-4-m",
    "expr-voice-4-f",
    "expr-voice-5-m",
    "expr-voice-5-f",
]


def run(*, model: str, voice: str, output: str, text: str) -> datetime.timedelta:
    from kittentts import KittenTTS
    import soundfile as sf

    m = KittenTTS(model)
    t0 = datetime.datetime.now()
    audio = m.generate(text, voice=voice)
    if output == "-":
        # sf requires a seekable buffer for writing.
        bio = io.BytesIO()
        sf.write(bio, audio, 24000, format="WAV", subtype="PCM_16")
        sys.stdout.buffer.write(bio.getvalue())
    else:
        sf.write(output, audio, 24000)
    t1 = datetime.datetime.now()
    return t1 - t0


def main() -> None:
    ap = argparse.ArgumentParser(prog="kittentts", description="Run Kitten TTS model")
    ap.add_argument("--model", default="KittenML/kitten-tts-nano-0.1", help="Model to use")
    ap.add_argument("--text", required=True, help="Text to synthesize")
    ap.add_argument("--voice", default="expr-voice-2-f", help="Voice to use", choices=voices)
    ap.add_argument("--output", help="Output audio file (- for stdout; use with care)")

    args = ap.parse_args()

    if not args.output:
        ts = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
        args.output = f"{args.voice}-{ts}.wav"

    gen_time = run(
        model=args.model,
        voice=args.voice,
        output=args.output,
        text=args.text,
    )
    print(f"Generated audio in {gen_time}, saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""KittenTTS 2 — a Gradio demo in one file.

    pip install kittenml gradio
    python gradio_demo.py                       # then open http://localhost:7860
    python gradio_demo.py --share               # also print a public link
    python gradio_demo.py --backend vllm        # NVIDIA GPU, needs pip install "kittenml[vllm]"

All of the speech side goes through the `kittenml` package. This file only builds
the interface: built-in voices, cloning from an uploaded or recorded clip, the
decoding presets, and the expression tags.
"""
import argparse
import base64
import gzip
import re
import tempfile
import time

import gradio as gr
import numpy as np
import soundfile as sf

from kittenml import KittenTTS
from kittenml.kittentts2.text import EXPRESSION_TAGS, VOCAL_EVENT_TAGS

TITLE = "KittenTTS 2 Demo"
DEFAULT_TEXT = "Hello! This is a quick test of in-context voice cloning."

# Non-English voices are named after their language ("German", "GermanProfessor").
# Picking one swaps in a sample line in that language and turns off the English
# text normaliser; every other voice is English.
LANGUAGE_SAMPLES = {
    "Arabic": "صباح الخير. أتمنى لك يوماً رائعاً.",
    "Chinese": "早上好。祝你度过美好的一天。",
    "Dutch": "Goedemorgen. Ik wens je een fijne dag.",
    "French": "Bonjour. Je te souhaite une excellente journée.",
    "German": "Guten Morgen. Ich wünsche dir einen wunderschönen Tag.",
    "Hindi": "सुप्रभात। आपका दिन शुभ हो।",
    "Indonesian": "Selamat pagi. Semoga harimu menyenangkan.",
    "Italian": "Buongiorno. Ti auguro una splendida giornata.",
    "Japanese": "おはようございます。素敵な一日をお過ごしください。",
    "Korean": "좋은 아침입니다. 오늘도 좋은 하루 보내세요.",
    "Polish": "Dzień dobry. Życzę ci wspaniałego dnia.",
    "Portuguese": "Bom dia. Desejo-lhe um dia maravilhoso.",
    "Russian": "Доброе утро. Желаю тебе прекрасного дня.",
    "Spanish": "Buenos días. Espero que tengas un día maravilloso.",
    "Turkish": "Günaydın. Sana harika bir gün dilerim.",
}
SAMPLE_TEXTS = {DEFAULT_TEXT, *LANGUAGE_SAMPLES.values()}
ALL_LANGUAGES = "All languages"

DECODING_MODES = ["Expressive", "Advanced"]
# Recordings longer than this are cut to their opening seconds before cloning:
# the model wants roughly 5-30 seconds of a single speaker.
MAX_REFERENCE_S, CLIPPED_REFERENCE_S = 30.0, 20.0

MODEL = None    # the KittenTTS instance, loaded in main()

# ── expression tags ───────────────────────────────────────────────────────────
_TAG_SPAN = re.compile(r"\[(?:" + "|".join(sorted(EXPRESSION_TAGS)) + r")\]"
                       r"|<(?:" + "|".join(sorted(VOCAL_EVENT_TAGS)) + r")>", re.IGNORECASE)
_EMPHASIS_SPAN = re.compile(r"\(\(\([^()\n]{1,80}\)\)\)")
TAG_COLORS = {"emotion": "purple", "vocal event": "blue", "emphasis": "orange"}


def tag_preview(text):
    """Highlight the tags the model will act on; hidden while the text has none."""
    text = str(text or "")
    marks = [(m.start(), m.end(), "emotion" if m.group().startswith("[") else "vocal event")
             for m in _TAG_SPAN.finditer(text)]
    marks += [(m.start(), m.end(), "emphasis") for m in _EMPHASIS_SPAN.finditer(text)]
    if not marks:
        return gr.update(value=None, visible=False)
    segments, pos = [], 0
    for start, end, label in sorted(marks):
        if start > pos:
            segments.append((text[pos:start], None))
        segments.append((text[start:end], label))
        pos = end
    if pos < len(text):
        segments.append((text[pos:], None))
    return gr.update(value=segments, visible=True)


def tags_help():
    emotions = " ".join(f"`[{t}]`" for t in sorted(EXPRESSION_TAGS))
    events = " ".join(f"`<{t}>`" for t in sorted(VOCAL_EVENT_TAGS))
    return (
        "**Emotion** — a leading tag sets the emotion for the whole line: " + emotions + "\n\n"
        "**Vocal events** — an inline tag adds a non-verbal beat anywhere: " + events + "\n\n"
        "**Emphasis** — wrap a word or short phrase in triple parentheses to stress it.\n\n"
        "Example: `[joyful] We actually won the grant <laugh> I can (((hardly))) believe it!`\n\n"
        "Emotion control is early beta: it biases the delivery rather than guaranteeing it. "
        "Anything else inside brackets is read as plain text.")


# ── generation ────────────────────────────────────────────────────────────────
def synthesize(text, mode, top_k, min_p, top_p, temperature, max_new_tokens,
               repetition_penalty, repetition_window, seed, normalize, **speaker):
    if not str(text or "").strip():
        raise gr.Error("Type some text to speak.")
    sampling = {}
    if mode == "Advanced":
        sampling = dict(top_k=int(top_k), min_p=float(min_p), top_p=float(top_p),
                        temperature=float(temperature))
    advanced = {"repetition_penalty": float(repetition_penalty),
                "repetition_window": int(repetition_window),
                "seed": None if seed is None else int(seed)}
    started = time.time()
    try:
        audio = MODEL.generate(text, preset="expressive",
                               max_new_tokens=int(max_new_tokens), normalize=bool(normalize),
                               advanced=advanced, **sampling, **speaker)
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error
    elapsed = time.time() - started
    seconds = len(audio) / MODEL.sample_rate
    return (MODEL.sample_rate, to_int16(audio)), f"{seconds:.1f} s of audio in {elapsed:.1f} s"


def generate_builtin(voice, text, *settings):
    if voice not in MODEL.voices:
        raise gr.Error(f"Unknown voice {voice!r}. Pick one from the list.")
    return synthesize(text, *settings, voice=voice)


def generate_clone(source, uploaded, recorded, transcript, text, *settings):
    reference = recorded if source == "Record with mic" else uploaded
    if not reference:
        raise gr.Error("Add a reference clip first: upload a file or record one.")
    return synthesize(text, *settings, reference=reference,
                      reference_text=(transcript or "").strip() or None)


# ── voices and references ─────────────────────────────────────────────────────
def to_int16(audio):
    """What gr.Audio plays; converting here keeps Gradio from warning on every take."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def voice_language(voice):
    return next((lang for lang in LANGUAGE_SAMPLES if voice.startswith(lang)), "English")


def voices_for(language):
    voices = MODEL.available_voices
    if language == ALL_LANGUAGES:
        return voices
    return [v for v in voices if voice_language(v) == language]


def first_voice(voices):
    """The repository's default voice, else a language's own namesake voice, else the first."""
    for preferred in (MODEL.config.get("default_voice"), *LANGUAGE_SAMPLES):
        if preferred in voices:
            return preferred
    return voices[0] if voices else None


def on_language(language):
    voices = voices_for(language)
    return gr.update(choices=voices, value=first_voice(voices))


def on_voice(voice, text):
    """Preview the voice's own reference clip, and follow its language."""
    spec = MODEL.voices.get(voice)
    reference = None
    if spec:
        # Handed over as audio data, not a path: the clip lives in the Hugging Face
        # cache, which Gradio will not serve files from.
        audio, rate = sf.read(spec["reference"], dtype="float32", always_2d=True)
        reference = (rate, to_int16(audio.mean(axis=1)))
    language = voice_language(voice or "")
    new_text = LANGUAGE_SAMPLES.get(language, DEFAULT_TEXT)
    if text not in SAMPLE_TEXTS:    # never overwrite something the user typed
        new_text = text
    return reference, new_text, language == "English"


def prepare_reference(path):
    """Cut a long recording to its opening seconds, and clear the old transcript."""
    if not path:
        return None, ""
    try:
        info = sf.info(path)
    except RuntimeError as error:
        raise gr.Error(f"Could not read that audio file: {error}") from error
    if info.duration <= MAX_REFERENCE_S:
        return path, ""
    audio, rate = sf.read(path, frames=int(CLIPPED_REFERENCE_S * info.samplerate), dtype="float32")
    clipped = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(clipped, audio, rate)
    gr.Info(f"Using the first {CLIPPED_REFERENCE_S:.0f} seconds of your "
            f"{info.duration:.0f}-second clip.")
    return clipped, ""


def transcribe(source, uploaded, recorded):
    reference = recorded if source == "Record with mic" else uploaded
    if not reference:
        raise gr.Error("Add a reference clip first.")
    return MODEL.transcriber(reference)


def show_source(source):
    recording = source == "Record with mic"
    return gr.update(visible=not recording), gr.update(visible=recording)


# ── controls shared by both tabs ──────────────────────────────────────────────
def decoding_controls():
    """The Decoding radio plus every generation setting, in `synthesize` order."""
    expressive = MODEL.decode_presets["expressive"]
    mode = gr.Radio(DECODING_MODES, value="Expressive", label="Decoding",
                    info="Expressive: the model's lively default. "
                         "Advanced: Expressive's settings with the sliders unlocked.")
    sliders = [
        gr.Slider(0, 200, value=expressive["top_k"], step=1, label="top_k (0 = off)", visible=False),
        gr.Slider(0.0, 0.5, value=expressive.get("min_p", 0.0), step=0.01, label="min_p", visible=False),
        gr.Slider(0.5, 1.0, value=expressive["top_p"], step=0.01, label="top_p", visible=False),
        gr.Slider(0.3, 2.0, value=expressive["temperature"], step=0.05, label="temperature", visible=False),
    ]
    tip = gr.Markdown("**Want it more expressive?** Set `top_k` to 0, raise `temperature` above "
                      "1.0 and `top_p` to 0.9 or more. The further you push, the livelier — "
                      "and less predictable — the delivery.", visible=False)
    # A fresh update per output: Gradio has dropped repeats of one identical update object.
    mode.change(lambda m: [gr.update(visible=m == "Advanced") for _ in range(5)], mode, [*sliders, tip])
    with gr.Accordion("More settings", open=False):
        max_tokens = gr.Slider(250, 3000, value=1000, step=250,
                               label="Max new tokens per chunk (25 tokens ≈ 1 second)")
        penalty = gr.Slider(1.0, 2.0, value=MODEL.advanced_options["repetition_penalty"], step=0.05,
                            label="Repetition penalty",
                            info="1.0 = off. Try 1.2 or 1.3 to discourage repeated audio.")
        window = gr.Slider(1, 250, value=MODEL.advanced_options["repetition_window"] or 50, step=1,
                           label="Repetition window (audio tokens)",
                           info="Only generated audio counts. 50 tokens ≈ 2 seconds.")
        seed = gr.Number(value=17, precision=0, minimum=0, label="Seed",
                         info="Keep it to repeat a take; change it, or clear it, for a new one.")
        normalize = gr.Checkbox(value=True, label="Normalize text",
                                info="Spell out numbers, dates and abbreviations. English only.")
    return [mode, *sliders, max_tokens, penalty, window, seed, normalize]


def text_input():
    with gr.Accordion("Expression tags (early beta)", open=False):
        gr.Markdown(tags_help())
    text = gr.Textbox(label="Text to speak", lines=3, value=DEFAULT_TEXT)
    preview = gr.HighlightedText(visible=False, show_legend=False, label="Detected controls",
                                 color_map=TAG_COLORS, combine_adjacent=False)
    text.change(tag_preview, text, preview)
    return text


def output_column():
    audio = gr.Audio(label="Generated speech", type="numpy", interactive=False)
    status = gr.Markdown(elem_id="kitten-status")
    return audio, status


def build_demo():
    voices = MODEL.available_voices
    with gr.Blocks(title=TITLE) as demo:
        gr.HTML(f'<div id="kitten-header">{_LOGO_SVG}<span class="wordmark">KittenML</span></div>'
                f'<div id="kitten-tagline">{TITLE}<span class="sub">Pick one of {len(voices)} '
                'built-in voices, or clone your own from a short reference clip.</span></div>',
                padding=False)

        with gr.Tab("Built-in voices"):
            with gr.Row():
                with gr.Column():
                    languages = sorted({voice_language(v) for v in voices} - {"English"})
                    language = gr.Dropdown([ALL_LANGUAGES, "English", *languages], value="English",
                                           label="Language (filters the voice list)",
                                           visible=bool(languages))
                    english = voices_for("English")
                    # Custom values allowed so the API accepts any voice, not only the
                    # English ones the list opens on; generate_builtin checks the name.
                    voice = gr.Dropdown(english, value=first_voice(english), label="Voice",
                                        allow_custom_value=True)
                    reference = gr.Audio(label="Reference clip (this voice)", type="numpy",
                                         interactive=False)
                    text = text_input()
                    settings = decoding_controls()
                    go = gr.Button("Generate", variant="primary")
                with gr.Column():
                    audio, status = output_column()
            language.change(on_language, language, voice)
            voice.change(on_voice, [voice, text], [reference, text, settings[-1]])
            demo.load(on_voice, [voice, text], [reference, text, settings[-1]])
            go.click(generate_builtin, [voice, text, *settings], [audio, status],
                     api_name="generate_builtin")

        with gr.Tab("Upload your voice"):
            gr.Markdown("Give the model 5 to 30 seconds of a single speaker. **Whisper** "
                        "transcribes it, or type the transcript yourself, then generate.")
            with gr.Row():
                with gr.Column():
                    source = gr.Radio(["Upload a file", "Record with mic"], value="Upload a file",
                                      label="Reference source")
                    # Two components rather than one whose `sources` is swapped: updating
                    # `sources` on a mounted Audio does not re-mount the recorder.
                    uploaded = gr.Audio(label="Reference voice — upload an audio file",
                                        type="filepath", sources=["upload"])
                    recorded = gr.Audio(label="Reference voice — press record, then stop",
                                        type="filepath", sources=["microphone"], visible=False)
                    transcribe_btn = gr.Button("Transcribe reference (Whisper)")
                    transcript = gr.Textbox(label="Reference transcript (optional, edit if needed)",
                                            lines=2, placeholder="Left empty, Whisper fills it in "
                                                                 "when you generate.")
                    clone_text = text_input()
                    clone_settings = decoding_controls()
                    clone_go = gr.Button("Generate", variant="primary")
                with gr.Column():
                    clone_audio, clone_status = output_column()
            source.change(show_source, source, [uploaded, recorded]).then(
                None, None, None, js="() => window.__kittenStopMic?.()")
            uploaded.upload(prepare_reference, uploaded, [uploaded, transcript])
            recorded.stop_recording(prepare_reference, recorded, [recorded, transcript]).then(
                None, None, None, js="() => window.__kittenStopMic?.()")
            recorded.clear(None, None, None, js="() => window.__kittenStopMic?.()")
            transcribe_btn.click(transcribe, [source, uploaded, recorded], transcript,
                                 api_name="transcribe")
            clone_go.click(generate_clone,
                           [source, uploaded, recorded, transcript, clone_text, *clone_settings],
                           [clone_audio, clone_status], api_name="generate_clone")

        gr.HTML('<div id="kitten-footer">KittenML — open-source tiny frontier models for edge '
                'devices &nbsp;·&nbsp; <a href="https://kittenml.com/" target="_blank" '
                'rel="noopener">kittenml.com</a> &nbsp;·&nbsp; <a href="https://github.com/'
                'KittenML/KittenTTS" target="_blank" rel="noopener">GitHub</a></div>', padding=False)
    return demo


# ── look: kittenml.com's monochrome system ────────────────────────────────────
# White ground, near-black ink, no accent colour, 2px black outlines with a 12px
# radius instead of fills or shadows, Source Sans 3 throughout.
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Source Sans 3"), "-apple-system", "BlinkMacSystemFont", "sans-serif"],
).set(
    body_background_fill="#FFFFFF",
    body_text_color="#1A1A1A",
    body_text_color_subdued="#6B7280",
    block_background_fill="#FFFFFF",
    block_border_width="2px",
    block_border_color="#1A1A1A",
    block_radius="12px",
    block_label_text_color="#1A1A1A",
    block_title_text_color="#1A1A1A",
    input_background_fill="#FFFFFF",
    input_border_color="#1A1A1A",
    input_border_width="2px",
    input_radius="10px",
    button_large_radius="12px",
    button_small_radius="10px",
    button_primary_background_fill="#1A1A1A",
    button_primary_background_fill_hover="#000000",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#1A1A1A",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F5F5F5",
    button_secondary_text_color="#1A1A1A",
    button_secondary_border_color="#1A1A1A",
)

# Gradio follows prefers-color-scheme by stamping `.dark` on the root, which would
# repaint the page near-black; the site is light-only, so keep stripping it. Also
# release the microphone when a recording ends: Gradio stops the MediaRecorder but
# leaves the stream's tracks live, so the browser keeps showing the mic as in use.
JS = """
() => {
  if (!window.__kittenMicPatched && navigator.mediaDevices?.getUserMedia) {
    window.__kittenMicPatched = true;
    window.__kittenStreams = [];
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      const stream = await original(constraints);
      if (constraints && constraints.audio) window.__kittenStreams.push(stream);
      return stream;
    };
    window.__kittenStopMic = () => {
      window.__kittenStreams.forEach(s => s.getTracks().forEach(t => t.stop()));
      window.__kittenStreams = [];
    };
  }
  const light = () => {
    document.documentElement.classList.remove('dark');
    document.body && document.body.classList.remove('dark');
  };
  light();
  new MutationObserver(light).observe(document.documentElement,
      {attributes: true, attributeFilter: ['class']});
}
"""

CSS = """
body, .dark body { background: transparent !important; }
.dark { --body-background-fill: #FFFFFF; --background-fill-primary: #FFFFFF;
        --block-background-fill: #FFFFFF; --body-text-color: #1A1A1A;
        --border-color-primary: #1A1A1A; }
/* The faded cat-doodle backdrop: a translucent white wash over the tiled artwork,
   both on <html> so they cover the whole canvas and stay fixed while scrolling. */
html, .dark html {
  background-color: #FFFFFF !important;
  background-image: linear-gradient(rgba(255,255,255,0.93), rgba(255,255,255,0.93)),
                    url("PATTERN_URI") !important;
  background-repeat: repeat, repeat !important;
  background-size: auto, 640px auto !important;
  background-attachment: fixed, fixed !important;
}
gradio-app, .gradio-container, .app, .gradio-container > .main,
.gradio-container .main, .gradio-container .wrap.contain { background: transparent !important; }
.gradio-container { width: 100% !important; max-width: 1150px !important; margin: 0 auto !important; }
.gradio-container, .gradio-container * { letter-spacing: -0.01em; }

#kitten-header { display: flex; align-items: center; justify-content: center; gap: 14px;
                 padding: 8px 4px 0; }
#kitten-header svg { height: 44px; width: auto; flex-shrink: 0; }
#kitten-header .wordmark { font-size: 1.45rem; font-weight: 400; letter-spacing: -0.02em;
                           color: #1A1A1A; line-height: 1.1; }
#kitten-tagline { font-size: 1.15rem; font-weight: 300; line-height: 1.4; letter-spacing: -0.02em;
                  color: #1A1A1A; max-width: 620px; margin: 6px auto 18px; text-align: center; }
#kitten-tagline .sub { color: #6B7280; font-size: 0.86em; display: block; margin-top: 4px; }
#kitten-footer a { color: #1A1A1A; text-decoration: underline;
                     text-underline-offset: 3px; }
#kitten-status { color: #6B7280; font-size: .9rem; }

/* the site's signature: a 2px black outline, 12px radius, and a lift on hover */
.gradio-container .tabitem { border: 2px solid #1A1A1A !important; border-radius: 12px !important;
                             padding: 12px !important; }
button.lg, button.md, button.sm { font-weight: 500 !important; transition: transform .18s ease; }
button.lg:hover, button.md:hover, button.sm:hover { transform: translateY(-2px); }
.tab-wrapper button, .tab-nav button { font-weight: 400 !important; color: #6B7280 !important;
                                       border-bottom: 2px solid transparent !important; }
.tab-wrapper button.selected, .tab-nav button.selected {
  color: #1A1A1A !important; font-weight: 600 !important;
  border-bottom: 2px solid #1A1A1A !important; }
footer { display: none !important; }
#kitten-footer { text-align: center; color: #6B7280; font-size: .87rem; padding: 18px 0 6px; }
"""


def main():
    global MODEL
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="KittenML/kitten-tts-2",
                        help="Hugging Face repository or local directory")
    parser.add_argument("--backend", default="auto", choices=["auto", "torch", "vllm", "mlx"],
                        help="auto picks MLX on an Apple-silicon Mac with mlx installed, else torch")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (auto-detected)")
    parser.add_argument("--weights", default=None, help="packed (default), emb4 or full")
    parser.add_argument("--decoder", default=None, help="default or student_w4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()

    MODEL = KittenTTS(args.model, backend=args.backend, device=args.device,
                      weights=args.weights, decoder=args.decoder)
    print(f"Loaded {args.model} · backend {MODEL.backend} · device {MODEL.device}", flush=True)
    pattern = "data:image/svg+xml;base64," + base64.b64encode(_CAT_PATTERN_SVG.encode()).decode()
    # One generation at a time: every request shares the one model.
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        theme=THEME, css=CSS.replace("PATTERN_URI", pattern), js=JS)


# ── brand art ─────────────────────────────────────────────────────────────────
# The KittenML logo and the cat-doodle backdrop from kittenml.com, gzipped and
# base64-encoded so the demo stays a single file with no assets to fetch.
def _unpack(*chunks):
    return gzip.decompress(base64.b64decode("".join(chunks))).decode()


_LOGO_SVG = _unpack(
    "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"
    "jzg+atL65i9zX/8DHma0uMEbAAA="
)

_CAT_PATTERN_SVG = _unpack(
    "H4sIAAAAAAACA7Wdy5Ic15GmXyUNs2mZQc1zv7SJWkxsMS8wO3WLImlGNdtIjqh++/Hv/0+hkhSAygLVi8JBRkVGnIvf/Xev"
    "P/z4t69vf//rd//545dvvvnpp//6ty+++Pnnn//15/qv3//w9RclpfRF3PHm9rdvv/r5f3//9y/fpFu65bh8Wym9uf387Z9/"
    "+ubLN1x4c/vmq2+//uanL9/wmz/+4c9f/eXHP/7hx//+679//93t2z9/+eY/8q8e01Zpt7bnfnP7/m9f/fCX777/+cs3f/v2"
    "x2///buv4gH/9aefvrn95dvvvvvyzc/ffPvTV29uP/3wp//88S/f//DXL9/89U8//fDt3/9l3HiQ/v3dm1u85P/E/96tOc6/"
    "O7V36fyb/u+bL85TH3lQb/O269V7fL/eei8eui6uFpuQb33PW571NmI/8t7XiGslLX0upcRYbqWmuF5jjM+530o8cOQVY79G"
    "jesr30Yrt5r3bdR9qz3ur+tWR1yv41ZnfC/uq6tc/rx9/4j3tLg+2jViWrW224i51NRvM+cY1zVjDjHeJt/L4za5v4xrjvjc"
    "43qL7+0Yy7w1rsccWp3x/XprLa6ncWs91hXvbJv3zVsvMeYUWzGuEe/po2pdnX3I8fuVr1HYk34bhZ+YVxnx3XgOe8D6Y30j"
    "dnS0+B37MOL/o2v+vEs/O+Yf8xix18xzsi7GVi6Nk3nu22LfVrmtwnNqjOzjjNH7vPKK8+K+FnMNumXOMY8V+9ML4771HGO8"
    "qyfGfrU1+N6tzaTnt94Yr9Z4buxVzHvFM1rtMc6r8f0WZ7fis86m6HqNPVxxLlX3xZnwvhLX4Z8StBIkulK7lRFrXPUqsScz"
    "3l2CDuaaQUOTz1fQVoxBc6x55luOdczYzxznydllzixoLFf2rsf1WHuccW5BQ316bCuuQ4P1FmQ74vV7xGn1i2F2uDo2iDEO"
    "scfNQTA9FpXjwHssmIfGGJNhDEJm4yH0zEH2S4QfRBlCIw5v6/czz6sUCIxFxaHF/0tsYoxXYTI7Ft+DOHowAgRQ8lUg7Mym"
    "bBFRmWI0Xe8xVzar873B9RLPb57/DiKMg8rxnB7EzKI7mxLPClHjMeapMeaagxBrbHZeQewpxfzjYDPrCaKvMVYxwVXi4Fps"
    "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"
    "4zyfj8OnOtlulFByMOvM/7G1FoNNz2jgyvMoIA6WhH6fhfByp3csC5WuOBWkABw0r1oyg2KFwFRQY0hT9xNVVA3wdgpOv29q"
    "vuDqe8/jQQuWwEJ2y+HT6FvAIONJrmde2m6BLJ7KsgCleOBNeEIeK3gDV/sLESXZIq0sBJk+q9rfNaXxvsfmiPVBveDJMbtO"
    "2Dn4GhaG6AgLNTlXq+sru654GbSjemEySaofloVgbM1yVUjdRd0Q3J7ZHfVYi9oio/glDx0MU4Ne2go31yhqrXud8dQbPz1/"
    "2jqN+T60Vnlhs7w7o7y69hS4WwdUrR46xQhStXVut3P/gx61cEQXRQ14BnrHPh5dqgpqKg26D6rV9z84f6Vn3HyMMrriwJfS"
    "JFTneLw9j9nXk/Pv45zxec5j71RUo15uaFadaXwq3xonRYPnrmYl+TQ2rJ+TI88yjdX4ch0XCletSFm8EzyZqFhyV+aR/Oc2"
    "+vaflOiCOWZ3HcF9WGpzIRQcKcm+3HOavAbKWL9vBiW4iFb97X1/d15Z16dNZLlqzY0EPGpeF264QArvx3muFwMHTv98PQfX"
    "Wp+XgTjO8xtI08/17r8Ppz911JZryroFsYqCBQiaauAsFzHbMBJYgpy4IoNFuW0V7Xpff5kq/cJ/QfDr+wPRf7/7009f/cva"
    "b2f73e2H73/i0+9z/93Np/X7/Db/7s0f//D/fvzq9s0PX/3lyzf/i788+Pcv3/y+rze3/45x3P3lwjyf/3BhLlkz+Pqjb82r"
    "v13j/Wtz/fWbmt9Up9/EeN406/OL4v+ffk9Yq29HfX5P+diK9llRvlvRul9ReeFNwf9vd3r/pvKRBfXhFzG+f1G5e1F+6UVj"
    "tPsX/b48cGC1nm1s7986xvNLx3rhnXOOt+OZSnL71DvPSkc5W1qeV1rq/Za+dHg7t7fzmUh+3z6yutHOm+72tNyTY32JHHd9"
    "m3d+Xt8nmaD6rS35re2ZCVa+++OdL70zDNS3Ja871vuHN5V/Fs2EAfo2nKXnd9WPLOpD/Nbu+K298KKeVryo3+/kK87snuHq"
    "i3zQ6tuS7vbvk4d2trIdRmB8OrQ7Son/f/qlK5V46TMnjAfe+SQtGd8f37g/vvESebb+y+PbHxNjZ3m93f1J2X73qvQSp4/8"
    "NqzW5zetRxZ4OKE/c8Lezy+N/7/ACTm9DZP9Xry8sLx+t7xf7GR6iRFai/WlB5iuHfHVnsXXuuOE9TInlLe1PFNKe0CkPGkh"
    "xg/q1fwSfY7K+tYntrJ+nBHuCCX+/4JGGCyvPMYIRyO0fN75rGTXnRhbL3H8LkEn/Y44x0fopB3h0p7pZN3Jlv0SlfTxNqy0"
    "T7DboZF6OLs+c/a8I8f5El+XlN6Gb/QJ86T9c8RyDVn5ixXlj53Rb+XltmJJd6r00+bJ00vP8vq8o/t7UyHXF+l+x2ufJeQj"
    "YutDp3fHAvNFFljlbRvjQavhiQeOLm/Pem/f8cB+kQfaDJK544GPc/gHBNirOHy9HaU/uLp/FkPkeOmdTv+omfJkzz5Lr3G3"
    "ttFfdAva25H3p/jh6U3tHy3n9QrLGbcgvMB7Fn/Av2of0HO/0AP5ASdhlGeG/+j6PkSPd5y36wMGUV+/ME0e0HOHNPuda5Da"
    "vZnSHlAFIz+zQR4fs01+s0qNA5x3DPeQQfQhg/1V8iyvEW/tj701/2pX692u3r81jRfdhPp25nnHFuljErt+4F3jNe9qkxXm"
    "x+yj/CtiXR8m1pfeOWaOd64HifVJYvd/MCb2nZ+384uWEhGW+aCf8rTSJ7J9NpX2vXeUXvRoQ+PP+gsB8E8ME3zx49++/uP/"
    "B9tNgYbzggAA"
)

if __name__ == "__main__":
    main()

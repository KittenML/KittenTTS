"""KittenTTS 2: a speech language model with in-context voice cloning.

Unlike the ONNX KittenTTS models, which map phonemes to a waveform directly,
KittenTTS 2 is an autoregressive language model over a shared text/audio
vocabulary. It reads the text to speak and writes S3 codec tokens, which a
separate vocoder turns into audio. A voice is supplied two ways at once: a
speaker embedding through the model's own projection head, and the reference clip
itself as codec tokens in the prompt.

    from kittenml import KittenTTS

    m = KittenTTS("/path/to/kitten-tts-2")
    audio = m.generate("Hello there.", voice="Bruno")
    audio = m.generate("Hello there.", reference="my_voice.wav")
"""

import contextlib
import json
import pathlib

import numpy as np
import torch

from . import text as text_utils
from .logits import build_logits_processors
from .prompt import build_generation_prompt, build_reference_prefix
from .speaker import (SPEAKER_EMBEDDING_DIM, WHISPER_MODEL, ReferenceTranscriber,
                      SpeakerEncoder, attach_speaker_head)
from .tokens import TokenMap
from .vocoder import S3Codec, resolve_reference_path

# The codec runs at 25 Hz, so this is the conversion between a token budget and
# seconds of audio.
TOKENS_PER_SECOND = 25
# Measured delivery rate for this checkpoint. Used to size a per-chunk token budget
# from the chunk's length, which caps a runaway without truncating honest speech.
CHARS_PER_SECOND = 20.0
CHUNK_BUDGET_SLACK = 1.8
MIN_CHUNK_BUDGET = 200

# Knobs behind the `advanced=` argument. Everything here has a working default and
# most callers never touch it; they are exposed because the values that steer
# runaway generations and chunking are worth being able to tune per request.
# `None` means "use whatever the model repository configured".
ADVANCED_OPTIONS = {
    # Penalty applied to tokens already seen. Raising it suppresses loops harder,
    # at the cost of clipping legitimately repeated sounds.
    "repetition_penalty": 1.1,
    # Limit that penalty to the last N generated tokens. None scans everything,
    # which lets a long reference clip suppress valid target audio.
    "repetition_window": None,
    # Escalating penalty once one codec token repeats `grace` times in a row —
    # the main guard against the model sticking on silence or a held phoneme.
    "token_run_penalty": 1.3,
    "token_run_grace": 10,
    # Chunking. Shorter chunks are steadier but more seams; see text.py.
    "chunk_chars": None,
    "chunk_min_chars": None,
    "chunk_gap_s": text_utils.CHUNK_GAP_S,
    # Emotion conditioning. None auto-enables it whenever the text carries an
    # expression tag; True or False force it, to A/B the effect.
    "use_emotion": None,
    # Seed the sampler for a reproducible generation.
    "seed": None,
}


@contextlib.contextmanager
def _quiet_unexpected_keys():
    """Quieten transformers' loading chatter for one from_pretrained call."""
    import logging

    # ERROR rather than silence, and only for this one call: a genuine failure
    # still raises, and anything the caller loads later logs normally again. Covers
    # the unexpected-key report and the tensor-parallel notes, neither of which
    # says anything useful here.
    transformers_logger = logging.getLogger("transformers")
    previous = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        transformers_logger.setLevel(previous)


class KittenTTS2:
    """A loaded KittenTTS 2 model.

    Instantiated by `kittenml.KittenTTS` when a repository's config declares
    `"type": "KITTEN2"`; construct it directly only if you want to bypass that
    dispatch.
    """

    def __init__(self, repo_dir, config, device=None, cache_dir=None, hf_token=None,
                 decoder=None, weights=None):
        self.repo_dir = pathlib.Path(repo_dir)
        self.config = config
        self.cache_dir = cache_dir

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.token_map = TokenMap.from_config(config)
        generation = config.get("generation", {})
        self.decode_presets = config.get("decode_presets", {})
        self.default_preset = config.get("default_preset", "stable")
        self.emotion_control = generation.get("emotion_control")
        self.repetition_window = generation.get("repetition_window")
        self.use_reference_prompt = generation.get("use_reference_prompt", True)
        self.chunk_chars = generation.get("chunk_chars", text_utils.CHUNK_CHARS)
        self.chunk_min_chars = generation.get("chunk_min_chars", text_utils.CHUNK_MIN_CHARS)

        self.weights = weights or config.get("default_weights", "packed")
        self._load_language_model()
        self.decoders = config.get("decoders", {})
        self.decoder = decoder or config.get("default_decoder", "default")
        self.codec = S3Codec(device=self.device, cache_dir=cache_dir,
                             student_weights=self._decoder_weights(self.decoder))
        self.speaker_encoder = SpeakerEncoder(
            self.repo_dir / config.get("speaker_embedding", "speaker/model.safetensors"),
            device=self.device)
        self.transcriber = ReferenceTranscriber(
            device=self.device, cache_dir=cache_dir,
            model_name=config.get("transcriber", WHISPER_MODEL))

        self.voices = self._load_voices()
        self._reference_cache = {}

    @property
    def sample_rate(self):
        return self.codec.sample_rate

    def _decoder_weights(self, name):
        """Path to a named decoder's weights, or None for the built-in flow."""
        if name == "default":
            return None
        if name not in self.decoders:
            raise ValueError(f"Unknown decoder {name!r}. "
                             f"Choose from: {self.available_decoders}")
        return self.repo_dir / self.decoders[name]

    @property
    def available_decoders(self):
        """Decoder names accepted by the `decoder` argument."""
        return ["default", *self.decoders]

    @property
    def advanced_options(self):
        """The `advanced=` knobs and the values this model would use for them."""
        return self._resolve_advanced(None)

    @property
    def available_voices(self):
        """Names accepted by the `voice` argument."""
        return list(self.voices)

    # ── loading ───────────────────────────────────────────────────────────────
    def _load_language_model(self):
        from transformers import AutoTokenizer

        lm_dir = self.repo_dir / self.config.get("lm_dir", "lm")
        if not lm_dir.is_dir():
            raise FileNotFoundError(f"language model directory not found: {lm_dir}")

        self.dtype = torch.bfloat16 if self.config.get("dtype", "bf16") == "bf16" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(str(lm_dir))
        # The weights on disk are already the folded ternary codeword — every weight
        # group is exactly {-gamma, 0, +gamma}, not a latent full-precision tensor.
        # Re-quantizing them would shrink each nonzero weight by its group's sparsity
        # fraction, so this loads as an ordinary model on purpose.
        # The checkpoint carries spk_proj.* alongside the LM weights, which the
        # architecture knows nothing about; attach_speaker_head installs them below.
        state = self._load_lm_weights(lm_dir)
        with _quiet_unexpected_keys():
            model = self._build_lm(lm_dir, state)
        model = model.to(self.device)
        # `state` already holds spk_proj, so pass it rather than re-reading the file.
        attach_speaker_head(model, str(lm_dir), self.device, self.dtype,
                            spk_dim=self.config.get("speaker_embedding_dim",
                                                    SPEAKER_EMBEDDING_DIM),
                            state=state)
        del state
        model.eval()
        self.model = model

    def _build_lm(self, lm_dir, state):
        """Instantiate the architecture and install `state` into it.

        Built from the config rather than via from_pretrained, because
        from_pretrained insists on finding a weights file on disk even when handed a
        state dict — and with the packed weights there is no such file.

        Not built on the meta device, tempting as that is to skip initialising 1.7B
        values that are about to be overwritten: the rotary embedding's inv_freq is a
        non-persistent buffer computed in __init__, so it is absent from the state
        dict and would stay meta, which only surfaces later as "cannot copy out of
        meta tensor" when the model moves to the GPU.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        # transformers moved this between 4.x and 5.x, and it is only an
        # optimisation, so a version that has neither just pays the init cost.
        try:
            from transformers.initialization import no_init_weights
        except ImportError:
            try:
                from transformers.modeling_utils import no_init_weights
            except ImportError:
                no_init_weights = contextlib.nullcontext

        lm_config = AutoConfig.from_pretrained(str(lm_dir))
        for attention in ("flash_attention_2", "sdpa"):
            try:
                # Every parameter is replaced from the checkpoint a few lines down, so
                # randomly initialising 1.7B of them first is pure cost — about a
                # minute of it.
                with no_init_weights():
                    model = AutoModelForCausalLM.from_config(
                        lm_config, dtype=self.dtype, attn_implementation=attention)
                break
            except Exception:
                if attention == "sdpa":
                    raise
        missing, _ = model.load_state_dict(state, strict=False, assign=True)
        # lm_head is tied to the token embedding here, so it is legitimately absent
        # from the file. Anything else missing means the checkpoint is incomplete and
        # would otherwise run on random weights.
        model.tie_weights()
        unfilled = [n for n in missing if n != "lm_head.weight"]
        if unfilled:
            raise ValueError(f"weights missing from the checkpoint: {unfilled[:5]}")
        return model

    def _weights_path(self, name):
        """Where a named weight variant lives, or None to read the plain files."""
        if name == "full":
            return None
        key = {"packed": "lm_packed", "emb4": "lm_emb4"}.get(name)
        if key is None:
            raise ValueError(f"Unknown weights {name!r}. "
                             f"Choose from: {self.available_weights}")
        declared = self.config.get(key)
        if not declared:
            raise ValueError(f"this repository declares no {name!r} weights")
        return self.repo_dir / declared

    @property
    def available_weights(self):
        """Weight variants this repository offers."""
        names = ["full"]
        for name, key in (("packed", "lm_packed"), ("emb4", "lm_emb4")):
            if self.config.get(key):
                names.append(name)
        return names

    def _load_lm_weights(self, lm_dir):
        """The language model's weights, in whichever packing was selected.

        "packed" is a 1.58-bit packing that is exactly equivalent to
        model.safetensors at about a third of the size, and is the default.
        "emb4" is smaller again but quantises the embedding, which is lossy —
        see tl2_emb4.py. "full" reads the plain bf16 files.
        """
        from safetensors import safe_open
        from safetensors.torch import load_file

        path = self._weights_path(self.weights)
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(f"{self.weights!r} weights not found: {path}")
            if self.weights == "emb4":
                from .tl2_emb4 import load_state_dict
                return load_state_dict(path)
            with safe_open(str(path), framework="pt") as handle:
                metadata = handle.metadata() or {}
                tensors = {k: handle.get_tensor(k) for k in handle.keys()}
            from .ternary import unpack_state_dict
            return unpack_state_dict(tensors, metadata)

        state = {}
        for shard in sorted(pathlib.Path(lm_dir).glob("model*.safetensors")):
            if "-" in shard.stem:                 # a packed variant, not a plain shard
                continue
            state.update(load_file(str(shard)))
        if not state:
            raise FileNotFoundError(f"no plain model weights in {lm_dir}")
        return state

    def _load_voices(self):
        voices_path = self.repo_dir / self.config.get("voices", "voices/voices.json")
        if not voices_path.is_file():
            return {}
        with open(voices_path) as f:
            entries = json.load(f)
        root = voices_path.parent
        return {name: {**spec,
                       "reference": str(root / spec["reference"]),
                       "artifacts": str(root / spec["artifacts"])}
                for name, spec in entries.items()}

    # ── reference artifacts ───────────────────────────────────────────────────
    def _voice_reference(self, voice):
        """Prompt prefix, speaker embedding and vocoder conditioning for a built-in voice.

        The embedding and the reference's codec tokens are precomputed in the
        repository, so a built-in voice never loads the embedding model at all.
        """
        if voice not in self.voices:
            raise ValueError(
                f"Voice {voice!r} not available. Choose from: {self.available_voices}")
        if ("voice", voice) not in self._reference_cache:
            spec = self.voices[voice]
            artifacts = np.load(spec["artifacts"])
            self._reference_cache[("voice", voice)] = self._build_reference(
                embedding=artifacts["embedding"],
                reference_tokens=artifacts["reference_tokens"].tolist(),
                transcript=spec["transcript"],
                reference_wav=spec["reference"])
        return self._reference_cache[("voice", voice)]

    def _clone_reference(self, reference_wav, reference_text=None):
        """The same artifacts, derived on the fly from an arbitrary clip."""
        reference_wav = resolve_reference_path(reference_wav)
        key = ("clone", reference_wav, reference_text)
        if key not in self._reference_cache:
            transcript = reference_text
            if not (transcript and transcript.strip()):
                # The transcript is part of the prompt, and a caller should not have
                # to type it out; transcribe rather than refuse.
                transcript = self.transcriber(reference_wav)
                if not (transcript and transcript.strip()):
                    raise ValueError(
                        f"no speech found in the reference clip: {reference_wav}")
            self._reference_cache[key] = self._build_reference(
                embedding=self.speaker_encoder(reference_wav),
                reference_tokens=self.codec.encode_reference(reference_wav),
                transcript=transcript,
                reference_wav=reference_wav)
        return self._reference_cache[key]

    def _build_reference(self, embedding, reference_tokens, transcript, reference_wav):
        transcript_ids = self.tokenizer.encode(transcript.strip(), add_special_tokens=False)
        return {
            "prefix": build_reference_prefix(self.token_map, transcript_ids, reference_tokens),
            "embedding": torch.tensor(np.asarray(embedding, np.float32),
                                      device=self.device, dtype=self.dtype).unsqueeze(0),
            "conditioning": self.codec.reference_conditioning(reference_wav),
            "transcript": transcript,
        }

    # ── generation ────────────────────────────────────────────────────────────
    def _resolve_advanced(self, advanced):
        """Merge caller overrides over the defaults, rejecting unknown keys."""
        resolved = dict(ADVANCED_OPTIONS)
        for key, value in (advanced or {}).items():
            if key not in ADVANCED_OPTIONS:
                raise ValueError(f"unknown advanced option {key!r}. "
                                 f"Valid options: {sorted(ADVANCED_OPTIONS)}")
            if value is not None:
                resolved[key] = value
        # Fall back to the repository's configuration where nothing was given.
        if resolved["repetition_window"] is None:
            resolved["repetition_window"] = self.repetition_window
        if resolved["chunk_chars"] is None:
            resolved["chunk_chars"] = self.chunk_chars
        if resolved["chunk_min_chars"] is None:
            resolved["chunk_min_chars"] = self.chunk_min_chars
        return resolved

    def _resolve_preset(self, preset, overrides):
        if preset not in self.decode_presets:
            raise ValueError(f"Unknown preset {preset!r}. "
                             f"Choose from: {list(self.decode_presets)}")
        settings = dict(self.decode_presets[preset])
        settings.update({k: v for k, v in overrides.items() if v is not None})
        return settings

    def _generate_tokens(self, reference, chunk, settings, max_new_tokens,
                         use_reference_prompt, emotion_ids, advanced):
        """Run the LM for one chunk and return its audio codec ids (0-based)."""
        prompt = build_generation_prompt(
            self.token_map,
            self.tokenizer.encode(chunk, add_special_tokens=False),
            reference_prefix=reference["prefix"] if use_reference_prompt else None,
            emotion_ids=emotion_ids)

        processors = build_logits_processors(
            self.token_map,
            temperature=settings["temperature"], top_k=settings["top_k"],
            top_p=settings["top_p"], min_p=settings.get("min_p", 0.0),
            repetition_penalty=advanced["repetition_penalty"],
            repetition_window=advanced["repetition_window"],
            token_run_penalty=advanced["token_run_penalty"],
            token_run_grace=advanced["token_run_grace"],
            prompt_length=len(prompt))

        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        # Sampling is done entirely by `processors`; generate() gets neutral kwargs so
        # it does not add a second, unordered copy of the same warpers.
        self.model._pending_speaker_embedding = reference["embedding"]
        try:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids, max_new_tokens=int(max_new_tokens), do_sample=True,
                    temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                    logits_processor=processors,
                    eos_token_id=[self.token_map.speech_end_id, self.token_map.stop_id],
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None) or 0)
        finally:
            self.model._pending_speaker_embedding = None

        low, high = self.token_map.audio_id_base, self.token_map.audio_id_end
        generated = output[0].tolist()[len(prompt):]
        return [t - low for t in generated if low <= t < high]

    def _chunk_budget(self, chunk, max_new_tokens, single_chunk):
        """Cap a chunk's generation at what its text plausibly needs.

        A single chunk gets the full budget; within a longer piece, sizing the
        budget from the chunk's own length stops one runaway chunk from consuming
        the whole allowance.
        """
        if single_chunk:
            return int(max_new_tokens)
        expected = len(chunk) / CHARS_PER_SECOND * TOKENS_PER_SECOND * CHUNK_BUDGET_SLACK
        return max(MIN_CHUNK_BUDGET, min(int(max_new_tokens), int(expected)))

    def _synthesize(self, reference, target_text, preset, max_new_tokens, normalize,
                    use_reference_prompt, overrides, advanced):
        spoken = text_utils.normalize_text(target_text, normalize=normalize)
        if not spoken.strip():
            raise ValueError("no text to speak")
        chunks = text_utils.split_for_synthesis(spoken, max_chars=advanced["chunk_chars"],
                                                min_chars=advanced["chunk_min_chars"])
        settings = self._resolve_preset(preset, overrides)
        if advanced["seed"] is not None:
            torch.manual_seed(int(advanced["seed"]))

        use_emotion = advanced["use_emotion"]
        if use_emotion is None:
            use_emotion = text_utils.has_expression_tag(target_text)
        emotion_ids = None
        if self.emotion_control and use_emotion:
            emotion_ids = self.tokenizer.encode(self.emotion_control, add_special_tokens=False)

        # In-context cloning improves speaker fidelity on long text but destabilises
        # short prompts, where a long reference is extra context the model can loop
        # back into. The checkpoint's config decides which side of that trade it wants.
        if use_reference_prompt is None:
            use_reference_prompt = self.use_reference_prompt

        for chunk in chunks:
            budget = self._chunk_budget(chunk, max_new_tokens, single_chunk=len(chunks) == 1)
            audio_ids = self._generate_tokens(reference, chunk, settings, budget,
                                              use_reference_prompt, emotion_ids, advanced)
            wav = self.codec.decode(reference["conditioning"], audio_ids)
            if wav is not None:
                yield wav

    # ── public API ────────────────────────────────────────────────────────────
    def _speaker(self, voice, reference, reference_text):
        """Resolve the speaker: a built-in voice, or a reference recording."""
        if reference is not None and voice is not None:
            raise ValueError("give either voice= or reference=, not both")
        if reference is not None:
            return self._clone_reference(reference, reference_text)
        if reference_text is not None:
            raise ValueError("reference_text= only applies together with reference=")
        voice = voice or self.config.get("default_voice")
        if voice is None:
            raise ValueError("no voice given and the model declares no default voice")
        return self._voice_reference(voice)

    def generate(self, text, voice=None, reference=None, reference_text=None,
                 preset=None, max_new_tokens=1000, normalize=True,
                 use_reference_prompt=None, temperature=None, top_k=None,
                 top_p=None, min_p=None, advanced=None):
        """Synthesize `text` as a built-in voice, or as a cloned one.

        The speaker comes from exactly one of `voice` or `reference`:

            m.generate("Hello.", voice="Bruno")               # a built-in voice
            m.generate("Hello.", reference="my_voice.wav")    # cloned from a recording

        Args:
            text: what to speak. Supports a leading `[emotion]` tag, inline
                `<event>` tags and `(((emphasis)))` spans.
            voice: a name from `available_voices`. Defaults to the repository's
                own default voice when neither this nor `reference` is given.
            reference: path to roughly 5-30 seconds of a single speaker, to clone.
            reference_text: what that clip says. Transcribed with Whisper if omitted.
            preset: a decoding preset name, e.g. "stable" or "expressive".
            max_new_tokens: per-chunk token budget; 25 tokens is about one second.
            normalize: expand written forms ("$12.50" -> "twelve dollars fifty cents").
            use_reference_prompt: include the reference clip's own audio in the
                prompt. Defaults to the checkpoint's own setting.
            temperature, top_k, top_p, min_p: override individual preset values.
            advanced: dict of rarely-needed knobs — repetition and run penalties,
                chunking, emotion conditioning, and `seed` for a reproducible
                generation. See `ADVANCED_OPTIONS` for the full set and defaults;
                an unknown key raises ValueError.

        Returns:
            A float32 numpy waveform at `self.sample_rate`.
        """
        options = self._resolve_advanced(advanced)
        wavs = list(self._synthesize(
            self._speaker(voice, reference, reference_text), text,
            preset or self.default_preset, max_new_tokens, normalize,
            use_reference_prompt,
            dict(temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p),
            options))
        return self._join(wavs, options["chunk_gap_s"])

    def generate_stream(self, text, voice=None, reference=None, reference_text=None,
                        preset=None, max_new_tokens=1000, normalize=True,
                        use_reference_prompt=None, temperature=None, top_k=None,
                        top_p=None, min_p=None, advanced=None):
        """`generate`, yielding each chunk's audio as it is produced.

        Chunks are yielded untrimmed and unjoined — `generate` post-processes the
        seams, which a streaming caller cannot do without waiting for the next
        chunk. Text short enough to fit one chunk yields exactly once.
        """
        yield from self._synthesize(
            self._speaker(voice, reference, reference_text), text,
            preset or self.default_preset, max_new_tokens, normalize,
            use_reference_prompt,
            dict(temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p),
            self._resolve_advanced(advanced))

    def generate_to_file(self, text, output_path, **kwargs):
        """Synthesize and write a wav. Returns the path written."""
        import soundfile as sf
        audio = self.generate(text, **kwargs)
        sf.write(str(output_path), audio, self.sample_rate)
        return str(output_path)

    def normalize_text(self, text):
        """The spoken form the model will actually hear for `text`."""
        return text_utils.normalize_text(text)

    def _join(self, wavs, gap_s=text_utils.CHUNK_GAP_S):
        joined = text_utils.join_chunks(wavs, self.sample_rate, gap_s=gap_s)
        if joined is None:
            raise RuntimeError(
                "the model produced no audio tokens — try different text or a different preset")
        return joined

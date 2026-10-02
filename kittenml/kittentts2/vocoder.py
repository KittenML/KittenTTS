"""The audio codec ends of the KittenTTS 2 pipeline.

The language model neither reads nor writes waveforms — it consumes and produces
S3 codec tokens. This module owns both conversions:

    reference wav  --tokenizer-->  codec tokens    (for the in-context prompt)
    codec tokens   --s3gen------>  24 kHz waveform (the generated audio)

Both use ResembleAI's chatterbox-turbo s3gen, vendored in `s3gen.py` with its
weights fetched from Hugging Face on first use. Only s3gen is loaded; chatterbox's
own text-to-token model is unused, since that is precisely what KittenTTS 2
replaces.
"""

import pathlib

import librosa
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .s3gen import S3_SR, S3GEN_SIL, S3GEN_SR, SPEECH_VOCAB_SIZE, S3Gen

CHATTERBOX_REPO = "ResembleAI/chatterbox-turbo"
S3GEN_WEIGHTS = "s3gen_meanflow.safetensors"

# s3gen conditions on at most 10 s of reference audio, and its own conditioning
# path asserts the clip is longer than 5 s.
DECODER_COND_SECONDS = 10
DECODER_MIN_SECONDS = 6.0

# The meanflow decoder converges in two CFM steps; more cost time without changing
# the output audibly.
CFM_TIMESTEPS = 2

# The optional quantised decoder (see student_flow.py) replaces only the flow, and
# was distilled against a shorter, truncated voice prompt than the stock flow uses.
# These are its values, not ours to vary: conditioning it differently is not what
# it was trained on.
STUDENT_REF_SECONDS = 4.0
STUDENT_MAX_PROMPT_TOKENS = 112


class S3Codec:
    """Loads s3gen once and holds the per-reference conditioning it derives."""

    sample_rate = S3GEN_SR

    def __init__(self, device="cuda", cache_dir=None, student_weights=None):
        """
        Args:
            student_weights: path to a packed student flow. When given, the flow
                stage is replaced by it and the tokenizer, speaker encoder and
                vocoder are still used from s3gen.
        """
        self.device = device
        weights_path = hf_hub_download(repo_id=CHATTERBOX_REPO, filename=S3GEN_WEIGHTS,
                                       cache_dir=cache_dir)
        self.s3gen = S3Gen(meanflow=True)
        self.s3gen.load_state_dict(load_file(weights_path), strict=True)
        self.s3gen.to(device).eval()
        self._ref_cache = {}

        self.student = None
        if student_weights is not None:
            self.student = load_student_flow(student_weights, device=device)

    def encode_reference(self, wav_path):
        """Reference clip -> S3 codec token ids, for the in-context prompt."""
        wav16, _ = librosa.load(str(wav_path), sr=S3_SR, mono=True)
        with torch.inference_mode():
            tokens, lengths = self.s3gen.tokenizer(
                torch.from_numpy(wav16).float().unsqueeze(0).to(self.device))
        return tokens[0, :int(lengths[0].item())].long().cpu().tolist()

    def reference_conditioning(self, wav_path):
        """Reference clip -> the conditioning dict used when decoding.

        Cached per path: this is pure inference over a fixed clip, and a built-in
        voice is re-used on every request. The two decoders want different
        conditioning, so the cache is keyed on which one is active.
        """
        key = (str(wav_path), self.student is not None)
        if key not in self._ref_cache:
            seconds = STUDENT_REF_SECONDS if self.student is not None else DECODER_COND_SECONDS
            wav24 = _load_decoder_reference(str(wav_path))[:int(seconds * S3GEN_SR)]
            with torch.inference_mode():
                ref = self.s3gen.embed_ref(wav24, S3GEN_SR, device=self.device)
            self._ref_cache[key] = self._truncate_prompt(ref) if self.student else ref
        return self._ref_cache[key]

    def _truncate_prompt(self, ref):
        """The prompt the student flow expects: at most 112 tokens, mel to match.

        The mel runs at twice the token rate, so the two have to be cut together
        or the flow sees a prompt whose halves disagree.
        """
        n = min(ref["prompt_token"].shape[1], ref["prompt_feat"].shape[1] // 2,
                STUDENT_MAX_PROMPT_TOKENS)
        return {
            "prompt_token": ref["prompt_token"][:, :n],
            "prompt_token_len": torch.tensor([n], device=self.device),
            "prompt_feat": ref["prompt_feat"][:, :n * 2],
            "embedding": ref["embedding"],
        }

    def decode(self, ref_conditioning, audio_ids):
        """Generated codec tokens -> a 24 kHz waveform.

        Out-of-range ids are dropped rather than raising: a truncated or degenerate
        generation should still return the audio it did produce.
        """
        valid = [t for t in audio_ids if 0 <= t < SPEECH_VOCAB_SIZE]
        if not valid:
            return None
        tokens = torch.tensor(valid, dtype=torch.long, device=self.device)
        # s3gen drops its last few tokens as streaming lookahead, so pad with silence
        # to keep the final syllable of the generation.
        tokens = torch.cat([tokens, torch.full((3,), int(S3GEN_SIL),
                                               dtype=torch.long, device=self.device)])
        with torch.inference_mode():
            if self.student is not None:
                wav = self._decode_with_student(ref_conditioning, tokens)
            else:
                wav, _ = self.s3gen.inference(speech_tokens=tokens,
                                              ref_dict=ref_conditioning,
                                              n_cfm_timesteps=CFM_TIMESTEPS)
        return wav.squeeze().detach().cpu().float().numpy()

    def _decode_with_student(self, cond, tokens):
        """Student flow to mel in one step, then s3gen's own vocoder to audio."""
        from .student_flow import meanflow_forward

        tokens = tokens.unsqueeze(0)
        lengths = torch.tensor([tokens.shape[1]], device=self.device)
        mel, _, _ = meanflow_forward(self.student, tokens, lengths, n_steps=1, **cond)
        mel = mel.to(dtype=self.s3gen.dtype)
        cache = torch.zeros(1, 1, 0, device=self.device, dtype=self.s3gen.dtype)
        wav, _ = self.s3gen.mel2wav.inference(speech_feat=mel, cache_source=cache)
        # s3gen applies this fade itself on its own path; the student path calls the
        # vocoder directly, so it has to be applied here or the clip starts with a click.
        wav = wav.clone()
        wav[:, :len(self.s3gen.trim_fade)] *= self.s3gen.trim_fade
        return wav


def _load_decoder_reference(wav_path, min_seconds=DECODER_MIN_SECONDS):
    """Load a reference at 24 kHz, looping it if it is too short for s3gen.

    Short clips are tiled — the speech is repeated, never silence-padded, because
    silence in the conditioning degrades the vocoder's timbre estimate.
    """
    wav, _ = librosa.load(str(wav_path), sr=S3GEN_SR, mono=True)
    if len(wav) == 0:
        raise ValueError(f"reference audio is empty: {wav_path}")
    needed = int(min_seconds * S3GEN_SR)
    if len(wav) < needed:
        wav = np.tile(wav, int(np.ceil(needed / len(wav))))
    return wav.astype(np.float32)


def resolve_reference_path(path):
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"reference audio not found: {path}")
    return str(path)


def load_student_flow(weights_path, device="cuda"):
    """Rebuild a quantised student flow from packed integer codes and scales.

    The stored weights are already on the quantisation grid, so they are installed
    as-is rather than being quantised a second time — see the note in
    student_flow.py about why a dequantise/requantise round trip is not free.
    """
    from .student_flow import build_flow, convert_to_bitnet

    blob = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    flow = build_flow(**blob["arch"])
    convert_to_bitnet(flow, mode=blob["mode"], norm_input=blob["norm_input"],
                      prequantised=True)

    state = {}
    for name, entry in blob["quantised"].items():
        codes = (_unpack4(entry["packed"], entry["numel"]) if blob["bits"] == 4
                 else entry["codes"])
        scale = entry["scale"].reshape(-1, *([1] * (len(entry["shape"]) - 1)))
        state[name] = codes.reshape(entry["shape"]).float() / scale
    state.update({k: v.float() for k, v in blob["dense"].items()})

    missing, unexpected = flow.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"student flow weights do not match the architecture "
                         f"(missing={missing[:3]}, unexpected={unexpected[:3]})")
    return flow.to(device).eval()


def _unpack4(packed, numel):
    """Two 4-bit codes per byte, each biased by 8 when written."""
    high = (packed >> 4).to(torch.int16) - 8
    low = (packed & 0x0F).to(torch.int16) - 8
    return torch.stack([high, low], dim=1).reshape(-1)[:numel].to(torch.int8)

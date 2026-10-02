"""Token layout for the KittenTTS 2 speech language model.

The model shares one vocabulary between text and audio. Text occupies the base
Qwen3 BPE range, then six named special tokens, then 6561 S3 codec tokens at a
fixed offset, then optional marker rows appended at the end:

    [0, audio_id_base)                      base text vocab + named specials
    [audio_id_base, audio_id_base + 6561)   S3 audio codec tokens
    final_seg                               end-of-sample cue after the last text
    reference_*                             wrappers for the in-context reference

Unlike the text tokens, audio ids are never produced by the text tokenizer — the
prompt builder emits the raw integers. `TokenMap` is what keeps both sides
agreeing on where that range starts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenMap:
    """Resolved token ids for one checkpoint.

    Checkpoints differ in how they wrap the in-context reference, so the
    `reference_*` ids are optional: when they are 0 the reference reuses the
    plain text_start/speech_start/speech_end ids instead (see
    `prompt.build_reference_prefix`).
    """

    audio_id_base: int
    num_audio_tokens: int
    speech_start_id: int
    speech_end_id: int
    text_start_id: int
    text_end_id: int
    start_id: int
    stop_id: int
    final_seg_id: int = 0
    reference_speech_start_id: int = 0
    reference_speech_end_id: int = 0
    reference_text_start_id: int = 0
    reference_text_end_id: int = 0

    @property
    def audio_id_end(self) -> int:
        """One past the last audio codec id."""
        return self.audio_id_base + self.num_audio_tokens

    @property
    def uses_reference_role_tokens(self) -> bool:
        return bool(self.reference_text_start_id)

    @classmethod
    def from_config(cls, config: dict) -> "TokenMap":
        """Build from the `token_map` block of a KittenTTS 2 repo's config.json."""
        tm = config["token_map"]
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(tm) - known
        if unknown:
            raise ValueError(f"unknown token_map keys: {sorted(unknown)}")
        return cls(**tm)

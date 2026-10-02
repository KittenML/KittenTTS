"""Prompt assembly for KittenTTS 2.

A generation prompt has two parts. The reference prefix carries the voice being
cloned — the reference transcript plus its audio as codec tokens — and the target
segment carries the text to speak. The model then continues with audio tokens
until it emits `<speech_end>`.
"""


def build_reference_prefix(token_map, reference_text_ids, reference_audio_tokens):
    """Assemble the in-context reference portion of the prompt.

    Checkpoints wrap the reference differently depending on how they were trained,
    and prompting one with the other's layout measurably degrades cloning:

      reference-supervised checkpoints (reference-role tokens present):
        [START] <reference_text_start> text <reference_text_end>
                <reference_speech_start> audio <reference_speech_end>

      everything else, reusing the target segment's own wrappers:
        [START] <text_start> text <speech_start> audio <speech_end>

    The caller appends the target segment itself — see `build_generation_prompt`.
    """
    base = token_map.audio_id_base
    audio_ids = [base + t for t in reference_audio_tokens]
    if token_map.uses_reference_role_tokens:
        return [token_map.start_id,
                token_map.reference_text_start_id, *reference_text_ids,
                token_map.reference_text_end_id,
                token_map.reference_speech_start_id, *audio_ids,
                token_map.reference_speech_end_id]
    return [token_map.start_id,
            token_map.text_start_id, *reference_text_ids,
            token_map.speech_start_id, *audio_ids, token_map.speech_end_id]


def build_generation_prompt(token_map, target_text_ids, reference_prefix=None,
                            emotion_ids=None):
    """Assemble the full prompt for one chunk of target text.

    `reference_prefix=None` drops the in-context reference entirely, leaving the
    speaker embedding (through `spk_proj`) as the only identity signal. That
    trades a little speaker similarity for stability on short text, where a long
    reference is context the model can loop back into.

    `emotion_ids` are the tokenized `{emo: 1}` control, spliced in immediately
    before the target text as the checkpoint was trained to receive it.
    """
    head = list(reference_prefix) if reference_prefix else [token_map.start_id]
    final_marker = [token_map.final_seg_id] if token_map.final_seg_id else []
    return [*head,
            *(emotion_ids or []),
            token_map.text_start_id, *target_text_ids,
            *final_marker,
            token_map.speech_start_id]

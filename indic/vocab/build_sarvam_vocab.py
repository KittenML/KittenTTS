"""Rebuild the KittenTTS 2 checkpoint around a different text tokenizer.

The model shares one embedding table between text and audio, and that table is
tied to `lm_head`. Swapping the text tokenizer therefore is not a config change —
it is surgery on the one matrix that both reads text and writes audio:

    old (158,241 rows)                    new (~74,700 rows)
    [0, 151669)  Qwen3 BPE text   ──X──   [0, V_text)        Sarvam pieces
    151669-151674 named specials   ──→    V_text + 0..5      same six, copied
    151675+6561  S3 audio codec    ──→    V_text + 6..       same 6561, copied
    158236-158240 marker rows      ──→    tail               copied

Only the text rows are actually new. The audio rows carry the model's entire
ability to produce sound and are copied verbatim, so what the fine-tune has to
relearn is the mapping from new text ids to the *same* audio behaviour — not the
behaviour itself. Keeping the audio block contiguous and the markers at the tail
means `TokenMap`, `prompt.py` and the logit masking in `logits.py` all keep
working off `config.json` with no code change.

New text rows are initialised as the mean of the Qwen embeddings for the same
string. A Sarvam piece covers 5-6 Qwen tokens on average, so the mean is a real
signal rather than noise: the body already knows what "नम" looks like spread over
five BPE pieces, and the averaged row puts the new single piece near that point
in embedding space. `--init random` is the ablation that measures whether this
matters.

    python -m indic.vocab.build_sarvam_vocab --out /path/to/kitten-tts-2-sarvam

The ternary transformer weights are vocabulary-independent and are copied through
untouched, so this does not disturb the quantization.
"""

import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

SOURCE_REPO = "KittenML/kitten-tts-2"
SARVAM_REPO = "sarvamai/sarvam-1"

# The six named specials, in the order they must keep: `TokenMap` reads them by
# name from config.json, but the GGUF export and the vLLM loader both assume the
# block is contiguous and in this order.
NAMED_SPECIALS = ["speech_start_id", "speech_end_id", "text_start_id",
                  "text_end_id", "start_id", "stop_id"]
MARKERS = ["final_seg_id", "reference_speech_start_id", "reference_speech_end_id",
           "reference_text_start_id", "reference_text_end_id"]

# Pad the table to a multiple of this. Matmul kernels tile the vocab dimension, and
# a ragged final tile costs more than the unused rows do. The padding rows are
# unreachable: no tokenizer emits them and `logits.py` masks sampling to the audio
# range, so they can never be produced.
VOCAB_PAD = 128


def resolve_source(repo_or_path, cache_dir=None):
    """Return a local directory holding the source checkpoint."""
    path = pathlib.Path(repo_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download
    # The speaker encoder and the voice embeddings are copied into the output
    # verbatim -- they index no text ids, so the vocabulary swap leaves them valid.
    # They have to be fetched all the same: without `speaker/model.safetensors` the
    # rebuilt checkpoint cannot be speaker-conditioned, and `attach_speaker_head`
    # refuses to load it rather than silently generating in the wrong voice.
    # `lm/model-*.safetensors` are excluded: they are the packed and emb4 variants,
    # derived from the bf16 weights and stale the moment the table changes.
    return pathlib.Path(snapshot_download(
        repo_or_path, cache_dir=cache_dir,
        allow_patterns=["config.json", "lm/*", "voices/*", "speaker/*", "decoders/*"],
        ignore_patterns=["cpp/*"]))


def plan_layout(n_text):
    """Where every block lands in the new table, given `n_text` text rows.

    Mirrors the source layout exactly — specials, then audio, then markers — so
    that the only thing that changed is how many text rows precede it.
    """
    cursor = n_text
    specials = {name: cursor + i for i, name in enumerate(NAMED_SPECIALS)}
    cursor += len(NAMED_SPECIALS)
    audio_base = cursor
    cursor += 6561
    markers = {name: cursor + i for i, name in enumerate(MARKERS)}
    cursor += len(MARKERS)
    total = cursor
    padded = -(-total // VOCAB_PAD) * VOCAB_PAD
    return {
        "token_map": {"audio_id_base": audio_base, "num_audio_tokens": 6561,
                      **specials, **markers},
        "n_rows": total,
        "n_rows_padded": padded,
    }


def build_embedding(old_embed, old_map, new_map, src_tok, dst_tok, init="mean",
                    seed=0, report_every=5000):
    """Produce the new embedding matrix.

    Returns (weight, stats). Rows are filled in three passes so that a failure in
    the expensive text pass still leaves the audio rows diagnosable.
    """
    import torch

    n_rows = new_map["n_rows_padded"]
    hidden = old_embed.shape[1]
    dtype = old_embed.dtype

    # Scale of the random init matches the source table's own per-row norm rather
    # than config.initializer_range: the trained table has drifted from its init,
    # and seeding new rows at the wrong magnitude shows up as an immediate loss
    # spike that is easy to mistake for a bad mapping.
    old_std = old_embed.float().std().item()
    generator = torch.Generator().manual_seed(seed)
    new = (torch.randn(n_rows, hidden, generator=generator, dtype=torch.float32)
           * old_std).to(dtype)

    stats = {"embedding_std": old_std, "n_rows": n_rows, "init": init}

    # 1. audio block — the rows that carry the model's voice. Copied verbatim.
    n_audio = old_map["num_audio_tokens"]
    new[new_map["token_map"]["audio_id_base"]:
        new_map["token_map"]["audio_id_base"] + n_audio] = \
        old_embed[old_map["audio_id_base"]:old_map["audio_id_base"] + n_audio]

    # 2. named specials and markers — same meaning, new addresses.
    for name in NAMED_SPECIALS + MARKERS:
        old_id = old_map.get(name, 0)
        if old_id:
            new[new_map["token_map"][name]] = old_embed[old_id]

    # 3. text rows.
    if init.startswith("mean"):
        stats.update(_init_text_mean(new, old_embed, src_tok, dst_tok, report_every))
        if init == "mean_rescaled":
            stats.update(_rescale_text_rows(new, old_embed, old_map, dst_tok))
    else:
        stats["mapped"] = 0
    return new, stats


def _rescale_text_rows(new, old_embed, old_map, dst_tok):
    """Restore the norm scale that averaging destroys.

    Averaging k roughly-independent vectors shrinks the result's norm by about
    sqrt(k), and a Sarvam piece covers 5.4 Qwen tokens on average -- so the mean
    init lands the new rows at roughly 0.6x the norm of the trained table it is
    imitating (measured). That is not a cosmetic difference: `embed_tokens` is the
    input to layer 0, so uniformly short rows mean a uniformly weak text signal,
    and because the table is tied to `lm_head` the same rows also produce
    systematically smaller logits.

    One global factor rather than per-row normalisation, deliberately: averaging
    also makes a piece covering many sub-tokens shorter than one covering few, and
    that difference carries real information about how specific the piece is.
    Rescaling each row to a fixed norm would erase it.
    """
    import torch

    n_text = len(dst_tok)
    new_norm = new[:n_text].float().norm(dim=1).median()
    old_norm = old_embed[:old_map["speech_start_id"]].float().norm(dim=1).median()
    factor = (old_norm / new_norm.clamp(min=1e-8)).item()
    new[:n_text] = (new[:n_text].float() * factor).to(new.dtype)
    return {"rescale_factor": factor,
            "median_norm_before": new_norm.item(),
            "median_norm_target": old_norm.item()}


def _init_text_mean(new, old_embed, src_tok, dst_tok, report_every):
    """Each new text row = mean of the old rows for the same string.

    The piece is converted back to the text it represents before being re-encoded,
    which is what makes the two vocabularies comparable at all: `▁नम` is the
    string " नम", and Qwen has its own (longer) spelling of that same string.
    """
    import torch

    n_text = len(dst_tok)
    mapped = unmapped = 0
    covered = 0
    for new_id in range(n_text):
        piece = dst_tok.convert_ids_to_tokens(new_id)
        if piece is None:
            unmapped += 1
            continue
        text = _piece_to_text(piece, dst_tok, new_id)
        if not text:
            unmapped += 1
            continue
        old_ids = src_tok(text, add_special_tokens=False)["input_ids"]
        if not old_ids:
            unmapped += 1
            continue
        new[new_id] = old_embed[torch.tensor(old_ids)].float().mean(dim=0).to(new.dtype)
        mapped += 1
        covered += len(old_ids)
        if report_every and new_id and new_id % report_every == 0:
            print(f"  text rows {new_id:,}/{n_text:,}", flush=True)
    return {"mapped": mapped, "unmapped": unmapped,
            "mean_old_tokens_per_piece": covered / max(mapped, 1)}


def _piece_to_text(piece, tok, token_id):
    """The literal string a tokenizer piece stands for.

    Three cases have to be handled separately or the mapping silently degrades:
    byte-fallback pieces (`<0xE0>`) are raw bytes and must not be decoded as text,
    control symbols stand for nothing pronounceable, and ordinary pieces carry
    SentencePiece's `▁` word-boundary marker that has to become a real space.
    """
    if piece.startswith("<0x") and piece.endswith(">"):
        # A single byte. Decoding it in isolation is usually invalid UTF-8, so map
        # it through whatever the source tokenizer uses for that same byte.
        return None
    if piece in (tok.all_special_tokens or []):
        return None
    text = tok.convert_tokens_to_string([piece])
    if not text or not text.strip():
        # SentencePiece renders a bare `▁` as "", which would otherwise map to an
        # empty encode and leave the row at its random init.
        return " " if piece.startswith("▁") else None
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE_REPO, help="repo id or local dir")
    ap.add_argument("--tokenizer", default=SARVAM_REPO, help="replacement tokenizer")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--init", choices=["mean", "mean_rescaled", "random"],
                    default="mean_rescaled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the layout and exit without loading weights")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    dst_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    source = resolve_source(args.source, args.cache_dir)
    config = json.load(open(source / "config.json"))
    lm_config = json.load(open(source / "lm" / "config.json"))
    old_map = config["token_map"]

    n_text = len(dst_tok)
    layout = plan_layout(n_text)
    print(f"source      {source}")
    print(f"tokenizer   {args.tokenizer}  ({n_text:,} pieces, {type(dst_tok).__name__})")
    print(f"old vocab   {lm_config['vocab_size']:,} rows, audio_id_base "
          f"{old_map['audio_id_base']:,}")
    print(f"new vocab   {layout['n_rows']:,} rows -> padded {layout['n_rows_padded']:,}, "
          f"audio_id_base {layout['token_map']['audio_id_base']:,}")
    removed = (lm_config["vocab_size"] - layout["n_rows_padded"]) * lm_config["hidden_size"]
    verb = "removed from" if removed > 0 else "added to"
    print(f"embedding   {abs(removed) / 1e6:.1f}M parameters {verb} the table "
          f"({abs(removed) * 2 / 1e6:.0f} MB at bf16, tied with lm_head)")
    if args.dry_run:
        print(json.dumps(layout["token_map"], indent=2))
        return

    import torch
    from safetensors.torch import load_file, save_file

    src_tok = AutoTokenizer.from_pretrained(source / "lm")
    print("\nloading lm/model.safetensors ...", flush=True)
    state = load_file(str(source / "lm" / "model.safetensors"))
    embed_key = next(k for k in state if k.endswith("embed_tokens.weight"))
    old_embed = state[embed_key]

    print(f"building new embedding ({args.init} init) ...", flush=True)
    new_embed, stats = build_embedding(old_embed, old_map, layout, src_tok, dst_tok,
                                       init=args.init, seed=args.seed)
    state[embed_key] = new_embed
    # Tied weights: an explicit lm_head row set would now disagree with the table.
    state.pop("lm_head.weight", None)

    out = args.out
    (out / "lm").mkdir(parents=True, exist_ok=True)
    print(f"writing {out} ...", flush=True)
    save_file(state, str(out / "lm" / "model.safetensors"), metadata={"format": "pt"})

    lm_config["vocab_size"] = layout["n_rows_padded"]
    lm_config.pop("bos_token_id", None)
    lm_config.pop("eos_token_id", None)
    json.dump(lm_config, open(out / "lm" / "config.json", "w"), indent=2)

    config["token_map"] = layout["token_map"]
    # The packed and emb4 variants are derived from the bf16 weights, so they are
    # stale the moment the table changes. Dropping the keys is what stops
    # `weights="packed"` from silently loading a checkpoint with the old vocab.
    for stale in ("lm_packed", "lm_emb4", "cpp"):
        config.pop(stale, None)
    config["default_weights"] = "bf16"
    config["tokenizer_source"] = args.tokenizer
    config["embedding_init"] = args.init
    json.dump(config, open(out / "config.json", "w"), indent=2)

    dst_tok.save_pretrained(out / "lm")
    for extra in ("voices", "speaker", "decoders"):
        src_extra = source / extra
        if src_extra.exists():
            shutil.copytree(src_extra, out / extra, dirs_exist_ok=True)

    json.dump({**stats, **{k: v for k, v in layout.items() if k != "token_map"}},
              open(out / "vocab_surgery.json", "w"), indent=2)
    print("\n" + json.dumps(stats, indent=2))
    print(f"\ndone -> {out}")
    print("NOTE: voices/*.npz and speaker/ are unchanged; they do not depend on the "
          "text vocabulary.")


if __name__ == "__main__":
    main()

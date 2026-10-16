"""Check that a vocabulary-swapped checkpoint is correct, not merely written.

`build_sarvam_vocab.py` rewrites the one matrix that both reads text and writes
audio. If it gets the audio block's new address wrong by a single row, nothing
raises: the file saves, the model loads, generation runs, and the output is
noise — because every codec id now points at the embedding of its neighbour. That
failure is invisible until someone listens to a 40-hour training run.

So the checks here are the ones that would catch exactly that, and they compare
against the source checkpoint rather than against the config that was written
from the same assumptions:

  * the audio, special and marker rows are **bit-identical** to the rows they
    were copied from, read out of the source file at the source's own offsets;
  * the declared `token_map` parses through the real `TokenMap` and its audio
    range lands inside the table;
  * the new text rows look like trained embeddings rather than noise, by norm;
  * the shipped tokenizer round-trips Indic text and addresses only text rows.

    python -m indic.vocab.verify_surgery --checkpoint indic/data/kitten-tts-2-sarvam
"""

import argparse
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SOURCE_REPO = "KittenML/kitten-tts-2"


def load_token_map(config):
    """Parse through the real TokenMap, by file path to dodge the heavy package."""
    spec = importlib.util.spec_from_file_location(
        "_tokens", ROOT / "kittenml" / "kittentts2" / "tokens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TokenMap.from_config(config)


def check(results, name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=pathlib.Path,
                    default=ROOT / "indic" / "data" / "kitten-tts-2-sarvam")
    ap.add_argument("--source", default=SOURCE_REPO)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    out_dir = args.checkpoint
    config = json.load(open(out_dir / "config.json"))
    lm_config = json.load(open(out_dir / "lm" / "config.json"))
    results = []
    print(f"checkpoint {out_dir}\n")

    token_map = load_token_map(config)
    new_state = load_file(str(out_dir / "lm" / "model.safetensors"))
    embed_key = next(k for k in new_state if k.endswith("embed_tokens.weight"))
    new_embed = new_state[embed_key]
    rows, hidden = new_embed.shape

    check(results, "config token_map parses", True,
          f"audio_id_base {token_map.audio_id_base:,}")
    check(results, "vocab_size matches embedding rows",
          lm_config["vocab_size"] == rows, f"{lm_config['vocab_size']:,} vs {rows:,}")
    check(results, "audio range inside table",
          token_map.audio_id_end <= rows,
          f"audio ends at {token_map.audio_id_end:,}, table has {rows:,}")
    check(results, "lm_head is tied (no separate head stored)",
          "lm_head.weight" not in new_state)
    check(results, "stale packed-weight keys dropped",
          "lm_packed" not in config and "lm_emb4" not in config)

    # The comparison that matters: against the source file, at the source's offsets.
    from huggingface_hub import hf_hub_download
    src_cfg = json.load(open(hf_hub_download(args.source, "config.json")))
    src_map = load_token_map(src_cfg)
    src_embed = load_file(hf_hub_download(args.source, "lm/model.safetensors"))[embed_key]

    n_audio = src_map.num_audio_tokens
    old_audio = src_embed[src_map.audio_id_base:src_map.audio_id_base + n_audio]
    new_audio = new_embed[token_map.audio_id_base:token_map.audio_id_base + n_audio]
    check(results, "audio rows bit-identical to source",
          torch.equal(old_audio, new_audio), f"{n_audio:,} rows")

    # Off-by-one is the failure this is really hunting, so prove the alignment is
    # unique: shifting by one row must NOT also match.
    shifted = new_embed[token_map.audio_id_base + 1:token_map.audio_id_base + 1 + n_audio]
    check(results, "audio alignment is unique (off-by-one would differ)",
          not torch.equal(old_audio, shifted))

    named = ["speech_start_id", "speech_end_id", "text_start_id", "text_end_id",
             "start_id", "stop_id", "final_seg_id", "reference_speech_start_id",
             "reference_speech_end_id", "reference_text_start_id",
             "reference_text_end_id"]
    mismatched = [n for n in named
                  if getattr(src_map, n)
                  and not torch.equal(src_embed[getattr(src_map, n)],
                                      new_embed[getattr(token_map, n)])]
    check(results, "special and marker rows copied exactly",
          not mismatched, f"{len(named) - len(mismatched)}/{len(named)} rows match")

    # New text rows should sit in the same norm regime as the trained table. Noise
    # at the wrong scale is the other silent failure: it trains, just badly.
    n_text = token_map.speech_start_id
    text_norm = new_embed[:n_text].float().norm(dim=1)
    old_text_norm = src_embed[:src_map.speech_start_id].float().norm(dim=1)
    ratio = text_norm.median().item() / old_text_norm.median().item()
    check(results, "text-row norms in the trained regime",
          0.3 < ratio < 3.0,
          f"median norm {text_norm.median():.3f} vs source {old_text_norm.median():.3f} "
          f"(ratio {ratio:.2f})")
    check(results, "no NaN or Inf in the table",
          bool(torch.isfinite(new_embed.float()).all()))

    tok = AutoTokenizer.from_pretrained(out_dir / "lm")
    sample = "यह हिंदी वाक्य है। இது தமிழ். This is English."
    ids = tok(sample, add_special_tokens=False)["input_ids"]
    check(results, "shipped tokenizer round-trips mixed script",
          tok.decode(ids, skip_special_tokens=True).strip() == sample.strip())
    check(results, "tokenizer emits only text rows",
          max(ids) < token_map.speech_start_id,
          f"max id {max(ids):,} < first special {token_map.speech_start_id:,}")

    passed = sum(r["ok"] for r in results)
    print(f"\n{passed}/{len(results)} checks passed")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"checkpoint": str(out_dir), "passed": passed, "total": len(results),
             "results": results}, indent=2))
        print(f"wrote {args.out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

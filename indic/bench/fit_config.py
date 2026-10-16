"""Find the largest training configuration that actually fits on this GPU.

Memory estimates for a transformer are easy to write down and routinely wrong by
a factor of two: the optimizer state, the gradient-checkpointing boundaries and —
for this model especially — the cross-entropy over a 75k-row vocabulary all land
in different places than a parameter count suggests. So this measures instead of
predicting. It builds the real architecture at the real shapes, runs a real
forward/backward/step, and reads the allocator's high-water marks.

Weights are randomly initialised. Peak memory depends on tensor shapes and the
autograd graph, not on the values in them, so a random model measures the same
number as the real checkpoint while needing no download.

    python -m indic.bench.fit_config --vocab 74752
    python -m indic.bench.fit_config --vocab 158241 --mode lora   # the Qwen3 control

Each row reports both high-water marks: `max_memory_allocated` (bytes actually
handed to tensors) and `max_memory_reserved` (bytes the caching allocator holds
from the driver). The verdict is taken on *reserved*, because that is the one
that OOMs -- the allocator serves tensors out of blocks it has already claimed,
and a request it cannot fit into the pool fails even when the allocated total
still looks comfortable. The gap between the two is fragmentation overhead, and
on this workload it runs over a gigabyte. A config is only recommended if it
keeps `--reserve` GB free, because a laptop GPU is also driving a display and an
OOM three hours into a run costs the run.

An OOM row is a statement about this allocator state, not a hard bound. Rows
sitting within a few hundred MB of the line have been observed to flip with
allocation history, so treat the near-miss ones as "do not plan on it" rather
than as a measured ceiling.
"""

import argparse
import contextlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# What one training example looks like, in tokens. Derived from the sequence layout
# in prompt.py at the rates the codec and tokenizer actually run:
#   reference text (~32 Sarvam pieces) + reference audio (6 s x 25 Hz = 150)
#   + target text (~32) + target audio (10 s x 25 Hz = 250) + ~10 marker tokens
# Rounded up to leave room for the long tail. With Qwen3 the text part is 4-5x
# larger, which is itself part of what this compares.
SEQ_SARVAM = 512
SEQ_QWEN3 = 768


def build_model(vocab_size, device, dtype, grad_checkpointing=True):
    """A Qwen3 at KittenTTS 2's exact shape, with the vocabulary under test."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "qwen3", vocab_size=vocab_size, hidden_size=2048, intermediate_size=6144,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, max_position_embeddings=40960, rope_theta=1000000,
        rms_norm_eps=1e-6, tie_word_embeddings=True, attention_bias=False)
    # Materialise straight into `dtype`. Building at the default float32 and
    # casting afterwards puts the whole model on the card at 4 bytes/param on the
    # way through -- 6.4 GB for the Sarvam layout, 7.7 GB for Qwen3 -- and that
    # transient, not the training step, becomes the peak this script reports. It
    # is an artifact of how the probe gets its weights; a real run loads bf16
    # straight off disk and never pays it.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model = model.to_empty(device=device)
    # `spk_proj`: the trained speaker head (speaker.py:27-44). Small, but it is a
    # trainable full-precision module and it holds a gradient, so it belongs in
    # any honest measurement.
    model.spk_proj = torch.nn.Sequential(
        torch.nn.Linear(512, 2048), torch.nn.LayerNorm(2048)).to(device=device, dtype=dtype)
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def apply_mode(model, mode, lora_rank=32):
    """Set up which parameters train, and return the trainable count."""
    import torch

    for param in model.parameters():
        param.requires_grad_(False)

    if mode == "full":
        for param in model.parameters():
            param.requires_grad_(True)
    elif mode == "embed":
        # Phase A of the Sarvam experiment: only the new text rows have to learn
        # anything, so the body stays frozen and the embedding (tied to lm_head)
        # is the single trainable matrix.
        model.get_input_embeddings().weight.requires_grad_(True)
    elif mode == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        # The embedding is trained alongside the adapters, not wrapped by them: a
        # low-rank update cannot teach 68k brand-new rows what they mean.
        model.get_input_embeddings().weight.requires_grad_(True)
    else:
        raise ValueError(mode)

    for param in model.spk_proj.parameters():
        param.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable


def probe(vocab_size, mode, batch, seq, lora_rank=32, grad_checkpointing=True,
          optimizer="adamw8bit", device="cuda", steps=2):
    """Run real training steps and return peak VRAM in GB, or None on OOM."""
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = opt = None
    try:
        model = build_model(vocab_size, device, torch.bfloat16, grad_checkpointing)
        model, trainable = apply_mode(model, mode, lora_rank)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = make_optimizer(params, optimizer)

        embed = model.get_input_embeddings()
        for _ in range(steps):
            ids = torch.randint(0, vocab_size, (batch, seq), device=device)
            # The speaker vector is conditioning, not a token: `spk_proj` lifts a
            # 512-d reference embedding into hidden space and it is added to the
            # input embeddings. Routing it through the real path is the only way
            # spk_proj gets a gradient -- left outside the graph it would take no
            # optimizer state, and the measurement would quietly exclude it.
            spk = torch.randn(batch, 512, device=device, dtype=torch.bfloat16)
            inputs_embeds = embed(ids) + model.spk_proj(spk).unsqueeze(1)
            # Loss on target audio tokens only (training.md): roughly the last
            # 45% of the sequence. Masking the rest to -100 is not just
            # bookkeeping -- it is what the real run does, and an unmasked loss
            # would measure a different backward graph.
            labels = ids.clone()
            labels[:, :int(seq * 0.55)] = -100
            out = model(inputs_embeds=inputs_embeds, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        # Read after the optimizer step, so the state tensors Adam allocates on
        # its first step are inside the measurement. `reserved` is the binding
        # number -- see the module docstring -- and `peak` is kept beside it
        # because their difference is what fragmentation is costing us.
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        return {"peak_gb": peak, "reserved_gb": reserved,
                "trainable": trainable, "oom": False}
    except torch.cuda.OutOfMemoryError:
        return {"peak_gb": None, "reserved_gb": None, "trainable": None, "oom": True}
    finally:
        # The optimizer holds its own references to the parameters, so dropping
        # the model alone leaves the weights pinned and the empty_cache() below
        # releases nothing. Both have to go before the pool can actually shrink.
        del model, opt
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def make_optimizer(params, kind):
    """AdamW, optionally with 8-bit state.

    8-bit state matters more here than usual: the embedding is the biggest
    trainable tensor in every mode, so its optimizer state is the biggest single
    allocation after the frozen body, and quantising it is what moves a config
    from not-fitting to fitting.
    """
    import torch
    if kind == "adamw8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=1e-4)
        except ImportError:
            print("  (bitsandbytes missing; falling back to fp32 AdamW)", flush=True)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=1e-4, momentum=0.9)
    return torch.optim.AdamW(params, lr=1e-4)


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=int, default=74752,
                    help="74752 = Sarvam layout (padded), 158241 = stock Qwen3")
    ap.add_argument("--seq", type=int, default=0, help="0 picks from --vocab")
    ap.add_argument("--mode", default="", help="limit to one of embed/lora/full")
    ap.add_argument("--reserve", type=float, default=1.5,
                    help="GB to keep free; a config using more than (total - reserve) "
                         "is reported as not recommended")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    seq = args.seq or (SEQ_QWEN3 if args.vocab > 100000 else SEQ_SARVAM)
    budget = total - args.reserve

    print(f"device    {torch.cuda.get_device_name(0)}  {total:.1f} GB")
    print(f"vocab     {args.vocab:,}   seq {seq}   reserve {args.reserve} GB "
          f"-> budget {budget:.1f} GB\n")

    modes = [args.mode] if args.mode else ["embed", "lora", "full"]
    grid = [(m, b) for m in modes for b in (1, 2, 4, 8, 16)]

    print(f"{'mode':<8}{'batch':>6}{'trainable':>13}{'alloc':>9}"
          f"{'reserved':>10}{'headroom':>10}  verdict")
    results = []
    for mode, batch in grid:
        r = probe(args.vocab, mode, batch, seq, args.lora_rank)
        r.update(mode=mode, batch=batch, seq=seq, vocab=args.vocab,
                 tokens_per_step=batch * seq)
        results.append(r)
        if r["oom"]:
            print(f"{mode:<8}{batch:>6}{'-':>13}{'OOM':>9}{'-':>10}{'-':>10}  no")
            continue
        head = budget - r["reserved_gb"]
        verdict = "FITS" if head > 0 else "over budget"
        print(f"{mode:<8}{batch:>6}{r['trainable']/1e6:>11.1f}M"
              f"{r['peak_gb']:>9.2f}{r['reserved_gb']:>10.2f}{head:>+10.2f}  {verdict}")

    recommend(results, budget)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"device": torch.cuda.get_device_name(0), "total_gb": total,
             "budget_gb": budget, "results": results}, indent=2))
        print(f"\nwrote {args.out}")


def recommend(results, budget):
    """The largest batch per mode that stays inside the budget."""
    print("\n== Largest configuration that fits, per mode ==")
    for mode in dict.fromkeys(r["mode"] for r in results):
        fitting = [r for r in results if r["mode"] == mode and not r["oom"]
                   and r["reserved_gb"] < budget]
        if not fitting:
            print(f"  {mode:<8} nothing fits in budget")
            continue
        best = max(fitting, key=lambda r: r["batch"])
        print(f"  {mode:<8} batch {best['batch']:<3} "
              f"({best['tokens_per_step']:,} tokens/step)  "
              f"reserved {best['reserved_gb']:.2f} GB "
              f"(alloc {best['peak_gb']:.2f}), "
              f"{budget - best['reserved_gb']:.2f} GB spare")


if __name__ == "__main__":
    main()

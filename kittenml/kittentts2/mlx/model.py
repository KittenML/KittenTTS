"""The KittenTTS 2 language model as MLX modules.

Qwen3 decoder-only transformer, written against MLX directly rather than run
through transformers, for two reasons. The ternary weights stay packed: every
linear is an `mx.quantized_matmul` over the 4-bit codes `loader.py` produces, so
the 1.4B ternary parameters are read as 4 bits each instead of being expanded to
bf16 first. And the speaker head is part of the module rather than a forward hook
— KittenTTS 2 overwrites the first position's embedding with a projected speaker
embedding, which here is just an argument.

Only what this checkpoint uses is implemented: full attention on every layer, no
sliding window, no rope scaling, tied embeddings. `from_config` rejects anything
else rather than quietly ignoring it.
"""

import mlx.core as mx
import mlx.nn as nn

BITS = 4

# Every linear in the checkpoint, and the only ones this architecture has.
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class TernaryLinear(nn.Module):
    """A ternary linear layer, multiplied in its packed 4-bit form.

    `loader.py` explains why the 4 bits are exact rather than a quantisation of
    the published weights.
    """

    def __init__(self, codes, scales, biases, group_size):
        super().__init__()
        self.weight, self.scales, self.biases = codes, scales, biases
        self.group_size = group_size

    def __call__(self, x):
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                   transpose=True, group_size=self.group_size, bits=BITS)


class Attention(nn.Module):
    """Grouped-query attention with Qwen3's per-head q/k normalisation."""

    def __init__(self, args, build):
        super().__init__()
        self.n_heads = args["num_attention_heads"]
        self.n_kv_heads = args["num_key_value_heads"]
        self.head_dim = args["head_dim"]
        self.scale = self.head_dim ** -0.5
        self.q_proj = build("self_attn.q_proj")
        self.k_proj = build("self_attn.k_proj")
        self.v_proj = build("self_attn.v_proj")
        self.o_proj = build("self_attn.o_proj")
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args["rms_norm_eps"])
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args["rope_theta"])

    def __call__(self, x, mask, cache):
        batch, length, _ = x.shape
        queries = self.q_norm(self.q_proj(x).reshape(batch, length, self.n_heads, -1))
        keys = self.k_norm(self.k_proj(x).reshape(batch, length, self.n_kv_heads, -1))
        values = self.v_proj(x).reshape(batch, length, self.n_kv_heads, -1)
        queries, keys, values = (t.transpose(0, 2, 1, 3) for t in (queries, keys, values))

        offset = cache.offset
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        keys, values = cache.update_and_fetch(keys, values)

        out = mx.fast.scaled_dot_product_attention(queries, keys, values,
                                                   scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, build):
        super().__init__()
        self.gate_proj = build("mlp.gate_proj")
        self.up_proj = build("mlp.up_proj")
        self.down_proj = build("mlp.down_proj")

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, args, build):
        super().__init__()
        self.self_attn = Attention(args, build)
        self.mlp = MLP(build)
        self.input_layernorm = nn.RMSNorm(args["hidden_size"], eps=args["rms_norm_eps"])
        self.post_attention_layernorm = nn.RMSNorm(args["hidden_size"],
                                                   eps=args["rms_norm_eps"])

    def __call__(self, x, mask, cache):
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class SpeakerHead(nn.Module):
    """The checkpoint's `spk_proj`: a linear projection and a layer norm.

    Mirrors the `nn.Sequential(nn.Linear(...), nn.LayerNorm(...))` the Torch
    backend builds, so the checkpoint's `spk_proj.0.*` and `spk_proj.1.*` map
    across unchanged.
    """

    def __init__(self, spk_dim, hidden):
        super().__init__()
        self.proj = nn.Linear(spk_dim, hidden, bias=True)
        self.norm = nn.LayerNorm(hidden)

    def __call__(self, embedding):
        return self.norm(self.proj(embedding))


class KittenLM(nn.Module):
    """Qwen3 with KittenTTS 2's speaker head, over packed ternary weights."""

    def __init__(self, args, quantised, group_size, spk_dim):
        super().__init__()
        self.args = args
        self.n_layers = args["num_hidden_layers"]
        self.n_kv_heads = args["num_key_value_heads"]
        self.head_dim = args["head_dim"]

        def builder(index):
            def build(name):
                key = f"model.layers.{index}.{name}.weight"
                if key not in quantised:
                    raise ValueError(f"{key} is not ternary in this checkpoint; the MLX "
                                     "backend expects every linear to be packed")
                return TernaryLinear(*quantised[key], group_size)
            return build

        self.layers = [Layer(args, builder(i)) for i in range(self.n_layers)]
        unused = set(quantised) - {f"model.layers.{i}.{name}.weight"
                                   for i in range(self.n_layers)
                                   for name in LINEARS}
        if unused:
            raise ValueError(f"the checkpoint packs weights this architecture has no "
                             f"place for: {sorted(unused)[:5]}")
        self.norm = nn.RMSNorm(args["hidden_size"], eps=args["rms_norm_eps"])
        self.embed_tokens = nn.Embedding(args["vocab_size"], args["hidden_size"])
        self.spk_proj = SpeakerHead(spk_dim, args["hidden_size"])

    @classmethod
    def from_config(cls, config, quantised, group_size, spk_dim):
        """Build from the architecture `lm/config.json` declares."""
        if config.get("model_type") != "qwen3":
            raise ValueError(f"the MLX backend supports qwen3 KittenTTS 2 checkpoints, "
                             f"not {config.get('model_type')!r}")
        for key, expected in (("rope_scaling", None), ("sliding_window", None),
                              ("use_sliding_window", False), ("attention_bias", False),
                              ("tie_word_embeddings", True)):
            if config.get(key, expected) != expected:
                raise ValueError(f"the MLX backend does not implement {key}="
                                 f"{config.get(key)!r}")
        args = {key: config[key] for key in (
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_theta",
            "vocab_size")}
        return cls(args, quantised, group_size, spk_dim)

    def load_dense(self, dense):
        """Install the tensors the checkpoint does not pack.

        The names are the checkpoint's own, so a renamed or truncated checkpoint
        fails here rather than running on freshly initialised norms.
        """
        tree = {"embed_tokens": {"weight": dense["model.embed_tokens.weight"]},
                "norm": {"weight": dense["model.norm.weight"]},
                "spk_proj": {"proj": {"weight": dense["spk_proj.0.weight"],
                                      "bias": dense["spk_proj.0.bias"]},
                             "norm": {"weight": dense["spk_proj.1.weight"],
                                      "bias": dense["spk_proj.1.bias"]}},
                "layers": [
                    {"input_layernorm": {"weight": dense[f"model.layers.{i}.input_layernorm.weight"]},
                     "post_attention_layernorm": {
                         "weight": dense[f"model.layers.{i}.post_attention_layernorm.weight"]},
                     "self_attn": {
                         "q_norm": {"weight": dense[f"model.layers.{i}.self_attn.q_norm.weight"]},
                         "k_norm": {"weight": dense[f"model.layers.{i}.self_attn.k_norm.weight"]}}}
                    for i in range(self.n_layers)]}
        self.update(tree)
        installed = {"model.embed_tokens.weight", "model.norm.weight"}
        installed |= {f"spk_proj.{i}.{kind}" for i in (0, 1) for kind in ("weight", "bias")}
        installed |= {f"model.layers.{i}.{name}.weight" for i in range(self.n_layers)
                      for name in ("input_layernorm", "post_attention_layernorm",
                                   "self_attn.q_norm", "self_attn.k_norm")}
        left_over = set(dense) - installed
        if left_over:
            raise ValueError(f"the checkpoint holds unpacked weights this architecture "
                             f"has no place for: {sorted(left_over)[:5]}")
        return self

    def embed(self, ids):
        """Token ids -> input embeddings, as a (1, len, hidden) batch."""
        return self.embed_tokens(mx.array(ids, mx.uint32)[None])

    def speaker_embedding(self, embedding):
        """A 512-dim speaker embedding -> the hidden state it occupies at prefill."""
        return self.spk_proj(embedding.reshape(1, -1).astype(self.norm.weight.dtype))

    def __call__(self, embeddings, cache):
        """Logits for `embeddings`, in float32 for the sampler.

        The lm_head is tied to the token embedding, so the output projection is
        the embedding matrix itself.
        """
        mask = "causal" if embeddings.shape[1] > 1 else None
        hidden = embeddings
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(hidden, mask, layer_cache)
        hidden = self.norm(hidden[:, -1:, :])
        return self.embed_tokens.as_linear(hidden)[:, 0, :].astype(mx.float32)

    def new_cache(self):
        return [KVCache(self.n_kv_heads, self.head_dim) for _ in range(self.n_layers)]


class KVCache:
    """Keys and values for one layer, grown in blocks.

    Reallocating per step would copy the whole cache every token; a block means
    that happens once per `step` tokens instead.
    """

    step = 256

    def __init__(self, n_kv_heads, head_dim):
        self.n_kv_heads, self.head_dim = n_kv_heads, head_dim
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        length = keys.shape[2]
        if self.keys is None or self.offset + length > self.keys.shape[2]:
            capacity = ((self.offset + length + self.step - 1) // self.step) * self.step
            shape = (keys.shape[0], self.n_kv_heads, capacity, self.head_dim)
            grown_keys = mx.zeros(shape, keys.dtype)
            grown_values = mx.zeros(shape, values.dtype)
            if self.keys is not None:
                grown_keys[..., :self.offset, :] = self.keys[..., :self.offset, :]
                grown_values[..., :self.offset, :] = self.values[..., :self.offset, :]
            self.keys, self.values = grown_keys, grown_values

        self.keys[..., self.offset:self.offset + length, :] = keys
        self.values[..., self.offset:self.offset + length, :] = values
        self.offset += length
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

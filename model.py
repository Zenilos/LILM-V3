from dataclasses import dataclass
import math
import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int = 4388
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 64
    ffn_hidden: int = 1024
    context_length: int = 128


def count_parameters(model: nn.Module) -> dict:
    by_type: dict[str, int] = {}
    seen_ids: set[int] = set()

    def _count(obj):
        if isinstance(obj, dict):
            n = 0
            for v in obj.values():
                n += _count(v)[1]
            return obj, n
        if isinstance(obj, list):
            n = 0
            for x in obj:
                n += _count(x)[1]
            return obj, n
        if hasattr(obj, "size"):
            if id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                by_type["weight"] = by_type.get("weight", 0) + obj.size
            return obj, obj.size
        return obj, 0

    params = model.parameters()
    _count(params)
    total = sum(by_type.values())
    return {"total": total, "by_type": by_type}


def rope_freqs(dim: int, seq_len: int, base: float = 10000.0) -> mx.array:
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2) / dim))
    t = mx.arange(seq_len)
    freqs = mx.outer(t, freqs)
    return mx.cos(freqs), mx.sin(freqs)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    B, n_heads, T, head_dim = x.shape
    cos = mx.reshape(cos[:T], (1, 1, T, head_dim // 2))
    sin = mx.reshape(sin[:T], (1, 1, T, head_dim // 2))
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    cos_full = mx.concatenate([cos, cos], axis=-1)
    sin_full = mx.concatenate([sin, sin], axis=-1)
    return x * cos_full + rotated * sin_full


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight


class GQA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.group_size = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_model)
        self.scale = config.head_dim ** -0.5

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, T, _ = x.shape
        residual = x
        x = self.norm(x)

        q = mx.reshape(self.q_proj(x), (B, T, self.n_heads, self.head_dim))
        k = mx.reshape(self.k_proj(x), (B, T, self.n_kv_heads, self.head_dim))
        v = mx.reshape(self.v_proj(x), (B, T, self.n_kv_heads, self.head_dim))

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = mx.repeat(k, self.group_size, axis=1)
        v = mx.repeat(v, self.group_size, axis=1)

        attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        causal_mask = mx.full((T, T), -1e9)
        causal_mask = mx.triu(causal_mask, k=1)
        attn = attn + causal_mask
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (B, T, self.n_heads * self.head_dim))
        return residual + self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.norm(x)
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return residual + self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = GQA(config)
        self.ffn = SwiGLU(config)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = self.attn(x, cos, sin)
        x = self.ffn(x)
        return x


class StudentModel(nn.Module):
    def __init__(self, config: ModelConfig, tie_embeddings: bool = True):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.final_norm = RMSNorm(config.d_model)
        self.cos, self.sin = rope_freqs(config.head_dim, config.context_length)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        x = self.final_norm(x)
        # tied LM head shares the embedding weight (11.1M params total)
        return x @ self.embedding.weight.T


def create_model(config: ModelConfig | None = None) -> StudentModel:
    config = config or ModelConfig()
    model = StudentModel(config)
    mx.eval(model.parameters())
    return model

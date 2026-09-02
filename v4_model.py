"""V4: joint intent classification + BIO slot tagging model (atomic commands).

A small bidirectional Transformer encoder with two output heads:

    token IDs
        │
        ▼ [shared bidirectional encoder, CLS-prefixed]
        ├──────────► intent head → softmax(K intents)
        └──────────► slot head → per-token softmax(BIO classes)

This is the classic joint intent-slot architecture (Goo et al. 2018),
decomposed from the failing end-to-end generative FSM model: intent is a
multi-class decision, slots are BIO span labels, no autoregressive cascade.

All layers are plain fp16/fp32 (no ternary STE) so we can measure the ceiling
first; quantization is a later step.
"""

from __future__ import annotations

import math
import mlx.core as mx
import mlx.nn as nn

# --- label spaces -----------------------------------------------------------
INTENTS = ["MOVE", "CLEAN", "PLAY", "SHOW", "HANDOVER", "STOP", "WAIT", "UNAVAILABLE"]
INTENT_TO_ID = {i: k for k, i in enumerate(INTENTS)}

# shared BIO tag space by slot name (one intent uses each slot family cleanly)
SLOT_LABELS = ["O"]
for slot in ["location", "person", "object", "recipient", "message", "file", "duration"]:
    SLOT_LABELS += [f"B-{slot}", f"I-{slot}"]
SLOT_TO_ID = {l: k for k, l in enumerate(SLOT_LABELS)}
N_SLOT_CLASSES = len(SLOT_LABELS)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        norm = mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight


def rope_freqs(dim: int, seq_len: int, base: float = 10000.0):
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2) / dim))
    t = mx.arange(seq_len)
    fr = mx.outer(t, freqs)
    return mx.cos(fr), mx.sin(fr)


def apply_rope(x, cos, sin):
    B, n_heads, T, head_dim = x.shape
    cos = mx.reshape(cos[:T], (1, 1, T, head_dim // 2))
    sin = mx.reshape(sin[:T], (1, 1, T, head_dim // 2))
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    cos_full = mx.concatenate([cos, cos], axis=-1)
    sin_full = mx.concatenate([sin, sin], axis=-1)
    return x * cos_full + rotated * sin_full


class Attention(nn.Module):
    """Bidirectional self-attention (no causal mask) — a slot tagger needs
    full context on both sides of every token."""
    def __init__(self, d: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(d, n_heads * head_dim)
        self.k = nn.Linear(d, n_heads * head_dim)
        self.v = nn.Linear(d, n_heads * head_dim)
        self.o = nn.Linear(n_heads * head_dim, d)

    def __call__(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return self.o(out)


class FFN(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.norm = RMSNorm(d)
        self.w1 = nn.Linear(d, hidden)
        self.w2 = nn.Linear(hidden, d)
        self.w3 = nn.Linear(d, hidden)

    def __call__(self, x):
        r = self.norm(x)
        return self.w2(nn.silu(self.w1(r)) * self.w3(r))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, head_dim: int, ffn: int):
        super().__init__()
        self.attn_norm = RMSNorm(d)
        self.attn = Attention(d, n_heads, head_dim)
        self.ffn = FFN(d, ffn)

    def __call__(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(x)
        return x


class V4Model(nn.Module):
    def __init__(self, vocab_size: int, d: int = 192, n_layers: int = 2,
                 n_heads: int = 4, head_dim: int = 32, ffn: int = 384,
                 max_len: int = 64, n_intents: int = len(INTENTS),
                 n_slot: int = N_SLOT_CLASSES):
        super().__init__()
        self.d = d
        self.embedding = nn.Embedding(vocab_size, d)
        self.blocks = [Block(d, n_heads, head_dim, ffn) for _ in range(n_layers)]
        self.final_norm = RMSNorm(d)
        self.intent_head = nn.Linear(d, n_intents)
        self.slot_head = nn.Linear(d, n_slot)
        self.cos, self.sin = rope_freqs(head_dim, max_len)

    def __call__(self, x, mask=None):
        """x: [B, T] token ids. Returns (intent_logits [B, K], slot_logits [B, T, S]).
        Token 0 of each row is a [CLS] marker used for intent pooling."""
        B, T = x.shape
        h = self.embedding(x)
        for blk in self.blocks:
            h = blk(h, self.cos, self.sin)
        h = self.final_norm(h)
        # intent: mean-pool over non-padding tokens (CLS + utterance)
        if mask is None:
            mask = mx.ones((B, T))
        mask = mask[..., None]
        pooled = (h * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1.0)
        intent_logits = self.intent_head(pooled)
        slot_logits = self.slot_head(h)
        return intent_logits, slot_logits


def count_params(m: V4Model) -> int:
    return sum(math.prod(v.shape) for _, v in m.parameters().items())

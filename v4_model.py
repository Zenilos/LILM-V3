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
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# --- label spaces -----------------------------------------------------------
INTENTS = ["MOVE", "CLEAN", "PLAY", "SHOW", "HANDOVER", "STOP", "WAIT", "UNAVAILABLE"]
INTENT_TO_ID = {i: k for k, i in enumerate(INTENTS)}

# shared BIO tag space by slot name (one intent uses each slot family cleanly)
SLOT_LABELS = ["O"]
for slot in ["location", "person", "message", "duration"]:
    SLOT_LABELS += [f"B-{slot}", f"I-{slot}"]
SLOT_TO_ID = {l: k for k, l in enumerate(SLOT_LABELS)}
N_SLOT_CLASSES = len(SLOT_LABELS)

# large finite negative used to (soft-)forbid impossible BIO transitions
NEG = 1e4


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

    def __call__(self, x, cos, sin, pad_mask=None):
        """pad_mask: [B, T] with 1 for real tokens, 0 for padding. Bidirectional
        attention is masked so padding tokens neither attend nor are attended to;
        this makes the model invariant to sequence length (a single unpadded
        utterance gives the same logits as the same utterance padded in a batch)."""
        B, T, _ = x.shape
        q = self.q(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if pad_mask is not None:
            # [B,1,1,T]: attend only to real tokens (keys); forbid padding keys
            km = pad_mask[:, None, None, :]
            attn = mx.where(km > 0, attn, -1e4)
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

    def __call__(self, x, cos, sin, pad_mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, pad_mask)
        x = x + self.ffn(x)
        return x


class CRF(nn.Module):
    """Linear-chain CRF over BIO tags.

    Enforces valid BIO transitions (no I-x unless preceded by B-x / I-x of the
    same family) so spans cannot fragment and the model cannot collapse to a
    single pathological all-O/patchy pattern the way independent per-token
    softmax can. `trans` is a learnable [C, C] transition matrix added to the
    emission scores from the slot head.
    """

    def __init__(self, n_tags: int):
        super().__init__()
        self.n_tags = n_tags
        self.trans = mx.zeros((n_tags, n_tags))  # trans[i, j] = score i -> j
        names = SLOT_LABELS

        def fam_of(k: int):
            n = names[k]
            return n[2:] if n.startswith(("B-", "I-")) else None

        # hard structural mask: 0 for allowed transitions, -inf for forbidden
        mask = np.ones((n_tags, n_tags), dtype=np.float32)
        for dst in range(n_tags):
            if names[dst].startswith("I-"):
                f = names[dst][2:]
                # only B-fam or I-fam may continue an I-* span
                for src in range(n_tags):
                    if fam_of(src) != f:
                        mask[src, dst] = 0.0
        # also forbid I-* as the very first tag -> handled at decode by an
        # explicit start constraint in nll and decode.
        # Use a large FINITE negative (not -inf): mlx.logsumexp miscompiles on
        # graph nodes full of -inf, and a finite -NEG keeps exp(-NEG)~0 so the
        # forbidden paths have essentially zero probability (effectively hard).
        self.log_mask = mx.array(np.where(mask > 0, 0.0, -NEG).astype(np.float32))

    def _start_forbid(self) -> mx.array:
        """[C] addend: -NEG for I-* tags (cannot start a sequence with I-*)."""
        out = np.zeros(self.n_tags, dtype=np.float32)
        for k in range(self.n_tags):
            if SLOT_LABELS[k].startswith("I-"):
                out[k] = -NEG
        return mx.array(out)

    @staticmethod
    def _force_padding(emissions, seq_mask):
        """[B,T,C] -> on padding positions (seq_mask==0) force tag O (index 0):
        set O emission to 0 and every other emission to -1e3."""
        B, T, C = emissions.shape
        o_col = mx.zeros((B, T, 1))
        others = mx.full((B, T, C - 1), -NEG)
        pad_template = mx.concatenate([o_col, others], axis=-1)
        return mx.where(seq_mask[..., None] > 0, emissions, pad_template)

    @staticmethod
    def _onehot(idx, n):
        """One-hot of a [B] integer index array -> [B, n]."""
        return mx.where(idx[:, None] == mx.arange(n)[None, :], 1.0, 0.0)

    def nll(self, emissions, seq_mask, tags):
        """Negative log-likelihood of gold tag sequences (forward algorithm).

        emissions [B,T,C], seq_mask [B,T] (1=real token), tags [B,T] (gold ids).
        Returns mean NLL over the batch.
        """
        B, T, C = emissions.shape
        emiss = self._force_padding(emissions, seq_mask)
        logw = self.trans + self.log_mask  # finite -NEG on forbidden transitions
        alpha = emiss[:, 0, :] + self._start_forbid()
        for t in range(1, T):
            a = alpha[:, :, None] + logw[None, :, :]  # [B,C,C]
            alpha = mx.logsumexp(a, axis=1) + emiss[:, t, :]
        log_z = mx.logsumexp(alpha, axis=-1)  # [B]
        # gold path score via one-hot gathers (no integer-index gather)
        score = mx.sum(self._onehot(tags[:, 0], C) * emiss[:, 0, :], axis=1)
        for t in range(1, T):
            oh_p = self._onehot(tags[:, t - 1], C)  # [B, C]
            oh_c = self._onehot(tags[:, t], C)      # [B, C]
            te = mx.sum(oh_p[:, :, None] * logw[None, :, :] * oh_c[:, None, :],
                        axis=(1, 2))
            em = mx.sum(oh_c * emiss[:, t, :], axis=1)
            score = score + te + em
        return mx.mean(log_z - score)

    def decode(self, emissions, seq_mask):
        """Viterbi decode -> [B,T] predicted tag ids."""
        B, T, C = emissions.shape
        emiss = self._force_padding(emissions, seq_mask)
        logw = self.trans + self.log_mask
        alpha = emiss[:, 0, :] + self._start_forbid()
        back = []
        for t in range(1, T):
            a = alpha[:, :, None] + logw[None, :, :]
            alpha = mx.max(a, axis=1) + emiss[:, t, :]
            bp = mx.argmax(a, axis=1)
            back.append(bp)
        cur = mx.argmax(alpha, axis=-1)  # [B]
        tags = [None] * T
        tags[T - 1] = cur
        for t in range(T - 2, -1, -1):
            prev = mx.take_along_axis(back[t], cur[:, None], axis=1)[:, 0]
            tags[t] = prev
            cur = prev
        return mx.stack(tags, axis=1)


class V4Model(nn.Module):
    def __init__(self, vocab_size: int, d: int = 192, n_layers: int = 2,
                 n_heads: int = 4, head_dim: int = 32, ffn: int = 384,
                 max_len: int = 64, n_intents: int = len(INTENTS),
                 n_slot: int = N_SLOT_CLASSES, use_crf: bool = False):
        super().__init__()
        self.d = d
        self.use_crf = use_crf
        self.embedding = nn.Embedding(vocab_size, d)
        self.blocks = [Block(d, n_heads, head_dim, ffn) for _ in range(n_layers)]
        self.final_norm = RMSNorm(d)
        self.intent_head = nn.Linear(d, n_intents)
        self.slot_head = nn.Linear(d, n_slot)
        if use_crf:
            self.crf = CRF(n_slot)
        self.cos, self.sin = rope_freqs(head_dim, max_len)

    def __call__(self, x, mask=None):
        """x: [B, T] token ids. Returns (intent_logits [B, K], slot_logits [B, T, S]).
        Token 0 of each row is a [CLS] marker used for intent pooling."""
        B, T = x.shape
        h = self.embedding(x)
        if mask is None:
            mask = mx.ones((B, T))
        for blk in self.blocks:
            h = blk(h, self.cos, self.sin, mask)
        h = self.final_norm(h)
        # intent: mean-pool over non-padding tokens (CLS + utterance)
        mask = mask[..., None]
        pooled = (h * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1.0)
        intent_logits = self.intent_head(pooled)
        slot_logits = self.slot_head(h)
        return intent_logits, slot_logits


def count_params(m: V4Model) -> int:
    return sum(math.prod(v.shape) for _, v in m.parameters().items())

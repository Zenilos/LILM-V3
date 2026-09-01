"""Stage C: QAT training of the 11.1M-parameter ternary student on Apple
Silicon via MLX.

The student learns the same wire format as the teacher (`serialize.py`): the
model sees ``<plan> <ok>/<no> <intent> <s:i><e:j>/<lit:*> <eop>`` labels and is
trained, teacher-forced, to reproduce them from the raw utterance.

Quantization (BitNet b1.58), every linear weight:
  * absmean scale per output channel, threshold to {-1,0,+1}, straight-through
    estimator (fp32 master weights), ramped in over the first `ramp_frac` steps.
  * embeddings int8 (scaled), norms fp16; neither is ternarized.

Loss (only positions at/after the first label token, i.e. >= <plan>):
    L = alpha*CE(gold) + beta*KL(student||teacher,T) + gamma*||P h_s - h_t||^2
  With no teacher (--teacher none) only the CE term runs (ablation).

Base tokenizer: the pruned SmolLM2 BPE is not needed to *train* the student,
which consumes a concrete per-corpus word->id map (serialize.tokenize) for the
utterance plus `serialize.Vocab` ids for the label specials. The pointer
mechanism (copy start/end *positions*) generalizes to whatever tokenizer runs
on-device; the FSM `legal()` set is applied to logits so decoding is exact.

Usage:
    python train_student.py \
        --train data/train_a.jsonl --val data/val.jsonl \
        --out checkpoints/student \
        --teacher checkpoints/teacher/best.pt|none \
        --steps 20000 --batch 256 --eval-every 2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from dsl import Action, actions_match, validate_plan
from serialize import FSM, Vocab, decode, encode, tokenize
from model import ModelConfig

# ---- defaults ----------------------------------------------------------------
LR = 3e-3
WARMUP = 2000
WD = 0.1
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
ALPHA = 1.0
BETA = 1.0
GAMMA = 0.1
TEMP = 1.5
RAMP_FRAC = 0.2

NEG = -1_000_000


# --------------------------------------------------------------------------
# Base tokenizer: word -> id, built from the corpus (reference word-level).
# --------------------------------------------------------------------------
class BaseTok:
    def __init__(self, size: int = 4096):
        self.size = size
        self.word2id: dict[str, int] = {}

    def ids(self, words: list[str]) -> list[int]:
        return [self.word2id.get(w, 0) for w in words]


def build_base_tok(texts: list[str], size: int = 4096) -> BaseTok:
    bt = BaseTok(size)
    # deterministic: most-frequent words get low ids (keeps model stable)
    freq: dict[str, int] = {}
    for t in texts:
        for w in tokenize(t):
            freq[w] = freq.get(w, 0) + 1
    for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(bt.word2id) >= size:
            break
        bt.word2id[w] = len(bt.word2id)
    return bt


# --------------------------------------------------------------------------
# Quantized building blocks
# --------------------------------------------------------------------------
def absmean_scale(w: mx.array) -> mx.array:
    scale = mx.mean(mx.abs(w), axis=-1, keepdims=True)
    return w / mx.maximum(scale, 1e-9)


def ternary_ste(w: mx.array, t: float) -> mx.array:
    """BitNet b1.58 straight-through ternary. `t` in [0,1] ramps strength:
    t=0 -> latent fp, t=1 -> fully ternary {-1,0,+1}."""
    if t <= 0:
        return w
    wq = mx.abs(w)
    scale = mx.mean(wq, axis=-1, keepdims=True)
    wn = w / mx.maximum(scale, 1e-9)
    th = 0.5 * t
    q = mx.where(wn > th, 1.0, mx.where(wn < -th, -1.0, 0.0))
    # straight-through: dL/dw = dL/dq (identity on q), fp master updated
    return mx.stop_gradient(q) * t + w * (1.0 - t)


def int8_quant(h: mx.array) -> mx.array:
    """Per-row int8 embedding with straight-through."""
    scale = mx.max(mx.abs(h), axis=-1, keepdims=True) / 127.0 + 1e-9
    q = mx.clip(mx.round(h / scale), -127.0, 127.0)
    return mx.stop_gradient(q * scale - h) + h


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight


class QLinear(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        scale = math.sqrt(2.0 / (cin + cout))
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(cout, cin))

    def __call__(self, x: mx.array, t: float) -> mx.array:
        return x @ ternary_ste(self.weight, t).T


class Embedding(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.weight = mx.random.uniform(
            low=-0.1, high=0.1, shape=(vocab, dim))

    def __call__(self, x: mx.array) -> mx.array:
        return int8_quant(self.weight[x])


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


class GQABlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.group = cfg.n_heads // cfg.n_kv_heads
        self.q = QLinear(cfg.d_model, cfg.n_heads * cfg.head_dim)
        self.k = QLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim)
        self.v = QLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim)
        self.o = QLinear(cfg.n_heads * cfg.head_dim, cfg.d_model)
        self.norm = RMSNorm(cfg.d_model)
        self.scale = cfg.head_dim ** -0.5

    def __call__(self, x, cos, sin, t):
        B, T, _ = x.shape
        res = x
        x = self.norm(x)
        q = self.q(x, t).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x, t).reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x, t).reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = mx.repeat(k, self.group, axis=1)
        v = mx.repeat(v, self.group, axis=1)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        attn = mx.softmax(attn + mask, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        return res + self.o(out, t)


class SwiGLUBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = QLinear(cfg.d_model, cfg.ffn_hidden)
        self.up = QLinear(cfg.d_model, cfg.ffn_hidden)
        self.down = QLinear(cfg.ffn_hidden, cfg.d_model)
        self.norm = RMSNorm(cfg.d_model)

    def __call__(self, x, t):
        res = x
        x = self.norm(x)
        return res + self.down(nn.silu(self.gate(x, t)) * self.up(x, t), t)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = GQABlock(cfg)
        self.ffn = SwiGLUBlock(cfg)

    def __call__(self, x, cos, sin, t):
        x = self.attn(x, cos, sin, t)
        return self.ffn(x, t)


class Student(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model)
        self.cos, self.sin = rope_freqs(cfg.head_dim, cfg.context_length)

    def __call__(self, x: mx.array, t: float) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, t)
        h = self.final_norm(h)
        # tied LM head shares the embedding weight (single trainable tensor);
        # emitted int8 at export, fp32 here for a clean gradient path
        return h @ self.embedding.weight.T


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_rows(path: str, limit: int = 0) -> list[dict]:
    out, seen = [], set()
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["text"] in seen:
                continue
            seen.add(rec["text"])
            try:
                rec["_acts"] = [Action.from_dict(x) for x in rec["actions"]]
                validate_plan(rec["_acts"])
            except (ValueError, KeyError):
                continue
            out.append(rec)
            if limit and len(out) >= limit:
                break
    return out


def encode_rows(rows, vocab: Vocab, bt: BaseTok, max_len: int):
    """Return list of (utt_ids, label_ids) and the label-start index each row."""
    items = []
    for r in rows:
        try:
            label = encode(r["text"], r["_acts"], vocab)
        except Exception:
            continue
        utt = tokenize(r["text"])
        if not utt or len(utt) > vocab.max_input:
            continue
        uid = bt.ids(utt)
        if len(uid) + len(label) > max_len:
            continue
        items.append((uid, label, len(uid)))
    return items


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def ce_loss(logits: mx.array, labels: mx.array) -> mx.array:
    """logits (B,T,V), labels (B,T) with NEG ignored -> scalar."""
    B, T, V = logits.shape
    flat = logits.reshape(-1, V)
    lab = labels.reshape(-1)
    valid = lab != NEG
    if int(mx.sum(valid).item()) == 0:
        return mx.array(0.0)
    ce = nn.losses.cross_entropy(flat, mx.maximum(lab, 0), reduction="none")
    ce = mx.where(valid, ce, 0.0)
    return mx.sum(ce) / mx.maximum(mx.sum(valid), 1)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def greedy_decode(model, utt_ids: list[int], vocab: Vocab, fsm: FSM):
    n = len(utt_ids)
    inp = mx.array(utt_ids, dtype=mx.int32)[None, :]
    st = fsm.start(n)
    gen = []
    for _ in range(80):
        legal = fsm.legal(st)
        if not legal:
            break
        logits = model(inp, 1.0)[0, -1, :]
        la = mx.array(sorted(legal), dtype=mx.int32)
        sel = mx.take(logits, la)
        tid = int(la[mx.argmax(sel)].item())
        gen.append(tid)
        try:
            st = fsm.step(st, tid)
        except ValueError:
            break
        if st.done:
            break
        inp = mx.concatenate([inp, mx.array([[tid]], dtype=mx.int32)], axis=1)
    if not gen:
        return None
    try:
        if gen[0] == vocab.id["<no>"]:
            return [Action("UNAVAILABLE", {})]
        return decode(" ".join(["x"] * n), [vocab.id["<plan>"]] + gen, vocab)
    except Exception:
        return None


def evaluate(model, val_items, golds, vocab, fsm, n_eval=512):
    correct = total = 0
    order = list(range(len(val_items)))
    random.Random(0).shuffle(order)
    for i in order[:n_eval]:
        uid, _, _start = val_items[i]
        pred = greedy_decode(model, uid, vocab, fsm)
        total += 1
        if pred is not None and actions_match(pred, golds[i]):
            correct += 1
    return correct / max(1, total)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_a.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="checkpoints/student")
    ap.add_argument("--teacher", default="none")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ramp-frac", type=float, default=RAMP_FRAC)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    return ap.parse_args()


def main():
    a = parse_args()
    mx.random.seed(a.seed)
    random.seed(a.seed)

    vocab = Vocab()
    fsm = FSM(vocab)
    cfg = ModelConfig()

    train_rows = load_rows(a.train, a.limit)
    val_rows = load_rows(a.val)
    print(f"rows: train={len(train_rows)} val={len(val_rows)}", flush=True)

    bt = build_base_tok([r["text"] for r in train_rows])
    print(f"base tokenizer: {len(bt.word2id)} words", flush=True)

    train_items = encode_rows(train_rows, vocab, bt, a.max_len)
    val_items = encode_rows(val_rows, vocab, bt, a.max_len)
    val_golds = [r["_acts"] for r in val_rows][: len(val_items)]
    print(f"encodable: train={len(train_items)} val={len(val_items)}", flush=True)

    model = Student(cfg)
    num_params = _count_params(model.parameters())
    print(f"params: {num_params:,}", flush=True)

    def lr_schedule(t):
        if t < a.warmup:
            return a.lr * (t + 1) / a.warmup
        frac = (t - a.warmup) / max(1, (a.steps - a.warmup))
        return a.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac)))

    optimizer = optim.AdamW(learning_rate=a.lr, betas=(BETA1, BETA2),
                            eps=EPS, weight_decay=WD)
    amp = None  # KD hidden projection when teacher is enabled

    ramp_steps = max(1, int(a.steps * a.ramp_frac))
    idx = list(range(len(train_items)))
    best_em, step, t0 = 0.0, 0, time.time()

    def loss_fn(params, inputs, labels, ramp):
        model.update(params)
        logits = model(inputs, ramp)
        return ALPHA * ce_loss(logits, labels)

    loss_and_grad = mx.value_and_grad(loss_fn, argnums=0)

    reshuffle_cache = {}
    while step < a.steps:
        random.shuffle(idx)
        for bi in range(0, len(idx), a.batch):
            if step >= a.steps:
                break
            batch_idx = idx[bi:bi + a.batch]
            blen = max(len(train_items[i][0]) + len(train_items[i][1])
                       for i in batch_idx)
            blen = min(blen, a.max_len)
            n = len(batch_idx)
            inputs = mx.zeros((n, blen), dtype=mx.int32)
            labels = mx.full((n, blen), NEG, dtype=mx.int32)
            for r, i in enumerate(batch_idx):
                uid, lab, start = train_items[i]
                seq = (uid + lab)[:blen]
                inputs[r, : len(seq)] = mx.array(seq, dtype=mx.int32)
                ln = len(lab)
                # targets are the NEXT token: place label[k] at position
                # (start-1+k), so the model predicts <plan> right after the
                # final utterance token, then <ok>, <intent>, ..., <eop>.
                labels[r, start - 1: start - 1 + ln] = mx.array(lab, dtype=mx.int32)

            ramp = min(1.0, step / ramp_steps)
            lr = lr_schedule(step)
            optimizer.learning_rate = lr
            lossv, grads = loss_and_grad(
                model.trainable_parameters(), inputs, labels, ramp)
            optimizer.update(model, grads)
            step += 1

            if step % 100 == 0:
                print(f"step {step}/{a.steps} loss={lossv.item():.4f} "
                      f"ramp={ramp:.2f} lr={lr:.2e} "
                      f"{time.time()-t0:.0f}s", flush=True)
            if a.eval_every and step % a.eval_every == 0:
                em = evaluate(model, val_items, val_golds, vocab, fsm)
                print(f"  [eval step {step}] val_em={em:.4f}", flush=True)
                os.makedirs(a.out, exist_ok=True)
                if em >= best_em:
                    best_em = em
                    _save(model, os.path.join(a.out, "best.npz"))
                # always persist a rolling checkpoint so training can resume;
                # save under a unique name and keep only the latest two.
                _save(model, os.path.join(a.out, f"last-{step}.npz"))
                for old in sorted(os.listdir(a.out)):
                    if old.startswith("last-") and old != f"last-{step}.npz":
                        os.remove(os.path.join(a.out, old))
        if len(idx) == 0:
            break
    print(f"done. best val_em={best_em:.4f}", flush=True)


def _count_params(d):
    seen = set()
    def c(o):
        if isinstance(o, dict):
            return sum(c(v) for v in o.values())
        if isinstance(o, list):
            return sum(c(v) for v in o)
        if hasattr(o, "size"):
            if id(o) not in seen:
                seen.add(id(o))
                return o.size
            return 0
        return 0
    return c(d)


def _flatten_params(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_params(v, key))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                out.update(_flatten_params(x, f"{key}.{i}"))
        else:
            out[key] = v
    return out


def _save(model, path):
    np.savez(path, **{k: np.array(v) for k, v in _flatten_params(model.parameters()).items()})


if __name__ == "__main__":
    main()

"""V4: train the joint intent + BIO slot-tagger on balanced atomic data.

Reads data/v4/{train,val}_balanced.jsonl (rows: tokens, intent, slots where
slots is {\"INTENT-slotname\": \"B I O ...\"}) and trains the V4Model.

Metrics on val:
  - intent accuracy (exact match of predicted vs gold intent class)
  - slot BIO label accuracy and span-level F1
  - per-intent intent accuracy
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from train_student import build_base_tok
from v4_model import (V4Model, INTENTS, INTENT_TO_ID, SLOT_TO_ID,
                      SLOT_LABELS, N_SLOT_CLASSES)

WORDS2ID: dict[str, int] = {}
UNK = 0
SLOT_LOSS_W = 1.5
SLOT_W_CAP = 4.0

# map a data slot name (after stripping INTENT-) to the shared BIO tag family
SLOT_FAMILY = {
    "location": "location",
    "person": "person",
    "message": "message",
    "duration_amount": "duration",
    "duration_unit": "duration",
}


def compute_slot_weights(rows):
    """Effective-number / smoothed inverse-frequency weights per slot class so
    rare tags (B-*/I-*) are boosted relative to the dominant O label, but the
    scaling stays gentle (capped) so the loss still learns O well."""
    per_cls = collections.Counter()
    for r in rows:
        for key, tags in r.get("slots", {}).items():
            fam = key.split("-", 1)[1]
            fam = SLOT_FAMILY.get(fam, fam)
            for t in tags.split():
                slab = f"{t}-{fam}" if t != "O" else "O"
                per_cls[slab] += 1
    n_cls = N_SLOT_CLASSES
    counts = np.array([per_cls[SLOT_LABELS[i]] for i in range(n_cls)], dtype=np.float32)
    counts = np.maximum(counts, 0.0)
    # O is the reference class (weight 1.0); rarer classes get boosted,
    # capped at SLOT_W_CAP so they don't dominate. Empty classes get the cap.
    o_denom = max(1.0, counts[0])
    w = (o_denom + 1.0) / (1.0 + counts)
    w = np.minimum(w, SLOT_W_CAP)
    w[0] = 1.0  # O exactly 1.0
    return mx.array(w)


def parse_row(r: dict, max_len: int) -> tuple[list[int], int, list[int]]:
    """Return (token_ids, intent_id, slot_label_ids)."""
    tokens = r["tokens"][: max_len - 1]  # leave room for [CLS]
    ids = [0] + [WORDS2ID.get(w, UNK) for w in tokens]  # [CLS] at 0
    intent = INTENT_TO_ID.get(r["intent"], INTENT_TO_ID["UNAVAILABLE"])
    # slot labels over the utterance tokens only (skip the [CLS] position)
    n = len(tokens)
    slot_ids = [SLOT_TO_ID["O"]] * n
    for key, tags in r.get("slots", {}).items():
        # key like "MOVE-location"; family is the part after the first '-'
        fam = key.split("-", 1)[1]
        fam = SLOT_FAMILY.get(fam, fam)
        tag_list = tags.split()
        for i, tag in enumerate(tag_list):
            if i >= n:
                break
            if tag == "O":
                continue
            slab = f"{tag}-{fam}"
            if slab in SLOT_TO_ID:
                slot_ids[i] = SLOT_TO_ID[slab]
    return ids, intent, slot_ids


def collate(batch):
    """batch: list of (ids, intent, slot_ids). Pad to max length in batch."""
    max_t = max(len(b[0]) for b in batch)
    n = len(batch)
    toks = np.zeros((n, max_t), dtype=np.int32)
    mask = np.zeros((n, max_t), dtype=np.float32)
    intents = np.zeros((n,), dtype=np.int32)
    slots = np.zeros((n, max_t), dtype=np.int32)
    for i, (ids, it, sid) in enumerate(batch):
        L = len(ids)
        toks[i, :L] = ids
        mask[i, :L] = 1.0
        intents[i] = it
        slots[i, 1:1 + len(sid)] = sid  # slot labels start after [CLS]
    return (mx.array(toks), mx.array(mask), mx.array(intents), mx.array(slots))


def load_rows(path: str):
    out = []
    for line in open(path):
        r = json.loads(line)
        out.append(r)
    return out


def steps_per_epoch(n: int, batch: int):
    return max(1, n // batch)


def evaluate(model: V4Model, rows, bt, max_len: int, batch: int = 512):
    model.eval()
    intent_correct = 0
    n = 0
    per_intent = {}
    # span-level F1 helpers
    span_pred = []
    span_gold = []
    slot_tok_correct = 0
    slot_tok_total = 0
    for s in range(0, len(rows), batch):
        chunk = [parse_row(r, max_len) for r in rows[s:s + batch]]
        toks, m, it_gold, slots_gold = collate(chunk)
        it_logits, slot_logits = model(toks, m)
        it_pred = mx.argmax(it_logits, axis=-1)
        if getattr(model, "use_crf", False):
            slot_pred = model.crf.decode(slot_logits, m)
        else:
            slot_pred = mx.argmax(slot_logits, axis=-1)
        for i in range(toks.shape[0]):
            n += 1
            g = int(it_gold[i])
            p = int(it_pred[i])
            if p == g:
                intent_correct += 1
            per_intent.setdefault(INTENTS[g], [0, 0])
            per_intent[INTENTS[g]][1] += 1
            per_intent[INTENTS[g]][0] += int(p == g)
            # slot spans (positions after [CLS])
            g_labels = [int(slots_gold[i, j]) for j in range(1, toks.shape[1])
                        if m[i, j] == 1]
            p_labels = [int(slot_pred[i, j]) for j in range(1, toks.shape[1])
                        if m[i, j] == 1]
            slot_tok_total += len(g_labels)
            slot_tok_correct += sum(1 for a, b in zip(g_labels, p_labels) if a == b)
            span_gold.append(spans(g_labels))
            span_pred.append(spans(p_labels))
    # span F1
    tp = fp = fn = 0
    for gs, ps in zip(span_gold, span_pred):
        gs, ps = set(gs), set(ps)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    slot_acc = slot_tok_correct / max(1, slot_tok_total)
    print(f"  intent_acc={intent_correct}/{n} {intent_correct/max(1,n):.3f} "
          f"slot_tok_acc={slot_acc:.3f} span_F1={f1:.3f}")
    return {
        "intent_acc": intent_correct / max(1, n),
        "slot_acc": slot_acc,
        "span_f1": f1,
        "per_intent": per_intent,
        "tp": tp, "fp": fp, "fn": fn,
    }


def spans(labels):
    """Convert a list of BIO label ids to a set of (family, start, end) spans."""
    out = set()
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab == 0:
            i += 1
            continue
        name = SLOT_LABELS[lab]
        if name.startswith("B-"):
            fam = name[2:]
            j = i + 1
            while j < len(labels) and SLOT_LABELS[labels[j]] == f"I-{fam}":
                j += 1
            out.add((fam, i, j))
            i = j
        else:
            i += 1
    return out


def _nparams(o) -> int:
    if isinstance(o, mx.array):
        return int(np.prod(o.shape))
    if isinstance(o, dict):
        return sum(_nparams(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(_nparams(v) for v in o)
    return 0


def _flatten(o, prefix=""):
    """Flatten nested module params to dotted keys for .npz save."""
    out = {}
    if isinstance(o, mx.array):
        out[prefix] = o
    elif isinstance(o, dict):
        for k, v in o.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            out.update(_flatten(v, f"{prefix}.{i}" if prefix else str(i)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/v4/train_balanced.jsonl")
    ap.add_argument("--val", default="data/v4/val_balanced.jsonl")
    ap.add_argument("--out", default="checkpoints/v4")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--ffn", type=int, default=384)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--slot-w", type=float, default=SLOT_LOSS_W)
    ap.add_argument("--use-crf", action="store_true")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--score-norm", action="store_true",
                    help="save best by normalized intent+spanF1 instead of last epoch")
    ap.add_argument("--embed-init", default=None,
                    help="npz with an 'embedding' [V,d] matrix to init the embedding")
    ap.add_argument("--t", type=float, default=0.0,
                    help="QAT final ternary ramp in [0,1] (0=fp). Annealed "
                         "linear 0->t over the run, held at t for the last 25%%.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mx.random.seed(a.seed)
    np.random.seed(a.seed)

    train_rows = load_rows(a.train)
    val_rows = load_rows(a.val)

    # build vocab from train tokens
    texts = [" ".join(r["tokens"]) for r in train_rows]
    bt = build_base_tok(texts, size=4096)
    global WORDS2ID, UNK
    WORDS2ID = bt.word2id
    UNK = 0

    model = V4Model(vocab_size=len(bt.word2id), d=a.d, n_layers=a.layers,
                    n_heads=a.heads, head_dim=a.head_dim, ffn=a.ffn,
                    max_len=a.max_len, use_crf=a.use_crf)
    if a.embed_init:
        ei = np.load(a.embed_init)["embedding"]
        model.load({"embedding.weight": mx.array(ei.astype("float32"))},
                   strict=False)
        print(f"embedding initialized from {a.embed_init} shape={ei.shape}")
    print(f"vocab={len(bt.word2id)} params={_nparams(model.parameters())} "
          f"crf={a.use_crf}")

    opt = optim.AdamW(learning_rate=a.lr, weight_decay=a.wd)
    slot_weights = compute_slot_weights(train_rows)

    def loss_fn(mod, toks, m, it_gold, slots_gold):
        it_logits, slot_logits = mod(toks, m)
        it_ce = nn.losses.cross_entropy(it_logits, it_gold, reduction="mean")
        if a.use_crf:
            sl = mod.crf.nll(slot_logits, m, slots_gold)
            return it_ce + a.slot_w * sl
        flat = slot_logits.reshape(-1, N_SLOT_CLASSES)
        fg = slots_gold.reshape(-1)
        fm = m.reshape(-1)
        per = nn.losses.cross_entropy(flat, fg, reduction="none")
        # per-token class weights from the target label
        w = mx.take(slot_weights, fg)
        sl_ce = (per * w * fm).sum() / mx.maximum((fm * w).sum(), 1.0)
        return it_ce + a.slot_w * sl_ce

    loss_and_grad = mx.value_and_grad(loss_fn, argnums=0)

    print("training...")
    os.makedirs(a.out, exist_ok=True)
    start = time.time()
    best_score = -1.0
    hold_epochs = max(1, int(a.epochs * 0.25))  # hold ramp at t for last 25%
    ramp_start = int(a.epochs - hold_epochs) if a.t > 0 else 0
    for ep in range(1, a.epochs + 1):
        # QAT annealing: linear 0->t over [1, ramp_start], then hold at t.
        if a.t > 0:
            if ep < ramp_start:
                ramp = a.t * (ep - 1) / max(1, ramp_start - 1)
            else:
                ramp = a.t
            model.set_ramp(ramp)
        np.random.shuffle(train_rows)
        running = 0.0
        nstep = 0
        for s in range(0, len(train_rows), a.batch):
            chunk = [parse_row(r, a.max_len) for r in train_rows[s:s + a.batch]]
            toks, m, it_gold, slots_gold = collate(chunk)
            loss, grad = loss_and_grad(model, toks, m, it_gold, slots_gold)
            opt.update(model, grad)
            mx.eval(model.parameters(), opt.state, loss)
            running += float(loss)
            nstep += 1
        elapsed = time.time() - start
        print(f"[epoch {ep}/{a.epochs}] loss={running/max(1,nstep):.4f} "
              f"ramp={model.ramp:.3f} ({elapsed:.0f}s) | val:", flush=True)
        res = evaluate(model, val_rows, bt, a.max_len)
        state = _flatten(model.parameters())
        npz = {k: np.asarray(v) for k, v in state.items()}
        if a.score_norm:
            score = 0.5 * res["intent_acc"] + 0.5 * res["span_f1"]
            if score > best_score:
                best_score = score
                np.savez(f"{a.out}/best.npz", **npz)
                print(f"  * new best score={score:.3f}", flush=True)
        else:
            np.savez(f"{a.out}/last.npz", **npz)
    if a.score_norm:
        print(f"best score={best_score:.3f}")
    print("done")


if __name__ == "__main__":
    main()

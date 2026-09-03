"""Measure the accuracy budget of ternary quantization (PTQ) on a V4/V5 model.

No retraining: every linear `*.weight` is replaced by its channel-wise ternary
approximation (scale = mean|w| along the input/rightmost dim; threshold 0.5*ramp),
exactly as `export.py` packs for on-device. Biases, norms, embedding, CRF stay
fp32/fp16. Re-runs the exact `v4_eval.py` intent + slot-value evaluation so the
PTQ numbers are directly comparable to the fp checkpoint.

This tells us whether full QAT retraining is needed, or plain PTQ stays within
budget. Usage mirrors v4_eval.py:

    ~/p3.11/bin/python3 quant_eval.py --ckpt checkpoints/v5crf_mask/best.npz \
        --use-crf --ramp 1.0
"""
import argparse
import collections
import json

import mlx.core as mx
import numpy as np

from train_student import build_base_tok
from v4_model import (V4Model, INTENTS, INTENT_TO_ID, SLOT_LABELS, SLOT_TO_ID,
                      N_SLOT_CLASSES)
from v4_train import SLOT_FAMILY


def ternary_dim(w: mx.array, t: float = 1.0) -> mx.array:
    """Channel-wise ternary along the rightmost axis (matches export.ternary_channel)."""
    n = w.astype(mx.float32)
    scale = mx.abs(n).mean(axis=-1, keepdims=True)
    wn = n / mx.maximum(scale, 1e-9)
    th = 0.5 * t
    q = mx.where(wn > th, 1.0, mx.where(wn < -th, -1.0, 0.0))
    return q * mx.maximum(scale, 1e-9)


def load_ptq(ckpt_path: str, d: int, layers: int, use_crf: bool, ramp: float,
             w2i, fp: bool = False) -> V4Model:
    model = V4Model(vocab_size=len(w2i), d=d, n_layers=layers, use_crf=use_crf)
    data = np.load(ckpt_path, allow_pickle=False)

    root = {}
    for k in data.files:
        arr = data[k]
        # ternarize 2D linear weights only; embedding stays fp (int8 on device),
        # norms/CRF/biases stay fp
        do_quant = (not fp and k != "embedding.weight" and k.endswith(".weight")
                    and arr.ndim == 2 and not k.endswith("crf.trans"))
        if do_quant:
            root[k] = ternary_dim(mx.array(arr), ramp)
        else:
            root[k] = mx.array(arr)
    tree = {}
    for k, arr in root.items():
        parts = k.split(".")
        d_ = tree
        for i in range(len(parts) - 1):
            p, nxt = parts[i], parts[i + 1]
            if nxt.isdigit():
                d_ = d_.setdefault(p, [])
            elif p.isdigit():
                idx = int(p)
                while len(d_) <= idx:
                    d_.append({})
                d_ = d_[idx]
            else:
                d_ = d_.setdefault(p, {})
        d_[parts[-1]] = mx.array(arr)
    model.update(tree)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v5crf_mask/best.npz")
    ap.add_argument("--val", default="data/v4/val_balanced.jsonl")
    ap.add_argument("--train", default="data/v4/train_balanced.jsonl")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--use-crf", action="store_true")
    ap.add_argument("--ramp", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=0, help="limit to first N val rows (0 = all)")
    ap.add_argument("--fp", action="store_true",
                    help="no quantization (validate loader against v4_eval)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.val)]
    if a.n:
        rows = rows[:a.n]
    tr = [json.loads(l) for l in open(a.train)]
    bt = build_base_tok([" ".join(r["tokens"]) for r in tr], size=4096)
    w2i = bt.word2id

    model = load_ptq(a.ckpt, a.d, a.layers, a.use_crf, a.ramp, w2i, fp=a.fp)
    print("PTQ model loaded (ramp=%.2f, fp=%s)" % (a.ramp, a.fp))

    def decode(toks, labels):
        spans = collections.defaultdict(list)
        cur = None
        for j, tok in enumerate(toks):
            lab = SLOT_LABELS[int(labels[j + 1])]
            if lab == "O":
                cur = None
                continue
            bio, fam = lab.split("-", 1)
            if bio == "B":
                spans[fam].append([tok])
                cur = (fam, len(spans[fam]) - 1)
            elif bio == "I" and cur is not None and cur[0] == fam:
                spans[fam][cur[1]].append(tok)
            else:
                cur = None
        return {f: [" ".join(s) for s in v] for f, v in spans.items()}

    intent_corr = intent_tot = 0
    per_intent = collections.defaultdict(lambda: [0, 0])
    fam_p = collections.defaultdict(lambda: [0, 0, 0])

    for r in rows:
        intent_tot += 1
        it_id = INTENT_TO_ID.get(r["intent"], 7)
        ids = [0] + [w2i.get(w, 0) for w in r["tokens"][:39]]
        if len(ids) < 2:
            continue
        T = len(ids)
        mask = mx.ones((1, T))
        it_f, sl_f = model(mx.array([ids]), mask)
        it_pred = int(mx.argmax(it_f[0]))
        per_intent[r["intent"]][1] += 1
        if it_pred == it_id:
            intent_corr += 1
            per_intent[r["intent"]][0] += 1

        labels = model.crf.decode(sl_f, mask)[0] if a.use_crf else \
            mx.argmax(sl_f, axis=-1)[0]
        dec = decode(r["tokens"], labels)
        gold = collections.defaultdict(list)
        for key, tags in r.get("slots", {}).items():
            fam = SLOT_FAMILY.get(key.split("-", 1)[1])
            toks = r["tokens"][:40]
            tag_toks = tags.split()
            run = []
            for j, t in enumerate(tag_toks):
                if t != "O":
                    run.append(toks[j])
                else:
                    if run:
                        gold[fam].append(" ".join(run))
                        run = []
            if run:
                gold[fam].append(" ".join(run))
        for fam, gold_vals in gold.items():
            pred_vals = dec.get(fam, [])
            fam_p[fam][2] += len(gold_vals)
            for gv in gold_vals:
                fam_p[fam][1] += 1
                if gv in pred_vals:
                    fam_p[fam][0] += 1

    print(f"=== PTQ INTENT (n={intent_tot}) ramp={a.ramp} ===")
    print(f"exact intent_acc = {intent_corr}/{intent_tot} ({intent_corr/intent_tot:.1%})")
    for it in INTENTS:
        c, t = per_intent[it]
        print(f"  {it:<11} {c}/{t} ({c/t:.1%})" if t else f"  {it:<11} -")
    print("\n=== PTQ SLOT VALUE EXTRACTION ===")
    for fam in sorted(fam_p):
        c, t, g = fam_p[fam]
        acc = c / t if t else 0
        print(f"  {fam:<12} {c}/{t} ({acc:.1%})  [gold_vals={g}]")


if __name__ == "__main__":
    main()

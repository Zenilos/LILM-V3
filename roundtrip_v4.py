"""Round-trip verify an export_v4 blob: unpack model.tern via manifest.json,
rebuild a V4Model, and score intent + slot value exactly like quant_eval.py.

This is the reference decode the C firmware must match (firmware/v5_model.c
uses the identical 5-trits/byte + int8 + fp16 layout). If the blob's score
matches the direct QAT-checkpoint score, the exporter + packer contract is
correct end-to-end.

Usage:
    ~/p3.11/bin/python3 roundtrip_v4.py --dir build/export_v5 \
        --train data/v4/train_balanced.jsonl --val data/v4/val_balanced.jsonl
"""
import argparse, collections, json
import mlx.core as mx, numpy as np
from train_student import build_base_tok
from v4_model import V4Model, INTENTS, INTENT_TO_ID, SLOT_LABELS
from v4_train import SLOT_FAMILY

_TRITS_PER_BYTE = 5
_BASE3 = [1, 3, 9, 27, 81]


def unpack_trits(packed: np.ndarray, n: int) -> np.ndarray:
    pb = packed.astype(np.int64)
    codes = np.empty(n, dtype=np.int64)
    for i in range(5):
        sel = np.arange(i, n, 5)
        codes[sel] = (pb[sel // 5] // _BASE3[i]) % 3
    return (codes - 1).astype(np.int8)


def load_blob(manifest_path, blob_path):
    m = json.load(open(manifest_path))
    blob = np.fromfile(blob_path, dtype=np.uint8)
    tensors = m["tensors"]

    def raw(name):
        meta = tensors[name]
        return blob[meta["offset"]:meta["offset"] + meta["nbytes"]], meta

    def f16(name, n):
        rr, meta = raw(name)
        return np.frombuffer(rr[:n * 2], np.float16).astype(np.float32), meta

    def tern_to_w(name):
        """packed trits + .scale -> dequantized [out,in] fp array."""
        arr, meta = raw(name)
        n = int(np.prod(meta["shape"]))
        tri = unpack_trits(arr, n).reshape(meta["shape"])
        scale, _ = f16(name + ".scale", meta["shape"][0])
        return tri.astype(np.float32) * scale.reshape(-1, 1)

    def int8_to_w(name):
        qarr, meta = raw(name)
        n = int(np.prod(meta["shape"]))
        name_base = name[:-2]  # strip trailing '.q' -> base
        scale, _ = f16(name_base + ".scale", meta["shape"][0])
        i8 = np.frombuffer(qarr[:n], np.int8).astype(np.float32).reshape(meta["shape"])
        return i8 * scale.reshape(-1, 1)

    tree = {}
    for name in tensors:
        kind = tensors[name]["kind"]
        shape = tuple(tensors[name]["shape"])
        n = int(np.prod(shape)) if shape else 1
        if kind == "tern":
            if name.endswith(".scale"):
                continue
            tree[name] = tern_to_w(name)
        elif kind == "int8":
            if name.endswith(".scale"):
                continue
            # reassemble embedding: .q + .scale -> embedding.weight
            base = name[:-2]
            tree[base] = int8_to_w(name)
        elif kind == "fp16":
            if name.endswith(".scale"):
                continue  # consumed inside tern_to_w / int8_to_w
            arr, _ = f16(name, n)
            tree[name] = arr.reshape(shape)
    return m, tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--val", default="data/v4/val_balanced.jsonl")
    ap.add_argument("--train", default="data/v4/train_balanced.jsonl")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()

    manifest, tree_arrays = load_blob(os.path.join(a.dir, "manifest.json"),
                                      os.path.join(a.dir, "model.tern"))
    cfg = manifest["config"]
    rows = [json.loads(l) for l in open(a.val)]
    if a.n:
        rows = rows[:a.n]
    tr = [json.loads(l) for l in open(a.train)]
    bt = build_base_tok([" ".join(r["tokens"]) for r in tr], size=4096)
    w2i = bt.word2id

    model = V4Model(vocab_size=cfg["vocab"], d=cfg["d"], n_layers=cfg["n_layers"],
                    use_crf=True)
    # dotted keys from manifest -> nested {blocks:[{attn:{...}}], ...} (v4_eval fmt)
    tree = {}
    for k, v in tree_arrays.items():
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
        d_[parts[-1]] = mx.array(v)
    model.update(tree)

    def decode(toks, labels):
        spans = collections.defaultdict(list); cur = None
        for j, tok in enumerate(toks):
            lab = SLOT_LABELS[int(labels[j + 1])]
            if lab == "O": cur = None; continue
            bio, fam = lab.split("-", 1)
            if bio == "B": spans[fam].append([tok]); cur = (fam, len(spans[fam]) - 1)
            elif bio == "I" and cur and cur[0] == fam: spans[fam][cur[1]].append(tok)
            else: cur = None
        return {f: [" ".join(s) for s in v] for f, v in spans.items()}

    ic = ittot = 0
    fam_p = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        ittot += 1
        it_id = INTENT_TO_ID.get(r["intent"], 7)
        ids = [0] + [w2i.get(w, 0) for w in r["tokens"][:39]]
        if len(ids) < 2: continue
        T = len(ids); mask = mx.ones((1, T))
        it_f, sl_f = model(mx.array([ids]), mask)
        if int(mx.argmax(it_f[0])) == it_id: ic += 1
        labels = model.crf.decode(sl_f, mask)[0]
        dec = decode(r["tokens"], labels)
        gold = collections.defaultdict(list)
        for key, tags in r.get("slots", {}).items():
            fam = SLOT_FAMILY.get(key.split("-", 1)[1])
            toks = r["tokens"][:40]; tt = tags.split(); run = []
            for j, t in enumerate(tt):
                if t != "O": run.append(toks[j])
                elif run: gold[fam].append(" ".join(run)); run = []
            if run: gold[fam].append(" ".join(run))
        for fam, gv in gold.items():
            pv = dec.get(fam, []); fam_p[fam][2] += len(gv)
            for g in gv:
                fam_p[fam][1] += 1
                if g in pv: fam_p[fam][0] += 1

    print(f"=== ROUNDTRIP (from blob) intent n={ittot} ===")
    print(f"intent_acc = {ic}/{ittot} ({ic/ittot:.1%})")
    print("slot value extraction:")
    for fam in sorted(fam_p):
        c, t, g = fam_p[fam]
        print(f"  {fam:<12} {c}/{t} ({c/t:.1%})" if t else f"  {fam:<12} -")


import os
if __name__ == "__main__":
    main()

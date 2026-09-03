"""Build an nn.Embedding init matrix from qwen semantic word categories.

Reads data/v4/vocab_cats.json (word -> LOCATION/PERSON/OBJECT/FILE/MESSAGE/
DURATION/ACTION/FUNCTION/OTHER) produced by classify_vocab.py, plus the vocab
from train_balanced.jsonl. Writes an init matrix [V, d] where words in the same
semantic category start near a shared category centroid (plus per-word noise),
so the model can generalize "the thing after 'bring' is an object" across all
object words instead of learning each one independently from a random init.
"""
import json
import numpy as np
from train_student import build_base_tok

CATS = ["LOCATION", "PERSON", "OBJECT", "FILE", "MESSAGE",
        "DURATION", "ACTION", "FUNCTION", "OTHER"]
CAT_IDX = {c: i for i, c in enumerate(CATS)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/v4/train_balanced.jsonl")
    ap.add_argument("--cats", default="data/v4/vocab_cats.json")
    ap.add_argument("--out", default="data/v4/embed_init.npz")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    tr = [json.loads(l) for l in open(a.train)]
    bt = build_base_tok([" ".join(r["tokens"]) for r in tr], size=4096)
    w2i = bt.word2id
    cats = json.load(open(a.cats))

    V = len(w2i)
    d = a.d
    init = rng.normal(0, 1, (V, d)).astype(np.float32)

    # one random centroid direction per category; scale to unit-ish norm
    centroids = {}
    for c in CATS:
        v = rng.normal(0, 1, d).astype(np.float32)
        v /= np.linalg.norm(v)
        centroids[c] = v * (d ** 0.5)  # ~ unit L2 per-dim similar to Embedding init

    unk_as = "OTHER"
    for w, wid in w2i.items():
        c = cats.get(w, unk_as)
        if c not in CAT_IDX:
            c = unk_as
        base = centroids[c]
        noise = rng.normal(0, a.noise, d).astype(np.float32)
        init[wid] = base + noise

    # scale overall to match default nn.Embedding magnitude (~N(0,1))
    init = init * 1.0
    np.savez(a.out, embedding=init)
    print(f"wrote {a.out} shape={init.shape} "
          f"cats={ {c: sum(1 for w in w2i if cats.get(w,'OTHER')==c) for c in CATS} }")


if __name__ == "__main__":
    main()

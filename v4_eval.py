"""Eval V4 checkpoint: per-intent intent acc, per-family slot VALUE accuracy
(exact span-equality) so we know what the robot can actually act on."""
import json, collections
import mlx.core as mx, numpy as np
from train_student import build_base_tok
from v4_model import (V4Model, INTENTS, INTENT_TO_ID, SLOT_LABELS,
                      SLOT_TO_ID, N_SLOT_CLASSES)
from v4_train import SLOT_FAMILY

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v4w/last.npz")
    ap.add_argument("--val", default="data/v4/val_balanced.jsonl")
    ap.add_argument("--train", default="data/v4/train_balanced.jsonl")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--use-crf", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.val)]
    tr = [json.loads(l) for l in open(a.train)]
    # vocab MUST be built from the same texts used at train time (id ordering)
    bt = build_base_tok([" ".join(r["tokens"]) for r in tr], size=4096)
    w2i = bt.word2id

    model = V4Model(vocab_size=len(w2i), d=a.d, n_layers=a.layers,
                    use_crf=a.use_crf)
    data = np.load(a.ckpt, allow_pickle=False)
    tree = {}
    for k in data.files:
        parts = k.split(".")
        d = tree
        for i in range(len(parts) - 1):
            p, nxt = parts[i], parts[i + 1]
            if nxt.isdigit():
                d = d.setdefault(p, [])
            elif p.isdigit():
                idx = int(p)
                while len(d) <= idx:
                    d.append({})
                d = d[idx]
            else:
                d = d.setdefault(p, {})
        d[parts[-1]] = mx.array(data[k])
    model.update(tree)
    print("model loaded")

    def decode(toks, labels):
        """BIO decode into {family: [value_str...]} honoring adjacency constraints.
        toks = raw tokens (no CLS); labels = [CLS] + per-token label ids."""
        spans = collections.defaultdict(list)
        cur = None
        for j, tok in enumerate(toks):
            lab = SLOT_LABELS[int(labels[j + 1])]  # skip CLS
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

    intent_corr = 0
    intent_tot = 0
    per_intent = collections.defaultdict(lambda: [0, 0])
    # value-level accuracy per family: fraction of gold value-tuples fully extracted
    val_stats = collections.defaultdict(lambda: [0, 0, 0])  # correct, total_pairs, total_utts_with
    # span token precision/recall (class-free) for F1
    tp = fp = fn = 0
    fam_p = collections.defaultdict(lambda: [0, 0, 0])  # exact-match, tries, golds

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
        pred_intent = INTENTS[it_pred]
        if it_pred == it_id:
            intent_corr += 1
            per_intent[r["intent"]][0] += 1
        per_intent[r["intent"]][1] += 1

        if a.use_crf:
            labels = model.crf.decode(sl_f, mask)[0]
        else:
            labels = mx.argmax(sl_f, axis=-1)[0]
        dec = decode(r["tokens"], labels)
        # gold value tuples per family (drop O-token-only entries)
        gold = collections.defaultdict(list)
        for key, tags in r.get("slots", {}).items():
            fam = SLOT_FAMILY.get(key.split("-", 1)[1])
            toks = r["tokens"][:40]
            tag_toks = tags.split()
            # extract contiguous non-O runs
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
            # exact set match of the (usually single) value
            got = set(pred_vals) >= set(gold_vals)
            # count each
            for gv in gold_vals:
                fam_p[fam][1] += 1
                if gv in pred_vals:
                    fam_p[fam][0] += 1

    print(f"\n=== INTENT (n={intent_tot}) ===")
    print(f"exact intent_acc = {intent_corr}/{intent_tot} ({intent_corr/intent_tot:.1%})")
    for it in INTENTS:
        c, t = per_intent[it]
        print(f"  {it:<11} {c}/{t} ({c/t:.1%})" if t else f"  {it:<11} -")

    print("\n=== SLOT VALUE EXTRACTION (exact value-string match) ===")
    for fam in sorted(fam_p):
        c, t, g = fam_p[fam]
        acc = c / t if t else 0
        print(f"  {fam:<12} {c}/{t} ({acc:.1%})  [gold_vals={g}]")

if __name__ == "__main__":
    main()

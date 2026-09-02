"""Diagnose V4 slot failures: class skew + per-class pred stats + examples."""
import json, collections
import mlx.core as mx, numpy as np
from train_student import build_base_tok
from v4_model import V4Model, SLOT_LABELS, INTENT_TO_ID, INTENTS

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v4/last.npz")
    ap.add_argument("--val", default="data/v4/val_balanced.jsonl")
    a = ap.parse_args()
    SLOTS_IDX = {l: i for i, l in enumerate(SLOT_LABELS)}

    rows = [json.loads(l) for l in open(a.val)]
    bt = build_base_tok([" ".join(r["tokens"]) for r in rows], size=4096)
    w2i = bt.word2id

    data = np.load(a.ckpt, allow_pickle=False)
    model = V4Model(vocab_size=len(w2i), d=192, n_layers=2)
    tree = {}
    for k in data.files:
        parts = k.split(".")
        d = tree
        for i in range(len(parts) - 1):
            p = parts[i]
            nxt = parts[i + 1]
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

    # global non-O fraction + per-class confusion
    non_o_total = 0
    tot_tokens = 0
    per_fam = collections.Counter()
    for r in rows:
        tags = []
        for key, s in r.get("slots", {}).items():
            fam = key.split("-", 1)[1]
            L = s.split()
            for i, t in enumerate(L):
                if t != "O":
                    per_fam[fam] += 1
                    non_o_total += 1
                    tags.append(fam)
        tot_tokens += len(r["tokens"])
    print(f"total val tokens: {tot_tokens}, non-O (slot) tokens: {non_o_total} "
          f"({non_o_total/max(1,tot_tokens):.1%})")
    print("per slot-family non-O counts:", dict(per_fam))

    # show some example predictions
    shown = 0
    print("\nExample predictions (tokens=model slots | gold):")
    for r in rows:
        if shown >= 12:
            break
        ids = [0] + [w2i.get(w, 0) for w in r["tokens"][:39]]
        T = len(ids)
        mask = mx.ones((1, T))
        it_f, sl_f = model(mx.array([ids]), mask)
        it_pred = INTENTS[int(mx.argmax(it_f[0]))]
        if it_pred != r["intent"]:
            continue
        ps = []
        for j in range(1, len(ids)):
            lab = SLOT_LABELS[int(mx.argmax(sl_f[0, j]))]
            ps.append(lab)
        gs = {}
        for key, s in r.get("slots", {}).items():
            gs[key] = s.split()
        gold_list = []
        for j in range(len(ps)):
            gold_list.append("O")
        for key, s in gs.items():
            fam = key.split("-", 1)[1]
            for idx, t in enumerate(s):
                if idx < len(gold_list) and t != "O":
                    gold_list[idx] = f"{t}-{fam}"
        toks = r["tokens"][:len(ps)]
        print(f"  [{r['intent']}] {r['text'] if 'text' in r else ' '.join(toks)}")
        print(f"      pred: {list(zip(toks, ps))}")
        print(f"      gold: {list(zip(toks, gold_list))}")
        shown += 1

if __name__ == "__main__":
    main()

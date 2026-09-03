"""Verify approach-A chain segmenter end-to-end with the V5 atomic model.

Pipeline:
  1. Regenerate pair/triple chains (corpus.generate) with gold actions.
  2. Ground-truth atomic clauses via v4_decompose.decompose_chain (label-aware).
  3. Segment the SAME chain gold-free via chain_seg.segment.
  4. Run each segmented sentence through the V5 model -> (intent, slots).
  5. Check each segmented sentence's (intent, required slots) against the
     corresponding gold atomic clause it was derived from.

This measures the full path: does breaking a chain into sentences, then
classifying each, recover the correct atomic actions?
"""
from __future__ import annotations
import argparse, json
import mlx.core as mx, numpy as np
import corpus as corpus_mod
import v4_decompose as dec
import chain_seg
from v4_model import V4Model, INTENTS, SLOT_LABELS
from v4_train import load_rows, SLOT_FAMILY
from train_student import build_base_tok


def load_model(ckpt):
    tr = load_rows('data/v4/train_balanced.jsonl')
    bt = build_base_tok([' '.join(r['tokens']) for r in tr], size=4096)
    w2i = bt.word2id
    model = V4Model(vocab_size=len(w2i), use_crf=True)
    data = np.load(ckpt, allow_pickle=False)
    tree = {}
    for k in data.files:
        parts = k.split('.'); d = tree
        for i in range(len(parts)-1):
            p, nxt = parts[i], parts[i+1]
            if nxt.isdigit(): d = d.setdefault(p, [])
            elif p.isdigit():
                idx = int(p)
                while len(d) <= idx: d.append({})
                d = d[idx]
            else: d = d.setdefault(p, {})
        d[parts[-1]] = mx.array(data[k])
    model.update(tree)
    return model, w2i, bt


def decode(toks, labels, w2i):
    """BIO decode into {family: [value,...]} honoring adjacency."""
    import collections
    spans = collections.defaultdict(list)
    cur = None
    for j, tok in enumerate(toks):
        lab = SLOT_LABELS[int(labels[j+1])]  # skip CLS
        if lab == "O":
            cur = None; continue
        bio, fam = lab.split("-", 1)
        if bio == "B":
            spans[fam].append([tok]); cur = (fam, len(spans[fam])-1)
        elif bio == "I" and cur is not None and cur[0] == fam:
            spans[fam][cur[1]].append(tok)
        else:
            cur = None
    return {f: [" ".join(s) for s in v] for f, v in spans.items()}


def classify(model, w2i, sentence):
    toks = sentence.split()
    ids = [0] + [w2i.get(w, 0) for w in toks[:39]]
    T = len(ids)
    mask = mx.ones((1, T))
    it_f, sl_f = model(mx.array([ids]), mask)
    it_pred = INTENTS[int(mx.argmax(it_f[0]))]
    if getattr(model, 'use_crf', False):
        labels = model.crf.decode(sl_f, mask)[0]
    else:
        labels = mx.argmax(sl_f, axis=-1)[0]
    dec = decode(toks, labels, w2i)
    return it_pred, dec


def gold_slots(clause, action):
    """Gold (intent, required slot values) for a decomposed atomic clause."""
    a = action
    req = {}
    for slot in ("location", "person", "message", "duration_amount", "duration_unit"):
        v = a.get("slots", {}).get(slot)
        if v:
            req[SLOT_FAMILY.get(slot, slot)] = v
    return a["intent"], req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v5crf_mask/best.npz")
    ap.add_argument("--n", type=int, default=400, help="chains to evaluate")
    ap.add_argument("--mix", default=(0.0, 0.7, 0.3, 0.0))
    a = ap.parse_args()
    model, w2i, bt = load_model(a.ckpt)

    rng = __import__('random').Random(7)
    rows, _ = corpus_mod.generate(a.n, 'train', seed=7, mix=a.mix, dedup=False)

    tot_clauses = 0
    intent_ok = 0
    full_ok = 0
    n_seg_ok = 0      # correct number of clauses
    n_space_join = 0  # chains joined by the degenerate bare-space connective
    model_intent_ok = 0
    model_full_ok = 0
    model_clauses = 0
    fails = []
    for r in rows:
        text = r["text"]
        acts = r["actions"]
        gold = dec.decompose_chain(text, acts)
        sents = chain_seg.segment(text)
        if len(sents) == len(gold):
            n_seg_ok += 1
        space_join = False
        # was this chain joined by the bare-space connective (no delimiter word)?
        for i in range(len(gold)):
            pass
        for (clause, act), sent in zip(gold, sents):
            tot_clauses += 1
            gintent, greq = gold_slots(clause, act)
            pintent, pdec = classify(model, w2i, sent)
            i_ok = (pintent == gintent)
            s_ok = all(gv in pdec.get(fam, []) for fam, gv in greq.items()) if greq else True
            intent_ok += i_ok
            if i_ok and s_ok:
                full_ok += 1
            # model-only score on correctly-segmented clauses
            if len(sents) == len(gold):
                model_clauses += 1
                model_intent_ok += i_ok
                if i_ok and s_ok:
                    model_full_ok += 1
            if not (i_ok and s_ok):
                if len(fails) < 15:
                    fails.append((sent, gintent, greq, pintent, pdec))

    print(f"chains={len(rows)}  clauses={tot_clauses}")
    print(f"seg-count-correct:  {n_seg_ok}/{len(rows)} ({n_seg_ok/len(rows):.1%})")
    print(f"intent/clause:      {intent_ok}/{tot_clauses} ({intent_ok/tot_clauses:.1%})")
    print(f"intent+slots/clause:{full_ok}/{tot_clauses} ({full_ok/tot_clauses:.1%})")
    if model_clauses:
        print(f"--- model-only (on correctly-segmented clauses) ---")
        print(f"intent:             {model_intent_ok}/{model_clauses} "
              f"({model_intent_ok/model_clauses:.1%})")
        print(f"intent+slots:       {model_full_ok}/{model_clauses} "
              f"({model_full_ok/model_clauses:.1%})")
    print("\n--- sample failures (sent | gold | pred) ---")
    for s, gi, gr, pi, pd in fails:
        grs = ",".join(f"{k}={v}" for k, v in gr.items()) or "-"
        pds = ",".join(f"{k}:{v}" for k, v in pd.items()) or "-"
        print(f"  {s!r}\n    gold({gi},{grs}) pred({pi},{pds})")


if __name__ == "__main__":
    main()

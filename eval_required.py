"""Intent + required-slot eval (ignore optional slots).

For each (pred, gold) action pair:
  - intent must match
  - all required slots must be present and value-match
  - optional slots are IGNORED

Reports per-intent accuracy and overall chain EM.
"""

from __future__ import annotations
import argparse, json

import mlx.core as mx
import numpy as np

from dsl import Action, SLOTS, _is_set, _values_match
from eval import load_model, greedy_decode
from train_student import build_base_tok
from serialize import Vocab, FSM, tokenize


def _req_match(pred: Action, gold: Action) -> bool:
    if pred.intent != gold.intent:
        return False
    ps = {k: v for k, v in pred.slots.items() if _is_set(v)}
    for slot in SLOTS[gold.intent]["required"]:
        if slot not in ps:
            return False
        if not _values_match(slot, ps[slot], gold.slots[slot]):
            return False
    return True


def _chain_match(pred, gold):
    if pred is None or len(pred) != len(gold):
        return False
    return all(_req_match(p, g) for p, g in zip(pred, gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--t", type=float, default=0.0)
    a = ap.parse_args()

    vocab = Vocab()
    fsm = FSM(vocab)
    model = load_model(a.model)

    seen = set()
    rows = []
    with open(a.val) as f:
        for line in f:
            r = json.loads(line)
            if r["text"] in seen:
                continue
            seen.add(r["text"])
            acts = [Action.from_dict(d) for d in r["actions"]]
            rows.append((r["text"], acts))

    bt = build_base_tok([t for t, _ in rows])

    i_tot = {}
    i_req = {}
    chain_ok = 0

    for text, gold in rows:
        words = text.split()
        uid = bt.ids(words)
        pred = greedy_decode(model, words, uid, vocab, fsm, a.t)

        chain_ok += int(_chain_match(pred, gold))

        if pred is not None and len(pred) == len(gold):
            for p, g in zip(pred, gold):
                i_tot.setdefault(g.intent, [0, 0])
                i_req.setdefault(g.intent, [0, 0])
                i_tot[g.intent][1] += 1
                i_req[g.intent][1] += 1
                if p.intent == g.intent:
                    i_tot[g.intent][0] += 1
                    ps = {k: v for k, v in p.slots.items() if _is_set(v)}
                    ok = all(
                        slot in ps and _values_match(slot, ps[slot], g.slots[slot])
                        for slot in SLOTS[g.intent]["required"]
                    )
                    if ok:
                        i_req[g.intent][0] += 1
        else:
            for g in gold:
                i_tot.setdefault(g.intent, [0, 0])
                i_req.setdefault(g.intent, [0, 0])
                i_tot[g.intent][1] += 1
                i_req[g.intent][1] += 1

    n = len(rows)
    print(f"\nModel: {a.model}  t={a.t}  n={n}")
    print(f"{'Intent':<14} {'Intent acc':>11} {'Req+intent':>12} {'Count':>6}")
    print("-" * 48)
    ti = tr = tn = 0
    for intent in SLOTS:
        if intent not in i_tot:
            continue
        ic, it = i_tot[intent]
        rc, rt = i_req[intent]
        ti += ic; tr += rc; tn += it
        ia = f"{ic}/{it} ({ic/it:.1%})" if it else "-"
        ra = f"{rc}/{rt} ({rc/rt:.1%})" if rt else "-"
        print(f"{intent:<14} {ia:>12} {ra:>12} {it:>5}")
    print("-" * 48)
    print(f"{'TOTAL':<14} {ti}/{tn} ({ti/tn:.1%})  {tr}/{tn} ({tr/tn:.1%})  {tn:>5}")
    print(f"\nChain EM (intent+required only): {chain_ok}/{n} = {chain_ok/n:.1%}")


if __name__ == "__main__":
    main()

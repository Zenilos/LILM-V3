"""Standalone evaluation harness for the trained student model.

Loads a saved checkpoint (numpy .npz from train_student.py) and scores the
validation set against every metric in ARCHITECTURE.md section 5:

  * dsl.actions_match  -- exact sequence match, no partial credit (primary)
  * intent-sequence accuracy independent of slots
  * per-slot precision/recall (person & recipient reported separately, since
    optional slots penalize in both directions)
  * EM stratified by chain length (atomic / pair / triple)
  * EM on held-out entities (running split=val sees only unseen names)
  * `<ok>`/`<no>` confusion, with false-accept as the safety metric

Usage:
    python eval.py --model checkpoints/student/best.npz \
        --val data/val.jsonl

The checkpoint must share the tokenizer/data conventions of the training run
(the same corpus word->id map is rebuilt from the val + a train tails file if
given, otherwise from the val texts -- exact for held-out eval).
"""

from __future__ import annotations

import argparse
import json

import mlx.core as mx
import numpy as np

from dsl import Action, actions_match, normalize_slot
from serialize import FSM, Vocab, decode, encode, tokenize
from train_student import Student, build_base_tok, ModelConfig


def _unflatten(data):
    """Rebuild the nested module-parameter dict from dotted .npz keys."""
    tree = {}
    for k in data.files:
        parts = k.split(".")
        d = tree
        for i, p in enumerate(parts):
            last = (i == len(parts) - 1)
            if last:
                d[p] = mx.array(data[k])
                break
            nxt = parts[i + 1]
            if nxt.isdigit():
                d = d.setdefault(p, [])
            elif p.isdigit():
                p = int(p)
                while len(d) <= p:
                    d.append({})
                d = d[p]
            else:
                d = d.setdefault(p, {})
    return _norm(tree)


def _norm(o):
    if isinstance(o, dict):
        d = {}
        for key, val in o.items():
            d[int(key) if key.isdigit() else key] = _norm(val) if isinstance(val, dict) else val
        return d
    if isinstance(o, list):
        return [_norm(x) if isinstance(x, dict) else x for x in o]
    return o


def load_model(path):
    data = np.load(path, allow_pickle=False)
    model = Student(ModelConfig())
    model.update(_unflatten(data))
    return model


def greedy_decode(model, utt_words, utt_ids, vocab, fsm, t=1.0):
    n = len(utt_ids)
    inp = mx.array(utt_ids, dtype=mx.int32)[None, :]
    st = fsm.start(n)
    gen = []
    for _ in range(80):
        legal = fsm.legal(st)
        if not legal:
            break
        logits = model(inp, t)[0, -1, :]
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
        return decode(" ".join(utt_words), [vocab.id["<plan>"]] + gen, vocab)
    except Exception:
        return None


def run_eval(model, rows, vocab, fsm, bt, t=1.0):
    golds = []
    for r in rows:
        try:
            encode(r["text"], r["_acts"], vocab)
            if not tokenize(r["text"]):
                continue
        except Exception:
            continue
        golds.append(r)

    total = em = intent_ok = 0
    by_kind_em = {}
    by_kind_n = {}
    slot_tp = slot_fp = slot_fn = 0
    pers_tp = pers_fp = pers_fn = 0
    ok_tp = ok_fp = ok_tn = ok_fn = 0

    for r in golds:
        acts = r["_acts"]
        rejected = acts[0].intent == "UNAVAILABLE"
        kind = "reject" if rejected else {1: "atomic", 2: "pair", 3: "triple"}[len(acts)]
        words = tokenize(r["text"])
        uid = bt.ids(words)
        try:
            pred = greedy_decode(model, words, uid, vocab, fsm, t)
        except Exception:
            pred = None
        total += 1
        # ok/no confusion (false-accept is the safety metric)
        if rejected:
            if pred is not None and pred[0].intent == "UNAVAILABLE":
                ok_tn += 1
            else:
                ok_fp += 1
        else:
            if pred is not None and pred[0].intent != "UNAVAILABLE":
                ok_tp += 1
            else:
                ok_fn += 1
        if actions_match(pred, acts):
            em += 1
            intent_ok += 1
            by_kind_em[kind] = by_kind_em.get(kind, 0) + 1
            # slot-level stats
            for p, g in zip(pred, acts):
                for slot, gv in g.slots.items():
                    pv = p.slots.get(slot)
                    hit = pv is not None and (
                        normalize_slot(slot, pv) == normalize_slot(slot, gv)
                        or normalize_slot(slot, pv).replace(" ", "")
                           == normalize_slot(slot, gv).replace(" ", ""))
                    if hit:
                        slot_tp += 1
                    else:
                        slot_fn += 1
                    if slot in ("person", "recipient"):
                        pers_tp += 1 if hit else 0
                        pers_fn += 0 if hit else 1
                for slot in p.slots:
                    if slot not in g.slots:
                        slot_fp += 1
                        if slot in ("person", "recipient"):
                            pers_fp += 1
        elif pred is not None and len(pred) == len(acts) and \
                all(p.intent == g.intent for p, g in zip(pred, acts)):
            intent_ok += 1
        by_kind_n[kind] = by_kind_n.get(kind, 0) + 1

    intent_denom = max(1, ok_tp + ok_fn)
    return {
        "em": em / max(1, total),
        "n": total,
        "intent_acc": intent_ok / intent_denom,
        "slot_precision": slot_tp / max(1, slot_tp + slot_fp),
        "slot_recall": slot_tp / max(1, slot_tp + slot_fn),
        "pers_precision": pers_tp / max(1, pers_tp + pers_fp),
        "pers_recall": pers_tp / max(1, pers_tp + pers_fn),
        "by_kind_em": by_kind_em,
        "by_kind_n": by_kind_n,
        "ok_tp": ok_tp, "ok_fp": ok_fp, "ok_tn": ok_tn, "ok_fn": ok_fn,
        "false_accept": ok_fp / max(1, ok_fp + ok_tn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--train-words", default="",
                    help="a train jsonl to build the base tokenizer from "
                         "(reproduces training tokenizer exactly)")
    ap.add_argument("--t", type=float, default=1.0,
                    help="decode quantization: 0.0=fp (use for --fp runs), "
                         "1.0=ternary (default, QAT runs)")
    a = ap.parse_args()

    vocab = Vocab()
    fsm = FSM(vocab)
    model = load_model(a.model)

    val = []
    seen = set()
    for line in open(a.val):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec["text"] in seen:
            continue  # dedupe by text, matching train_student.load_rows
        seen.add(rec["text"])
        try:
            rec["_acts"] = [Action.from_dict(x) for x in rec["actions"]]
        except Exception:
            pass
        val.append(rec)
    # rebuild tokenizer from all texts seen (val) or train for parity
    texts = [r["text"] for r in val]
    if a.train_words:
        with open(a.train_words) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
    bt = build_base_tok(texts)

    m = run_eval(model, val, vocab, fsm, bt, t=a.t)
    print("=== Student Evaluation ===")
    print(f"exact-match (actions_match): {m['em']:.4f}  (n={m['n']})")
    print(f"intent-seq accuracy:         {m['intent_acc']:.4f}")
    print(f"slot precision/recall:       {m['slot_precision']:.4f} / {m['slot_recall']:.4f}")
    print(f"person+recipient P/R:        {m['pers_precision']:.4f} / {m['pers_recall']:.4f}")
    print("EM by chain length:")
    for k in ("atomic", "pair", "triple", "reject"):
        n = m["by_kind_n"].get(k, 0)
        e = m["by_kind_em"].get(k, 0)
        print(f"  {k:8s}: {e}/{n} = {e/max(1,n):.4f}" if n else f"  {k:8s}: -")
    print(f"ok/no confusion: TP={m['ok_tp']} FP={m['ok_fp']} TN={m['ok_tn']} FN={m['ok_fn']}")
    print(f"false-accept rate (safety):  {m['false_accept']:.4f}")


if __name__ == "__main__":
    main()

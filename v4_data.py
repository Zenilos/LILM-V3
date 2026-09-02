"""V4 balanced atomic dataset builder.

Generates a balanced set of atomic (single-action) utterances and emits
labels in a joint intent + BIO-slot-tagging format:

    JSONL, one row per utterance:
      {
        "tokens": ["go", "to", "my", "desk"],
        "intent": "MOVE",
        "slots": {"MOVE-location": "B I", "MOVE-person": "O O O O"}  # BIO per slot
      }

The corpus generator (corpus.py) is deterministic and already splits
train/val entities and templates, so its gold slots appear verbatim in the
surface text. We re-generate a large atomic pool, then downsample per intent
to a target so every intent class is balanced (STOP/WAIT/PLAY no longer
starved).

UNAVAILABLE (reject) rows are included so the intent classifier can learn
the "can't do this" class.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import corpus as corpus_mod
from dsl import Action, SLOTS, _is_set

# label to emitted intents (an atomic example maps to exactly one)
INTENTS = ["MOVE", "CLEAN", "PLAY", "SHOW", "HANDOVER", "STOP", "WAIT", "UNAVAILABLE"]


def tokens_of(text: str) -> list[str]:
    return text.split()


def span_bio(tokens: list[str], value: str) -> list[str]:
    """Return BIO tags over `tokens` for a gold slot value. Value is a
    multi-word phrase that appears verbatim in tokens; greedy match from the
    right so a repeated word is bound to the correct occurrence when possible."""
    vt = value.split()
    out = ["O"] * len(tokens)
    if not vt:
        return out
    # greedy left-to-right, prefer the earliest start
    i = 0
    while i <= len(tokens) - len(vt):
        if tokens[i:i + len(vt)] == vt:
            out[i] = "B"
            for j in range(i + 1, i + len(vt)):
                out[j] = "I"
            break
        i += 1
    return out


def build(rows: list[dict], per_intent: int, rng: random.Random) -> list[dict]:
    """Downsample atomic rows to a balanced per-intent target."""
    by_intent: dict[str, list] = {}
    for r in rows:
        acts = r["actions"]
        if len(acts) != 1:
            continue
        a = acts[0]
        if a["intent"] == "UNAVAILABLE":
            intent = "UNAVAILABLE"
        else:
            intent = a["intent"]
        by_intent.setdefault(intent, []).append(r)

    out = []
    for intent in INTENTS:
        pool = by_intent.get(intent, [])
        if len(pool) > per_intent:
            pool = rng.sample(pool, per_intent)
        for r in pool:
            tokens = tokens_of(r["text"])
            if intent == "UNAVAILABLE":
                slot_tags = {}
            else:
                a = r["actions"][0]
                slot_tags = {}
                for slot in SLOTS[intent]["required"] + SLOTS[intent]["optional"]:
                    if slot in a["slots"]:
                        bio = span_bio(tokens, a["slots"][slot])
                        slot_tags[f"{intent}-{slot}"] = " ".join(bio)
            out.append({
                "tokens": tokens,
                "intent": intent,
                "slots": slot_tags,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-atomic", type=int, default=300000,
                    help="how many atomic rows to generate before balancing")
    ap.add_argument("--per-intent", type=int, default=5000,
                    help="target rows per real intent after balancing")
    ap.add_argument("--reject-jsonl", default="data/train_a.jsonl",
                    help="source of UNAVAILABLE rows (existing corpus)")
    ap.add_argument("--reject-n", type=int, default=5000,
                    help="target UNAVAILABLE rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out", default="-")
    a = ap.parse_args()

    rng = random.Random(a.seed)

    # generate a large all-atomic pool for the real intents
    rows, dropped = corpus_mod.generate(
        a.n_atomic, a.split, a.seed, mix=(1.0, 0.0, 0.0, 0.0), dedup=False,
    )
    balanced = build(rows, a.per_intent, rng)

    # append UNAVAILABLE from the existing corpus's reject rows
    reject_rows = []
    seen = set()
    for line in open(a.reject_jsonl):
        r = json.loads(line)
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        if len(r["actions"]) == 1 and r["actions"][0]["intent"] == "UNAVAILABLE":
            reject_rows.append(r)
    if len(reject_rows) > a.reject_n:
        reject_rows = rng.sample(reject_rows, a.reject_n)
    for r in reject_rows:
        tokens = tokens_of(r["text"])
        balanced.append({"tokens": tokens, "intent": "UNAVAILABLE", "slots": {}})

    # finally cap every real intent at per_intent (in case any exceeded)
    final = []
    per_int = collections.Counter()
    for r in balanced:
        if r["intent"] != "UNAVAILABLE" and per_int[r["intent"]] >= a.per_intent:
            continue
        per_int[r["intent"]] += 1
        final.append(r)

    print(f"generated={len(rows)} dropped={dropped} out={len(final)}",
          file=sys.stderr)
    f = open(a.out, "w") if a.out != "-" else sys.stdout
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

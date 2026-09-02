"""V4: decompose chain (pair/triple) rows into atomic (single-action) rows.

Each chain utterance is built by concatenating atomic clause surfaces with a
connective from corpus.CONNECTIVES. This decomposer splits a chain back into
atomic utterances, aligning each clause to its gold action, so we can:

  1. Reuse the ~640k chain clauses as extra atomic training data now, and
  2. Save the decomposition for a future chain->atomics task.

Labeling notes:
  - intent is taken from the gold action (always right).
  - a slot's BIO tags are emitted ONLY when its value span appears verbatim in
    the clause (e.g. MOVE.location in "go to the sunroom"). Anaphor clauses
    ("and clean it") or sentinel values ("here"/"everywhere") carry the intent
    but no span, which is the correct, honest atomic target.
  - WAIT stores value as (duration_amount, duration_unit); we tag whichever
    spans appear (the number and the unit word) under a single WAIT-duration
    slot.
"""

from __future__ import annotations

import argparse
import json
import random

import corpus as corpus_mod
from dsl import Action, SLOTS, _is_set

REAL_INTENTS = ["MOVE", "CLEAN", "PLAY", "SHOW", "HANDOVER", "STOP", "WAIT"]
CONNECTIVES = corpus_mod.CONNECTIVES


def split_once(text: str, word: str) -> tuple[str, str]:
    """Split text at the first connective `word`, preferring the earliest one
    to keep clauses in action order. Returns (before, after) or (None, None)."""
    idx = text.find(word)
    if idx < 0:
        return None, None
    return text[:idx], text[idx + len(word):]


def decompose_chain(text: str, actions: list) -> list[tuple[str, dict]]:
    """Return [(clause_text, action_dict), ...] for a chain utterance.

    Uses a left-to-right greedy split guided by corpus.CONNECTIVES (longest
    first so ' and then ' wins over ' and '). Falls back to splitting at commas
    or spaces when needed. The last action may absorb the remainder.
    """
    remaining = text
    clauses = []
    n = len(actions)
    # longest-first connective order
    conns = sorted(set(CONNECTIVES) - {" "}, key=len, reverse=True)
    for i in range(n):
        if i == n - 1:
            clauses.append(remaining.strip())
            break
        # try connectives in order, take the earliest position among them
        best = None
        for conn in conns:
            idx = remaining.find(conn)
            if idx >= 0 and (best is None or idx < best[0]):
                best = (idx, conn)
        if best is None:
            # fallback: split in half and hope
            clauses.append(remaining.strip()[:len(remaining) // 2])
            remaining = remaining[len(remaining) // 2:]
        else:
            idx, conn = best
            clauses.append(remaining[:idx].strip())
            remaining = remaining[idx + len(conn):]
    # clip to n clauses
    result = []
    for c, a in zip(clauses, actions):
        if c:
            result.append((c, a))
    return result


def tokens_of(text: str) -> list[str]:
    return text.split()


def span_bio(tokens: list[str], value: str) -> list[str]:
    vt = value.split()
    out = ["O"] * len(tokens)
    if not vt:
        return out
    i = 0
    while i <= len(tokens) - len(vt):
        if tokens[i:i + len(vt)] == vt:
            out[i] = "B"
            for j in range(i + 1, i + len(vt)):
                out[j] = "I"
            break
        i += 1
    return out


def atomic_row(clause: str, action: dict) -> dict:
    tokens = tokens_of(clause)
    a = Action.from_dict(action)
    intent = a.intent
    slot_tags = {}
    if intent != "UNAVAILABLE":
        for slot in SLOTS[intent]["required"] + SLOTS[intent]["optional"]:
            v = a.slots.get(slot)
            if v and _is_set(v):
                bio = span_bio(tokens, v)
                if "B" in bio:
                    slot_tags[f"{intent}-{slot}"] = " ".join(bio)
    return {"tokens": tokens, "intent": intent, "slots": slot_tags,
            "source_clause": clause}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", default="data/train_a.jsonl")
    ap.add_argument("--out", default="-")
    ap.add_argument("--max", type=int, default=0, help="0=all")
    a = ap.parse_args()

    out = []
    n_atom = 0
    seen = set()
    for line in open(a.in_jsonl):
        r = json.loads(line)
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        acts = r["actions"]
        kind = r.get("kind")
        if len(acts) == 1 and acts[0]["intent"] != "UNAVAILABLE":
            n_atom += 1
            out.append(atomic_row(r["text"], acts[0]))
        elif len(acts) > 1:
            for clause, act in decompose_chain(r["text"], acts):
                n_atom += 1
                out.append(atomic_row(clause, act))
        if a.max and len(out) >= a.max:
            break
    f = open(a.out, "w") if a.out != "-" else sys.stdout
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"atomic rows out: {len(out)}", file=sys.stderr)


if __name__ == "__main__":
    import sys
    main()

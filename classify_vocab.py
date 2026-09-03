"""Build data/v4/vocab_cats.json (word -> semantic category) offline.

Reads the entity pools and templates directly from corpus.py (locations,
persons, messages, durations) plus the training utterance vocab, and assigns
each vocab word exactly one semantic category. No external model required:

  CATEGORIES (schema of build_embed_init.CATS):
    LOCATION   -- content words of location entity phrases
    PERSON     -- content words of person entity phrases
    MESSAGE    -- content words of message entity phrases
    DURATION   -- duration amounts + units ("30", "minutes", "half", ...)
    ACTION     -- verb/noun tokens of the templated action verbs
    FUNCTION   -- connective / structural / generic tokens
    OTHER      -- anything uncategorised

Multi-word entities map per-word ("the kitchen" -> "the":FUNCTION,
"kitchen":LOCATION). Atomic words that appear only as a slot value are
tagged with that slot family regardless of role in a template.
"""
import argparse
import json
from collections import Counter

from serialize import tokenize
from train_student import build_base_tok
from corpus import POOLS, FUZZY, EXACT, T, COMPOSITE

CATS = ["LOCATION", "PERSON", "OBJECT", "FILE", "MESSAGE",
        "DURATION", "ACTION", "FUNCTION", "OTHER"]


def _content_words(phrases):
    stop = {"the", "a", "an", "my", "our", "your", "his", "her", "their",
            "in", "on", "at", "over", "by", "of", "to", "for", "is", "are",
            "who", "that", "which", "and", "me", "us", "it", "this"}
    out = []
    for p in phrases:
        for w in tokenize(p):
            if w not in stop:
                out.append(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/v4/train_balanced.jsonl")
    ap.add_argument("--out", default="data/v4/vocab_cats.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.train)]
    bt = build_base_tok([" ".join(r["tokens"]) for r in rows], size=4096)
    w2i = bt.word2id

    cats: dict[str, Counter] = {w: Counter() for w in w2i}

    # 1) entity pools -> family-specific content words
    for ph in _content_words(POOLS["location"]["train"] + POOLS["location"]["val"]):
        if ph in w2i:
            cats[ph]["LOCATION"] += 1
    for ph in _content_words(POOLS["person"]["train"] + POOLS["person"]["val"]):
        if ph in w2i:
            cats[ph]["PERSON"] += 1
    for ph in _content_words(POOLS["message"]["train"] + POOLS["message"]["val"]):
        if ph in w2i:
            cats[ph]["MESSAGE"] += 1

    # 2) durations: fuzzy phrases + exact amounts/units
    for phrase, (amt, unit) in FUZZY.items():
        for w in tokenize(phrase) + [amt, unit]:
            if w in w2i:
                cats[w]["DURATION"] += 1
    for amt, unit in EXACT:
        for w in [amt, unit, unit[:-1]]:
            if w in w2i:
                cats[w]["DURATION"] += 1

    # 3) action verbs from the single-action templates
    for t in T:
        toks = tokenize(t.pattern.replace("{", " ").replace("}", " "))
        actions = {"MOVE": "go cruise travel roll walk head move navigate trundle "
                           "drive find see get make way over",
                   "CLEAN": "clean vacuum tidy hoover sweep mop wipe dust",
                   "PLAY": "play put cue start turn music song",
                   "SHOW": "show display flash print screen tell let know",
                   "HANDOVER": "bring hand take pass deliver give fetch grab get "
                               "this it that me",
                   "STOP": "stop halt abort freeze stand cut pull enough",
                   "WAIT": "wait pause hold stay hang"}
        if t.intent in actions:
            for w in toks:
                if any(w == a or w in a.split() for a in actions[t.intent].split()):
                    if w in w2i:
                        cats[w]["ACTION"] += 1

    out = {}
    for w, w2i_id in w2i.items():
        if cats[w]:
            # slot family wins over ACTION; then ACTION over FUNCTION
            for k in ("LOCATION", "PERSON", "MESSAGE", "DURATION"):
                if cats[w].get(k, 0):
                    out[w] = k
                    break
            else:
                if cats[w].get("ACTION", 0):
                    out[w] = "ACTION"
                else:
                    out[w] = "FUNCTION"
        else:
            out[w] = "OTHER"

    json.dump(out, open(a.out, "w"), indent=0, sort_keys=True)
    print(f"wrote {a.out} ({len(out)} words)")
    print("category histogram:",
          {c: sum(1 for w in w2i if out[w] == c) for c in CATS})


if __name__ == "__main__":
    main()

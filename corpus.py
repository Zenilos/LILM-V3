"""Stage A: programmatic corpus generation. Gold is exact by construction.

Entity pools are split train/val so held-out room and person names measure
whether copy-pointer decoding really does generalize to unseen values.
Templates are split too, so a val example shares neither surface form nor
entity with anything in train.

The intended train mix (atomic / pair / triple / reject) is enforced with
deterministic per-kind targets. Because single-action diversity is bounded by
the template × entity space, the generator relaxes strict global dedup once a
kind's distinct space is exhausted -- otherwise the rare kinds (PLAY, STOP,
atomic generally) would be starved out of the file entirely, exactly as they
were with a hard `seen` cap. On-disk repeats are acceptable: they just
reinforce the common cases, and the same action across paraphrases is the
whole point of Stage B anyway.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field

from dsl import Action, validate_plan
from serialize import Unencodable, Vocab, encode, tokenize

# --- entity pools -------------------------------------------------------------
POOLS = {
    "location": {
        "train": [
            "the kitchen", "the living room", "my room", "the garage", "the hallway",
            "the bedroom", "the office", "upstairs", "the dining room", "the basement",
            "my desk", "the front door", "the balcony", "the laundry room",
            # extra train-only locations to widen atomic diversity
            "the attic", "the pantry", "the sunroom", "the workshop", "the nursery",
            "the study", "the playroom", "the den", "the closet", "the foyer",
            "the terrace", "the guest room", "the cellar", "the rooftop", "the porch",
            "the library", "the gym", "the driveway", "the courtyard", "the corridor",
            "the mudroom", "the utility room", "the greenhouse", "the sauna",
        ],
        "val": [
            "the conservatory", "the mezzanine", "the storeroom", "the veranda",
            "the annexe", "the loft", "the outhouse", "the sickbay",
        ],
    },
    "person": {
        "train": [
            "john", "my wife", "the kids", "sarah", "my daughter", "my son",
            "grandma", "mr. olsen", "anna", "my brother", "my father", "my mother",
            "grandpa", "mc", "olivia", "lucas", "mia", "noah", "emma", "liam",
            "sofia", "ethan", "ava", "mason", "isabella", "james", "charlotte",
            "ben", "nora", "oscar", "the baby", "the children", "my nephew",
            "my niece", "aunt linda", "uncle ray", "the gardener", "the nanny",
        ],
        "val": [
            "freja", "dr. patel", "the neighbours", "ms. toft", "the postman",
            "mr. chavez", "priya", "the caretaker",
        ],
    },
    "message": {
        "train": [
            "dinner is ready", "time to go", "i am busy", "the door is open",
            "call me back", "we are leaving in ten minutes",
            "breakfast is on the table", "the laundry is done", "coffee is ready",
            "dinner will be late", "the mail arrived", "your taxi is waiting",
            "that is the alarm", "the game is about to start", "please knock",
            "it is starting to rain", "the oven is hot", "visitors are here",
            "the power went out", "start the meeting",
        ],
        "val": [
            "the bath is running", "your parcel arrived", "the garage door is open",
            "supper is served", "the lights were left on",
        ],
    },
}

# Fuzzy quantities resolved to the closed duration vocab at label time, so the
# model never has to emit a value outside DURATION_AMOUNTS.
FUZZY = {
    "half an hour": ("30", "minutes"), "an hour": ("1", "hours"),
    "a couple of minutes": ("2", "minutes"), "a few minutes": ("3", "minutes"),
    "a few seconds": ("5", "seconds"), "a minute": ("1", "minutes"),
    "a moment": ("10", "seconds"), "a bit": ("30", "seconds"),
}
EXACT = [("5", "seconds"), ("10", "seconds"), ("30", "seconds"), ("2", "minutes"),
         ("5", "minutes"), ("10", "minutes"), ("15", "minutes"), ("20", "minutes"),
         ("45", "minutes"), ("1", "hours"), ("2", "hours")]


@dataclass
class Tmpl:
    tid: str
    pattern: str                 # {slot} placeholders
    intent: str
    slots: dict[str, str] = field(default_factory=dict)   # slot -> placeholder name
    split: str = "both"


# --- single-action templates --------------------------------------------------
T: list[Tmpl] = []


def _t(tid, pattern, intent, slots, split="both"):
    T.append(Tmpl(tid, pattern, intent, slots, split))


# MOVE
for i, p in enumerate(["go to {location}", "head over to {location}", "move to {location}",
                       "drive to {location}", "can you go to {location}",
                       "please go to {location}", "walk over to {location}",
                       "go on over to {location}", "make your way to {location}",
                       "head to {location}", "get over to {location}",
                       "cruise to {location}", "travel to {location}"]):
    _t(f"move{i}", p, "MOVE", {"location": "location"})
for i, p in enumerate(["navigate to {location}", "roll to {location}",
                       "trundle over to {location}"]):
    _t(f"move_v{i}", p, "MOVE", {"location": "location"}, split="val")

# MOVE + person
for i, p in enumerate(["go to {person} who is by {location}",
                       "go to {person}, she is at {location}",
                       "find {person} near {location}",
                       "would you go to {person} which is standing by {location}",
                       "go to where {person} is, over by {location}",
                       "head to {person} at {location}",
                       "go see {person} over at {location}"]):
    _t(f"movep{i}", p, "MOVE", {"location": "location", "person": "person"})

# CLEAN
for i, p in enumerate(["clean {location}", "vacuum {location}", "tidy up {location}",
                       "give {location} a clean", "hoover {location}", "sweep {location}",
                       "mop {location}", "wipe down {location}", "dust {location}"]):
    _t(f"clean{i}", p, "CLEAN", {"location": "location"})
for i, p in enumerate(["clean up", "start cleaning", "vacuum in here", "clean this room",
                       "tidy up in here", "do some cleaning", "have a clean around"]):
    _t(f"cleanh{i}", p, "CLEAN", {"location": "@here"})
_t("cleane0", "clean the whole house", "CLEAN", {"location": "@everywhere"})

# PLAY (slot-less)
for i, p in enumerate(["play music", "put on some music", "start playing music",
                       "play something", "cue up some music", "turn on the music",
                       "play a song", "start the music", "play"]):
    _t(f"play{i}", p, "PLAY", {})

# SHOW
for i, p in enumerate(["show {message}", "display {message}", "put {message} on the screen",
                       "flash up {message}", "print {message}", "screen {message}"]):
    _t(f"show{i}", p, "SHOW", {"message": "message"})
for i, p in enumerate(["tell {person} {message}", "let {person} know {message}",
                       "show {person} {message}", "please tell {person} {message}"]):
    _t(f"showp{i}", p, "SHOW", {"message": "message", "person": "person"})

# HANDOVER: object-less; `person` filled ONLY for third parties.
for i, p in enumerate(["bring this to {person}", "hand this to {person}",
                       "take it over to {person}", "pass this to {person}",
                       "deliver this to {person}", "give this to {person}",
                       "hand {person} this", "bring it to {person}"]):
    _t(f"handr{i}", p, "HANDOVER", {"person": "person"})
# Speaker is the recipient: object-less, no person slot.
for i, p in enumerate(["bring it", "bring me that", "hand it to me", "get me that",
                       "fetch it for me", "grab me that", "pass me it"]):
    _t(f"hand{i}", p, "HANDOVER", {})

# STOP (closed class by nature)
for i, p in enumerate(["stop", "stop everything", "halt", "abort", "stop what you are doing",
                       "freeze", "stand down", "cut it out", "pull up", "that is enough"]):
    _t(f"stop{i}", p, "STOP", {})

# WAIT
for i, p in enumerate(["wait {duration}", "pause for {duration}", "hold on for {duration}",
                       "wait for {duration}", "stay put for {duration}",
                       "hang tight for {duration}", "wait {duration} please"]):
    _t(f"wait{i}", p, "WAIT", {"duration_amount": "@dur"})

# --- composite templates: cross-action coreference ----------------------------
COMPOSITE = [
    ("comp0", "go to {location} and clean it", ["MOVE:location", "CLEAN:location"]),
    ("comp1", "go to {location} then vacuum it", ["MOVE:location", "CLEAN:location"]),
    ("comp2", "head to {location} and give it a clean", ["MOVE:location", "CLEAN:location"]),
    ("comp3", "go to {location}, clean it, then stop", ["MOVE:location", "CLEAN:location", "STOP:"]),
]

CONNECTIVES = [" and ", " and then ", " then ", ", ", ", then ", ", after that ", " also ", " "]

# --- out-of-scope ---------------------------------------------------------------
REJECT = ["make me a coffee", "order me a pizza", "what is the weather today",
          "call my mom", "turn on the tv", "book a table for two", "send an email to john",
          "what time is it", "remind me to buy milk", "play chess with me",
          "open the window", "set the thermostat to 21", "tell me a joke",
          "who won the game last night", "unlock the front door", "cook me dinner",
          "water the lawn", "take a photo", "record a video", "scan the documents",
          "print the receipt", "make a payment", "check my heartbeat", "do my homework",
          "feed the cat", "walk the dog", "translate this to spanish", "summarize the news"]
REJECT_VAL = ["water the plants", "feed the dog", "translate this into danish",
              "bake a cake", "find me a flight"]


def _pool(split, kind):
    return POOLS[kind]["train" if split == "train" else "val"]


def _pick(rng, kind, split):
    return rng.choice(_pool(split, kind))


def _render_duration(rng):
    if rng.random() < 0.35:
        phrase = rng.choice(list(FUZZY))
        amt, unit = FUZZY[phrase]
        return phrase, amt, unit
    amt, unit = rng.choice(EXACT)
    phrase = f"{amt} {unit}" if rng.random() < 0.7 else f"{amt} {unit[:-1]}"
    return phrase, amt, unit


def _fill(rng, t: Tmpl, split: str):
    """Render one template into (text, Action)."""
    text, slots = t.pattern, {}
    for slot, ph in t.slots.items():
        if ph == "@here":
            slots["location"] = "here"
        elif ph == "@everywhere":
            slots["location"] = "everywhere"
        elif ph == "@dur":
            phrase, amt, unit = _render_duration(rng)
            text = text.replace("{duration}", phrase)
            slots["duration_amount"], slots["duration_unit"] = amt, unit
        else:
            v = _pick(rng, ph, split)
            text = text.replace("{" + slot + "}", v)
            slots[slot] = v
    return text, Action(t.intent, slots)


def _composite(rng, split):
    tid, pattern, spec = rng.choice(COMPOSITE)
    loc = _pick(rng, "location", split)
    text = pattern.replace("{location}", loc)
    acts = []
    for s in spec:
        intent, slot = s.split(":")
        acts.append(Action(intent, {slot: loc} if slot else {}))
    return tid, text, acts


def _text_key(text):
    return hashlib.blake2b(" ".join(tokenize(text)).encode(), digest_size=12).hexdigest()


def _row(text, acts, kind, tids, split, vocab):
    text = " ".join(text.split()).lower()
    try:
        validate_plan(acts)
        encode(text, acts, vocab)               # wire-format gate
    except (ValueError, Unencodable):
        return None, None
    return text, {"text": text, "actions": [a.to_dict() for a in acts],
                  "kind": kind, "templates": tids, "split": split}


def _single(rng, split):
    """One atomic example: returns (key, row) or (None, None)."""
    pool = [t for t in T if t.split == "both" or t.split == split]
    t = rng.choice(pool)
    text, a = _fill(rng, t, split)
    return _row(text, [a], "atomic", [t.tid], split, Vocab())


def _chain(rng, split, k, vocab):
    """One chained example of k atomic actions (k=2..3), deduped locally so no
    repeated identical action and STOP is last."""
    pool = [t for t in T if t.split == "both" or t.split == split]
    for _ in range(40):
        tid, text, acts = _composite(rng, split)
        if len(acts) == k and all(a.intent != "STOP" for a in acts[:-1]):
            return _row(text, acts, {2: "pair", 3: "triple"}[k], [tid], split, vocab)
        parts, acts = [], []
        ok = True
        for _ in range(k):
            t = rng.choice(pool)
            txt, a = _fill(rng, t, split)
            if any(a.intent == "STOP" for a in acts):
                ok = False
                break
            parts.append(txt)
            acts.append(a)
        if not ok:
            continue
        if any(acts[i].to_dict() == acts[i + 1].to_dict() for i in range(len(acts) - 1)):
            continue
        text = parts[0] + "".join(rng.choice(CONNECTIVES) + p for p in parts[1:])
        return _row(text, acts, {2: "pair", 3: "triple"}[k], [], split, vocab)
    return None, None


def generate(n: int, split: str, seed: int = 0,
             mix=(0.55, 0.28, 0.09, 0.08), dedup: bool = True) -> tuple[list[dict], int]:
    """mix = (atomic, pair, triple, reject), enforced with target counts so a
    rare kind can never be starved out by the dedup set.

    `dedup` controls whether already-seen utterances are skipped. It defaults
    to True (each distinct surface emitted at most once per run) but once a
    kind exhausts its distinct space, repeats are allowed so the requested mix
    is still honoured -- this is what keeps atomic (whose diversity is bounded
    by the template x entity space) from vanishing from large runs.
    """
    rng = random.Random(seed)
    vocab = Vocab()
    targets = [int(n * m) for m in mix]
    targets[-1] = n - sum(targets[:-1])     # reject absorbs rounding
    done = [0, 0, 0, 0]                     # emitted per kind (0..3)
    out, dropped, seen = [], 0, set()
    stale = [0, 0, 0, 0]                    # consecutive dup misses per kind
    exhausted = [False] * 4                 # kind whose distinct space is full
    guard = 0
    while sum(done) < n and guard < n * 200:
        guard += 1
        # choose the kind that is furthest from its target (fills the slowest)
        k = min(range(4), key=lambda i: done[i] / max(1, targets[i]))
        if k == 3:                          # reject
            src = REJECT if split == "train" else REJECT_VAL
            text = rng.choice(src)
            acts = [Action("UNAVAILABLE", {})]
            tids = ["reject"]
            # near-miss mixed chain: valid clause + out-of-scope tail
            if rng.random() < 0.45 and split == "train":
                t = rng.choice([x for x in T if x.split == "both"
                                and x.intent in ("CLEAN", "MOVE", "HANDOVER")])
                pre, _ = _fill(rng, t, split)
                text = pre + rng.choice([" and ", " then ", ", and "]) + text
                tids = ["reject_mixed", t.tid]
            text, row = _row(text, acts, "reject", tids, split, vocab)
        else:
            kk = k + 1                      # 0->atomic(1), 1->pair(2), 2->triple(3)
            if kk == 1:
                text, row = _single(rng, split)
            else:
                text, row = _chain(rng, split, kk, vocab)

        if row is None:
            dropped += 1
            continue
        key = _text_key(text)
        if dedup and not exhausted[k] and key in seen:
            stale[k] += 1
            # give up on uniqueness for this kind once it clearly saturated
            if stale[k] >= 64:
                exhausted[k] = True
            continue
        seen.add(key)
        stale[k] = 0
        out.append(row)
        done[k] += 1
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="-")
    ap.add_argument("--balanced", action="store_true",
                    help="equal atomic/pair/triple/reject, for evaluation")
    ap.add_argument("--no-dedup", action="store_true",
                    help="allow repeated utterances (for large runs)")
    a = ap.parse_args()
    mix = (0.25, 0.25, 0.25, 0.25) if a.balanced else (0.55, 0.28, 0.09, 0.08)
    rows, dropped = generate(a.n, a.split, a.seed, mix, dedup=not a.no_dedup)
    f = open(a.out, "w") if a.out != "-" else sys.stdout
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.out != "-":
        f.close()
    print(f"{len(rows)} rows, {dropped} dropped at the wire-format gate",
          file=sys.stderr)


if __name__ == "__main__":
    main()

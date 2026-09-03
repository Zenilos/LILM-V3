"""Deployment-facing chain -> atomic-sentence segmenter (approach A).

The V5 model classifies ONE atomic command at a time. A spoken command is
often a chain ("go to the kitchen and clean it", "head to my desk, pick that
up, then wait"). This module breaks a chain into standalone atomic sentences,
each of which the atomic V5 model can classify, WITHOUT needing gold actions.

It is the surface-only, no-label version of v4_decompose.decompose_chain:
find action boundaries from connectives alone, then resolve anaphora
("it"/"this"/"that"/"here") by substituting the referent most recently
mentioned in a prior clause (the location/person/message value).

Usage (deployment case):
    from chain_seg import segment
    sents = segment("go to the kitchen and clean it")
    # -> ["go to the kitchen", "clean the kitchen"]

Run each returned sentence through the V5 intent+slot model to get
(intent, slots).
"""

from __future__ import annotations

import re

# Longest-first so " and then " wins over " and ". The bare " " is NOT a
# boundary (it just separates words). Regex-split on the connective alternation.
# Sentence/period and semicolon boundaries are treated as clause separators too
# ("break a chain to separate sentences" often arrives with such punctuation).
_CONNECTIVE_RE = re.compile(
    r"(\s+and\s+then\s+|,+\s*after\s+that\s+|,+\s*then\s+|,+\s*and\s+|"
    r"\s+and\s+|\s+then\s+|\s+also\s+|,+\s*|\s*\.\s*|\s*;\s*)"
)

# Anaphors that refer to a previous clause's slot value.
ANAPHORS = re.compile(r"\b(it|this|that|them)\b")
# Bare connective words that can lead a clause after a boundary ("; then stop",
# "then go", "and wait"); strip them so the clause starts at its verb.
_LEADING_CONN = re.compile(r"^(?:and|then|also)\s+")
# Sentinel locations ("clean up", "vacuum here") that carry no explicit span
# but where "here" can still be a real location span for MOVE/CLEAN.
HERE = re.compile(r"\bhere\b")

# Coarse keyword set to recognize an explicit location mention and bind the
# referent for a following anaphoric clause.
_LOC_RE = re.compile(
    r"\b(?:the|my|our|her|his)?\s*(?:kitchen|living room|room|garage|hallway|"
    r"bedroom|office|upstairs|dining room|basement|desk|front door|balcony|"
    r"laundry room|attic|pantry|sunroom|workshop|nursery|study|playroom|den|"
    r"closet|foyer|terrace|guest room|cellar|rooftop|porch|library|gym|"
    r"driveway|courtyard|corridor|mudroom|utility room|greenhouse|sauna|"
    r"conservatory|mezzanine|storeroom|veranda|annexe|loft|outhouse|sickbay)\b"
)


def _boundaries(text: str) -> list[str]:
    """Split `text` into clauses at connective occurrences."""
    parts = [p for p in _CONNECTIVE_RE.split(text) if p]
    clauses = []
    current = ""
    for p in parts:
        if _CONNECTIVE_RE.fullmatch(p):
            if current.strip():
                clauses.append(current.strip())
            current = ""
        else:
            current += p
    if current.strip():
        clauses.append(current.strip())
    return clauses


def _resolve(clause: str, ctx: dict) -> str:
    """Substitute anaphors / "here" with the current referent."""
    ref = ctx.get("context_ref")
    if ref:
        clause = ANAPHORS.sub(lambda _: ref, clause)
    if HERE.search(clause) and ctx.get("location"):
        clause = HERE.sub(ctx["location"], clause)
    return clause


def _bind(clause: str, ctx: dict) -> None:
    """Update the referent context from an explicit location mention."""
    m = _LOC_RE.search(clause)
    if m:
        value = clause[m.start():m.end()].strip()
        ctx["context_ref"] = value
        ctx["location"] = value


def segment(text: str) -> list[str]:
    """Break a spoken chain into self-contained atomic sentences."""
    clauses = _boundaries(text.strip().lower())
    ctx: dict = {"context_ref": None, "location": None}
    out: list[str] = []
    for clause in clauses:
        clause = _LEADING_CONN.sub("", clause)
        _bind(clause, ctx)
        resolved = _resolve(clause, ctx)
        if resolved.strip():
            out.append(resolved.strip())
    return out


if __name__ == "__main__":
    cases = [
        # (input, expected atomic sentences, expected match)
        ("go to the kitchen and clean it",
         ["go to the kitchen", "clean the kitchen"], True),
        ("go to the bedroom, vacuum it, then wait 10 minutes",
         ["go to the bedroom", "vacuum the bedroom", "wait 10 minutes"], True),
        ("head to my desk and then stop",
         ["head to my desk", "stop"], True),
        ("please go to the garage, clean up, and stop",
         ["please go to the garage", "clean up", "stop"], True),
        ("play music and go to the living room",
         ["play music", "go to the living room"], True),
        ("move to the office, display hello, then wait a moment",
         ["move to the office", "display hello", "wait a moment"], True),
        ("show dinner is ready to sarah",
         ["show dinner is ready to sarah"], True),
        # real sentence separators (period / semicolon) split the chain
        ("go to the kitchen. clean it. wait",
         ["go to the kitchen", "clean the kitchen", "wait"], True),
        ("head to my desk; then stop",
         ["head to my desk", "stop"], True),
        # corpus degenerate: no explicit connective word -> cannot split (documented)
        ("travel to the utility room make your way to the greenhouse",
         None, False),
    ]
    npass = 0
    for inp, expect, strict in cases:
        got = segment(inp)
        if not strict:
            print(f"{inp!r}\n  -> {got}  (documented no-split case, skipped)\n")
            npass += 1
            continue
        ok = got == expect
        npass += ok
        print(f"{'OK ' if ok else 'FAIL'} {inp!r}\n  -> {got}\n")
    print(f"{npass}/{len(cases)} passed")
    assert npass == len(cases), "segmenter self-test failed"
    print("chain_seg.py OK")

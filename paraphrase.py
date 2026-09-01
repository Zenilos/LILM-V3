"""Stage B: naturalness via a local LLM, with gold carried over from Stage A.

The LLM never produces labels. It rewrites an utterance whose gold is already
known, and the rewrite is kept only if every slot value survives as a literal
span. That inverts the usual reliability problem: a model that drifts produces
a DROPPED example, not a mislabelled one, so teacher quality bounds yield
rather than accuracy.

Talks to an OpenAI-compatible endpoint (Ollama, mlx_lm.server, etc.):
    ollama serve          (default: http://localhost:11434)
    mlx_lm.server --model <local-path> --port 8080
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request

from dsl import Action
from serialize import Unencodable, Vocab, encode, tokenize

SYSTEM = (
    "You rewrite short spoken commands to a home robot so they sound like different "
    "people talking. Output ONLY the rewritten command, lowercase, no quotes, no "
    "explanation.\n"
    "HARD RULE: every phrase listed under KEEP must appear in your rewrite exactly, "
    "character for character. Do not pluralize, abbreviate, reorder words inside "
    "them, or swap synonyms. Everything around them may change freely.\n"
    "Keep the same actions in the same order. Do not add or remove actions."
)

STYLES = [
    "make it more polite and indirect",
    "make it terse, like a command",
    "make it casual and conversational, with a filler word",
    "phrase it as a question",
    "add a short irrelevant aside before the request",
    "use a different verb for the same action",
]


def build_prompt(text: str, keep: list[str], style: str) -> str:
    lines = "\n".join(f"- {k}" for k in keep) or "- (none)"
    return f"COMMAND: {text}\nKEEP:\n{lines}\nSTYLE: {style}\nREWRITE:"


def keep_phrases(actions: list[Action]) -> list[str]:
    """Values that must survive verbatim. duration_* are excluded: they are
    literal tokens, so the surface form is free to change ('half an hour' ->
    '30 minutes') as long as the canonical value is unchanged."""
    out = []
    for a in actions:
        for slot, v in a.slots.items():
            if slot.startswith("duration_") or v in ("here", "everywhere"):
                continue
            out.append(v)
    return sorted(set(out), key=len, reverse=True)


def call(host: str, model: str, prompt: str, temp: float, timeout: int = 60) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "temperature": temp, "max_tokens": 80,
    }).encode()
    req = urllib.request.Request(f"{host}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def verify(text: str, actions: list[Action], vocab: Vocab) -> bool:
    """The gate. A rewrite is usable only if it still encodes to the same plan."""
    try:
        encode(text, actions, vocab)
    except Unencodable:
        return False
    return len(tokenize(text)) <= vocab.max_input


def run(rows, host, model, per_row, temp, echo=None):
    """echo: injectable fake client for offline testing."""
    V = Vocab()
    client = echo or (lambda p: call(host, model, p, temp))
    seen = set()
    kept = dropped = 0
    for r in rows:
        acts = [Action.from_dict(a) for a in r["actions"]]
        keep = keep_phrases(acts)
        for i in range(per_row):
            style = STYLES[(hash(r["text"]) + i) % len(STYLES)]
            try:
                cand = client(build_prompt(r["text"], keep, style))
            except Exception as e:                      # noqa: BLE001
                print(f"request failed: {e}", file=sys.stderr)
                continue
            cand = " ".join(cand.split()).lower().strip('"')
            key = hashlib.blake2b(" ".join(tokenize(cand)).encode(),
                                  digest_size=12).hexdigest()
            if key in seen or cand == r["text"] or not verify(cand, acts, V):
                dropped += 1
                continue
            seen.add(key)
            kept += 1
            yield {**r, "text": cand, "source": "paraphrase",
                   "origin": r["text"], "style": style}
    print(f"paraphrase: kept {kept}, dropped {dropped} "
          f"({dropped / max(1, kept + dropped):.0%} rejection rate)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="http://localhost:11434",
                    help="OpenAI-compatible endpoint (default: Ollama at :11434)")
    ap.add_argument("--model", default="smollm2:135m",
                    help="Model name (default: smollm2:135m)")
    ap.add_argument("--per-row", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    with open(a.inp) as f:
        rows = [json.loads(l) for l in f]
    if a.limit:
        rows = rows[:a.limit]
    with open(a.out, "w") as f:
        for r in run(rows, a.host, a.model, a.per_row, a.temp):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

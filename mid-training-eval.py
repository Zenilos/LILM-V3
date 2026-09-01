"""Interactive single-prompt inference against a student checkpoint.

Lets you inspect what the model emits for an arbitrary utterance at any point
during training, without a full eval.py run. Loads a .npz checkpoint, rebuilds
the word tokenizer (same convention as eval.py), then reads one prompt at a time.

The decode temperature flag `t` matches the training/eval mode:
  * t=0.0  -- fp (latent weights, no ternary snapping). Use with --fp runs
              (e.g. checkpoints/fp_v2/best.npz), the un-quantized baseline.
  * t=1.0  -- fully ternary quantization. Use with QAT runs (e.g.
              checkpoints/student/best.npz) where weights were trained ternary.

Usage:
    python mid-training-eval.py [--model ckpt] [--t 0.0|1.0] [--val data/val.jsonl]

Type prompts at the prompt> line; a blank line exits.

Note: FSM-constrained greedy decode guarantees any non-None output decodes to a
structurally valid plan, so failures always show up as wrong values/actions
rather than unparseable garbage.
"""

from __future__ import annotations

import argparse
import json

import mlx.core as mx
import numpy as np

from dsl import Action
from serialize import FSM, Vocab, decode, tokenize
from train_student import Student, build_base_tok, ModelConfig

from eval import _unflatten


def load_model(path: str) -> Student:
    data = np.load(path, allow_pickle=False)
    model = Student(ModelConfig())
    model.update(_unflatten(data))
    return model


def greedy_decode(model: Student, words: list[str], uid: list[int],
                  vocab: Vocab, fsm: FSM, t: float = 0.0):
    inp = mx.array(uid, dtype=mx.int32)[None, :]
    st = fsm.start(len(uid))
    gen: list[int] = []
    for _ in range(80):
        legal = fsm.legal(st)
        if not legal:
            break
        logits = model(inp, t)[0, -1, :]
        la = mx.array(sorted(legal), dtype=mx.int32)
        tid = int(la[mx.argmax(mx.take(logits, la))].item())
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
    if gen[0] == vocab.id["<no>"]:
        return [Action("UNAVAILABLE", {})]
    return decode(" ".join(words), [vocab.id["<plan>"]] + gen, vocab)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="checkpoints/fp_v2/best.npz")
    ap.add_argument("--t", type=float, default=0.0,
                    help="decode temperature 0.0=fp, 1.0=ternary "
                         "(default 0.0 for fp baselines)")
    ap.add_argument("--val", default="data/val.jsonl",
                    help="data used to rebuild the word tokenizer")
    a = ap.parse_args()

    vocab = Vocab()
    fsm = FSM(vocab)
    model = load_model(a.model)

    with open(a.val) as f:
        rows = [json.loads(l) for l in f]
    bt = build_base_tok([r["text"] for r in rows])
    print(f"model: {a.model}  t={a.t}  tokenizer from {a.val}"
          f" ({len(bt.word2id)} words)", flush=True)
    print("type a prompt, blank line to exit", flush=True)

    while True:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            break
        if not prompt:
            break
        words = tokenize(prompt)
        if not words:
            print("  (no tokens)", flush=True)
            continue
        pred = greedy_decode(model, words, bt.ids(words), vocab, fsm, a.t)
        print(f"  tokens: {words}", flush=True)
        print(f"  pred  : {pred}", flush=True)


if __name__ == "__main__":
    main()

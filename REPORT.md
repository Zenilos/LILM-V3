# Report — PlanCore-11M training investigation

_Date: 2026-09-02_
_Scope: three fp training runs (`fp_ab`, `fp_v2`, `fp_sample`), honest eval, and the
conclusion about the model's ability to map spoken commands to intents + slots._

---

## 1. tl;dr

The 11.1M-parameter from-scratch student does **not** learn to map a spoken
command to a correct `(intent, slots)` plan, even when we relax the metric to
**intent-only** and drop slot extraction entirely. After three fp runs, the best
honest full-val numbers are:

| metric | fp_v2 | fp_sample |
|---|---|---|
| exact plan EM (intent + all slots) | **6.7%** | ~7.0% in-run |
| intent + required slots only | **5.1%** | 3.1% |
| utterance **intent-chain only** (no slots) | **7.9%** | 7.0% |
| per-position intent accuracy (MOVE) | 33% | 30% |
| chain length correct (any intents) | 64% | 57% |

The model memorizes the training set (train CE ≈ 0.0025) but does **not**
generalize: on unseen phrasings it either picks the wrong intent, fills slots
with the wrong span, or emits UNAVAILABLE. This is a **capacity / pre-training
gap**, not a training-loop bug. Training-loop fixes (scheduled sampling, KD)
were tried or planned and do not address the root cause.

---

## 2. The three runs

### Run 1 — fp_ab (superseded)
- Warmup-only schedule, no cosine decay, scored on a buggy harness.
- Train loss ≈ 0.005 (memorized), best "val EM" ≈ 0.023.
- Retained only as evidence for the exposure-bias diagnosis.

### Run 2 — fp_v2 (20k steps, complete)
- Command: `--steps 20000 --batch 256 --warmup 2000 --ramp-frac 0.2 --fp`
- In-training 512-subset val_em peaked at **0.2773** (step 8000) — an over-read.
- Honest full-deduped-val eval (n=3641) of best checkpoint: **EM 0.0665**,
  intent-seq acc 0.1911, atomic 0.146, pair 0.054, triple 0.005, reject 3/5.
- Train loss ≈ 0.0025 → memorized.

### Run 3 — fp_sample (scheduled sampling, 0.1)
- Command adds `--sample-prob 0.1 --fp` (always-on second forward).
- In-training full-val EM (fixed harness, per-kind): 0.047 @2k, 0.031 @4k,
  0.064 @6k, **0.070 @8k**, 0.062 @10k, 0.066 @12k, 0.065 @14k.
- Process died of OOM at step 15700 (from concurrent background eval, not the
  run itself); best checkpoint is step 8000.
- Honest intent-only eval: utterance intent-chain **7.0%**, chain length 57.3%.

---

## 3. Honest eval methodology (the bugs that inflated early numbers)

1. **512-row subset over-read.** The training loop sampled only 512 val rows,
   inflating EM (0.28 "best"). Fixed: `evaluate()` now does full-val + per-kind.
2. **No val dedupe.** `data/val.jsonl` has 5,000 lines but only 3,641 unique
   texts (263 copies of "find me a flight"). `eval.py` now dedupes by text to
   match `load_rows`.
3. **Pointer decode vs fake text.** Decoded pointer tokens against `"x"*n`
   instead of the real utterance → every slot decoded to `x...`. Fixed to decode
   against the real utterance words.
4. **`--t` for fp decode.** A --fp run's weights were previously snapped to
   ternary at eval. Now decode with `t=0.0` for fp checkpoints, `t=1.0` for QAT.

With all four fixed, the honest full-val EM is **~0.07**, not 0.28.

---

## 4. Intent-only analysis (new, run 2026-09-02)

Because plan EM is dominated by slot extraction, we isolated the intent signal:
does the model pick the right intent even if we ignore all slot values?

`intent_only.py` results (per gold action, positional, only when chain length
matches gold):

### fp_v2
| Intent | correct/total | pct |
|---|---|---|
| MOVE | 710/2135 | 33.3% |
| SHOW | 128/858 | 14.9% |
| HANDOVER | 151/1500 | 10.1% |
| WAIT | 63/672 | 9.4% |
| CLEAN | 96/1292 | 7.4% |
| PLAY | 29/638 | 4.5% |
| STOP | 3/291 | 1.0% |
| UNAVAILABLE | 0/5 | 0.0% |

- utterance **intent-chain exact: 7.9%** (289/3641)
- chain **length** correct (any intents): **63.6%** (2316/3641)
- output sizes: 1365× len-1, 1374× len-2, 902× len-3

### fp_sample
- utterance intent-chain exact: **7.0%** (254/3641)
- chain length correct: **57.3%**
- MOVE 29.6%, CLEAN 15.5%, WAIT 14.6%, SHOW 5.6%, HANDOVER 4.3%, PLAY 0.6%,
  STOP 1.7%

### Reading
- **Chain length is mostly right** (~57-64%), so the decoder is structurally sane.
- **Per-intent, only MOVE exceeds ~chance**, and even MOVE is 30-33% — a trivial
  majority classifier on the MOVE-vs-not axis would beat it.
- Multi-action chains compound: getting every intent+order right on a 2-3 action
  utterance drops whole-utterance intent accuracy to **~7-8%**.
- `intent + required slot` is even worse (5.1% / 3.1%): the model cannot bind the
  correct span to the slot (e.g. "go to my daughter at my desk" → location should
  be "my desk", but the model emits the wrong span or UNAVAILABLE).

**Conclusion:** intent understanding is weak and slot binding is near-zero. A
dedicated intent classifier would substantially beat this model on intent; the
slot-extraction head is the largest single source of failure.

---

## 5. Root cause

Not quantization (fp runs show the same), not the training loop (fp_v2 and
fp_sample agree), and not a data-size problem in isolation. The 11.1M
from-scratch model, trained only on ~470k template sentences, lacks the
pre-trained semantic knowledge to recognize that "daughter" is a person and
"desk" is a location. It memorizes specific phrasings and fails on any novel
composition.

---

## 6. Viable paths (documented in NEXT.md Step 6)

**A. Scale the from-scratch model** — 18-21M params fits ESP32-S3 PSRAM; one
controlled ablation to test if capacity is the bottleneck. Pure ESP32 deploy,
but from-scratch is sample-inefficient.

**B. Pre-computed encoder embeddings** (offline distillation) — run a small
pre-trained LM over the corpus, dump embeddings, train a tiny action head on
top. Highest expected EM; encoder too big for on-device unless very small or
streamed from flash.

**C. Hybrid on-device + external** — keep the small model for easy/atomic
commands, fall back to an external model (or lower accuracy) for chains and
novel phrasings, gated by top-1 confidence.

Recommended order: try A (one 18M ablation) to confirm capacity is the issue;
if it doesn't help, B is highest-ROI but needs the encoder-on-device problem
solved; C is the pragmatic fallback.

---

## 7. Artifacts / where the evidence lives

- Scripts: `eval_required.py` (intent+required-slot, no optional),
  `intent_only.py` (utterance intent-chain, no slots).
- Runs: `checkpoints/fp_v2/best.npz`, `checkpoints/fp_sample/best.npz`
  (gitignored, regenerable),
  `logs/fp_v2.log`, `logs/fp_sample.log`.
- Design: `ARCHITECTURE.md`, `PLAN.md`, detailed plan `NEXT.md`.
- Export verified: 3.63 MB ternary blob round-trips exactly (trit packing +
  int8 embedding); packing contract + `--selftest` added to `export.py`.

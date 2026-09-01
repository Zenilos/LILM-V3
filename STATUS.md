# STATUS — PlanCore-11M (training investigation)

_Last updated: 2026-09-01_

This file documents what exists, what we found while debugging training, and
the concrete next steps. It is the working record for the solo session; see
`PLAN.md` (roadmap) and `ARCHITECTURE.md` (normative design) for the full plan.

---

## Where things stand

The **full student-training pipeline is built and runs end-to-end**, but does
**not yet produce a model with meaningful exact-match (EM) on the validation
set**. We found the precise failure mode via an fp-vs-QAT ablation; we have not
yet fixed it. Nothing is currently training.

| Component | Status |
|---|---|
| Data generation (`corpus.py`) | Done. `data/train_a.jsonl` (470,113 unique) + `data/val.jsonl` (3,641 unique). 55/28/9/8 mix. |
| DSL + wire serialization (`dsl.py`, `serialize.py`) | Done, unit-verified (vocab=4388). |
| Student model (`model.py`) | Done. 11,135,360 params, tied embed/LM-head, GQA, SwiGLU, RoPE. |
| QAT training (`train_student.py`) | Runs; **failure mode identified** (below). |
| Eval harness (`eval.py`) | Done; loader fixed (unflattens dot-key `.npz`, `model.update`). |
| Export (`export.py`) | 3.63 MB blob verified on a checkpoint. |
| FSM in C (`fsm.h`, `fsm.c`) | Done; self-test passes (vocab=4388, mask=549B). |
| Teacher fine-tune (`train_teacher.py`) | Written, **not run** (needs BPE-aware encode + HF stack). |
| Paraphrase (`paraphrase.py`) | Written; **blocked** (Ollama not reachable on :11434). |

---

## Bugs found and fixed (in `train_student.py`)

1. **Label off-by-one / copy-shortcut bug.** Labels were placed so the target
   equaled the input token at each label position, letting the model minimize
   CE by copying instead of learning (loss collapsed to ~0.0002). Fixed: labels
   shift the target to the *next* token (`labels[start-1 : start-1+ln] = lab`).
   Verified the loss now descends normally (~2.8 start).

2. **Inverted cosine LR schedule.** LR saturated to ~0 right after warmup and
   *rose* toward the end. Fixed to peak at warmup then cosine-decay to ~0.

3. Added `--ramp-frac` and `--warmup` args (used for the fp ablation; a useful
   diagnostic feature, not a behavioral change for the default QAT run).

4. `eval.py` checkpoint loading was broken (used `model.load` with flat keys).
   Fixed with dot-key unflatten + `model.update`. Verified.

---

## Key experimental finding — the failure is EXPOSURE BIAS, not quantization

We ran a controlled **fp (non-quantized) ablation**:
`--ramp-frac 100 --warmup 3000 --steps 3000` (`logs/fp_ab.log`).

**Results (fp model, no quantization, your debug-harness tokenizer fixed to the
full train set):**

- Final training loss ≈ **0.005** → the model essentially **memorized the
  training set**.
- **Best val EM only ≈ 0.023 (**2.3%**)**; val EM nosedived to 0.006 by step 3000.
- **Train intents-only under FREE-RUN greedy decode: 2/30** — i.e. the model
  that memorizes train (teacher-forced loss ≈ 0) still gets ~7% of train
  intents right when decoding freely.

**Interpretation (confident):** this is **NOT** a quantization problem and
**NOT** a convergence problem. It is **exposure bias**: the model learns to
predict the next label token conditioned on the *gold label prefix*
(teacher-forced CE ≈ 0), but does **not** strongly condition on the *input
utterance*. In deployment (free-run), once one token is slightly off the 
label-pattern autoregression cascades → wrong intents/slots → ~0 EM, even on
training data.

Secondary contributor: the val split is deliberately out-of-distribution
(held-out entities), but that is not the primary cause — train free-run is also
bad.

The earlier QAT runs (before the fp ablation) showed the same low EM; that was
consistent with this, not an independent quantization failure.

---

## Next steps (in order of leverage)

1. **Fix exposure bias — implement scheduled sampling** in `train_student.py`:
   during training, with probability `eps` decaying over the run, feed the
   model its *own* previously sampled token instead of the gold token at each
   label position. This directly addresses the confirmed failure mode. Simple,
   contained change (~20 lines).
   - Success criterion: fp train free-run intents-only accuracy jumps sharply
     (>> 2/30), and fp val EM climbs well above 0.023.

2. **Strengthen input-conditioning** if needed: prevent the model from
   shortcutting on label-pattern context (e.g. block attention from label
   positions to other label positions so intents/pointers must be read from the
   utterance), and/or weight intent tokens more heavily in `ce_loss`.

3. **Only after a working fp student** (fp val EM meaningful): re-enable QAT
   (`--ramp-frac 0.2`) and confirm the quantized model lands within the
   ~4-point EM drop budget from ARCHITECTURE.md. If QAT still collapses,
   revisit the ternary quantizer (BitNet-style scale factors / sub-2-bit pack).

4. Resume the full 20k-step run on the fixed recipe, then re-examine
   `export.py` output and the ESP32 FSM path.

---

## Standing commands / environment

- Python: `~/p3.11/bin/python3` (not the fish-venv activation; active.fish fails
  under bash).
- MLX 0.32.2 on the M1 GPU (16 GB). No `mlx.safetensors`; checkpoints are
  `.npz`.
- Student training run already validated at ~1.3 s/step, batch 256.
- **Monitoring cadence:** check active training roughly every 40 minutes, not
  every 10.

## Files not yet committed / generated artifacts

These are regenerable and excluded from git (see `.gitignore`):
`data/*.jsonl`, `checkpoints/*`, `logs/*`. Source under `lilmv3/`:
`ARCHITECTURE.md`, `PLAN.md`, `STATUS.md`, `corpus.py`, `dsl.py`,
`serialize.py`, `model.py`, `train_student.py`, `train_teacher.py`,
`paraphrase.py`, `eval.py`, `export.py`, `fsm.h`, `fsm.c`.

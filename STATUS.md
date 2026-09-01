# STATUS — PlanCore-11M (training investigation)

_Last updated: 2026-09-01_

This file documents what exists, what we found while debugging training, and
the concrete next steps. It is the working record for the solo session; see
`PLAN.md` (roadmap) and `ARCHITECTURE.md` (normative design) for the full plan.

---

## Where things stand

The **full student-training pipeline is built and runs end-to-end**, but does
**not yet produce a model with meaningful exact-match (EM) on the validation
set**. We found the precise failure mode via an fp-vs-QAT ablation and have
landed the fixes (corrected EM harness, schedule-sampled training, optional KD,
`--fp` mode). **A clean fp baseline is training now** (`checkpoints/fp_v2`,
PID on the M1); the first eval checkpoint scored **val_em=0.1934** at step 2000.

| Component | Status |
|---|---|
| Data generation (`corpus.py`) | Done. `data/train_a.jsonl` (470,113 unique) + `data/val.jsonl` (3,641 unique). 55/28/9/8 mix. |
| DSL + wire serialization (`dsl.py`, `serialize.py`) | Done, unit-verified (vocab=4388). |
| Student model (`train_student.py`; `model.py` trimmed to shared `ModelConfig`) | Done. 11.1M params, tied embed/LM-head, GQA, SwiGLU, RoPE. |
| QAT training (`train_student.py`) | **Training now** (fp_v2 baseline, 20k steps). Fixes landed: pointer-decode metric bug, gold alignment, scheduled sampling (`--sample-prob`), KD (`--teacher`), true-fp mode (`--fp`). First real result: **val EM 0.1934 @ step 2000** (vs 0.023 on the old buggy run). |
| Eval harness (`eval.py`) | Done; loader fixed (unflattens dot-key `.npz`, `model.update`); pointer decode + intent-denominator fixed. |
| Mid-training inspection (`mid-training-eval.py`) | New; interactive single-prompt eval against any checkpoint (`--t 0.0` fp / `1.0` ternary). |
| Export (`export.py`) | 3.63 MB blob verified on a checkpoint. |
| FSM in C (`fsm.h`, `fsm.c`) | Done; self-test passes (vocab=4388, mask=549B). |
| Teacher fine-tune (`train_teacher.py`) | Written, **not run** (needs BPE-aware encode + HF stack); student KD logit-distillation is implemented and gated on `--teacher`. |
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

5. **Pointer decode scored against fake input text (metric artifact).**
   `greedy_decode` decoded generated pointer tokens against `" ".join(["x"]*n)`
   instead of the real utterance, so every copy-pointer slot decoded to `"x..."`
   and could never match gold — the near-zero val EM was partly a *measurement*
   artifact, not pure model failure. Fixed in `train_student.py` and `eval.py`:
   decode against the real utterance words.

6. **Val golds / items misalignment.** `encode_rows` drops the occasional row,
   but `val_golds` was a blind truncation of `val_rows._acts`, so `evaluate`
   could match predictions to the wrong gold plans. `encode_rows` now carries
   `(uid, words, label, start, gold)` per item and `evaluate` scores each item
   against its own gold.

7. **RoPE buffer vs `--max-len` mismatch.** `ModelConfig.context_length=128`
   preallocated the RoPE tables while training defaulted to `--max-len 256` — a
   latent reshape crash on sequences >128. `context_length` is now 256.

8. **Dead duplicate architecture removed.** `model.py` reimplemented the student
   with different module names (`q_proj` vs `q`, etc.), so its `StudentModel`
   could never load a training checkpoint. `model.py` is trimmed to the shared
   `ModelConfig`; `train_student.py` is the single canonical architecture.

9. **Tokenizer-prune ID promise made explicit.** `build_pruned_tokenizer`
   renamed kept tokens by enumeration; changed to `{t: vocab[t]}` so original
   SmolLM2 IDs are preserved by contract (the KD `gather` premise).

10. Housekeeping: `torch.load(..., weights_only=True)`; duplicate pool entries
    in `corpus.py`; dead `intent_denom` and wrong intent denominator in
    `eval.py`; open file handle in `paraphrase.py`. All fixed.

11. **No true-fp mode existed.** Quantization strength is the ramp value `t`
    (0 = latent fp, 1 = fully ternary), and `greedy_decode` always decoded at
    `t=1.0` — so a "fp" run whose ramp stayed ~0 was *evaluated* with weights
    snapped to ternary {-1,0,+1}. Added `--fp`: forces `t=0` for both training
    AND decode, giving an honest unquantized baseline.

12. **`mid-training-eval.py` greed-select bug.** The standalone script picked
    its token by taking `mx.argmax` directly instead of indexing the
    legal-token list (`la[argmax]`), yielding token id 0 (unk) rather than the
    legal choice. Fixed to `la[mx.argmax(sel)]` (matching `train_student.py`).

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

**Measurement caveat:** until fix #5 above, the val EM figure was itself
depressed by the pointer-decode artifact. The exposure-bias diagnosis — train
teacher-forced ≈ 0.005 with free-run intents 2/30 — is unaffected (free-run
intent counts don't depend on decoding pointers), but the fp ablation must be
re-run on the fixed harness to get a true EM number.

Secondary contributor: the val split is deliberately out-of-distribution
(held-out entities), but that is not the primary cause — train free-run is also
bad.

The earlier QAT runs (before the fp ablation) showed the same low EM; that was
consistent with this, not an independent quantization failure.

---

## New fp baseline (in progress)

A clean fp run on the **fixed** harness and **correct** schedule is training:
`--steps 20000 --batch 256 --warmup 2000 --ramp-frac 0.2 --fp` (`logs/fp_v2.log`,
`checkpoints/fp_v2/`). This supersedes the stale `fp_ab.log` (that run was 100%
warmup with no cosine decay and was scored on the buggy pointer decode).

**Result so far:**
- **val_em = 0.1934 @ step 2000** (measured at LR peak, still early). First
  truly valid EM number; ~8× the 0.023 the old buggy run capped at.
- Training loss already ~0.005–0.013 (approaching train memorization).

**Caveat:** ~2000 training steps is far short of convergence; the cosine decay
covers the remaining 18k steps. The decisive number is mid-/late-run EM (steps
4000, 6000, ... 20000), not the step-2000 spike. Mid-run inspection with
`mid-training-eval.py` confirms the model still produces valid-but-wrong plans
(e.g. `MOVE{location:"my desk"}` + a spurious second action) — expected at LR
peak.

---

## Next steps (in order of leverage)

1. **Let the fp_v2 baseline finish** (20k steps, ~7–8 h total). Read val EM
   trend across eval checkpoints. If EM peaks then collapses, that confirms
   exposure bias and points to schedule-sampling / input-conditioning (below);
   if it keeps climbing, the earlier near-zero EM was dominated by the metric
   artifact + a never-decaying schedule, and plain CE may already work.

2. **Scheduled sampling — implemented, awaiting a validation run.** Live as
   `--sample-prob` (default 0). At each label position the model's own greedy
   argmax is fed back as input instead of the gold token, with probability
   ramping in over the first half of the run; loss labels stay gold. If the
   plain fp baseline shows the exposure-bias signature (EM peak-then-collapse),
   rerun with `--sample-prob 0.1` and compare.

3. **Strengthen input-conditioning** if needed: prevent the model from
   shortcutting on label-pattern context (e.g. block attention from label
   positions to other label positions so intents/pointers must be read from the
   utterance), and/or weight intent tokens more heavily in `ce_loss`.

4. **Only after a working fp student** (fp val EM meaningful): re-enable QAT
   (`--ramp-frac 0.2`) and confirm the quantized model lands within the
   ~4-point EM drop budget from ARCHITECTURE.md. If QAT still collapses,
   revisit the ternary quantizer (BitNet-style scale factors / sub-2-bit pack).

5. Resume the full 20k-step run on the fixed recipe, then re-examine
   `export.py` output and the ESP32 FSM path.

---

## Standing commands / environment

- Python: `~/p3.11/bin/python3` (not the fish-venv activation; active.fish fails
  under bash).
- MLX 0.32.2 on the M1 GPU (16 GB). No `mlx.safetensors`; checkpoints are
  `.npz`. Student training validated at ~1.3 s/step, batch 256.
- **Monitoring cadence:** check active training roughly every 40 minutes, not
  every 10.
- **Git remote:** `origin  git@github.com-Zenilos:Zenilos/LILM-V3.git`
  (uses the `github.com-Zenilos` SSH alias → `id_github_personal`; the default
  `id_ed25519` authenticates as a different account and is denied).

## Files not yet committed / generated artifacts

These are regenerable and excluded from git (see `.gitignore`):
`data/*.jsonl`, `checkpoints/*`, `logs/*`. Source under `lilmv3/`:
`ARCHITECTURE.md`, `PLAN.md`, `STATUS.md`, `NEXT.md`, `corpus.py`, `dsl.py`,
`serialize.py`, `model.py`, `train_student.py`, `train_teacher.py`,
`paraphrase.py`, `eval.py`, `mid-training-eval.py`, `export.py`, `fsm.h`,
`fsm.c`.

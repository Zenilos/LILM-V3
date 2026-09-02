# NEXT.md — What to do now

_Last updated: 2026-09-02_

This is a working plan, not a design doc. It picks up after commit `0cd1536`
(all bug fixes, scheduled sampling, KD, model.py reconciliation) and the
**completed fp_v2 baseline** (Step 1, see below).

---

## Where we actually are

Step 1 (clean fp baseline) is **done**. All pipeline code is written and
runnable. known correctness bugs are fixed. Scheduled sampling is implemented
(`--sample-prob`). KD distillation is implemented (`--teacher`). Current facts:

- **`fp_v2` baseline ran 20k steps to completion** on the fixed harness with
  the correct cosine schedule. Best val_em = **0.2773 @ step 8000**, final
  0.1230, train loss ≈ 0.0025 (memorized train). Curve oscillates and peaks
  mid-run → exposure bias is confirmed and dominant. This is the number every
  subsequent experiment compares against.
- **`fp_ab.log` was a warmup-only, buggy-harness run** — superseded, keep only
  as historical evidence for the exposure-bias diagnosis.
- **No teacher checkpoint exists.** The teacher script has never been run
  (needs torch/HF + network for the SmolLM2-135M download).
- **No paraphrase data exists.** Ollama is not reachable.
- **nothing is currently training** (fp_v2 finished).

**Bottom line:** plain CE fp training memorizes the teacher-forced objective
but does not condition on the utterance in free-run. The next lever is
**scheduled sampling** (Step 2), already implemented and unproven.

---

## Step 1 — DONE: fp baseline with correct recipe

Ran to 20k steps: `--steps 20000 --batch 256 --warmup 2000 --ramp-frac 0.2
--fp` (`logs/fp_v2.log`, `checkpoints/fp_v2/`). Val EM curve:

| step | val_em | | step | val_em |
|------|--------|-|------|--------|
| 2000  | 0.1934 | | 12000 | 0.1738 |
| 4000  | 0.2012 | | 14000 | 0.2422 |
| 6000  | 0.1484 | | 16000 | 0.1016 |
| 8000  | **0.2773** | | 18000 | 0.1582 |
| 10000 | 0.1875 | | 20000 | 0.1230 |

**Outcome: exposure bias confirmed.** The 0.28 best is in the `0.10–0.40`
band of the decision table → scheduled sampling / KD is the next lever, NOT
QAT. Spot checks of the best checkpoint via `mid-training-eval.py` fail cleanly
on simple prompts (wrong intent, wrong span, spurious second action).

---

## Step 2 — fp + scheduled sampling (NEXT ACTION)

Step 1 confirmed exposure bias is dominant, so rerun with `--sample-prob 0.1`
(keep `--fp` for a clean fp comparison):

```bash
~/p3.11/bin/python3 train_student.py \
    --train data/train_a.jsonl \
    --val data/val.jsonl \
    --out checkpoints/fp_sample \
    --steps 20000 \
    --batch 256 \
    --eval-every 2000 \
    --ramp-frac 0.2 \
    --warmup 2000 \
    --sample-prob 0.1 \
    --teacher none \
    --fp
```

The sampling probability ramps from 0 to 0.1 over the first 10k steps, then
holds at 0.1. Two forwards per batch = ~2× compute (≈ 2.6 s/step, ~15 h).
Compare directly to Step 1: success = the EM curve exceeds fp_v2's 0.2773 best
and stays up (fp_v2 peaked then collapsed).

If sampling helps, also try `--sample-prob 0.2` to find the sweet spot. If it
*doesn't* help (or hurts), move on to KD (Step 3) without it.

---

## Step 3 — fp + KD (teacher must be trained first)

Before Step 3, train the teacher. This runs on PyTorch/HF, not MLX, and
requires network for the SmolLM2-135M download (~270 MB, one-time). Run on
the full 470k train set:

```bash
~/p3.11/bin/python3 train_teacher.py \
    --train data/train_a.jsonl \
    --val data/val.jsonl \
    --out checkpoints/teacher \
    --epochs 3 \
    --batch-size 32 \
    --grad-accum 8 \
    --max-seq-len 256 \
    --lr 2e-5
```

**Kill point:** if the teacher's val EM is below 95% after 3 epochs, the wire
format or data conventions have a problem. Fix the data before distilling.

With a trained teacher, rerun the best fp student recipe from Step 1 with KD:

```bash
~/p3.11/bin/python3 train_student.py \
    --train data/train_a.jsonl \
    --val data/val.jsonl \
    --out checkpoints/fp_kd \
    --steps 20000 \
    --batch 256 \
    --eval-every 2000 \
    --ramp-frac 0.2 \
    --warmup 2000 \
    --sample-prob 0.0 \
    --teacher checkpoints/teacher/best.pt
```

Compare `fp_v2` (Step 1) vs `fp_kd`. If KD adds <1 point EM, drop it — it's
complexity that hasn't earned its place and adds a torch dependency at training
time.

---

## Step 4 — QAT (ternary) from the best fp recipe

Once we have a fp student whose EM is meaningful, enable the ternary
quantizer. Use the identical recipe but change `--ramp-frac 0.2`:

```bash
~/p3.11/bin/python3 train_student.py \
    --train data/train_a.jsonl \
    --val data/val.jsonl \
    --out checkpoints/qat \
    --steps 20000 \
    --batch 256 \
    --eval-every 2000 \
    --ramp-frac 0.2 \
    --warmup 2000 \
    --sample-prob <best from Step 2, or 0.0> \
    --teacher <best.pt or none>
```

**Expected drop:** ARCHITECTURE.md budgets ~4 points from fp to ternary. If
the drop is larger, the ramp schedule needs revisiting.

Run the export on the best checkpoint and verify `fsm.c` self-test:

```bash
~/p3.11/bin/python3 export.py --model checkpoints/qat/best.npz --out build/model.tern
cc -DFSM_TEST -Wall -Wextra -o fsm_test fsm.c && ./fsm_test
```

---

## Step 5 — scale data, paraphrase, full run

With the student recipe validated (Step 1–4 above), scale up:

1. **Unblock paraphrase.** Either:
   - Start Ollama: `ollama serve`, then `paraphrase.py --in data/train_a.jsonl --out data/train_b.jsonl --limit 150000 --per-row 2`
   - Or skip Stage B for now: the template corpus is already 470k, and
     paraphrase only adds naturalness diversity. For the f1 build, pure
     template data is fine.

2. **Scale Stage A** to 1M if the template corpus proves too small for the
   student to generalize (measured by the held-out entity gap in eval.py).

3. **Full training run** on the best recipe, 20k+ steps, with the final data.

---

## What NOT to do yet

- **Don't deploy to ESP32.** That is Phase 5 in PLAN.md. Wait until the fp
  student has meaningful EM and the quantized student is within budget.
- **Don't tune the hidden-state γ term.** Deferred by design (BPE vs word
  alignment). Only revisit if logit-level KD (Step 3) is clearly insufficient.
- **Don't touch the FSM or serializer.** Both self-test pass. Fix them only
  if a real val example fails to decode.
- **Don't commit the STATUS.md "Next steps" section's incremental updates
  until an actual run completes.** Update STATUS.md in-place as steps land,
  but only commit when a run boundary is reached.

---

## Running in parallel (while Step 2 trains)

- Start Ollama and run `paraphrase.py` on a 20k subset as a smoke test.
- Review `export.py` in detail: verify the int8-embedding export path and
  the trit-packing logic. This hasn't been tested end-to-end and is the most
  likely place a latent off-by-one hides.
- Draft the ESP32 firmware skeleton (build system, partition table, PSRAM
  init) while the student trains — no dependency on model quality.

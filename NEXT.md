# NEXT.md — What to do now

_Last updated: 2026-09-01_

This is a working plan, not a design doc. It picks up after commit `0cd1536`
(all bug fixes, scheduled sampling, KD, model.py reconciliation). Nothing is
currently training.

---

## Where we actually are

All pipeline code is written and runnable. All known correctness bugs are fixed.
Scheduled sampling is implemented (`--sample-prob`). KD distillation is
implemented (`--teacher`). The training log is stale:

- **`fp_ab.log`**: ran with `--ramp-frac 100 --warmup 3000 --steps 3000`, so
  the *entire* 3000-step run was still in warmup (lr never decayed) and the
  quantizer never ramped. This means the "exposure bias" diagnosis is valid
  (train EM ~0.005 with free-run 2/30 intents), but the run never exercised the
  cosine schedule or the ternary ramp — it is not a fair fp baseline.
- **val_em=0.0234 was measured on the buggy harness** (pointer decode against
  `"x"` text). The true EM with the fixed harness could be higher or lower.
- **No teacher checkpoint exists.** The teacher script has never been run.
- **No paraphrase data exists.** Ollama is not reachable.
- **Full training has never been run for 20k+ steps.** The 3000-step run
  stopped at 6400/20000 in a prior QAT attempt that was also on the buggy
  harness.

**Bottom line:** we have never run the student on the *correct* recipe
(cosine decay + quantizer ramp + correct eval harness). That is the single
most important thing to do next.

---

## Step 1 — fp ablation with correct recipe (highest priority)

Run the fp student (no quantization) through a complete training run with the
right schedule, and measure val EM on the fixed harness. This is the baseline
against which everything else is compared.

```bash
~/p3.11/bin/python3 train_student.py \
    --train data/train_a.jsonl \
    --val data/val.jsonl \
    --out checkpoints/fp_v2 \
    --steps 20000 \
    --batch 256 \
    --eval-every 2000 \
    --ramp-frac 0.2 \
    --warmup 2000 \
    --sample-prob 0.0 \
    --teacher none \
    --max-len 256
```

**Why this is the right command:**
- `--ramp-frac 0.2`: quantizer ramps in over the first 4000 steps. In fp mode
  the ramp doesn't matter (ramp only affects ternary weights, which are just
  regular weights in fp), but this keeps the config canonical so the fp and QAT
  runs are identical except for `--ramp-frac`.
- `--warmup 2000`: 2000-step warmup, then cosine decay over 18k steps. This is
  the schedule the paper specifies. The old fp_ab.log never left warmup.
- `--sample-prob 0.0`: no scheduled sampling yet — this isolates the baseline
  exposure bias on the fixed harness.
- `--teacher none`: no KD yet — pure hard-label CE.

**Run this before anything else.** It takes ~7.5 hours at ~1.3s/step. Run it
overnight. The expected outcomes:

| EM result | interpretation | next step |
|-----------|----------------|-----------|
| val_em ≈ 0 | harness still broken or model can't learn at all | debug harness, not the model |
| val_em 0.02–0.10 | exposure bias remains dominant | enable scheduled sampling (Step 2) |
| val_em 0.10–0.40 | meaningful learning, ceiling room for scheduled sampling | run KD ablation (Step 3) |
| val_em > 0.40 | student is learning; exposure bias largely solved | move straight to QAT (Step 4) |

**Monitor** `best_em` and the per-kind EM breakdown (`atomic` should be
strongest, `triple` weakest). Also watch whether EM peaks mid-run then declines
(overfitting → lower the lr or increase warmup).

---

## Step 2 — fp + scheduled sampling (if Step 1 EM < 0.10)

If exposure bias is still dominant, rerun with `--sample-prob 0.1`:

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
    --teacher none
```

The sampling probability ramps from 0 to 0.1 over the first 10k steps, then
holds at 0.1. Two forwards per batch = ~2× compute. Compare directly to
Step 1.

If sampling helps, also try `--sample-prob 0.2` to find the sweet spot. If it
*doesn't* help (or hurts), move on to KD (Step 3) without it — the exposure
bias diagnosis might already be overcome by the fixed harness and proper
schedule.

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

## Running in parallel (while Step 1 trains)

- Start Ollama and run `paraphrase.py` on a 20k subset as a smoke test.
- Review `export.py` in detail: verify the int8-embedding export path and
  the trit-packing logic. This hasn't been tested end-to-end and is the most
  likely place a latent off-by-one hides.
- Draft the ESP32 firmware skeleton (build system, partition table, PSRAM
  init) while the student trains — no dependency on model quality.

# NEXT.md — What to do now

_Last updated: 2026-09-03 (V5 fp16 deployment bring-up)_

This is a working plan, not a design doc. It picks up after commit `0cd1536`
(all bug fixes, scheduled sampling, KD, model.py reconciliation) and the
**completed fp_v2 baseline** (Step 1, see below).

---

## V5 deployment bring-up — next steps

The deployment path is now plain fp16, not ternary/QAT. fp16 storage is
lossless for `v5crf_mask/best.npz`; the 1,414,502-byte `model.bin` is decoded
to fp32 for computation. `export_v4.py` emits `model.toc`, and the C runtime
loads checkpoint `cos`/`sin` verbatim rather than recomputing RoPE.

1. **Fix host parity.** Run `~/p3.11/bin/python3
   firmware/tests/run_host_test.py --dir /tmp/export_v5_fp16 --n 1500`. The
   current result is 56.7% intent versus the 59.3% reference, with large
   C-vs-MLX logit errors. Isolate the first block-0 divergence in this order:
   RMSNorm, q/k/v, RoPE, attention scores/softmax, value combine, o-projection,
   then block 1 and the heads.
2. **Lock acceptance.** Require 59.3% intent, 39.8% person, and approximately
   1e-3 maximum C-vs-MLX logit error with matching argmaxes. Re-run at both
   `-O0` and `-O2`.
3. **Clean the host path.** Remove temporary dump programs and debug-only
   output, while retaining the host regression harness.
4. **Wire ESP-IDF.** Add the component, model data partition, embedded
   `model.bin`/`model.toc`, tokenizer, and a `main` smoke test.
5. **Flash and validate.** Build and flash the connected XIAO ESP32-S3 N8R8;
   compare serial predictions against the host reference.
6. **V6 later.** If unseen entities matter, replace the fixed word vocabulary's
   UNK path with subword or character n-gram tokenization. Keep this separate
   from deployment bring-up.

---

## V5 — COMPLETE (schema simplification + length-invariance + chain→sentence)

All V5 work is done and committed on branch `V5`:

1. **Schema simplification** — dropped `object`/`file`, merged
   `recipient`→`person`. Families: `location/person/message/duration`
   (9 BIO tags). Retrained CRF → `checkpoints/v5crf/`.
2. **Length-invariance fix** — attention key-padding mask. Model now gives
   identical logits unpadded vs padded (was a padding-leak that made on-device
   single-utterance inference wrong: "stop"→SHOW). Retrained →
   `checkpoints/v5crf_mask/best.npz`. Honest intent **90.2%**, span F1 0.630.
3. **Chain → sentence segmentation** — `chain_seg.py` breaks a chain into
   atomic sentences gold-free; `chain_eval.py` verifies end-to-end:
   intent/clause 95.9%, **model-only 98.2% intent / 94.5% intent+slots**.
   Approach B (learned boundaries) not needed.

See STATUS.md for full detail and the honest numbers.

### Next steps (in order of leverage)

1. **RESOLVED — "OOD / MOVE-HANDOVER confusion" is a UNK-vocab artifact, not
   semantic.** `classify_vocab.py` (offline, no Qwen) + `build_embed_init.py`
   now produce `embed_init.npz`, but analysis shows it CANNOT fix the val gap:
   17.9% of val tokens are OOV and collapse to the single UNK id, so unseen
   entities/verbs are invisible to the model no matter the embedding. Per-intent
   OOV share (MOVE 100%, SHOW 88%, CLEAN/HANDOVER 53%, PLAY/STOP/WAIT 0%)
   exactly matches accuracy ordering — the model is perfect on in-vocab input.
   See STATUS.md "Model-quality investigation". Decision pending on whether to
   (a) accept the hard-val stance, or (b) add subword/char tokenization for OOD
   named entities.
2. **IN PROGRESS — QAT / ternary export → ESP32.** Measured: V5 is only ~706k
   params → ~0.23 MB ternary blob (vs old 3.63 MB). **Plain PTQ ternary
   collapses the model (0% intent, verified via new `quant_eval.py`) — so full
   QAT retraining is mandatory.** There is no PTQ shortcut. Plan: add `--t`
   ramp (straight-through ternarize of linears) to `v4_train.py`, anneal 0→1,
   validate vs fp within ~2 pts intent / ~0.02 span F1, then write `export_v4.py`
   (maps `V4Model` → the existing blob/packing contract incl. CRF). See STATUS
   "Deployment" section.
3. **ESP32 firmware skeleton — DONE (review).** `firmware/` README + `v5_model.h/
   .c` skeleton implements the numeric quant/primitives (5-trits/byte unpack,
   int8 embed, fp16 scales, RMSNorm, ternary matmul, CRF scaffold), compiles
   clean. NOT yet wired: manifest loader / tensor pointer mapping pending
   `export_v4.py`; IDF component + partition + `main` stub. See firmware/README
   TODO gates.


### Chain → sentence decomposition — RESOLVED (approach A)

Approach A (symbolic splitter) implemented and validated — see STATUS.md.
Approach B (learned sentence-boundary model) evaluated and **declined**:
connective set is effectively closed, model-only clause accuracy is ~98%, and
the only segmentation gaps are corpus artifacts / a comma-appositive template
that a learned boundary head would not robustly fix. Reconsider only with real
deployment evidence of unseen connectives.

---

## V4 — joint intent + slot-tagger (DONE, superseded by V5)

The generative FSM student can't learn semantic roles. On `V4` we switched to a
joint intent + BIO slot-tagging model over **atomic** commands. Status:

- **Intent works: 85.9% atomic intent** (STOP 100%, WAIT 99%, SHOW 96%,
  MOVE 89%; weak spots PLAY 62%, HANDOVER 71%). Model = 879k params,
  d=192/L=2, 64k balanced rows. This is the branch's win.
- **CRF slot head (the chosen fix) is DONE:** span F1 ~0.26→~0.52, value
  extraction location 5%→90%, message 0%→38%, person 14%→34%. Committed as
  `873e6b9`. Remaining zero-families (file/object) drove the V5 schema change.

### Re-usable pieces

`v4_decompose.py` (chain→atomic, alignment-verified), `v4_data.py` (per-intent
balanced atomic set), `v4_model.py` / `v4_train.py` (trainer with CRF slot
loss + `--use-crf/--wd/--score-norm/--embed-init`, best-checkpoint saving),
`v4_eval.py` (per-intent + per-family value extraction, `--use-crf` Viterbi),
`build_embed_init.py` (semantic-category embedding init),
`classify_vocab.py` (offline vocab→category builder from `corpus.py`, no Qwen).

Docs updated: STATUS.md V4 + V5 sections; NEXT.md V5 section.

---

<!-- main-branch generative investigation continues below -->

## Where we actually are

Step 1 (clean fp baseline) is **done**. All pipeline code is written and
runnable. known correctness bugs are fixed. Scheduled sampling is implemented
(`--sample-prob`). KD distillation is implemented (`--teacher`). Current facts:

- **`fp_v2` baseline ran 20k steps to completion** on the fixed harness with
  the correct cosine schedule. Full-val EM on the best checkpoint = **0.0665**
  (deduped n=3641; intent-acc 0.19, triple 0.005), train loss ≈ 0.0025
  (memorized). The in-run 512-subset "0.2773 best" was a large over-read — the
  eval loop is fixed to full-val + per-kind now, and eval.py dedupes val by
  text (5000 lines → 3641 unique) exactly like `load_rows`.
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

## Step 2 — DONE: fp + scheduled sampling

Ran with `--sample-prob 0.1 --fp` (`logs/fp_sample.log`). Died at step 15700
(OOM from concurrent eval work), relaunched from step 14000. In-training full-val
EM trajectory (training-loop evaluate, bundles reject as atomic):

| step | val_em |
|------|--------|
| 2000  | 0.0467 |
| 4000  | 0.0313 |
| 6000  | 0.0637 |
| 8000  | **0.0703** |
| 10000 | 0.0621 |
| 12000 | 0.0662 |
| 14000 | 0.0651 |

**Outcome: marginal over fp_v2.** The in-training eval shows ~0.07 peak vs
fp_v2's 0.0665 honest full-val. The scheduled sampling with ramp=1.0 and
sample-prob=0.1 did not significantly improve generalization. The model still
memorizes rather than learning slot-filling rules.

**Root cause identified:** the 11.1M from-scratch model cannot learn semantic
roles (person/location/object) from 470k template sentences. Spot-checks show
the model either guesses wrong intent, fills slots with wrong spans, or gives
up (UNAVAILABLE). This is a capacity + pre-training gap, not a training loop
issue.

**honest intent+required-slot eval:** ran `eval_required.py` (ignores optional
slots) on both checkpoints. See `eval_required.md` for full per-intent breakdown.

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

**Outcome: exposure bias confirmed (and in-run numbers were over-reads).**
The training loop's `n_eval=512` subset showed 0.2773 "best" and a
non-deduped full run showed 0.16 — the honest deduped full-val EM on the best
checkpoint is **0.0665** (intent-acc 0.19, triple 0.005). Spot checks of the
best checkpoint fail cleanly (wrong intent/span/extra action). The decision:
**fix exposure bias first — scheduled sampling / KD, NOT QAT.**

---

## Step 2 — DONE: fp + scheduled sampling

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
Compare directly to Step 1: success = full-val EM clearly exceeds fp_v2's
**0.0665** and stays up (fp_v2 collapsed with chain length).

If sampling helps, also try `--sample-prob 0.2` to find the sweet spot. If it
*doesn't* help (or hurts), move on to KD (Step 3) without it.

**Result: marginal improvement, model hits semantic ceiling.** In-training peak
EM 0.0703 at step 8000 (vs fp_v2 honest 0.0665). The model cannot generalize
slot-filling to unseen phrasings — this is a capacity/pre-training issue, not
a training loop issue. See Step 6 for viable paths.

---

## Step 3 — KD (deferred, high uncertainty)

Train a teacher on PyTorch/HF, then distill into the student. This could help
the student learn smoother probability distributions over actions/slots, but
the root issue (no semantic understanding) may not be addressable by KD alone.

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

---

## Step 4 — QAT (ternary) — deferred

Only viable after Steps 1-3 produce a meaningful fp baseline. The ternary
quantizer (export.py) is ready and tested. The quantization-aware training
ramp and export pipeline are complete. Once the fp student EM is meaningful,
enable ternary and verify the drop is within the ~4 point budget.

---

## Step 5 — data scale — deferred

Paraphrase augmentation (Ollama) and template expansion (1M row corpus). Only
worth pursuing if the model has the capacity to benefit from more data.

---

## Step 6 — Viable ESP32 paths forward

The core finding: the 11.1M from-scratch model cannot learn semantic roles
(person/location/object) from 470k template sentences. Training loop
improvements (scheduled sampling, KD) cannot fix this — the model lacks
the foundational understanding to generalize. Three viable paths:

### Path A: Scale the from-scratch model

Push the architecture to fit ESP32-S3 N16R8 limits:

| Config | d_model | L | heads | ffn | params | flash (tern) | PSRAM |
|--------|---------|---|-------|-----|--------|-------------|-------|
| Current | 384 | 6 | 6/2 | 1024 | 11.1M | ~3.6 MB | ~5 MB |
| Mid | 448 | 8 | 7/2 | 1120 | ~18M | ~5.8 MB | ~7.5 MB |
| Max | 512 | 8 | 8/2 | 1280 | ~21M | ~6.7 MB | ~8.5 MB |

**Pros:** no external dependencies, pure ESP32 deployment, export pipeline
already works. **Cons:** from-scratch learning is sample-inefficient; may still
not learn semantic generalization at 21M. Worth one controlled ablation (18M vs
11.1M) to confirm whether capacity is the bottleneck.

### Path B: Pre-compute encoder embeddings (offline distillation)

1. Run a pre-trained LM (SmolLM2-135M or similar) on the full training corpus
   offline (on a server/Mac), dump hidden states to disk.
2. Train only the small action head (3 layers, ~2M params) on top of the frozen
   embeddings. This fits in MLX easily.
3. On-device: run the frozen encoder + small head together. The encoder
   processes the utterance, the head extracts intent + slots from the
   representations.

**Pros:** the encoder already understands semantic roles; the head only needs
to learn the action grammar. Highest expected EM. **Cons:** encoder too large
for ESP32-S3 PSRAM (SmolLM2-135M = ~270 MB). Would need a much smaller
encoder (e.g., 10-20M params) that fits in flash (~6-7 MB ternary), or
streaming decode from flash. More firmware complexity.

### Path C: Hybrid (small on-device + external for hard cases)

1. Keep the 11.1M model for common/atomic commands (MOVE, STOP, WAIT).
2. For multi-action chains and novel phrasings, fall back to an external model
   (server/API call) or accept lower accuracy.
3. Route by confidence: if the on-device model's top-1 logit margin is below
   a threshold, send to external.

**Pros:** pragmatic, works with current model quality. **Cons:** requires
connectivity for hard cases; adds latency for fallback; the threshold needs
tuning.

### Recommendation

**Try Path A first** (one ablation at 18M to confirm capacity is the
bottleneck). If 18M doesn't improve semantic generalization, Path B is the
highest-ROI path but requires solving the encoder-on-device problem. Path C
is the pragmatic fallback if neither path works within budget.

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

## Running in parallel (while waiting)

- Review `export.py` self-test and packing contract (done, commit pushed).
- Draft the ESP32 firmware skeleton (build system, partition table, PSRAM
  init) — no dependency on model quality.
- Run `eval_required.py` on all checkpoints to get intent+required-slot stats
  (not full EM, but shows where the model succeeds/fails per intent).

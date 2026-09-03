# STATUS — PlanCore-11M (training investigation)

_Last updated: 2026-09-03_

This file documents what exists, what we found while debugging training, and
the concrete next steps. It is the working record for the solo session; see
`PLAN.md` (roadmap) and `ARCHITECTURE.md` (normative design) for the full plan.

---

## Deployment — fp16 host bring-up (IN PROGRESS)

The current deployment path uses plain fp16 storage and fp32 computation.
`export_v4.py --mode fp16` writes `model.bin`, `manifest.json`, and binary
`model.toc`; `firmware/src/v5_model.c` decodes the blob and runs the encoder,
heads, and CRF. The checkpoint's `cos`/`sin` are loaded verbatim because the
checkpoint frequencies differ from the default RoPE helper.

The host kernel is not at parity yet. The latest run is 56.7% intent (851/1500)
versus the 59.3% reference, with intent-logit max error 6.477 and slot-logit
max error 8.864. The first known divergence is in block-0 attention. Do not
wire or flash until the stage is isolated and fixed.

Required gates:

1. Compare C against NumPy/MLX for RMSNorm, q/k/v, RoPE, scores/softmax, value
   combine, o-projection, block 1, and the output heads.
2. Re-run the host test at `-O0` and `-O2` after the fix.
3. Require 59.3% intent, 39.8% person, and roughly 1e-3 C-vs-MLX logit error.
4. Remove temporary dump helpers, then add the IDF component, model partition,
   tokenizer, and `main` smoke test.
5. Build, flash the XIAO ESP32-S3 N8R8, and compare serial predictions with the
   host reference.

The fixed 310-word vocabulary maps unseen words to one UNK id. Subword or
character n-gram tokenization is the separate V6 quality track.

---

## V5 — drop file/object, merge recipient→person (COMPLETE + finding)

Branch `V5` implemented the schema simplification: PLAY became intent-only,
HANDOVER dropped `object` and merged `recipient` → `person`, and `object`/
`file` were deleted. Final families: `location`, `person`, `message`,
`duration` → `SLOT_LABELS` = 9 BIO tags (was 15).

### V5 results (retrained with CRF, d=192/L=2, 707k params, 14 epochs)

`checkpoints/v5crf/best.npz`, measured with the same **batched/padded**
inference as training (`v4_train.evaluate`):

| metric | value |
|---|---|
| **intent_acc** | **47817/47819 = 99.996%** |
| slot token acc | 0.915 |
| span F1 | 0.769 |
| value extraction | duration 100%, location 85%, message 62%, person 37% |

This is a dramatic jump from V4 (intent 85.9%, span F1 ~0.5, location 5%
without CRF): dropping the 0%-capable families (file/object) let the model
focus and it now extracts all four remaining families.

### CRITICAL finding — model is NOT length-invariant (padding leak) — FIXED

The pre-fix training-loop `v4_train.evaluate` reported ~100% intent, but
`v4_eval.py` (each row **unpadded**, natural length = the on-device case)
reported only **20% STOP** and 89% overall. Root cause and fix:

- **Root cause:** intent pooling masks padding out of the *pool*, but the
  bidirectional self-attention was NOT masked over padding, so padding tokens
  (all id 0 = CLS/UNK) leaked positional signal into the real tokens. Training
  always padded via `collate`, so the model embedded "there are pads after me";
  run unpadded (one utterance at a time, as on-device) it behaved differently.
  `"stop"` unpadded → SHOW; padded → STOP.
- **Fix (v4_model.py):** threaded a key-padding mask through
  `Attention.__call__` / `Block.__call__` / `V4Model.__call__`, applying
  `-1e4` to attention scores over padding keys (`pad_mask [B,1,1,T]`). The
  model is now **length-invariant**: verified unpadded forward == padded
  forward (logit diff = 0.0).

### V5 results AFTER the attention-mask fix (honest, consistent)

Retrained CRF with masking → `checkpoints/v5crf_mask/best.npz`. Both the
batched (training-style) and unpadded (`v4_eval`) measurements now **agree
exactly per-intent** — full length-invariance confirmed. Honest numbers:

| metric | value |
|---|---|
| **intent_acc** | **90.2%** (STOP 100, CLEAN 100, PLAY 100, WAIT 100, HANDOVER 96, SHOW 86, MOVE 58) |
| span F1 | 0.630 |
| value extraction | duration 100%, message 74.5%, person 44.6%, location 32.3% |

**The previous ~100% was inflated** by the padding leak; ~90% is the honest
capability. The gap (esp. location 32.3%, MOVE 58%) is dominated by **OOD
generalization**: the val split is deliberately seen-never (held-out entities
& templates). MOVE's val utterances ("go to the conservatory", "roll to the
annexe") use wholly novel surface forms, so a small from-scratch model can't
generalize them — it over-predicts HANDOVER. This is documented OOD behavior,
not a training bug.

---



1. `dsl.py` — `SLOTS`: PLAY `()` , HANDOVER `optional ("person")`; `ALL_SLOTS`
   drops object/recipient/file; `SLOT_DESC` folds recipient→person; `INTENT_DESC`
   updated; `SLOT_ORDER["HANDOVER"]==("person",)`; tests updated.
2. `corpus.py` — PLAY templates slot-less; HANDOVER object-less with
   `person`+speaker variants; removed `object`/`file` entity pools.
3. `v4_model.py` — `SLOT_LABELS` → `["location","person","message","duration"]`
   (N_SLOT_CLASSES 15→9).
4. `v4_train.py` / `v4_eval.py` — `SLOT_FAMILY` drops object/recipient/file
   (v4_eval auto-adapts).
5. `v4_data.py` — fixed missing `collections` import (pre-existing latent bug
   that would crash `main()`).
6. Regenerated balanced data (train 61k / val 47.8k rows, 9 BIO tags) and
   retrained `checkpoints/v5crf/`.

---

## Chain → sentence segmentation (approach A, DONE)

`chain_seg.py` breaks a spoken multi-action chain into standalone atomic
sentences WITHOUT gold labels, each fed to the (now length-invariant) V5 model.

Splits on connectives (`and`, `and then`, `then`, `also`, commas, periods,
semicolons — longest first), strips leading bare connectives, resolves
anaphora ("it"/"this"/"that" → most recent location/person, "here" → referent).

**End-to-end verification** (`chain_eval.py`, 600 corpus chains, gold-free
segmentation → decompose → classify each clause with `v5crf_mask/best.npz`):

| metric | value |
|---|---|
| segmentation count correct | 93.7% |
| intent / clause | 95.9% |
| intent + slots / clause | 91.2% |
| **model-only intent** (on correctly-segmented) | **98.2%** |
| **model-only intent + slots** | **94.5%** |

**Decision: approach B NOT needed.** The remaining segmentation failures are
(i) corpus bare-space joins — degenerate concatenation with no connective word
("travel to the utility room make your way to the greenhouse"), not realistic
spoken input; and (ii) appositive relative clauses from a movep template
("go to mia, she is at my desk" — one MOVE, my segmenter over-splits at the
comma). Neither is helped by a learned boundary model; the connective set is
closed and model-only accuracy is already ~98%. Revisit B only if a real
deployment surfaces unseen boundaries.

Self-test: `~/p3.11/bin/python3 chain_seg.py` (10 cases, all pass).
Eval: `~/p3.11/bin/python3 chain_eval.py --ckpt checkpoints/v5crf_mask/best.npz`

---

## Model-quality investigation — the OOD gap is a UNK-vocab artifact, not embeddable

The pre-existing "semantic embedding init" idea (`build_embed_init.py`) presumes
a `data/v4/vocab_cats.json` (word → LOCATION/PERSON/... category) produced by
`classify_vocab.py` from a Qwen/Ollama model. **Neither file exists in this
repo**, and I verified Qwen is not present. Pivot: `classify_vocab.py` now
builds `vocab_cats.json` **fully offline** by reading the entity pools /
durations / action verbs directly from `corpus.py` (no LLM). `build_embed_init.py`
then emits `data/v4/embed_init.npz` (310×192, matches the train vocab).

**But the pivot revealed the approach is orthogonal to the real problem.** The
word vocab is tiny (310) because the corpus is template-based, and the val set
is adversarial by design (entity pools split train/val, plus val-only `move_v`
templates using verbs "navigate/roll/trundle"). Quantified:

- 17.9% of all val tokens are **OOV** → all collapse to the single UNK id (0)
  at inference (`v4_eval.py:81`), so the model literally cannot see the word.
- Per-intent OOV row share: **MOVE 100%**, SHOW 88%, CLEAN/HANDOVER 53%,
  **PLAY/STOP/WAIT 0%**.

This **exactly** matches the measured accuracy ordering (PLAY/STOP/WAIT 100%,
CLEAN 100%, HANDOVER 96%, SHOW 86%, MOVE 58%): the model is flawless on
in-vocabulary input and degrades precisely as unseen tokens rise. Because OOV
entities all share one UNK embedding, **no embedding init can separate them** —
it only re-labels in-vocab words, which are already handled.

**Conclusion:** the documented "MOVE-HANDOVER confusion / weak location span"
is a **tokenization-vocabulary artifact**, not a semantic-learning gap. Real
options, in order of value:
1. **Decide the eval stance.** The val pool-split deliberately tests unseen
   named entities/verbs; a from-scratch fixed-vocab model is legitimately
   expected to miss them. If the goal is realistic commands (known rooms/names),
   the model is already strong on those surfaces.
2. **Subword / char n-gram tokenization** so novel names decompose ("conservatory"
   → known substrings) instead of collapsing to UNK. Larger change; only worth
   it if OOD named entities are a real deployment target.
 3. Semantic embed init (`--embed-init embed_init.npz`) only as a minor
    regularization for in-vocab slot-role disambiguation; **not** a fix for OOD.

---

## Deployment — QAT/ternary export + ESP32 (PLAN; measure first, user to review)

### Model footprint (huge win)

V5 (`v5crf_mask/best.npz`) is only **~706k params**. Ternary + int8 + fp16
blob ≈ **0.23 MB** — vs the old 11.1M generative model at 3.63 MB. Fits
ESP32-S3 internal flash easily with no PSRAM pressure. `export.py` currently
targets the old generative `ModelConfig`; it does NOT understand `v4_model.V4Model`
(8 linear heads, `{q,k,v,o}`/`w1,w2,w3`, CRF transition, `blocks.*` naming).

### Measured: plain PTQ ternary collapses the model → QAT is mandatory

`quant_eval.py` ternarizes every 2D linear weight channel-wise at the same
contract `export.ternary_channel` uses, then re-runs the exact `v4_eval`
intent/slot harness on the V5 model.

| config | intent (MOVE subset) | person value |
|---|---|---|
| fp (validation) | 59.3% | 39.8% |
| ternary ramp=1.0 | **0%** | 13.8% |
| ternary ramp=0.5 | **0%** | 10.5% |

`--fp` reproduces `v4_eval` (loader verified), so the 0% is a genuine
collapse: the fp-trained V5 weights are not ternarizable without retraining.
**There is no PTQ shortcut — full QAT retraining is required.**

### QAT plan (QAT support in the V4/V5 path, currently absent)

`v4_model.py`/`v4_train.py` are pure-fp; the QAT ramp (`ternary_ste`,
`--fp/--t`) exists only in `train_student.py`/`model.py` for the old
generative model. Design:

1. Add a `--t <ramp>` flag to `v4_train.py` that replaces each linear
   weight with `ternary_ste(w, t)` (straight-through: forward uses the
   ternary approx, backward passes the fp gradient) — fused as `x @
   ternary_ste(w,t).T` in `Attention`/`FFN`/`intent_head`/`slot_head`.
2. Anneal `t` 0.0 → 1.0 over the run (freeze last few epochs at 1.0) so
   weights settle into ternarizable values; keep embedding fp, norms fp.
3. Validate the QAT checkpoint with `quant_eval.py --ramp 1.0` plus the fp
   `v4_eval.py` — **budget target: intent within ~2 points of fp, span F1
   within ~0.02** (measured per the conversation's acceptance bar).
4. Write a small `export_v4.py` (or extend `export.py`) that maps `V4Model`
   tensors to the existing blob: ternarized linears via `ternary_channel`,
   embedding int8, norms/biases fp16, and the CRF `trans`/`log_mask`/start-
   forbid collapsed into an exported `crf` struct (additive 9×9 fp16 + a
   `SLOT_LABELS` family table so Viterbi runs on device).
5. Round-trip test: unpack via `unpack_trits`/`int8_rows` reference and
   re-run `quant_eval` to confirm the packed number matches the QAT-checkpoint
   score.

`quant_eval.py` (new, committed) is the harness for steps 1-3.

---



## V4 branch — joint intent + slot-tagger pivot (atomic commands)

The generative end-to-end FSM student ceilings at ~7-8% even intent-only
(section below). On the `V4` branch we replaced it with the classic **joint
intent classification + BIO slot-tagging** architecture (shared bidirectional
encoder, two heads) targeting **atomic** commands only; chains are deferred.

Data: decomposed the original pair/triple chains into atomic clauses
(`v4_decompose.py`, 0 alignment mismatches on 2000-chain validation) giving a
850k-clause atomic pool, then balanced 8,000 rows per intent (`v4_data.py`):
64k train / 2,035 val. Labels are per-family BIO tags (duration_amount+unit
share one `duration` family).

Results (d=192/L=2, 879k params, class-weighted + slot@1.5 loss):

| metric | value |
|---|---|
| **intent acc (atomic, exact)** | **85.9%** |
| MOVE / CLEAN / SHOW / WAIT / STOP | 89 / 86 / 96 / 99 / 100% |
| HANDOVER / PLAY / UNAVAILABLE | 71 / 62 / 40% |
| slot token acc | ~70% |
| slot span F1 | **~0.19-0.31 (unstable)** |
| slot value extraction | duration 100%; person 14%, recipient 21%, location 5%, message 4%, **object 0, file 0** |

**Verdict:** the intent head is the breakthrough — 86% atomic intent vs the
generative model's 7-8% — and is small enough for ESP32 (879k params, ~1199
word vocab). The **slot head does not reliably extract spans**: it collapses
toward predicting `O` everywhere (under-detection), so value extraction for
object/file/location/message is ~0% while duration (single short span) hits
100%. Span F1 is unstable across epochs (0.18-0.31), the signature of
periodically re-entering the all-`O` local minimum. Raising slot-loss weight
(`--slot-w 4.0`) degraded intent to 76-81% without fixing spans; raising
capacity (d=256/L=3, 1.89M) gave a noisy 0.31 peak but no stable gain.

**Next levers:** CRF/structured slot loss to enforce valid BIO transitions,
self-attention over slot head or a dedicated slot decoder, larger
pretrained-backed encoder, or accept intent-only (UNAVAILABLE fallback for
low-confidence slot extraction). See NEXT.md V4 section.

Relevant files: `v4_model.py`, `v4_train.py`, `v4_data.py`, `v4_decompose.py`,
`v4_eval.py`, `v4_inspect.py`. Data is gitignored; rerun `v4_data.py` /
`v4_decompose.py` to regenerate. Eval note: must tokenize with the vocab
built from train rows (word ids differ by frequency ordering) or load yields
random results.

---

## Where things stand

The **full student-training pipeline is built and runs end-to-end**, but the
student does **not learn to map a spoken command to a correct plan**, even when
the metric is relaxed to **intent-only**. Three fp runs (clean baseline, then
scheduled sampling) all plateau at the same ceiling:

| metric | fp_v2 | fp_sample |
|---|---|---|
| exact plan EM | 6.7% | ~7.0% in-run |
| intent + required slots | 5.1% | 3.1% |
| intent-chain only (no slots) | 7.9% | 7.0% |
| chain length correct | 64% | 57% |
| MOVE intent (best intent) | 33% | 30% |

The model memorizes the training set (train CE ≈ 0.0025) but does not
generalize to unseen phrasings: it can't bind the correct span to a slot or
recognize semantic roles (person vs location vs object). This is a **capacity /
pre-training gap**, not a training-loop bug. Scheduled sampling (Step 2) was
run and did NOT help. See REPORT.md for the full writeup; NEXT.md Step 6 lists
the viable paths forward (scale to ~18M, pre-computed encoder embeddings, or
hybrid on-device + external).

| Component | Status |
|---|---|
| Data generation (`corpus.py`) | Done. `data/train_a.jsonl` (470,113 unique) + `data/val.jsonl` (3,641 unique). 55/28/9/8 mix. |
| DSL + wire serialization (`dsl.py`, `serialize.py`) | Done, unit-verified (vocab=4388). |
| Student model (`train_student.py`; `model.py` trimmed to shared `ModelConfig`) | Done. 11.1M params, tied embed/LM-head, GQA, SwiGLU, RoPE. |
| QAT training (`train_student.py`) | **fp_v2 baseline complete**: full-val EM **0.0665** (deduped n=3641; 512-subset + non-deduped running over-read to 0.28/0.16), train loss ~0.0025. **fp_sample (scheduled sampling 0.1) complete**: in-run peak 0.070 @8k, no meaningful gain. Exposure bias is NOT the bottleneck — semantic capacity is. Eval loop reports full-val + per-kind EM; eval.py dedupes like load_rows and gained `--t` (fp decode). |
| Eval harness (`eval.py`) | Done; loader fixed (unflattens dot-key `.npz`, `model.update`); pointer decode + intent-denominator fixed. `eval_required.py` = intent+required-slot (no optional); `intent_only.py` = utterance intent chain (no slots). |
| Mid-training inspection (`mid-training-eval.py`) | New; interactive single-prompt eval against any checkpoint (`--t 0.0` fp / `1.0` ternary). |
| Export (`export.py`) | 3.63 MB blob verified on a checkpoint; trit-packing + int8 embedding round-trip exact; packing contract + `--selftest` added. |
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

## fp_v2 baseline run — COMPLETE

A clean fp run on the **fixed** harness and the **correct** schedule ran to
completion: `--steps 20000 --batch 256 --warmup 2000 --ramp-frac 0.2 --fp`
(`logs/fp_v2.log`, `checkpoints/fp_v2/`). This supersedes the stale
`fp_ab.log` (that run was 100% warmup with no cosine decay and was scored on
the buggy pointer decode).

**Result — val EM curve (fixed harness):**

| step | val_em | | step | val_em |
|------|--------|-|------|--------|
| 2000  | 0.1934 | | 12000 | 0.1738 |
| 4000  | 0.2012 | | 14000 | 0.2422 |
| 6000  | 0.1484 | | 16000 | 0.1016 |
| 8000  | **0.2773** | | 18000 | 0.1582 |
| 10000 | 0.1875 | | 20000 | 0.1230 |

**CAVEAT — the in-run numbers above are a 512-row subset AND the val file is
pre-deduped here.** `data/val.jsonl` has 5,000 lines but only **3,641 unique
texts** (263 copies of "find me a flight"); `load_rows` dedupes, so the honest
universe is 3,641. `eval.py` now dedupes identically. The honest **full-val
EM** on the best checkpoint (step 8000) is:

- **EM 0.0665** (n=3641) | intent-seq acc 0.1911
- by chain: atomic **0.146** | pair **0.054** | triple **0.005** | reject 3/5
- reject class is only **5 unique examples** in val — false-accept stats are
  statistically meaningless at that n (the earlier 0.588 was an artifact of
  counting duplicate reject lines)
- train loss ≈ 0.0025 → memorized the training set.

**Verdict: exposure bias is real and dominant.** The honest full-val EM is
~0.07, far below the 512-subset over-read (0.2773). EM collapses with chain
length (composition requires reading the whole utterance) and the model
overfits the teacher-forced objective without conditioning on the utterance in
free-run. Per-prompt spot checks of the best checkpoint fail (wrong intent /
wrong span / spurious extra action).

---

## Next steps (in order of leverage)

1. **DONE — fp_v2 baseline.** Clean fp run, 20k steps. Honest full-val EM
   0.0665 (deduped n=3641); train loss ~0.0025 (memorized). EM collapses with
   chain length.

2. **DONE — fp_sample (scheduled sampling, `--sample-prob 0.1`).** In-training
   full-val EM peaked at 0.070 @ step 8000, then ~0.06. Did NOT meaningfully
   beat fp_v2. Confirms the bottleneck is capacity/pre-training, not exposure
   bias.

3. **Semantic-ceiling verdict.** Even intent-only (no slots) reaches only ~8%
   whole-utterance; per-intent only MOVE beats chance. The model cannot learn
   semantic roles from 470k templates at 11.1M params. Cross off further
   training-loop tuning (scheduled-sampling sweet spot, KD) as primary levers.

4. **Viable paths forward (see NEXT.md Step 6):**
   - **A. Scale to ~18M** (d=448/L=8): one controlled ablation to confirm
     capacity is the bottleneck. Pure ESP32 deploy.
   - **B. Pre-computed encoder embeddings (offline distillation):** highest
     expected EM; needs encoder-on-device problem solved (very small encoder or
     flash streaming).
   - **C. Hybrid:** small on-device model for easy/atomic commands, external
     fallback for hard cases, gated by top-1 confidence.

5. **QAT / deployment only after a model with meaningful EM exists.** The
   ternary export pipeline (export.py) is done and round-trip verified; it is
   ready to consume whatever model emerges.

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
`ARCHITECTURE.md`, `PLAN.md`, `STATUS.md`, `NEXT.md`, `REPORT.md`,
`corpus.py`, `dsl.py`, `serialize.py`, `model.py`, `train_student.py`,
`train_teacher.py`, `paraphrase.py`, `eval.py`, `eval_required.py`,
`intent_only.py`, `mid-training-eval.py`, `export.py`, `fsm.h`, `fsm.c`.
V4 branch adds: `v4_model.py`, `v4_train.py`, `v4_data.py`,
`v4_decompose.py`, `v4_eval.py`, `v4_inspect.py`, `intent_atomic.py`.
V5 branch adds: `chain_seg.py` (chain→sentence segmenter),
`chain_eval.py` (end-to-end chain→classify harness).

# PlanCore v1 — chained intent extraction on ESP32-S3

Utterance in, up to 3 ordered actions out, no JSON anywhere. The model emits a
short token sequence in which slot values are span pointers into the input; the
firmware decodes that into `Action` objects via `serialize.decode`.

## 1. Model

Decoder-only, same family as v0 so the existing kernels carry over.

| | v1 | why |
|---|---|---|
| d_model | 384 | |
| layers | 6 | |
| heads | 6 query / 2 KV (GQA) , head_dim 64 | KV cache is 1/3 of MHA |
| FFN | SwiGLU, hidden 1024 | |
| Norm / pos | RMSNorm / RoPE | |
| Context (train) | **256** | training window (`--max-len 256`); gives the RoPE buffers headroom over the wire format's real needs |
| Context (on-device) | **128** | `FSM_MAX_INPUT`/`serialize.MAX_INPUT_TOKENS` cap the input at 128; worst-case 33-token utterance + 18 output tokens fits with room. Sized so the KV cache stays in internal SRAM (below). |
| Vocab | 4388 = 4096 pruned BPE + 292 special | |
| Params | 9.44M ternary linears + 1.69M embedding = **11.1M** | |

The 292 special tokens are 4 control, 7 intent, 25 literal, 256 pointer
(`<s:0..127>`, `<e:0..127>`). Embedding is tied with the LM head.

### Storage (measured)

An actual `export.py` run on a trained checkpoint produced the ground-truth
blob. `model.tern` = **3,632,692 bytes (3.63 MB)**:

| | bytes |
|---|---|
| Linears, ternary packed 5 trits/byte (1.6 bit/weight) | 1.89 MB |
| Embedding, int8 | 1.68 MB |
| Per-output-channel fp16 scales + norms | ~0.06 MB |
| **Total (measured)** | **3.63 MB** |

### The binding constraint is memory bandwidth, not compute or flash

3.6 MB of weights are touched once per decoded token. At ~9.4M MACs/token the
S3's dual-core SIMD is not the bottleneck; moving 3.6 MB is. Reading from
flash over quad SPI (~40 MB/s) gives ~90 ms/token. Copying the weights into
octal PSRAM at boot (~120 MB/s effective) gives ~30 ms/token.

**So: load weights into PSRAM at boot, don't `mmap` from flash.** 3.6 MB of
your 8 MB. This supersedes the mmap suggestion from earlier — it only made
sense when I thought flash was the tight resource.

Memory map (ESP32-S3 N16R8: 16 MB flash, 8 MB octal PSRAM, ~327 KB usable
internal SRAM):

- PSRAM: weights 3.63 MB + working buffers ~1.0 MB ≈ **5.0 MB of 8 MB** (✔ ~3 MB headroom)
- Internal SRAM: KV cache **192 KB** (6 × 128 × 2 heads × 64 × 2 tensors, int8)
- Flash: weight partition 3.63 MB + tokenizer ~200 KB ≈ **5.3 MB of 16 MB** (✔ ample)

KV cache sizing: at context 256 the KV cache would be 393 KB, which exceeds
internal SRAM and would force PSRAM-KV (slower). At context 128 it is 192 KB
and stays in internal SRAM. Since the wire format caps input at 128 and real
utterances are ≤33 tokens, context 128 is the right on-device choice.

Worst case output is **18 tokens** (`serialize.budget()`), typical 8–9. At 30
ms/token that's ~0.3 s decode plus one prefill pass, so roughly **0.5 s
end-to-end**. Prefill is a GEMM over the whole prompt and costs one weight
sweep regardless of prompt length.

Because latency scales with model *bytes*, size is the dial. 384/6 is the
starting point; if measured tok/s leaves headroom, 448/8 (~5.4 MB) is the next
rung and roughly doubles latency. Flash is nowhere near a limit at 16 MB, so
don't let it drive the sizing.

## 2. Constrained decoding

`serialize.FSM` derives the legal-next-token set from `dsl.SLOTS`. Mask logits
to that set at every step. Consequences:

- Every sampled sequence decodes to a valid plan. Unparseable output stops
  being a failure mode; the worst case becomes a wrong value.
- Required slots cannot be skipped, `<eop>` cannot fire mid-action, an end
  pointer can never precede its start, and `>3` actions are unreachable.
- Greedy decode only. No beam, no sampling.

Two shortcuts worth taking: when the legal set has one member, skip the forward
pass entirely; and the `<ok>`/`<no>` gate is a binary decision over the prefill
state, so it can be a small classifier head instead of a decode step.

## 3. Data

`corpus.py` (Stage A) → `paraphrase.py` (Stage B).

Stage A is templates × entity pools, gold exact by construction, verified
through `encode()` before emission so an unrepresentable example is dropped
rather than shipped. Stage B rewrites Stage A utterances with your local model
while **carrying the gold over unchanged**; a rewrite is kept only if every
slot value survives as a literal span. The LLM therefore cannot mislabel, only
lower yield.

```bash
python corpus.py --n 1000000 --out train_a.jsonl
python corpus.py --n 5000 --split val --seed 99 --balanced --out val.jsonl
mlx_lm.server --model <path> --port 8080
python paraphrase.py --in train_a.jsonl --out train_b.jsonl --limit 150000 --per-row 2
```

Splits are scenario-level: val draws only from held-out entity pools and
val-only templates, so it measures generalization to unseen room and person
names. The only legitimate overlap is closed-class atomic utterances (`stop`,
`abort`).

**Volume.** 11M params against ~45 tokens per example means ~1M Stage A
examples at 4–5 epochs to reach a sane token budget. Stage A costs nothing, so
scale it; Stage B is the expensive part, so paraphrase a 150k subset and let
the template diversity carry the rest.

Mixture: 55% atomic / 28% pair / 9% triple / 8% reject for training, balanced
for evaluation so triples don't hide behind the prior. Roughly half the reject
examples are near-miss mixed chains (`clean the kitchen and order me a pizza`)
— without those the model learns "starts with a valid clause → `<ok>`".

**Conventions the generator enforces** (change them here, not in the data):

- Anaphora resolves: `go to the kitchen and clean it` → `MOVE{kitchen},
  CLEAN{kitchen}`. Each action is self-contained, so the executor needs no state.
- `recipient` is filled only for third parties. `bring me the cup` leaves it
  empty; `bring John the cup` fills it. Keyed to the text, no world knowledge.
- `duration_*` are always literal tokens from a closed set; fuzzy quantities
  are canonicalized at label time (`half an hour` → 30 / minutes).
- Locations with no span use `<lit:here>` / `<lit:everywhere>`.
- `STOP` is a barrier and appears last only. No repeated identical actions.
- Rejection is whole-utterance and is `[UNAVAILABLE]`, never `[]`, so a
  degenerate empty output can never score as a correct refusal.

## 4. Distillation

**Teacher.** Fine-tune SmolLM2-135M on the *identical* wire format with the
*same* pruned tokenizer. Prune rather than retrain the tokenizer: keep
SmolLM2's original token IDs for the surviving subset, and teacher→student
logit alignment becomes a `gather` over the kept index list. A fresh 8k BPE
would forfeit distribution-level KD. Measure the teacher's exact-match first —
it is the ceiling.

**Student.** QAT from step zero, never post-training quantization.

- BitNet b1.58 forward quantization, absmean scale per output channel,
  straight-through estimator, fp32 master weights.
- Ramp the quantizer in over the first ~20% of steps (latent fp → ternary).
  Ternary training is much more stable with the ramp than without.
- Embeddings int8, norms fp16. Do not ternarize either.

**Loss**, computed only on positions after `<plan>`:

```
L = α·CE(gold) + β·KL(student ‖ teacher, T≈1.5) + γ·‖P·h_s − h_t‖²
```

with `P` a learned 576→384 projection discarded at export. Apply the **FSM mask
to both teacher and student logits before the softmax** in the KL term, so
capacity goes into the decisions that are actually live rather than into
learning to suppress tokens the decoder already forbids.

**Implemented so far:** the β·KL term is live in `train_student.py` under
`--teacher <path>` (temp-scaled, masked to supervised positions, lazy torch
import, graceful no-teacher fallback). The γ hidden-state term is **deferred**:
hidden-position alignment is ill-defined because the student is word-level
(`serialize.tokenize`) while the teacher is BPE-level, so there is no stable
`h_t` position to supervise. Revisit only if logit-level KD proves
insufficient.

AdamW, lr 3e-3 (ternary tolerates higher than fp), cosine schedule, 2k warmup,
wd 0.1 off norms and embeddings, batch 256 sequences.

**Ablations worth running**, because each one is complexity that has to earn
its place: KD vs hard-label only; pointer vs generated values; on-device
context 128 vs 256 (KV budget); FSM-masked KL vs plain KL; with and without the
hidden-state term.

## 5. Evaluation

Primary metric is `dsl.actions_match` — exact sequence match, no partial credit.
Underneath it:

- intent-sequence accuracy independent of slots
- per-slot precision/recall, `person` and `recipient` separately since optional
  slots are penalized in both directions
- EM stratified by chain length (atomic / pair / triple)
- EM on held-out entities vs seen entities — the gap is what the pointer format
  is supposed to close
- `<ok>`/`<no>` confusion, with **false-accept rate as the safety metric**. A
  confident wrong `MOVE` is worse than an honest refusal, so threshold on
  `P(<ok>)` and refuse when marginal.

Then device parity: run the same val set through the C runtime and require EM
drift ≤ 0.5% against Python, with per-layer golden vectors to localize kernel
bugs. Report tok/s, p95 end-to-end latency, and PSRAM high-water alongside
accuracy.

## 6. Order of work

1. Scale Stage A to 1M, paraphrase 150k, hand-write a 500-utterance eval set
   that no template produced.
2. Fine-tune the teacher, record its EM. If it can't clear ~95%, the format or
   the conventions are wrong and no amount of distillation will fix it.
3. Train the fp student at the same size. This isolates quantization loss from
   capacity loss.
4. Turn on QAT. Expect a few points; if it's more than ~4, revisit the ramp.
5. Export, write the FSM in C, parity-test, then measure on device.

Steps 2 and 3 are the cheap kill points. Both happen before any embedded work.

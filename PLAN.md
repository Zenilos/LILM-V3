# PlanCore v1 — Implementation Plan

## Overview

Build a tiny LLM (11.1M params) that runs on ESP32-S3 (16MB flash, 8MB PSRAM) and extracts up to 3 chained intents from natural language commands. The model emits a compact token sequence, not JSON.

---

## Phase 1: Data Pipeline

### 1.1 Generate Stage A Corpus (corpus.py)

```bash
# Training: 1M examples, template-diverse
python corpus.py --n 1000000 --out data/train_a.jsonl

# Validation: balanced splits, held-out entities
python corpus.py --n 5000 --split val --seed 99 --balanced --out data/val.jsonl
```

**Verify:**
- Run `python dsl.py` and `python serialize.py` to confirm the wire format is sound
- Check distribution: 55% atomic / 28% pair / 9% triple / 8% reject

### 1.2 Paraphrase with Ollama (Stage B)

Start Ollama with a suitable model (e.g., `smollm2-135m` or `qwen2.5-0.5b`):

```bash
ollama serve  # or however you run it
```

Then run:

```bash
python paraphrase.py \
  --in data/train_a.jsonl \
  --out data/train_b.jsonl \
  --limit 150000 \
  --per-row 2 \
  --host http://localhost:11434
```

**Note:** The paraphrase script hits OpenAI-compatible endpoint. Ollama exposes this at `http://localhost:11434/v1`. Update `paraphrase.py` call URL if needed.

**Yield check:** Expect 30-50% rejection rate. The gate ensures no mislabelled examples.

### 1.3 Merge Splits

```bash
# Final training set: Stage A non-paraphrased + Stage B paraphrased
cat data/train_a.jsonl data/train_b.jsonl | shuf > data/train.jsonl
```

---

## Phase 2: Teacher Fine-tuning

### 2.1 Setup

- **Base model:** SmolLM2-135M (or similar small decoder-only)
- **Tokenizer:** Prune to 4096 tokens, keeping original IDs for surviving subset
- **Wire format:** Same as `serialize.py` — model learns to emit `<plan> <ok> <MOVE> <s:X><e:Y> ...`

### 2.2 Training Config

```python
# teacher_config.yaml
model_name: "HuggingFaceTB/SmolLM2-135M"
max_seq_length: 256  # 128 input + 128 output
learning_rate: 2e-5
num_epochs: 3
batch_size: 32
gradient_accumulation: 8
```

### 2.3 Evaluation

- Run teacher on `val.jsonl`
- Target: **≥95% exact-match** on `dsl.actions_match`
- If below 95%, iterate on data conventions before proceeding

---

## Phase 3: Student Training (MLX)

### 3.1 Architecture

| Parameter | Value |
|-----------|-------|
| d_model | 384 |
| layers | 6 |
| heads | 6 query / 2 KV (GQA) |
| head_dim | 64 |
| FFN | SwiGLU, hidden 1024 |
| Norm | RMSNorm |
| Pos | RoPE |
| Context | 256 |
| Vocab | 4388 (4096 BPE + 292 special) |
| Params | 11.1M (9.44M ternary + 1.69M embedding) |

### 3.2 QAT from Step Zero

```python
# Key settings
quantization: "ternary"  # BitNet b1.58
quantizer_ramp_steps: 0.2  # Ramp over first 20% of training
embedding_dtype: "int8"
norm_dtype: "fp16"
master_weights: "fp32"
```

### 3.3 Loss Function

```
L = α·CE(gold) + β·KL(student ‖ teacher, T≈1.5) + γ·‖P·h_s − h_t‖²
```

- `P`: learned 576→384 projection (discarded at export)
- Apply **FSM mask** to both teacher and student logits before softmax in KL term
- Loss computed only on positions after `<plan>`

### 3.4 Training Config

```python
# student_config.yaml
optimizer: "adamw"
learning_rate: 3e-3  # Ternary tolerates higher LR
scheduler: "cosine"
warmup_steps: 2000
weight_decay: 0.1  # Off norms and embeddings
batch_size: 256
```

### 3.5 Ablations to Run

| Ablation | Purpose |
|----------|---------|
| KD vs hard-label only | Is distillation worth the complexity? |
| Pointer vs generated values | Do copy pointers help generalization? |
| Context 128 vs 256 | Is 256 the right on-device KV budget? |
| FSM-masked KL vs plain KL | Does masking help capacity allocation? |
| With/without hidden-state term | Is the projection loss useful? |

---

## Phase 4: Export & Validation

### 4.1 Export Weights

```python
# Export flow
1. Export ternary linears packed 5 trits/byte (1.6 bit/weight) → 1.89 MB
2. Export int8 embeddings → 1.69 MB
3. Export fp16 scales and norms → ~40 KB
4. Total: ~3.6 MB
```

### 4.2 FSM in C

Port `serialize.py` FSM to C for constrained decoding on device:

```c
// Key functions to port
fsm_state_t fsm_start(int n_input);
token_set_t fsm_legal(fsm_state_t state);
fsm_state_t fsm_step(fsm_state_t state, int token);
```

### 4.3 Parity Testing

- Run same `val.jsonl` through Python and C runtime
- Require **EM drift ≤ 0.5%**
- Per-layer golden vectors to localize kernel bugs

---

## Phase 5: ESP32 Deployment

### 5.1 Memory Map (ESP32-S3 N16R8: 16 MB flash, 8 MB PSRAM)

Measured weight blob = **3.63 MB** (`export.py`). Whole working set fits.

| Region | Usage | Size |
|--------|-------|------|
| PSRAM | Weights | 3.63 MB |
| PSRAM | Working buffers / activations | ~1.0 MB |
| PSRAM | **Subtotal** | **~5.0 MB of 8 MB** (✔) |
| Internal SRAM | KV cache (int8, ctx 128) | 192 KB |
| Internal SRAM | Activations | varies |
| Flash | Weight partition | 3.63 MB |
| Flash | Tokenizer | ~200 KB |
| Flash | **Subtotal** | **~5.3 MB of 16 MB** (✔) |

KV cache lives in internal SRAM only at context 128 (192 KB); context 256 would
push it to 393 KB, over the ~327 KB internal budget, forcing slower PSRAM-KV.

### 5.2 Boot Sequence

1. Copy weights from flash to PSRAM at boot (~120 MB/s → ~30ms)
2. Load tokenizer from flash
3. Initialize KV cache in internal SRAM
4. Ready for inference

### 5.3 Inference Flow

```
Input tokens → Prefill (GEMM over prompt) → Decode loop:
  1. Compute logits
  2. Apply FSM mask (legal next tokens)
  3. Greedy argmax
  4. Update state
  5. Repeat until <eop> or <no>
```

**Expected performance:**
- Prefill: ~30ms (one weight sweep)
- Decode: ~30ms/token × 18 tokens max = ~540ms
- **Total: ~0.5s end-to-end**

---

## Phase 6: Evaluation

### 6.1 Metrics

| Metric | Description |
|--------|-------------|
| `dsl.actions_match` | Exact sequence match (primary) |
| Intent accuracy | Independent of slot values |
| Per-slot P/R | Separate for `person` and `recipient` |
| EM by chain length | Atomic / pair / triple |
| EM by entity | Held-out vs seen (pointer generalization) |
| `<ok>`/`<no>` confusion | False-accept rate (safety metric) |

### 6.2 Device Metrics

- tok/s
- p95 latency
- PSRAM high-water mark

---

## Files to Create/Modify

| File | Status | Purpose |
|------|--------|---------|
| `train_teacher.py` | New | Fine-tune SmolLM2 teacher |
| `train_student.py` | New | MLX QAT training loop |
| `model.py` | New | Student architecture definition |
| `export.py` | New | Export weights to binary format |
| `fsm.c` | New | C port of FSM for ESP32 |
| `paraphrase.py` | Modify | Update for Ollama endpoint |
| `eval.py` | New | Evaluation harness |

---

## Critical Path

```
corpus.py → paraphrase.py (Ollama) → train_teacher.py → train_student.py (MLX QAT) → export.py → fsm.c → ESP32
```

**Kill points:**
1. If teacher < 95% EM → fix data conventions
2. If fp student < teacher - 2% → architecture issue
3. If QAT drop > 4% → revisit quantizer ramp

---

## Next Immediate Steps

1. [ ] Generate Stage A corpus (1M examples)
2. [ ] Run paraphrase.py with Ollama
3. [ ] Write train_teacher.py
4. [ ] Fine-tune teacher, measure EM
5. [ ] Write model.py (student architecture in MLX)
6. [ ] Write train_student.py with QAT
7. [ ] Train fp student, compare to teacher
8. [ ] Enable QAT, measure loss
9. [ ] Write export.py
10. [ ] Port FSM to C
11. [ ] Parity test
12. [ ] Deploy to ESP32

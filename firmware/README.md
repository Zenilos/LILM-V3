# firmware/ - ESP32 runtime for the V5 joint intent + slot model

This is the V5 fp16 deployment runtime. `export_v4.py --mode fp16` writes a
raw fp16 `model.bin` and binary `model.toc`; the C loader decodes weights to
fp32 and performs inference in fp32. Host parity is still being debugged
before IDF wiring or flashing.

## Model

The model is a bidirectional transformer encoder with intent and BIO slot
heads, followed by local CRF Viterbi decoding. It is not the autoregressive
PlanCore model in `fsm.c`. Input is `[CLS]` (id 0) followed by word-vocabulary
ids, with OOV words mapped to id 0. The fp16 blob is 1.41 MB and fits the XIAO
ESP32-S3 N8R8 flash.

Checkpoint RoPE `cos`/`sin` values are exported and loaded verbatim. They must
not be recomputed on-device because the checkpoint frequencies differ from the
default helper in `v4_model.py`.

## Storage contract

- All tensors, including `cos`/`sin`, are IEEE fp16.
- `v5_load()` decodes tensors into one owned fp32 allocation.
- `model.toc` contains tensor names, offsets, byte counts, and nine config
  values, replacing device-side JSON parsing.

## Host test

```bash
~/p3.11/bin/python3 firmware/tests/run_host_test.py \
  --dir /tmp/export_v5_fp16 --n 1500
```

The current kernel is not at parity: it reaches 56.7% intent versus the 59.3%
reference, with the first known divergence in block-0 attention. Compare
RMSNorm, q/k/v, RoPE, attention scores/softmax, value combine, o-projection,
block 1, and heads in that order. Acceptance is 59.3% intent, 39.8% person,
and approximately 1e-3 maximum C-vs-MLX logit error.

## IDF layout

```text
firmware/
  CMakeLists.txt
  include/v5_model.h
  src/v5_model.c
  partitions.csv
  main/
```

## TODO

- [ ] Fix host parity and rerun at both `-O0` and `-O2`.
- [ ] Remove temporary dump helpers while retaining the regression harness.
- [ ] Add IDF component, model partition/data embedding, tokenizer, and main
      smoke test.
- [ ] Build, flash, and validate on the XIAO ESP32-S3 N8R8.
- [ ] Keep V6 subword/character tokenization separate from deployment bring-up.

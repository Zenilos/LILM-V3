# firmware/ — ESP32 runtime for the V5 joint intent + slot model

This is a **skeleton** (plan + header/source scaffolding, not yet compiling /
not yet wired to a real `.tern` from the V5 exporter). It documents the
on-device architecture and the exact quantization/packing contract the C
runtime must honor. Review, then implement the exporter + this runtime together.

## What changed vs the old `fsm.c`

The V4/V5 model is a **joint intent + BIO slot tagger** (transformer encoder
→ frame-level `intent` + per-token `slot` logits, decoded by a linear-chain
CRF). It is **not** the autoregressive PlanCore generative model that `fsm.h`/
`fsm.c` / the old `model.tern` implement. On device:

1. Tokenize the utterance against the small (310) word vocab; OOV → id 0.
2. [CLS] = id 0 prepended.
3. One transformer encoder forward (mean-pool over non-padding → `intent_logits`;
   per-token → `slot_logits`).
4. CRF Viterbi over the BIO slot logits → span labels; decode into slots.
5. `intent` + `slots` drive action execution (same `Action` semantics as dsl.py).

No autoregressive decode; no KV cache; single-shot forward. Much smaller than
the old model: **~0.23 MB** blob vs 3.63 MB.

## Memory map (ESP32-S3 N16R8: 16 MB flash, 8 MB PSRAM)

| Region | Contents | Size |
|---|---|---|
| Flash | `v5_model.tern` + manifest | ~0.23 MB |
| Internal SRAM | activations + int8 embeddings (~310×192) | < 100 KB |
| (PSRAM optional) | none required | 0 |

Because the model is small enough for internal flash/SRAM, it can run entirely
in internal memory; PSRAM is not needed (unlike the old 3.63 MB model). This
kills the old KV-cache/PSRAM constraints.

## Quantization contract (match `export_v4.py` / `export.py`)

- **Linear weights** (`q/k/v/o`, `w1/w2/w3`, `intent_head`, `slot_head`):
  ternary {-1,0,+1}, per-output-channel absmean scale (fp16), packed **5
  trits/byte**. Decode: `digit_{5b+i} = (byte_b / 3^i) % 3`, then `-1`.
- **Embedding**: int8, per-row scale = max|w|/127.
- **RMSNorm / biases / CRF `trans`**: fp16.
- **Manifest**: `tensors` with `offset/nbytes/shape/kind` so the loader maps
  the blob into pointers without copying.

The C `unpack_trits` below is the reference decode of `export.pack_trits`.

## Build (skeleton — files not yet wired into an idf-component)

Intended layout (ESP-IDF):

```
firmware/
  CMakeLists.txt          # idf-component: v5_model.c
  include/v5_model.h
  src/v5_model.c
  partitions.csv          # flash: v5_model 0x20000 size auto
  main/                   # app: init periphs, load model, run
```

Public API (see `v5_model.h`):

- `v5_load(manifest_path, blob_base)` — parse manifest, set tensor pointers.
- `v5_forward(tokens, n)` → `intent_idx`, `slot_logits[n][9]`.
- `v5_decode_slots(slot_logits, n)` → span list via local CRF Viterbi.

## TODO (gates before this compiles/runs)

- [ ] `export_v4.py`: emit a V5 manifest/blob matching the contract above
      (see STATUS "Deployment" section, QAT plan step 4).
- [ ] QAT `v5crf_qat` checkpoint (`v4_train.py --t` ramp) — STAT plan step 1-3.
- [ ] C: wire `unpack_trits` + int8 embed + fp16 scales into `v5_forward`
      matmuls (skeleton has the byte-layout scaffold).
- [ ] C: CRF Viterbi — needs `SLOT_LABELS` family table export (additive `trans`
      + start-forbid for `I-*`).
- [ ] IDF component + partition + `main` stub (test on host first via a small
      ANSI-C driver mirroring `quant_eval.py` expectations).

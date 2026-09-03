/**
 * v5_model.h - ESP32 inference runtime for the V5 joint intent + slot model.
 *
 * Skeleton (see firmware/README.md). This is a transformer-encoder tagger,
 * NOT the generative PlanCore model in fsm.h. On-device contract:
 *
 *   tokens[0] = CLS (id 0), then word ids from the 310-word vocab (OOV -> 0).
 *   v5_forward runs the full encoder (embed -> 2 blocks -> norm -> heads),
 *   producing intent logits (frame-pooled) and per-token BIO slot logits.
 *   v5_decode_slots runs a local linear-chain CRF Viterbi over slot logits.
 *
 * Quantization (must match export_v4.py / export.py):
 *   linear weights  -> ternary {-1,0,+1}, per-row absmean scale (fp16),
 *                      packed 5 trits/byte (see unpack_trits)
 *   embedding       -> int8, per-row scale = max|w|/127
 *   norms / biases  -> fp16
 *   CRF trans       -> fp16 [C,C] additive transition (I-* start forbidden)
 */
#ifndef V5_MODEL_H
#define V5_MODEL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define V5_D         192      /* d_model */
#define V5_HEAD_DIM  32
#define V5_N_HEADS   4
#define V5_N_LAYERS  2
#define V5_FFN       384
#define V5_VOCAB     310
#define V5_N_INTENT  8
#define V5_N_TAG     9        /* BIO slot classes */
#define V5_MAX_LEN   64       /* includes CLS */

/* One decoded slot span. family indexes into a device-side SLOT_FAMILY table. */
typedef struct {
    int family;         /* 0=location 1=person 2=message 3=duration */
    int start, end;     /* inclusive token indices in the ORIGINAL (no-CLS) tokens */
} v5_span_t;

typedef struct {
    /* resolved from manifest; offsets into a single loaded blob */
    const void *blob;             /* model.tern base */
    const uint8_t *embed_q;       /* [VOCAB][D] int8 */
    const uint16_t *embed_scale;  /* [VOCAB] fp16 per-row scale */
    const uint8_t *attn_q; const uint16_t *attn_q_scale;   /* per layer, below */
    /* per-layer tensors are looked up by name at load; stored as arrays */
} v5_model_t;

/* ---- lifecycle ---- */
/* Parse manifest.json, map blob pointers. Returns 0 on success. */
int v5_load(v5_model_t *m, const char *manifest_path, const void *blob);

/* Forward a token sequence (tokens[0]==CLS). Fills intent_id (0..7) and
 * slot_logits[n][V5_N_TAG] (unquantized fp math, device buffers provided by
 * caller or internal fixed-buffer allocation). Returns 0 on success. */
int v5_forward(const v5_model_t *m, const int16_t *tokens, int n,
               int *intent_id, float *slot_logits);

/* CRF Viterbi decode of slot_logits into <V5_N_TAG> tag ids; then
 * v5_spans() turns tag ids into v5_span_t list. */
void v5_decode_slots(const v5_model_t *m, const float *slot_logits, int n,
                     uint8_t *tag_ids);
int v5_spans(const int16_t *tokens, int n_no_cls, const uint8_t *tag_ids,
             v5_span_t *out, int out_cap);

#ifdef __cplusplus
}
#endif

#endif /* V5_MODEL_H */

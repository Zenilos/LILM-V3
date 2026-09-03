/**
 * v5_model.h - ESP32 inference runtime for the V5 joint intent + slot model.
 *
 * Deployment (V5 fp16, no quantization):
 *   - Weights are stored fp16 (export_v4.py --mode fp16, model.bin) and decoded
 *     to fp32 at load time by v5_load(). All compute is fp32, which matches the
 *     validated reference (roundtrip_v4.py): fp16 storage of this model is
 *     lossless (59.3% MOVE / 39.8% person on n=1500 == fp32 baseline).
 *   - CRITICAL: RoPE cos/sin MUST be loaded verbatim from the blob. The
 *     checkpoint's RoPE differs from a recomputed rope_freqs(base=10000), and
 *     recomputing on device yields wrong logits. cos/sin are exported as fp16
 *     and decoded here.
 *
 * This is a transformer-encoder tagger, NOT the generative FSM model.
 * On-device contract:
 *   tokens[0] = CLS (id 0), then word ids from the 310-word vocab (OOV -> 0).
 *   v5_forward: embed -> 2 blocks -> final RMSNorm -> heads.
 *       intent_logits  = mean-pool h over all T tokens -> intent_head (8)
 *       slot_logits    = per-token -> slot_head (9 BIO classes)
 *   v5_decode_slots: linear-chain CRF Viterbi over slot_logits.
 *   v5_spans: tag ids -> v5_span_t list (B/I adjacency).
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

/* Upper bound on tokens (incl. CLS) for static scratch buffers. The V5
 * checkpoint RoPE caps at 40; this is a safe ceiling. n must be <= m->max_len. */
#define V5_MAX_LEN   64

/* One decoded slot span. family indexes into the SLOT_FAMILY table. */
typedef struct {
    int family;         /* 0=location 1=person 2=message 3=duration */
    int start, end;     /* inclusive token indices in the ORIGINAL (no-CLS) tokens */
} v5_span_t;

/* Per-transformer-layer weights, fp32 (decoded from fp16 blob at load). */
typedef struct {
    float *q_w, *q_b;      /* [128][192], [128]   (n_heads*head_dim, d) */
    float *k_w, *k_b;
    float *v_w, *v_b;
    float *o_w, *o_b;      /* [192][128], [192] */
    float *attn_norm_w;    /* [192] */
    float *ffn_norm_w;     /* [192] */
    float *w1_w, *w1_b;    /* [384][192], [384] */
    float *w2_w, *w2_b;    /* [192][384], [192] */
    float *w3_w, *w3_b;    /* [384][192], [384] */
} v5_layer_t;

typedef struct {
    /* ---- decoded fp32 weights ---- */
    float *embed_w;        /* [VOCAB][D] */
    v5_layer_t layers[V5_N_LAYERS];
    float *final_norm_w;   /* [D] */
    float *intent_w, *intent_b;   /* [N_INTENT][D], [N_INTENT] */
    float *slot_w, *slot_b;       /* [N_TAG][D], [N_TAG] */
    /* RoPE (VERBATIM from checkpoint; do not recompute) */
    float *cos, *sin;      /* [max_len][head_dim/2] */
    int    max_len;
    /* CRF */
    float *crf_trans;      /* [N_TAG][N_TAG] additive transition */
    float *crf_log_mask;   /* [N_TAG][N_TAG] 0 or -NEG structural */

    /* ---- scratch (caller-provided buffers of adequate size) ---- */
    int scratch_cap;
    float *scratch;

    /* one BIG allocation backing all the pointers above (for easy free) */
    void *owns;
} v5_model_t;

/* ---- lifecycle ---- */
/* Decode model.bin (fp16) + model.toc (binary index) into fp32 weights.
 * Returns 0 on success. blob/b_toc must remain valid only during the call. */
int v5_load(v5_model_t *m, const void *blob, const void *toc);

/* Forward a token sequence (tokens[0]==CLS). Fills intent_id (0..7) and
 * slot_logits[n][V5_N_TAG] (fp math). If intent_logits is non-NULL, fills it
 * with the V5_N_INTENT intent logits. Returns 0 on success. */
int v5_forward(const v5_model_t *m, const int16_t *tokens, int n,
               int *intent_id, float *intent_logits, float *slot_logits);

/* CRF Viterbi decode of slot_logits into tag_ids (n entries). */
void v5_decode_slots(const v5_model_t *m, const float *slot_logits, int n,
                     uint8_t *tag_ids);

/* tag_ids -> v5_span_t list. tokens are the ORIGINAL no-CLS tokens with
 * n_no_cls entries; tag_ids has n_no_cls+1 entries (CLS at index 0). */
int v5_spans(const int16_t *tokens, int n_no_cls, const uint8_t *tag_ids,
             v5_span_t *out, int out_cap);

/* Release the big allocation made by v5_load (m->owns). */
void v5_free(v5_model_t *m);

#ifdef __cplusplus
}
#endif

#endif /* V5_MODEL_H */

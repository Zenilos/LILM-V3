/**
 * v5_model.c - reference numeric ops for the V5 inference runtime.
 *
 * SKELETON: the quant primitives (unpack_trits, int8 embed, fp16 scale math,
 * RMSNorm, ternary matmul) are implemented and unit-testable in pure ANSI C;
 * the manifest loader and the per-layer tensor pointer mapping are stubbed
 * with TODO markers pending the finalized export_v4.py manifest.
 */
#include "v5_model.h"

#include <math.h>
#include <string.h>

/* ---- quant contract (mirrors export.pack_trits / int8_rows / ternary) ---- */

static const int BASE3[5] = {1, 3, 9, 27, 81};

/* Decode N trits ({-1,0,+1}) from bytes packed 5 trits/byte (base-3 digits,
 * digit_{5b+i} = (byte_b / 3^i) % 3, then -1). */
static void unpack_trits(const uint8_t *packed, int n, int8_t *out) {
    for (int i = 0; i < n; i++) {
        int b = i / 5;
        int d = (packed[b] / BASE3[i % 5]) % 3;
        out[i] = (int8_t)(d - 1);
    }
}

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) bits = sign;
        else { exp = 1; while (!(frac & 0x400)) { frac <<= 1; exp--; } frac &= 0x3ff; }
        bits = sign | ((exp + 127) << 23) | (frac << 13);
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (frac << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

/* y[out] = dequantized ternary line: y_o = sum_i q_trit[o*in+i] * scale[o] * x_i */
static void line_tern(const uint8_t *packed_w, const uint16_t *scale_fp16,
                      const float *x, int in, int out, float *y) {
    int8_t *trits = (int8_t *)__builtin_alloca(in);
    for (int o = 0; o < out; o++) {
        unpack_trits(packed_w + (size_t)o * ((in + 4) / 5), in, trits);
        float s = fp16_to_f32(scale_fp16[o]);
        float acc = 0.0f;
        for (int i = 0; i < in; i++) acc += (float)trits[i] * x[i];
        y[o] = acc * s;
    }
}

/* int8 embedding lookup with per-row fp16 scale. */
static void embed_lookup(const uint8_t *q, const uint16_t *scale, int id,
                         float *out) {
    float s = fp16_to_f32(scale[id]);
    const uint8_t *row = q + (size_t)id * V5_D;
    for (int i = 0; i < V5_D; i++) out[i] = s * (float)((int8_t)row[i]);
}

static void rmsnorm(const float *in, const uint16_t *w_fp16, int n, float *out) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += in[i] * in[i];
    float r = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++)
        out[i] = fp16_to_f32(w_fp16[i]) * (in[i] * r);
}

/* ---- TODO: system of record ----
 * Loader: parse export_v4 manifest.json, resolve the per-layer tensor
 * pointers (attn q/k/v/o, w1/w2/w3 + per-row fp16 scales, norms, heads, and
 * CRF trans / family table) from the single blob + store in v5_model_t arrays.
 * (Packing layout is identical to export.py; see firmware/README TODO gates.)
 * ---- stub ---- */
int v5_load(v5_model_t *m, const char *manifest_path, const void *blob) {
    (void)m; (void)manifest_path; (void)blob;
    return -1; /* TODO(export_v4 manifest): map tensor offsets -> pointers */
}

/* ---- forward: embed -> 2 blocks (attention w/ mask + swiglu ffn) -> heads ----
 * Numeric scaffold; the pointer plumbing (per-layer w/ scale) is filled by the
 * loader above. XXX finalized in the exporter pass. */
int v5_forward(const v5_model_t *m, const int16_t *tokens, int n,
               int *intent_id, float *slot_logits) {
    (void)m; (void)tokens; (void)n; (void)intent_id; (void)slot_logits;
    return -1; /* TODO: wire loader-resolved tensors + RoPE-free no-position
                * (model uses absolute sin/cos; see v4_model.rope_freqs) */
}

/* ---- CRF Viterbi + span extraction (family table exported by exporter) ---- */
static const char *SLOT_LABELS[V5_N_TAG] =
    {"O", "B-location", "I-location", "B-person", "I-person",
     "B-message", "I-message", "B-duration", "I-duration"};

static int bio_family(int tag) {
    if (tag <= 0) return -1;
    return (tag - 1) / 2; /* B/I pairs -> 0 loc,1 person,2 msg,3 dur */
}

void v5_decode_slots(const v5_model_t *m, const float *slot_logits, int n,
                     uint8_t *tag_ids) {
    (void)m;
    /* TODO: linear-chain CRF Viterbi using additive trans[9][9] (fp16, from
     * manifest) + forbid I-* at start; emissions = slot_logits. Placeholder:
     * greedy per-token argmax for now. */
    for (int t = 0; t < n; t++) {
        int best = 0;
        for (int c = 1; c < V5_N_TAG; c++)
            if (slot_logits[t * V5_N_TAG + c] > slot_logits[t * V5_N_TAG + best])
                best = c;
        tag_ids[t] = (uint8_t)best;
    }
}

int v5_spans(const int16_t *tokens, int n_no_cls, const uint8_t *tag_ids,
             v5_span_t *out, int out_cap) {
    int cnt = 0, cur_fam = -1, start = -1;
    /*
     * tag_ids is parallel to the input INCLUDING the CLS at index 0, so the
     * token at output position `j` is tokens[j-1] (see v4_eval decode).
     * Placeholder body (full adjacency-constrained decode mirrors v4_eval).
     */
    (void)tokens; (void)n_no_cls; (void)tag_ids; (void)cnt; (void)start;
    (void)cur_fam;
    return 0; /* TODO: enforce B-fam + I-fam adjacency; emit out[] spans */
}

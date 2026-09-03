/**
 * v5_model.c - V5 joint intent + slot inference kernel, fp16-deployment.
 *
 * Matches v4_model.py numerically (fp32 compute). Weights are decoded from the
 * fp16 blob (model.bin) via the binary index (model.toc) at v5_load(). RoPE
 * cos/sin are loaded VERBATIM (never recomputed).
 *
 * Baselines (V5 fp16 roundtrip, n=1500): intent 59.3%, person 39.8% == fp32.
 */
#include "v5_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V5_EPS 1e-6f
#define V5_NEG 1e4f      /* structural CRF penalty (matches mlx NEG) */

/* -- small helper: read LE u16/u32 -------------------------------------------------- */
static uint16_t rd_u16(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}
static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static int32_t rd_i32(const uint8_t *p) {
    return (int32_t)rd_u32(p);
}

/* fp16 (IEEE half) -> fp32 */
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) bits = sign;
        else {
            exp = 1;
            while (!(frac & 0x400)) { frac <<= 1; exp--; }
            frac &= 0x3ff;
            bits = sign | ((exp + 127) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (frac << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ---- linear: y[o] = sum_i x[i]*w[o*in+i] + b[o] (w rows are output dims) ---------- */
static void linear(const float *x, const float *w, const float *b,
                   int in, int out, float *y) {
    for (int o = 0; o < out; o++) {
        float acc = 0.0f;
        const float *wr = w + (size_t)o * in;
        for (int i = 0; i < in; i++) acc += wr[i] * x[i];
        y[o] = acc + (b ? b[o] : 0.0f);
    }
}

/* ---- RMSNorm: y = x/sqrt(mean(x^2)+eps) * w ---------------- ---- */
static void rmsnorm(const float *x, const float *w, int n, float *y) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + V5_EPS);
    for (int i = 0; i < n; i++) y[i] = x[i] * r * w[i];
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ---- softmax over `n` entries of x (in place) ------------------------------------ */
static void softmax(float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---- RoPE on a [T][proj] buffer (n_heads x head_dim) in place, matching
 * apply_rope: each head's head_dim slice is rotated independently with the
 * same per-position cos/sin. ------------------------------------------------ */
static void apply_rope(float *qk, int T, int proj, int n_heads, int head_dim,
                       const float *cos, const float *sin) {
    int half = head_dim / 2;
    float tmp[V5_MAX_LEN * V5_D];
    memcpy(tmp, qk, sizeof(float) * (size_t)T * proj);
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < T; t++) {
            const float *c = cos + (size_t)t * half;
            const float *s = sin + (size_t)t * half;
            float *row = qk + (size_t)t * proj + (size_t)h * head_dim;
            const float *tr = tmp + (size_t)t * proj + (size_t)h * head_dim;
            for (int i = 0; i < half; i++) {
                float x1 = tr[i];
                float x2 = tr[half + i];
                row[i]       = x1 * c[i] - x2 * s[i];
                row[half + i] = x2 * c[i] + x1 * s[i];
            }
        }
    }
}

/* ---- one attention layer (bidirectional), output written to `out` [T][d] ---------- */
static void attention(const v5_layer_t *L, const float *x, int T,
                      const float *cos, const float *sin, float *out) {
    static float q[V5_MAX_LEN * V5_D];
    static float k[V5_MAX_LEN * V5_D];
    static float v[V5_MAX_LEN * V5_D];
    static float o[V5_MAX_LEN * V5_D];
    static float score[V5_MAX_LEN * V5_MAX_LEN];
    static int call = 0;
    const int in = V5_D;
    const int proj = V5_N_HEADS * V5_HEAD_DIM;
    int mycall = call++;

    if (getenv("V5_DUMP")) {
        char fn[64]; snprintf(fn, sizeof fn, "c_attn_in%d.bin", mycall);
        FILE *f = fopen(fn, "wb"); fwrite(x, 4, (size_t)T * in, f); fclose(f);
    }
    for (int t = 0; t < T; t++)
        linear(x + (size_t)t * in, L->q_w, L->q_b, in, proj, q + (size_t)t * proj);
    for (int t = 0; t < T; t++)
        linear(x + (size_t)t * in, L->k_w, L->k_b, in, proj, k + (size_t)t * proj);
    for (int t = 0; t < T; t++)
        linear(x + (size_t)t * in, L->v_w, L->v_b, in, proj, v + (size_t)t * proj);

    if (getenv("V5_DUMP")) {
        char fn[64];
        snprintf(fn, sizeof fn, "c_q_proj%d.bin", mycall);
        FILE *f = fopen(fn, "wb"); fwrite(q, 4, (size_t)T * proj, f); fclose(f);
        snprintf(fn, sizeof fn, "c_k_proj%d.bin", mycall);
        f = fopen(fn, "wb"); fwrite(k, 4, (size_t)T * proj, f); fclose(f);
    }
    apply_rope(q, T, proj, V5_N_HEADS, V5_HEAD_DIM, cos, sin);
    apply_rope(k, T, proj, V5_N_HEADS, V5_HEAD_DIM, cos, sin);

    if (getenv("V5_DUMP")) {
        char fn[64];
        snprintf(fn, sizeof fn, "c_q_rope%d.bin", mycall);
        FILE *f = fopen(fn, "wb"); fwrite(q, 4, (size_t)T * proj, f); fclose(f);
        snprintf(fn, sizeof fn, "c_k_rope%d.bin", mycall);
        f = fopen(fn, "wb"); fwrite(k, 4, (size_t)T * proj, f); fclose(f);
    }

    for (int h = 0; h < V5_N_HEADS; h++) {
        /* attention scores [T][T] with scale = head_dim^-0.5, softmax over keys */
        float sc = 1.0f / sqrtf((float)V5_HEAD_DIM);
        for (int t = 0; t < T; t++) {
            const float *qr = q + (size_t)t * proj + (size_t)h * V5_HEAD_DIM;
            float *srow = score + (size_t)t * T;
            for (int j = 0; j < T; j++) {
                const float *kr = k + (size_t)j * proj + (size_t)h * V5_HEAD_DIM;
                float a = 0.0f;
                for (int d2 = 0; d2 < V5_HEAD_DIM; d2++) a += qr[d2] * kr[d2];
                srow[j] = a * sc;
            }
            softmax(srow, T);
        }
        /* out[t] = sum_j score[t][j] * v[j]  (concat into o) */
        for (int t = 0; t < T; t++) {
            float *orow = o + (size_t)t * proj + (size_t)h * V5_HEAD_DIM;
            const float *srow = score + (size_t)t * T;
            for (int d2 = 0; d2 < V5_HEAD_DIM; d2++) {
                float a = 0.0f;
                for (int j = 0; j < T; j++)
                    a += srow[j] * v[(size_t)j * proj + (size_t)h * V5_HEAD_DIM + d2];
                orow[d2] = a;
            }
        }
    }
    if (getenv("V5_DUMP")) {
        char fn[64]; snprintf(fn, sizeof fn, "c_attn_out%d.bin", mycall);
        FILE *h0 = fopen(fn, "wb"); fwrite(o, 4, (size_t)T * proj, h0); fclose(h0);
    }
    for (int t = 0; t < T; t++)
        linear(o + (size_t)t * proj, L->o_w, L->o_b, proj, in, out + (size_t)t * in);
}

/* ---- FFN (swiglu), output [T][d] -------------------------------------------------- */
static void ffn(const v5_layer_t *L, const float *x, int T, float *out) {
    static float r[V5_MAX_LEN * V5_D];
    static float w1o[V5_MAX_LEN * V5_FFN];
    static float w3o[V5_MAX_LEN * V5_FFN];
    static float hbig[V5_MAX_LEN * V5_FFN];
    for (int t = 0; t < T; t++) {
        rmsnorm(x + (size_t)t * V5_D, L->ffn_norm_w, V5_D, r + (size_t)t * V5_D);
        linear(r + (size_t)t * V5_D, L->w1_w, L->w1_b, V5_D, V5_FFN,
               w1o + (size_t)t * V5_FFN);
        linear(r + (size_t)t * V5_D, L->w3_w, L->w3_b, V5_D, V5_FFN,
               w3o + (size_t)t * V5_FFN);
    }
    for (int t = 0; t < T; t++) {
        float *hr = hbig + (size_t)t * V5_FFN;
        const float *a = w1o + (size_t)t * V5_FFN;
        const float *b = w3o + (size_t)t * V5_FFN;
        for (int i = 0; i < V5_FFN; i++) hr[i] = silu(a[i]) * b[i];
        linear(hr, L->w2_w, L->w2_b, V5_FFN, V5_D, out + (size_t)t * V5_D);
    }
}

int v5_forward(const v5_model_t *m, const int16_t *tokens, int n,
               int *intent_id, float *intent_logits, float *slot_logits) {
    if (n < 1 || n > m->max_len) return -1;

    static float h[V5_MAX_LEN * V5_D];
    static float res[V5_MAX_LEN * V5_D];
    int T = n;
    const int D = V5_D;

    /* embed */
    for (int t = 0; t < T; t++) {
        int id = tokens[t];
        if (id < 0 || id >= V5_VOCAB) id = 0;
        memcpy(h + (size_t)t * D, m->embed_w + (size_t)id * D,
               sizeof(float) * D);
    }

    const char *v5_dump = getenv("V5_DUMP");
    if (v5_dump) {
        FILE *f = fopen("c_embed.bin", "wb"); fwrite(h, 4, (size_t)T * D, f); fclose(f);
    }

    /* 2 blocks: h = h + attn(rmsnorm(h)); h = h + ffn(h) */
    for (int l = 0; l < V5_N_LAYERS; l++) {
        const v5_layer_t *L = &m->layers[l];
        for (int t = 0; t < T; t++)
            rmsnorm(h + (size_t)t * D, L->attn_norm_w, D, res + (size_t)t * D);
        if (v5_dump && l == 0) {
            FILE *f = fopen("c_res0.bin", "wb"); fwrite(res, 4, (size_t)T * D, f); fclose(f);
        }
        attention(L, res, T, m->cos, m->sin, res);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < D; i++) h[t * D + i] += res[t * D + i];
        if (v5_dump) {
            char fn[64]; snprintf(fn, sizeof fn, "c_postattn%d.bin", l);
            FILE *f = fopen(fn, "wb"); fwrite(h, 4, (size_t)T * D, f); fclose(f);
        }
        ffn(L, h, T, res);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < D; i++) h[t * D + i] += res[t * D + i];
        if (v5_dump) {
            char fn[64]; snprintf(fn, sizeof fn, "c_postffn%d.bin", l);
            FILE *f = fopen(fn, "wb"); fwrite(h, 4, (size_t)T * D, f); fclose(f);
        }
    }
    for (int t = 0; t < T; t++)
        rmsnorm(h + (size_t)t * D, m->final_norm_w, D, res + (size_t)t * D);

    /* slot head: res -> per-token logits */
    for (int t = 0; t < T; t++)
        linear(res + (size_t)t * D, m->slot_w, m->slot_b, D, V5_N_TAG,
               slot_logits + (size_t)t * V5_N_TAG);

    /* intent: mean-pool over all T tokens -> intent_head */
    float pooled[V5_D];
    for (int i = 0; i < V5_D; i++) pooled[i] = 0.0f;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < V5_D; i++) pooled[i] += res[t * D + i];
    for (int i = 0; i < V5_D; i++) pooled[i] /= (float)T;

    float ilogits[V5_N_INTENT];
    linear(pooled, m->intent_w, m->intent_b, V5_D, V5_N_INTENT, ilogits);
    if (intent_logits) memcpy(intent_logits, ilogits, sizeof(ilogits));
    int bi = 0;
    for (int i = 1; i < V5_N_INTENT; i++)
        if (ilogits[i] > ilogits[bi]) bi = i;
    if (intent_id) *intent_id = bi;
    return 0;
}

/* ---- CRF Viterbi (matches CRF.decode) --------------------------------------------- */
void v5_decode_slots(const v5_model_t *m, const float *slot_logits, int n,
                     uint8_t *tag_ids) {
    static float alpha[V5_N_TAG];
    static float anew[V5_N_TAG];
    static int   bp[V5_MAX_LEN][V5_N_TAG];
    const int C = V5_N_TAG;

    /* start addend: -NEG for I-* tags (indices 2,4,6,8) */
    static float start_forbid[V5_N_TAG];
    /* logw[i][j] = trans[i][j] + log_mask[i][j] */
    static float logw[V5_N_TAG][V5_N_TAG];
    for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++)
            logw[i][j] = m->crf_trans[i * C + j] + m->crf_log_mask[i * C + j];
    for (int j = 0; j < C; j++)
        start_forbid[j] = ((j & 1) == 0 && j != 0) ? -V5_NEG : 0.0f;

    if (n < 1) return;
    for (int j = 0; j < C; j++) alpha[j] = slot_logits[j] + start_forbid[j];

    for (int t = 1; t < n; t++) {
        for (int j = 0; j < C; j++) {
            float best = alpha[0] + logw[0][j];
            int bi = 0;
            for (int i = 1; i < C; i++) {
                float v = alpha[i] + logw[i][j];
                if (v > best) { best = v; bi = i; }
            }
            anew[j] = best + slot_logits[t * C + j];
            bp[t][j] = bi;
        }
        memcpy(alpha, anew, sizeof(alpha));
    }

    int cur = 0;
    for (int j = 1; j < C; j++) if (alpha[j] > alpha[cur]) cur = j;
    tag_ids[n - 1] = (uint8_t)cur;
    for (int t = n - 2; t >= 0; t--) {
        cur = bp[t + 1][cur];
        tag_ids[t] = (uint8_t)cur;
    }
}

/* ---- spans (B/I adjacency), family table 0=location 1=person 2=message 3=duration - */
int v5_spans(const int16_t *tokens, int n_no_cls, const uint8_t *tag_ids,
             v5_span_t *out, int out_cap) {
    int cnt = 0;
    for (int j = 0; j < n_no_cls; j++) {
        int tag = tag_ids[j + 1];          /* CLS at index 0 */
        if (tag <= 0) continue;            /* O */
        /* tag: 0=O,1=B-loc,2=I-loc,... family=(tag-1)/2; B at odd tags */
        const char *label = (const char *[]){
            "O","B-location","I-location","B-person","I-person",
            "B-message","I-message","B-duration","I-duration"}[tag];
        if (label[0] == 'O') continue;
        int family = (tag - 1) / 2;        /* B/I pairs */
        int isB = (label[1] == 'B');
        if (cnt < out_cap) {
            if (isB || cnt == 0 || out[cnt - 1].family != family ||
                out[cnt - 1].end != j - 1) {
                out[cnt].family = family;
                out[cnt].start = j;
                out[cnt].end = j;
                cnt++;
            } else {
                out[cnt - 1].end = j;
            }
        }
    }
    return cnt;
}

/* ---- loader: decode fp16 blob via binary toc into fp32 v5_model_t ---------------- */
int v5_load(v5_model_t *m, const void *blob, const void *toc) {
    const uint8_t *t = (const uint8_t *)toc;
    const uint8_t *b = (const uint8_t *)blob;
    uint32_t n_tens = rd_u32(t);
    const uint8_t *p = t + 4;

    /* count total fp32 bytes we need = 2x fp16 blob bytes (each fp16 -> 4B) */
    uint32_t total_tensor_bytes = 0;
    for (uint32_t i = 0; i < n_tens; i++) {
        uint16_t nl = rd_u16(p); p += 2 + nl;
        uint32_t off = rd_u32(p); (void)off;
        uint32_t nb = rd_u32(p + 4); p += 8;
        total_tensor_bytes += nb;
    }
    const int32_t *cfg = (const int32_t *)p; /* 9 ints (i32) at end */
    (void)cfg;

    /* read config tail to get max_len */
    size_t cfg_off = (size_t)(p - t);
    int32_t c[9];
    memcpy(c, p, sizeof(c));

    memset(m, 0, sizeof(*m));

    /* one big fp32 arena (2x each fp16 tensor) + pointers for embedding/heads/etc.
     * We allocate 2*total_tensor_bytes; offsets tracked by decoding in blob order. */
    size_t arena_bytes = (size_t)total_tensor_bytes * 2u;
    uint8_t *arena = malloc(arena_bytes);
    if (!arena) return -1;
    m->owns = arena;
    m->max_len = (int)c[5];
    m->scratch = NULL;

    /* route each tensor to its fp32 slot */
    const uint8_t *tp = t + 4;
    for (uint32_t i = 0; i < n_tens; i++) {
        uint16_t nl = rd_u16(tp); tp += 2;
        char name[64];
        memcpy(name, tp, nl); name[nl] = 0; tp += nl;
        uint32_t off = rd_u32(tp); uint32_t nb = rd_u32(tp + 4); tp += 8;

        float *dst = NULL;
        size_t elems = nb / 2;   /* fp16 elements */

        if (!strcmp(name, "embedding.weight")) dst = (float *)arena, arena += nb * 2, m->embed_w = dst;
        else if (!strcmp(name, "final_norm.weight")) dst = (float *)arena, arena += nb * 2, m->final_norm_w = dst;
        else if (!strcmp(name, "cos")) dst = (float *)arena, arena += nb * 2, m->cos = dst;
        else if (!strcmp(name, "sin")) dst = (float *)arena, arena += nb * 2, m->sin = dst;
        else if (!strcmp(name, "crf.trans")) dst = (float *)arena, arena += nb * 2, m->crf_trans = dst;
        else if (!strcmp(name, "crf.log_mask")) dst = (float *)arena, arena += nb * 2, m->crf_log_mask = dst;
        else if (!strcmp(name, "intent_head.weight")) dst = (float *)arena, arena += nb * 2, m->intent_w = dst;
        else if (!strcmp(name, "intent_head.bias")) dst = (float *)arena, arena += nb * 2, m->intent_b = dst;
        else if (!strcmp(name, "slot_head.weight")) dst = (float *)arena, arena += nb * 2, m->slot_w = dst;
        else if (!strcmp(name, "slot_head.bias")) dst = (float *)arena, arena += nb * 2, m->slot_b = dst;
        else if (!strncmp(name, "blocks.", 7)) {
            int li = name[7] - '0';
            v5_layer_t *L = &m->layers[li];
            const char *rest = name + 9;
            if      (!strcmp(rest, "attn.q.weight")) { dst = (float *)arena; arena += nb*2; L->q_w = dst; }
            else if (!strcmp(rest, "attn.q.bias"))   { dst = (float *)arena; arena += nb*2; L->q_b = dst; }
            else if (!strcmp(rest, "attn.k.weight")) { dst = (float *)arena; arena += nb*2; L->k_w = dst; }
            else if (!strcmp(rest, "attn.k.bias"))   { dst = (float *)arena; arena += nb*2; L->k_b = dst; }
            else if (!strcmp(rest, "attn.v.weight")) { dst = (float *)arena; arena += nb*2; L->v_w = dst; }
            else if (!strcmp(rest, "attn.v.bias"))   { dst = (float *)arena; arena += nb*2; L->v_b = dst; }
            else if (!strcmp(rest, "attn.o.weight")) { dst = (float *)arena; arena += nb*2; L->o_w = dst; }
            else if (!strcmp(rest, "attn.o.bias"))   { dst = (float *)arena; arena += nb*2; L->o_b = dst; }
            else if (!strcmp(rest, "attn_norm.weight")) { dst=(float*)arena; arena += nb*2; L->attn_norm_w = dst; }
            else if (!strcmp(rest, "ffn.norm.weight")) { dst=(float*)arena; arena += nb*2; L->ffn_norm_w = dst; }
            else if (!strcmp(rest, "ffn.w1.weight")) { dst = (float *)arena; arena += nb*2; L->w1_w = dst; }
            else if (!strcmp(rest, "ffn.w1.bias"))   { dst = (float *)arena; arena += nb*2; L->w1_b = dst; }
            else if (!strcmp(rest, "ffn.w2.weight")) { dst = (float *)arena; arena += nb*2; L->w2_w = dst; }
            else if (!strcmp(rest, "ffn.w2.bias"))   { dst = (float *)arena; arena += nb*2; L->w2_b = dst; }
            else if (!strcmp(rest, "ffn.w3.weight")) { dst = (float *)arena; arena += nb*2; L->w3_w = dst; }
            else if (!strcmp(rest, "ffn.w3.bias"))   { dst = (float *)arena; arena += nb*2; L->w3_b = dst; }
        }
        if (dst) {
            const uint8_t *src = b + off;
            for (size_t e = 0; e < elems; e++)
                dst[e] = fp16_to_f32(rd_u16(src + e * 2));
        } else if (getenv("V5_AUDIT")) {
            fprintf(stderr, "UNROUTED TENSOR: %s (off=%u nb=%u)\n", name, off, nb);
        }
        (void)cfg_off;
    }
    return 0;
}

void v5_free(v5_model_t *m) {
    if (m->owns) free(m->owns);
    m->owns = NULL;
}

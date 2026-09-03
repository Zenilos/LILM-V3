/**
 * test_v5_host.c - host test for the V5 fp16 inference kernel.
 *
 * Reads:
 *   cases.bin : u32 n_rows; then per row: u16 n_tokens, n_tokens x int16 ids
 *               (tokens[0] == CLS(0), then word ids)
 *   model.bin / model.toc  : fp16 blob + binary index
 * Writes:
 *   out.bin   : per row: u8 intent_id, u8 n_tokens, n_tokens x u8 tag_ids
 *
 * Built/run from run_host_test.py which generates cases.bin and scores out.bin
 * against the roundtrip_v4 reference metric.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "v5_model.h"

static uint8_t *load_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc((size_t)sz);
    if (fread(p, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    *len = (size_t)sz;
    return p;
}

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : ".";
    char path[512];
    size_t bloblen, toclen, caselen;
    snprintf(path, sizeof path, "%s/model.bin", dir); uint8_t *blob = load_file(path, &bloblen);
    snprintf(path, sizeof path, "%s/model.toc", dir); uint8_t *toc = load_file(path, &toclen);
    uint8_t *cases = load_file("cases.bin", &caselen);  /* from cwd */

    v5_model_t m;
    if (v5_load(&m, blob, toc)) { fprintf(stderr, "v5_load failed\n"); return 2; }

    FILE *out = fopen("out.bin", "wb");
    if (!out) { fprintf(stderr, "cannot open out.bin\n"); return 2; }

    const uint8_t *p = cases;
    uint32_t n_rows = rd_u32(p); p += 4;

    static int16_t tokens[V5_MAX_LEN];
    static uint8_t tag_ids[V5_MAX_LEN];
    float slot_logits[V5_MAX_LEN * V5_N_TAG];
    static float ilogits[V5_N_INTENT];

    FILE *ilout = fopen("intent_logits.bin", "wb");
    if (!ilout) { fprintf(stderr, "cannot open intent_logits.bin\n"); return 2; }
    FILE *slotout = fopen("slot_logits.bin", "wb");
    if (!slotout) { fprintf(stderr, "cannot open slot_logits.bin\n"); return 2; }

    for (uint32_t r = 0; r < n_rows; r++) {
        uint16_t n = (uint16_t)(p[0] | (p[1] << 8)); p += 2;
        for (int i = 0; i < n; i++) { int16_t v; memcpy(&v, p, 2); tokens[i] = v; p += 2; }
        int intent;
        if (v5_forward(&m, tokens, (int)n, &intent, ilogits, slot_logits)) { fprintf(stderr, "fwd fail\n"); return 2; }
        v5_decode_slots(&m, slot_logits, (int)n, tag_ids);
        fputc((uint8_t)intent, out);
        fputc((uint8_t)n, out);
        fwrite(tag_ids, 1, (size_t)n, out);
        fwrite(ilogits, sizeof(float), V5_N_INTENT, ilout);
        fwrite(slot_logits, sizeof(float), (size_t)n * V5_N_TAG, slotout);
    }
    fclose(ilout);
    fclose(slotout);
    fclose(out);
    v5_free(&m);
    free(blob); free(toc); free(cases);
    printf("wrote out.bin for %u rows\n", n_rows);
    return 0;
}

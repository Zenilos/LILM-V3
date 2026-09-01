/**
 * fsm.c - constrained decoder for the PlanCore wire format (ESP32).
 *
 * Portable port of serialize.FSM. Greedy decoding samples only from the legal
 * next-token set, so every emitted sequence decodes to a valid plan.
 *
 * Vocabulary layout (must match serialize.Vocab, ARCHITECTURE.md):
 *   base 4096, <plan><ok><no><eop> -> 4096..4099
 *   MOVE CLEAN PLAY SHOW HANDOVER STOP WAIT -> 4100..4106
 *   <lit:here><lit:everywhere> -> 4107..4108
 *   <lit:seconds/minutes/hours> -> 4109..4111
 *   <num:...> (20)              -> 4112..4131
 *   <s:0..127>  -> 4132..4259
 *   <e:0..127>  -> 4260..4387
 *   vocab_size  = 4388
 */

#include "fsm.h"
#include <string.h>

/* Per-intent arity, in WIRE_INTENTS order:
 * MOVE CLEAN PLAY SHOW HANDOVER STOP WAIT */
static const int INTENT_NREQ[FSM_N_INTENTS] = {1, 1, 1, 1, 1, 0, 2};
static const int INTENT_NSLOT[FSM_N_INTENTS] = {2, 1, 1, 2, 2, 0, 2};

typedef enum { SLOT_LOCATION, SLOT_COPY, SLOT_AMOUNT, SLOT_UNIT } slot_kind;

/* Slot kinds per (intent, slot index); -1 terminates the order. */
static const slot_kind INTENT_SK[FSM_N_INTENTS][2] = {
    {SLOT_LOCATION, SLOT_COPY}, /* MOVE    */
    {SLOT_LOCATION, -1},        /* CLEAN   */
    {SLOT_COPY,     -1},        /* PLAY    */
    {SLOT_COPY,     SLOT_COPY}, /* SHOW    */
    {SLOT_COPY,     SLOT_COPY}, /* HANDOVER*/
    {-1,            -1},        /* STOP    */
    {SLOT_AMOUNT,   SLOT_UNIT}, /* WAIT    */
};

static inline bool is_intent_tok(int tok) { return tok >= FSM_ID_INT0 && tok < FSM_ID_LIT0; }
static inline bool is_ptr_s_tok(int tok)  { return tok >= FSM_ID_PTR_S && tok < FSM_ID_PTR_E; }

bool fsm_is_intent(int tok) { return is_intent_tok(tok); }
bool fsm_is_ptr_e(int tok)  { return tok >= FSM_ID_PTR_E && tok < FSM_VOCAB_SIZE; }

static inline void set_bit(uint8_t *m, int tok) { m[tok >> 3] |= (uint8_t)(1u << (tok & 7)); }

void fsm_start(fsm_state_t *st, int n_input) {
    memset(st, 0, sizeof(*st));
    st->intent = -1;
    st->open_start = -1;
    st->n_input = n_input < FSM_MAX_INPUT ? n_input : FSM_MAX_INPUT;
}

bool fsm_legal_has(const uint8_t *mask, int tok) {
    return (mask[tok >> 3] >> (tok & 7)) & 1u;
}

void fsm_legal(const fsm_state_t *st, const fsm_ctx_t *ctx, uint8_t *mask) {
    if (st->done) return;
    int n_in = st->n_input;
    if (!st->gated) {
        set_bit(mask, FSM_ID_OK);
        set_bit(mask, FSM_ID_NO);
        return;
    }
    if (st->open_start >= 0) {
        for (int j = st->open_start; j < n_in; j++) set_bit(mask, FSM_ID_PTR_E + j);
        return;
    }
    if (st->intent < 0) {
        if (st->n_actions < FSM_MAX_ACTIONS)
            for (int i = 0; i < FSM_N_INTENTS; i++) set_bit(mask, FSM_ID_INT0 + i);
        if (st->n_actions >= 1) set_bit(mask, FSM_ID_EOP);
        return;
    }
    const int it = st->intent;
    if (st->slot_i >= INTENT_NSLOT[it]) {
        fsm_state_t n = *st; n.intent = -1;
        fsm_legal(&n, ctx, mask);
        return;
    }
    switch (INTENT_SK[it][st->slot_i]) {
        case SLOT_LOCATION:
            for (int i = 0; i < n_in; i++) set_bit(mask, FSM_ID_PTR_S + i);
            for (int i = 0; i < FSM_N_SENT; i++) set_bit(mask, FSM_ID_LIT0 + i);
            break;
        case SLOT_COPY:
            for (int i = 0; i < n_in; i++) set_bit(mask, FSM_ID_PTR_S + i);
            break;
        case SLOT_AMOUNT:
            for (int i = 0; i < FSM_N_AMOUNTS; i++) set_bit(mask, FSM_ID_LIT0 + FSM_N_SENT + FSM_N_UNITS + i);
            break;
        case SLOT_UNIT:
            for (int i = 0; i < FSM_N_UNITS; i++) set_bit(mask, FSM_ID_LIT0 + FSM_N_SENT + i);
            break;
        default: break;
    }
    /* optional slot (slot_i >= n_required) may terminate the action */
    if (st->slot_i >= INTENT_NREQ[it]) {
        fsm_state_t n = *st; n.intent = -1;
        fsm_legal(&n, ctx, mask);
    }
    (void)ctx;
}

int fsm_step(fsm_state_t *st, int tok) {
    if (st->done) return -1;
    if (!st->gated) {
        if (tok != FSM_ID_OK && tok != FSM_ID_NO) return -1;
        st->gated = true;
        if (tok == FSM_ID_NO) st->done = true;
        return 0;
    }
    if (tok == FSM_ID_EOP) { st->done = true; return 0; }
    if (is_intent_tok(tok)) {
        st->intent = tok - FSM_ID_INT0;
        st->slot_i = 0;
        st->n_actions++;
        return 0;
    }
    if (is_ptr_s_tok(tok)) { st->open_start = tok - FSM_ID_PTR_S; return 0; }
    if (fsm_is_ptr_e(tok)) { st->open_start = -1; st->slot_i++; goto close; }
    st->slot_i++;   /* literal */
close:
    if (st->intent >= 0 && st->slot_i >= INTENT_NSLOT[st->intent])
        st->intent = -1;
    return 0;
}

/* ---- self-test mirroring serialize.py's FSM cases ------------------------ */
#ifdef FSM_TEST
#include <stdio.h>
#include <assert.h>

int main(void) {
    fsm_state_t s;
    fsm_ctx_t ctx = {0};
    uint8_t m[FSM_MASK_BYTES];
    (void)ctx;

    /* refuse intent token where a value is required: MOVE requires location */
    fsm_start(&s, 6);
    assert(fsm_step(&s, FSM_ID_OK) == 0);
    assert(fsm_step(&s, FSM_ID_INT0 + 0) == 0);      /* MOVE */
    memset(m, 0, sizeof m);
    fsm_legal(&s, &ctx, m);
    assert(!fsm_legal_has(m, FSM_ID_EOP));           /* location required */
    assert(!fsm_legal_has(m, FSM_ID_INT0 + 0));      /* no next intent yet */
    /* fill the required location with a pointer; person becomes optional */
    assert(fsm_step(&s, FSM_ID_PTR_S + 2) == 0);
    assert(fsm_step(&s, FSM_ID_PTR_E + 2) == 0);
    memset(m, 0, sizeof m);
    fsm_legal(&s, &ctx, m);
    assert(fsm_legal_has(m, FSM_ID_EOP));            /* person optional */
    /* end pointer can never precede its start */
    fsm_state_t s2;
    fsm_start(&s2, 10);
    assert(fsm_step(&s2, FSM_ID_OK) == 0);
    assert(fsm_step(&s2, FSM_ID_INT0 + 1) == 0);     /* CLEAN */
    assert(fsm_step(&s2, FSM_ID_PTR_S + 5) == 0);
    memset(m, 0, sizeof m);
    fsm_legal(&s2, &ctx, m);
    for (int j = 0; j < 5; j++)
        assert(!fsm_legal_has(m, FSM_ID_PTR_E + j)); /* only >= start allowed */
    assert(fsm_legal_has(m, FSM_ID_PTR_E + 5));

    printf("fsm.c self-test OK (vocab=%d, mask=%dB)\n", FSM_VOCAB_SIZE, FSM_MASK_BYTES);
    return 0;
}
#endif

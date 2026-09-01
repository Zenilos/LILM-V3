/**
 * fsm.h - constrained decoder state for the PlanCore wire format.
 *
 * See ARCHITECTURE.md. The decoder can never emit an unparseable sequence;
 * fsm_legal() gives the mask of legal next tokens for a state, and greedy
 * argmax within that mask always decodes.
 */
#ifndef PLANCODE_FSM_H
#define PLANCODE_FSM_H

#include <stdint.h>
#include <stdbool.h>

#define FSM_MAX_ACTIONS 3
#define FSM_MAX_INPUT 128

/* Vocabulary layout (matches serialize.Vocab; see fsm.c). */
#define FSM_BASE_SIZE   4096
#define FSM_N_CTRL      4
#define FSM_N_INTENTS   7
#define FSM_N_SENT      2
#define FSM_N_UNITS     3
#define FSM_N_AMOUNTS   20
#define FSM_ID_PLAN     (FSM_BASE_SIZE + 0)
#define FSM_ID_OK       (FSM_BASE_SIZE + 1)
#define FSM_ID_NO       (FSM_BASE_SIZE + 2)
#define FSM_ID_EOP      (FSM_BASE_SIZE + 3)
#define FSM_ID_INT0     (FSM_BASE_SIZE + FSM_N_CTRL)
#define FSM_ID_LIT0     (FSM_ID_INT0 + FSM_N_INTENTS)
#define FSM_ID_PTR_S    (FSM_ID_LIT0 + FSM_N_SENT + FSM_N_UNITS + FSM_N_AMOUNTS)
#define FSM_ID_PTR_E    (FSM_ID_PTR_S + FSM_MAX_INPUT)
#define FSM_VOCAB_SIZE  (FSM_ID_PTR_E + FSM_MAX_INPUT)

/* One byte of bitmask covers 8 token ids. */
#define FSM_MASK_BYTES  ((FSM_VOCAB_SIZE + 7) / 8)

typedef struct {
    int n_input;      /* actual number of input tokens (<= FSM_MAX_INPUT) */
    bool gated;       /* passed the <ok>/<no> gate */
    bool done;
    int intent;       /* current intent index (0..6) or -1 between actions */
    int slot_i;       /* slot index within the current intent */
    int open_start;   /* open pointer start index, or -1 */
    int n_actions;
} fsm_state_t;

/* context passed to fsm_legal (holds runtime input length) */
typedef struct {
    int n_input;
} fsm_ctx_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Initial state for an utterance of n_input tokens. */
void fsm_start(fsm_state_t *st, int n_input);

/* Fill `mask` (FSM_MASK_BYTES bytes, pre-zeroed by the caller) with the legal
 * next-token set for `st`. zero the mask before each call. */
void fsm_legal(const fsm_state_t *st, const fsm_ctx_t *ctx, uint8_t *mask);

/* Returns true if token `tok` is in the legal mask. */
bool fsm_legal_has(const uint8_t *mask, int tok);

/* Advance the state with a validated token in the current legal set.
 * Returns 0 on success, nonzero if illegal. */
int fsm_step(fsm_state_t *st, int tok);

/* Helpers */
bool fsm_is_intent(int tok);
bool fsm_is_ptr_e(int tok);

#ifdef __cplusplus
}
#endif

#endif /* PLANCODE_FSM_H */

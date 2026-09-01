"""Wire format between the ESP32 model and the DSL.

The model never emits JSON. It emits a short token sequence in which slot
values are either COPY POINTERS into the input (a start/end token pair naming
token indices in the utterance) or LITERAL tokens from a small closed set.

    "go to my daughter who is by my desk"
    <plan> <ok> <MOVE> <s:7> <e:8> <s:3> <e:4> <eop>
                       ^location^  ^person^

Why pointers: slot values are almost always spans of the input, so copying
them makes value hallucination structurally impossible, generalizes to unseen
room and person names for free, and costs 2 tokens regardless of value length.

Slot order is implied by the intent (see dsl.SLOT_ORDER), so no key tokens are
needed. The optional slot's presence is signalled by whether a value token or
a terminator (next intent / <eop>) follows the required slots.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dsl import (
    SLOTS,
    SLOT_ORDER,
    DURATION_UNITS,
    LOCATION_SENTINELS,
    MAX_ACTIONS,
    Action,
    normalize_slot,
)

MAX_INPUT_TOKENS = 128

# Intents the model can emit. UNAVAILABLE is carried by <no> instead, so it
# never appears as an intent token on the wire.
WIRE_INTENTS = tuple(i for i in SLOTS if i != "UNAVAILABLE")

# Slots whose value is a copy pointer. Everything else is a literal.
COPY_SLOTS = ("location", "object", "recipient", "file", "message", "person")

# Closed literal sets. duration_* are ALWAYS literals, never copies, because
# they need canonicalization anyway ("half an hour" -> 30 minutes) and a single
# uniform rule removes an ambiguous supervision case.
DURATION_AMOUNTS = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                    "12", "15", "20", "25", "30", "45", "60", "90", "120")

CTRL = ("<plan>", "<ok>", "<no>", "<eop>")


class Vocab:
    """Special tokens appended in contiguous blocks after the base tokenizer,
    so the FSM can mask a whole block with two integer comparisons."""

    def __init__(self, base_size: int = 4096, max_input: int = MAX_INPUT_TOKENS):
        self.base_size = base_size
        self.max_input = max_input
        toks: list[str] = list(CTRL)
        toks += [f"<{i}>" for i in WIRE_INTENTS]
        toks += [f"<lit:{v}>" for v in LOCATION_SENTINELS]
        toks += [f"<lit:{v}>" for v in DURATION_UNITS]
        toks += [f"<num:{v}>" for v in DURATION_AMOUNTS]
        self.lit_end = base_size + len(toks)
        self.ptr_s0 = self.lit_end
        self.ptr_e0 = self.ptr_s0 + max_input
        toks += [f"<s:{i}>" for i in range(max_input)]
        toks += [f"<e:{i}>" for i in range(max_input)]
        self.specials = tuple(toks)
        self.id = {t: base_size + i for i, t in enumerate(toks)}
        self.tok = {v: k for k, v in self.id.items()}
        self.size = base_size + len(toks)
        self.intent_ids = {i: self.id[f"<{i}>"] for i in WIRE_INTENTS}
        self.id_intent = {v: k for k, v in self.intent_ids.items()}
        self.lit_start = base_size + len(CTRL) + len(WIRE_INTENTS)

    # --- block predicates -------------------------------------------------
    def is_intent(self, t: int) -> bool:
        return t in self.id_intent

    def is_lit(self, t: int) -> bool:
        return self.lit_start <= t < self.lit_end

    def is_ptr_s(self, t: int) -> bool:
        return self.ptr_s0 <= t < self.ptr_e0

    def is_ptr_e(self, t: int) -> bool:
        return self.ptr_e0 <= t < self.ptr_e0 + self.max_input

    def s(self, i: int) -> int:
        return self.ptr_s0 + i

    def e(self, i: int) -> int:
        return self.ptr_e0 + i

    def lit(self, slot: str, value: str) -> int:
        key = f"<num:{value}>" if slot == "duration_amount" else f"<lit:{value}>"
        if key not in self.id:
            raise Unencodable(f"no literal token for {slot}={value!r}")
        return self.id[key]


class Unencodable(ValueError):
    """The example cannot be expressed in the wire format. Used as the
    verification gate for LLM paraphrases: if a paraphrase no longer contains
    a slot value as a span, the pair is dropped rather than mislabelled."""


# --- tokenization -------------------------------------------------------------
# The real tokenizer is the pruned SmolLM2 BPE with return_offsets_mapping.
# This reference implementation is word-level so the format can be tested and
# the generator run without the model repo present. Both satisfy: tokenize(str)
# -> list[str], and a value's tokens appear contiguously if the value is a span.

_WORD_RE = re.compile(r"[\w.']+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def find_span(utt_toks: list[str], value: str) -> tuple[int, int]:
    """Inclusive token span of `value` in the utterance.

    Returns the FIRST occurrence. Which occurrence is chosen does not matter:
    the metric compares decoded strings, not spans, so any occurrence that
    decodes to the right value is equally correct. Picking the first makes the
    label deterministic.
    """
    v = tokenize(value)
    if not v:
        raise Unencodable(f"empty value {value!r}")
    for i in range(len(utt_toks) - len(v) + 1):
        if utt_toks[i:i + len(v)] == v:
            return i, i + len(v) - 1
    # tolerate a leading article the label dropped ("the kitchen" -> "kitchen")
    stripped = tokenize(normalize_slot("location", value))
    if stripped and stripped != v:
        for i in range(len(utt_toks) - len(stripped) + 1):
            if utt_toks[i:i + len(stripped)] == stripped:
                return i, i + len(stripped) - 1
    raise Unencodable(f"value {value!r} is not a span of the utterance")


# --- encode / decode ----------------------------------------------------------
def encode(text: str, actions: list[Action], vocab: Vocab) -> list[int]:
    utt = tokenize(text)
    if len(utt) > vocab.max_input:
        raise Unencodable(f"utterance is {len(utt)} tokens, limit {vocab.max_input}")
    out = [vocab.id["<plan>"]]
    if len(actions) == 1 and actions[0].intent == "UNAVAILABLE":
        return out + [vocab.id["<no>"]]      # <no> is terminal; no <eop>
    out.append(vocab.id["<ok>"])
    if not 1 <= len(actions) <= MAX_ACTIONS:
        raise Unencodable(f"{len(actions)} actions")
    for a in actions:
        if a.intent == "UNAVAILABLE":
            raise Unencodable("UNAVAILABLE cannot appear beside other actions")
        out.append(vocab.intent_ids[a.intent])
        for slot in SLOT_ORDER[a.intent]:
            val = a.slots.get(slot)
            if val is None:
                continue                      # absent optional slot: no token
            if slot in COPY_SLOTS and val not in LOCATION_SENTINELS:
                i, j = find_span(utt, val)
                out += [vocab.s(i), vocab.e(j)]
            else:
                out.append(vocab.lit(slot, normalize_slot(slot, val)))
    out.append(vocab.id["<eop>"])
    return out


def decode(text: str, toks: list[int], vocab: Vocab) -> list[Action]:
    utt = tokenize(text)
    it = iter(toks)

    def nxt():
        return next(it, None)

    t = nxt()
    if t == vocab.id["<plan>"]:
        t = nxt()
    if t == vocab.id["<no>"]:
        return [Action("UNAVAILABLE", {})]
    if t != vocab.id["<ok>"]:
        raise ValueError("expected <ok> or <no>")

    actions: list[Action] = []
    t = nxt()
    while t is not None and t != vocab.id["<eop>"]:
        if not vocab.is_intent(t):
            raise ValueError(f"expected intent token, got {vocab.tok.get(t, t)}")
        intent = vocab.id_intent[t]
        slots: dict[str, str] = {}
        order = SLOT_ORDER[intent]
        n_req = len(SLOTS[intent]["required"])
        t = nxt()
        for k, slot in enumerate(order):
            if t is None or vocab.is_intent(t) or t == vocab.id["<eop>"]:
                if k < n_req:
                    raise ValueError(f"{intent} truncated before {slot}")
                break                          # optional slot absent
            if vocab.is_ptr_s(t):
                i = t - vocab.ptr_s0
                t2 = nxt()
                if t2 is None or not vocab.is_ptr_e(t2):
                    raise ValueError("start pointer not followed by end pointer")
                j = t2 - vocab.ptr_e0
                if not i <= j < len(utt):
                    raise ValueError(f"span ({i},{j}) out of range")
                slots[slot] = " ".join(utt[i:j + 1])
            elif vocab.is_lit(t):
                lit = vocab.tok[t]
                slots[slot] = lit[lit.index(":") + 1:-1]
            else:
                raise ValueError(f"unexpected {vocab.tok.get(t, t)} in slot position")
            t = nxt()
        actions.append(Action(intent, slots))
    if not actions:
        raise ValueError("empty plan")
    return actions


# --- constrained decoding -----------------------------------------------------
@dataclass
class FSMState:
    n_input: int
    gated: bool = False
    done: bool = False
    intent: str | None = None
    slot_i: int = 0
    open_start: int | None = None
    n_actions: int = 0


class FSM:
    """Legal-next-token mask derived from dsl.SLOTS. Guarantees every sampled
    sequence decodes to a valid plan, which turns the entire class of
    unparseable-output failures into worst-case wrong-value failures. ~100
    lines of C on device."""

    def __init__(self, vocab: Vocab):
        self.v = vocab

    def start(self, n_input: int) -> FSMState:
        return FSMState(n_input=min(n_input, self.v.max_input))

    def legal(self, st: FSMState) -> set[int]:
        v = self.v
        if st.done:
            return set()
        if not st.gated:
            return {v.id["<ok>"], v.id["<no>"]}
        if st.open_start is not None:            # must close the pointer pair
            return {v.e(j) for j in range(st.open_start, st.n_input)}
        if st.intent is None:                    # between actions
            out = set(v.intent_ids.values()) if st.n_actions < MAX_ACTIONS else set()
            if st.n_actions >= 1:
                out.add(v.id["<eop>"])
            return out
        order = SLOT_ORDER[st.intent]
        n_req = len(SLOTS[st.intent]["required"])
        if st.slot_i >= len(order):              # action complete
            return self.legal(_between(st))
        slot = order[st.slot_i]
        out = self._value_tokens(slot, st)
        if st.slot_i >= n_req:                   # optional slot may be skipped
            out |= self.legal(_between(st))
        return out

    def _value_tokens(self, slot: str, st: FSMState) -> set[int]:
        v = self.v
        if slot == "duration_amount":
            return {v.id[f"<num:{a}>"] for a in DURATION_AMOUNTS}
        if slot == "duration_unit":
            return {v.id[f"<lit:{u}>"] for u in DURATION_UNITS}
        out = {v.s(i) for i in range(st.n_input)}
        if slot == "location":
            out |= {v.id[f"<lit:{s}>"] for s in LOCATION_SENTINELS}
        return out

    def step(self, st: FSMState, tok: int) -> FSMState:
        v = self.v
        if tok not in self.legal(st):
            raise ValueError(f"illegal token {v.tok.get(tok, tok)}")
        if not st.gated:
            st.gated = True
            if tok == v.id["<no>"]:
                st.done = True
            return st
        if tok == v.id["<eop>"]:
            st.done = True
            return st
        if v.is_intent(tok):
            st.intent, st.slot_i = v.id_intent[tok], 0
            st.n_actions += 1
            return st
        if v.is_ptr_s(tok):
            st.open_start = tok - v.ptr_s0
            return st
        if v.is_ptr_e(tok):
            st.open_start = None
            st.slot_i += 1
            return _maybe_close(st)
        st.slot_i += 1                            # literal
        return _maybe_close(st)


def _between(st: FSMState) -> FSMState:
    return FSMState(n_input=st.n_input, gated=True, intent=None,
                    n_actions=st.n_actions)


def _maybe_close(st: FSMState) -> FSMState:
    if st.intent is not None and st.slot_i >= len(SLOT_ORDER[st.intent]):
        return _between(st)
    return st


def budget(n_actions: int = MAX_ACTIONS) -> int:
    """Worst-case output length, for the latency budget."""
    worst = max(
        sum(2 if s in COPY_SLOTS else 1 for s in SLOT_ORDER[i]) + 1
        for i in WIRE_INTENTS
    )
    return 2 + n_actions * worst + 1


if __name__ == "__main__":
    V = Vocab()
    fsm = FSM(V)

    def rt(text, actions):
        toks = encode(text, actions, V)
        st = fsm.start(len(tokenize(text)))
        for t in toks[1:]:
            st = fsm.step(st, t)
        assert st.done, f"FSM not terminated for {text!r}"
        back = decode(text, toks, V)
        from dsl import actions_match
        assert actions_match(back, actions), (text, back, actions)
        return toks

    t = rt("go to my daughter who is by my desk",
           [Action("MOVE", {"location": "my desk", "person": "my daughter"})])
    assert [V.tok[x] for x in t[:2]] == ["<plan>", "<ok>"]
    assert len(t) == 8, [V.tok[x] for x in t]

    rt("go to the kitchen and clean it",
       [Action("MOVE", {"location": "kitchen"}), Action("CLEAN", {"location": "kitchen"})])
    rt("clean up", [Action("CLEAN", {"location": "here"})])
    rt("wait half an hour then vacuum the bedroom",
       [Action("WAIT", {"duration_amount": "30", "duration_unit": "minutes"}),
        Action("CLEAN", {"location": "the bedroom"})])
    rt("bring me the cup", [Action("HANDOVER", {"object": "the cup"})])
    rt("bring john the red cup",
       [Action("HANDOVER", {"object": "the red cup", "recipient": "john"})])
    rt("play alarm.wav and stop",
       [Action("PLAY", {"file": "alarm.wav"}), Action("STOP", {})])
    rt("order me a pizza", [Action("UNAVAILABLE", {})])
    rt("go to the kitchen, clean it, then wait 5 minutes",
       [Action("MOVE", {"location": "kitchen"}), Action("CLEAN", {"location": "kitchen"}),
        Action("WAIT", {"duration_amount": "5", "duration_unit": "minutes"})])

    # a value that is not a span is rejected rather than mislabelled
    try:
        encode("clean the lounge", [Action("CLEAN", {"location": "living room"})], V)
        raise AssertionError("should have raised")
    except Unencodable:
        pass

    # FSM refuses an intent token where a value is required
    st = fsm.start(6)
    st = fsm.step(st, V.id["<ok>"])
    st = fsm.step(st, V.intent_ids["MOVE"])
    assert V.id["<eop>"] not in fsm.legal(st), "MOVE.location is required"
    assert V.intent_ids["CLEAN"] not in fsm.legal(st)
    # ...but may terminate once the required slot is filled
    st = fsm.step(st, V.s(2))
    st = fsm.step(st, V.e(2))
    assert V.id["<eop>"] in fsm.legal(st), "MOVE.person is optional"

    # end pointer can never precede its start
    st2 = fsm.start(10)
    st2 = fsm.step(st2, V.id["<ok>"])
    st2 = fsm.step(st2, V.intent_ids["CLEAN"])
    st2 = fsm.step(st2, V.s(5))
    assert all(x >= V.e(5) for x in fsm.legal(st2))

    print(f"vocab={V.size} (base {V.base_size} + {len(V.specials)} special)")
    print(f"worst-case output = {budget()} tokens")
    print("serialize.py OK")

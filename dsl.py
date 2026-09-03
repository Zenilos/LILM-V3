from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# Single source of truth. INTENTS and the tool schema are both derived from
# this, so the intent list can no longer drift out of sync with the slot table.
SLOTS = {
    "MOVE": {"required": ("location",), "optional": ("person",)},
    "CLEAN": {"required": ("location",), "optional": ()},
    "PLAY": {"required": (), "optional": ()},
    "SHOW": {"required": ("message",), "optional": ("person",)},
    "HANDOVER": {"required": (), "optional": ("person",)},
    "STOP": {"required": (), "optional": ()},
    "WAIT": {"required": ("duration_amount", "duration_unit"), "optional": ()},
    "UNAVAILABLE": {"required": (), "optional": ()},
}

# Migration shim for JSONL labelled before GET/GIVE were merged. Applied in
# Action.from_dict only. Delete once the corpus is regenerated.
INTENT_ALIASES = {"GET": "HANDOVER", "GIVE": "HANDOVER"}

INTENTS = tuple(SLOTS)

# Canonical emission order per intent. The decoder FSM and the dataset
# generator both read this, so arity is defined in exactly one place.
SLOT_ORDER = {
    intent: tuple(spec["required"]) + tuple(spec["optional"])
    for intent, spec in SLOTS.items()
}

ALL_SLOTS = (
    "location",
    "duration_amount",
    "duration_unit",
    "message",
    "person",
)

# Prose for the tool schema. Kept beside SLOTS so a new intent can't ship
# without a description.
INTENT_DESC = {
    "MOVE": "go to a place",
    "CLEAN": "vacuum/mop a place",
    "PLAY": "play music",
    "SHOW": "display a message",
    "HANDOVER": "bring/hand something to a person",
    "STOP": "halt everything",
    "WAIT": "pause for a duration",
    "UNAVAILABLE": "command not understood or not doable",
}

SLOT_DESC = {
    "location": "place: kitchen, my room, here, garage, where I cook",
    "person": "person referred to: the navigation referent for MOVE, the addressee for SHOW, the recipient for HANDOVER",
    "duration_amount": "how long to wait, numeric value",
    "duration_unit": "unit for duration_amount",
    "message": "text to display",
}

MAX_ACTIONS = 3

# Locations with no corresponding span in the utterance ("clean up").
# Emitted as literal tokens by the model, not as copy pointers.
LOCATION_SENTINELS = ("here", "everywhere")

DURATION_UNITS = ("seconds", "minutes", "hours")

_UNIT_SYNONYMS = {
    "s": "seconds", "sec": "seconds", "secs": "seconds",
    "second": "seconds", "seconds": "seconds",
    "m": "minutes", "min": "minutes", "mins": "minutes",
    "minute": "minutes", "minutes": "minutes",
    "h": "hours", "hr": "hours", "hrs": "hours",
    "hour": "hours", "hours": "hours",
}

# Multi-word entries must be matched before their constituent words, so this
# is applied phrase-first against the whole string rather than per-token.
# NOTE: fuzzy quantities ("half an hour", "a couple of minutes") are NOT
# handled here. The generator canonicalizes those to a numeric
# duration_amount + duration_unit pair at label time.
_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
    "twenty": "20", "twenty five": "25", "thirty": "30",
    "forty": "40", "forty five": "45", "fifty": "50",
    "sixty": "60", "ninety": "90",
}

_NUMBER_RE = re.compile(
    r"\b(" + "|".join(
        re.escape(k) for k in sorted(_NUMBERS, key=lambda k: -len(k.split()))
    ) + r")\b"
)


def _is_set(v) -> bool:
    """Truthiness test that keeps 0 / "0" (a legitimate duration_amount)."""
    return v is not None and str(v).strip() != ""


@dataclass(frozen=True)
class Action:
    intent: str
    slots: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.intent not in SLOTS:
            raise ValueError(f"unknown intent: {self.intent}")
        allowed = set(SLOT_ORDER[self.intent])
        bad = set(self.slots) - allowed
        if bad:
            raise ValueError(f"slots {sorted(bad)} not allowed for {self.intent}")
        missing = [s for s in SLOTS[self.intent]["required"] if not _is_set(self.slots.get(s))]
        if missing:
            raise ValueError(f"{self.intent} missing required slots {missing}")
        # Coerce every value to str so a numeric duration_amount from JSON and
        # a string duration_amount from the DSL compare identically.
        object.__setattr__(
            self, "slots", {k: str(v) for k, v in self.slots.items() if _is_set(v)}
        )
        if "duration_amount" in self.slots:
            amt = normalize_slot("duration_amount", self.slots["duration_amount"])
            try:
                ok = float(amt) >= 0
            except ValueError:
                raise ValueError(
                    f"duration_amount not numeric: {self.slots['duration_amount']!r}"
                ) from None
            if not ok:
                raise ValueError(f"duration_amount must be >= 0, got {amt!r}")
        if "duration_unit" in self.slots:
            unit = normalize_slot("duration_unit", self.slots["duration_unit"])
            if unit not in DURATION_UNITS:
                raise ValueError(f"duration_unit must be one of {DURATION_UNITS}, got {unit!r}")

    def to_dict(self) -> dict:
        return {"intent": self.intent, "slots": dict(sorted(self.slots.items()))}

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        raw = d.get("slots") or {}
        intent = INTENT_ALIASES.get(d["intent"], d["intent"])
        return cls(intent=intent, slots={k: v for k, v in raw.items() if _is_set(v)})


def normalize_value(text: str) -> str:
    t = str(text).lower().strip()
    t = re.sub(r"[.,!?;:\"']+$", "", t)
    t = re.sub(r"^(please\s+)?(the|a|an)\s+", "", t)
    t = re.sub(r"\s+", " ", t)
    t = _NUMBER_RE.sub(lambda m: _NUMBERS[m.group(1)], t)
    return t.strip()


def _canon_amount(v) -> str:
    s = str(v).strip()
    try:
        f = float(s)
    except ValueError:
        return normalize_value(s)
    return f"{f:g}"


def normalize_slot(slot: str, value) -> str:
    """Slot-aware normalization. duration_* have canonical forms; the rest are
    free text and fall through to normalize_value."""
    if slot == "duration_amount":
        return _canon_amount(value)
    if slot == "duration_unit":
        base = normalize_value(value)
        return _UNIT_SYNONYMS.get(base, base)
    return normalize_value(value)


def _values_match(slot: str, a, b) -> bool:
    na, nb = normalize_slot(slot, a), normalize_slot(slot, b)
    if na == nb:
        return True
    return na.replace(" ", "") == nb.replace(" ", "")


def action_matches(pred: Action, gold: Action) -> bool:
    if pred.intent != gold.intent:
        return False
    pred_slots = {k: v for k, v in pred.slots.items() if _is_set(v)}
    for slot in SLOTS[gold.intent]["required"]:
        if slot not in pred_slots:
            return False
        if not _values_match(slot, pred_slots[slot], gold.slots[slot]):
            return False
    for slot in SLOTS[gold.intent]["optional"]:
        g, p = gold.slots.get(slot), pred_slots.get(slot)
        if _is_set(g) != _is_set(p):
            return False
        if _is_set(p) and not _values_match(slot, p, g):
            return False
    return True


def actions_match(pred: Optional[list[Action]], gold: list[Action]) -> bool:
    if pred is None or len(pred) != len(gold):
        return False
    return all(action_matches(p, g) for p, g in zip(pred, gold))


def validate_plan(actions: list[Action]) -> None:
    """Plan-level rules that no single Action can enforce."""
    if not actions:
        raise ValueError("empty plan; a rejected utterance is [UNAVAILABLE], not []")
    if len(actions) > MAX_ACTIONS:
        raise ValueError(f"{len(actions)} actions exceeds MAX_ACTIONS={MAX_ACTIONS}")
    if any(a.intent == "UNAVAILABLE" for a in actions) and len(actions) > 1:
        raise ValueError("UNAVAILABLE is whole-utterance; it cannot appear beside other actions")


def load_actions(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                rec["actions"] = [Action.from_dict(a) for a in rec["actions"]]
            except (ValueError, KeyError) as e:
                raise ValueError(f"{path}:{i}: {e}") from e
            try:
                validate_plan(rec["actions"])
            except ValueError as e:
                raise ValueError(f"{path}:{i}: {e}") from e
            out.append(rec)
    return out


def export_tool_schema() -> list[dict]:
    """Generate tool_schema_verbose.json from SLOTS so the two can't drift."""
    props = {
        "intent": {
            "type": "string",
            "enum": list(INTENTS),
            "description": "; ".join(f"{i} {INTENT_DESC[i]}" for i in INTENTS),
        }
    }
    for slot in ALL_SLOTS:
        if not any(slot in SLOT_ORDER[i] for i in INTENTS):
            continue
        if slot == "duration_amount":
            props[slot] = {"type": "number", "minimum": 0, "description": SLOT_DESC[slot]}
        elif slot == "duration_unit":
            props[slot] = {"type": "string", "enum": list(DURATION_UNITS),
                           "description": SLOT_DESC[slot]}
        else:
            props[slot] = {"type": "string", "description": SLOT_DESC[slot]}
    return [{
        "name": "robot_action",
        "description": ("Execute one atomic robot action. Call once per action, in "
                        "execution order. Omit slot fields that the command does not mention."),
        "parameters": {"type": "object", "properties": props, "required": ["intent"]},
    }]


if __name__ == "__main__":
    # --- case folding and article stripping -------------------------------
    a1 = Action("MOVE", {"location": "my room"})
    a2 = Action.from_dict({"intent": "MOVE", "slots": {"location": "My Room"}})
    assert action_matches(a2, a1), a2
    assert action_matches(Action("MOVE", {"location": "the kitchen"}),
                          Action("MOVE", {"location": "kitchen"}))
    assert not action_matches(Action("MOVE", {"location": "The Room"}),
                              Action("MOVE", {"location": "my room"}))
    assert actions_match([a1], [Action("MOVE", {"location": "My room"})])

    # --- WAIT uses duration_amount + duration_unit ------------------------
    w_num = Action("WAIT", {"duration_amount": "5", "duration_unit": "seconds"})
    w_word = Action("WAIT", {"duration_amount": "five", "duration_unit": "secs"})
    assert action_matches(w_word, w_num)
    assert not actions_match([w_num, a1], [a1, w_num])          # order matters
    assert actions_match([w_word, a1], [w_num, a2])
    assert normalize_value("forty five seconds") == "45 seconds"
    assert action_matches(Action("WAIT", {"duration_amount": "forty five", "duration_unit": "s"}),
                          Action("WAIT", {"duration_amount": 45, "duration_unit": "seconds"}))
    assert action_matches(
        Action.from_dict({"intent": "WAIT",
                          "slots": {"duration_amount": 30, "duration_unit": "minutes"}}),
        Action("WAIT", {"duration_amount": "30", "duration_unit": "min"}))
    z = Action("WAIT", {"duration_amount": 0, "duration_unit": "seconds"})
    assert z.slots["duration_amount"] == "0"
    assert not action_matches(z, w_num)

    # --- CLEAN.location required ------------------------------------------
    assert action_matches(Action("CLEAN", {"location": "the Kitchen"}),
                          Action("CLEAN", {"location": "kitchen"}))
    for sentinel in LOCATION_SENTINELS:
        Action("CLEAN", {"location": sentinel})

    # "go to the kitchen and clean it" -> two ordered actions, anaphora resolved
    unresolved = [Action("MOVE", {"location": "the kitchen"}), Action("CLEAN", {"location": "it"})]
    gold = [Action("MOVE", {"location": "kitchen"}), Action("CLEAN", {"location": "kitchen"})]
    assert not actions_match(unresolved, gold), "unresolved anaphora must not score as correct"
    assert actions_match([Action("MOVE", {"location": "kitchen"}),
                          Action("CLEAN", {"location": "kitchen"})], gold)

    # --- optional slots penalize in both directions -----------------------
    m_bare = Action("MOVE", {"location": "my desk"})
    m_ref = Action("MOVE", {"location": "my desk", "person": "my daughter"})
    assert not action_matches(m_bare, m_ref)
    assert not action_matches(m_ref, m_bare)
    assert action_matches(Action("MOVE", {"location": "My Desk", "person": "My Daughter"}), m_ref)
    assert not action_matches(Action("SHOW", {"message": "dinner is ready", "person": "John"}),
                              Action("SHOW", {"message": "dinner is ready"}))

    # --- HANDOVER (merged GET/GIVE) ---------------------------------------
    assert action_matches(Action("HANDOVER", {}),
                          Action("HANDOVER", {}))
    assert action_matches(Action("HANDOVER", {"person": "John"}),
                          Action("HANDOVER", {"person": "john"}))
    # "bring me the cup": speaker is the recipient, so the slot stays empty
    assert action_matches(Action("HANDOVER", {}),
                          Action.from_dict({"intent": "HANDOVER", "slots": {}}))
    # legacy GET/GIVE labels migrate on load
    assert Action.from_dict({"intent": "GET", "slots": {}}).intent == "HANDOVER"
    assert Action.from_dict({"intent": "GIVE",
                             "slots": {"person": "John"}}).intent == "HANDOVER"

    # --- plan-level rules --------------------------------------------------
    validate_plan([Action("UNAVAILABLE", {})])
    validate_plan([a1, Action("CLEAN", {"location": "here"}), Action("STOP", {})])
    for bad_plan in ([],
                     [Action("UNAVAILABLE", {}), a1],
                     [a1, Action("UNAVAILABLE", {})],
                     [a1, a1, a1, a1]):
        try:
            validate_plan(bad_plan)
            raise AssertionError("should have raised")
        except ValueError:
            pass
    # a degenerate empty prediction can never score as a correct rejection
    assert not actions_match([], [Action("UNAVAILABLE", {})])

    # --- structural validation --------------------------------------------
    for bad in (lambda: Action("CLEAN", {}),
                lambda: Action("STOP", {"location": "kitchen"}),
                lambda: Action("UNAVAILABLE", {"location": "kitchen"}),
                lambda: Action("MOVE", {"location": "here", "recipient": "John"}),
                lambda: Action("WAIT", {"duration_amount": "5", "duration_unit": "fortnights"}),
                lambda: Action("WAIT", {"duration_amount": "-3", "duration_unit": "seconds"}),
                lambda: Action("WAIT", {"duration_amount": "soon", "duration_unit": "seconds"}),
                lambda: Action("WAKEUP", {"person": "John"}),
                lambda: Action("GET", {"object": "cup"})):
        try:
            bad()
            raise AssertionError("should have raised")
        except ValueError:
            pass

    # --- invariants the FSM depends on ------------------------------------
    assert all(len(SLOTS[i]["optional"]) <= 1 for i in INTENTS), \
        "FSM assumes at most one optional slot per intent"
    assert INTENTS == tuple(SLOTS)
    assert SLOT_ORDER["HANDOVER"] == ("person",)
    assert SLOT_ORDER["PLAY"] == ()
    assert [i for i in INTENTS if SLOTS[i]["optional"]] == ["MOVE", "SHOW", "HANDOVER"]
    assert max(len(SLOT_ORDER[i]) for i in INTENTS) == 2, "FSM value-slot budget"

    json.dumps(export_tool_schema())
    print("dsl.py OK")

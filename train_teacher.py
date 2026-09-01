"""Fine-tune SmolLM2-135M as the distillation teacher.

The teacher learns the *identical* wire format the student will emit, so
teacher->student logit alignment collapses to a gather over the pruned vocab.
Every example is rendered to `serialize`'s label vocabulary:

    <plan> <ok>/<no> <intent> <pointers/literals> <eop>

Pointers index into the *pruned SmolLM2 BPE* tokenization of the utterance
(not serialize.py's word-level reference), because both the teacher and the
device decoder slice spans out of the same BPE tokens. We therefore carry our
own BPE-aware encode/decode mirrors of serialize.py.

Training is teacher-forced on the gold wire label (no FSM needed during
training). At the end of every epoch we score exact-match with
``dsl.actions_match`` under the FSM-constrained greedy decoder, and keep the
checkpoint with the best validation EM. Runs resume from the last checkpoint.

Usage:
    python train_teacher.py \
        --train data/train_a.jsonl \
        --val data/val.jsonl \
        --out checkpoints/teacher \
        [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
)
from tokenizers import Tokenizer as TkTokenizer
from tokenizers import models

from dsl import SLOTS, SLOT_ORDER, Action, actions_match, validate_plan
from serialize import FSM, LOCATION_SENTINELS, MAX_ACTIONS, Vocab

log = logging.getLogger("train_teacher")

# Default hyper-parameters (overridable on the CLI).
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
PRUNED_VOCAB = 4096
MAX_SEQ_LEN = 256
LR = 2e-5
EPOCHS = 3
BATCH_SIZE = 32
GRAD_ACCUM = 8

NEG = -100  # ignore index for the causal-LM loss


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@dataclass
class Example:
    text: str
    actions: list[Action]
    kind: str


def load_rows(path: str, limit: int = 0) -> list[Example]:
    out: list[Example] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                acts = [Action.from_dict(a) for a in rec["actions"]]
                validate_plan(acts)
            except (ValueError, KeyError) as e:
                raise ValueError(f"{path}:{i}: {e}") from e
            out.append(Example(rec["text"], acts, rec.get("kind", "")))
            if limit and len(out) >= limit:
                break
    return out


# --------------------------------------------------------------------------
# Pruned tokenizer
# --------------------------------------------------------------------------
def build_pruned_tokenizer(
    base: PreTrainedTokenizerFast, keep: int = PRUNED_VOCAB
) -> PreTrainedTokenizerFast:
    """Reduce SmolLM2's ~49k BPE vocab to `keep` tokens while preserving every
    surviving token's ORIGINAL id. Keeping the ids intact is what lets
    teacher->student logit alignment be a ``gather`` over the retained index
    list. Merges whose result token falls outside the kept set are dropped, so
    the pruned BPE reassembles exactly the surviving vocabulary."""
    with tempfile.TemporaryDirectory() as tmp:
        base.save_pretrained(tmp)
        src = TkTokenizer.from_file(os.path.join(tmp, "tokenizer.json"))

    bpe: models.BPE = src.model
    vocab = bpe.get_vocab()
    ordered = sorted(vocab.items(), key=lambda kv: kv[1])
    kept_tokens = [t for t, _ in ordered[:keep]]
    new_vocab = {t: vocab[t] for t in kept_tokens}

    # `merges` is a list of (left, right) pairs in most tokenizers builds, but
    # some ship it as a dict of "left right" -> rank. Normalize both to pairs.
    raw = bpe.merges
    pairs = list(raw.items()) if isinstance(raw, dict) else list(raw)

    survived = set(kept_tokens)
    kept_merges = []
    for m in pairs:
        a, b = m[0], m[1]
        res = a + b
        if a in survived and b in survived and res in new_vocab:
            kept_merges.append((a, b))
            survived.add(res)
    new_bpe = models.BPE(
        vocab=new_vocab,
        merges=kept_merges,
        unk_token=bpe.unk_token,
        dropout=bpe.dropout,
        continuing_subword_prefix=bpe.continuing_subword_prefix,
        end_of_word_suffix=bpe.end_of_word_suffix,
        fuse_unk=bpe.fuse_unk,
    )

    pruned = TkTokenizer(
        new_bpe,
        normalizer=src.normalizer,
        pre_tokenizer=src.pre_tokenizer,
        post_processor=src.post_processor,
        decoder=src.decoder,
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=pruned,
        unk_token=base.unk_token,
        bos_token=base.bos_token,
        eos_token=base.eos_token,
        pad_token=base.pad_token or base.eos_token,
    )


# --------------------------------------------------------------------------
# BPE-aware wire encode / decode (mirrors of serialize.py)
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    return " ".join(s.split()).lower()


def utt_ids(raw_text: str, vocab: Vocab,
            tok: PreTrainedTokenizerFast) -> tuple[list[int], list[tuple]]:
    """Tokenize an utterance, returning (BPE ids, character offset mapping).
    Both are truncated to vocab.max_input so pointer indices never exceed the
    on-device context window."""
    enc = tok(
        raw_text,
        add_special_tokens=False,
        truncation=True,
        max_length=vocab.max_input,
        return_offsets_mapping=True,
    )
    return enc["input_ids"], enc["offset_mapping"]


def find_span(raw_text: str, text_ids: list[int], offsets: list[tuple],
              tok: PreTrainedTokenizerFast, value: str) -> tuple[int, int]:
    """Inclusive BPE-token span [i, j] of `value` in the utterance.

    `text_ids`/`offsets` come from utt_ids(), and must be aligned. The value is
    located by character position in the lowercased utterance, then snapped to
    the enclosing token boundaries; the span is verified by decoding, and on a
    mismatch we fall back to an exhaustive window search. Returning the first
    occurrence is fine: the metric compares decoded strings, not spans.
    Raises ValueError when the value is not a literal span.
    """
    low_text = _norm(raw_text)
    for want in (value, _norm(value)):
        want = _norm(want)
        if not want:
            continue
        idx = low_text.find(want)
        if idx >= 0 and offsets:
            s, e = idx, idx + len(want)
            lo = next((k for k, (a, b) in enumerate(offsets) if b > s),
                      len(offsets) - 1)
            hi = next((k for k in range(len(offsets) - 1, -1, -1)
                       if offsets[k][0] < e), 0)
            if lo <= hi:
                span = tok.decode(text_ids[lo:hi + 1], skip_special_tokens=True)
                if _norm(span) == want or want in _norm(span):
                    return lo, hi
        # exhaustive window search over decoded token spans
        n = len(text_ids)
        for i in range(n):
            for j in range(i, n):
                span = tok.decode(text_ids[i:j + 1], skip_special_tokens=True)
                if _norm(span) == want:
                    return i, j
    raise ValueError(f"value {value!r} is not a span of the utterance")


def encode_bpe(raw_text: str, actions: list[Action], vocab: Vocab,
               tok: PreTrainedTokenizerFast) -> list[int]:
    """Gold label for a BPE-tokenized utterance (serialize.encode over BPE)."""
    text_ids, offsets = utt_ids(raw_text, vocab, tok)
    out = [vocab.id["<plan>"]]
    if len(actions) == 1 and actions[0].intent == "UNAVAILABLE":
        return out + [vocab.id["<no>"]]          # <no> is terminal; no <eop>
    out.append(vocab.id["<ok>"])
    if not 1 <= len(actions) <= MAX_ACTIONS:
        raise ValueError(f"{len(actions)} actions")
    for a in actions:
        if a.intent == "UNAVAILABLE":
            raise ValueError("UNAVAILABLE cannot appear beside other actions")
        out.append(vocab.intent_ids[a.intent])
        for slot in ["location", "object", "recipient", "file", "message",
                     "person"]:
            val = a.slots.get(slot)
            if val is None:
                continue
            if slot == "location" and val in LOCATION_SENTINELS:
                out.append(vocab.lit(slot, val))
            else:
                i, j = find_span(raw_text, text_ids, offsets, tok, val)
                out += [vocab.s(i), vocab.e(j)]
        for slot in ["duration_amount", "duration_unit"]:
            val = a.slots.get(slot)
            if val is not None:
                out.append(vocab.lit(slot, val))
    out.append(vocab.id["<eop>"])
    return out


def decode_bpe(text_ids: list[int], gen: list[int], vocab: Vocab,
               tok: PreTrainedTokenizerFast) -> list[Action]:
    """Decode generated label tokens back to Actions (serialize.decode over
    BPE). `gen` starts at the <plan>/<ok>/<no> gate."""
    it = iter(gen)

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
            raise ValueError(f"expected intent, got {vocab.tok.get(t, t)}")
        intent = vocab.id_intent[t]
        slots: dict[str, str] = {}
        n_req = len(SLOTS[intent]["required"])
        t = nxt()
        for k, slot in enumerate(SLOT_ORDER[intent]):
            if t is None or vocab.is_intent(t) or t == vocab.id["<eop>"]:
                if k < n_req:
                    raise ValueError(f"{intent} truncated before {slot}")
                break
            if vocab.is_ptr_s(t):
                i = t - vocab.ptr_s0
                t2 = nxt()
                if t2 is None or not vocab.is_ptr_e(t2):
                    raise ValueError("start pointer not followed by end pointer")
                j = t2 - vocab.ptr_e0
                if not i <= j < len(text_ids):
                    raise ValueError(f"span ({i},{j}) out of range")
                span = tok.decode(text_ids[i:j + 1], skip_special_tokens=True)
                slots[slot] = " ".join(span.split())
            elif vocab.is_lit(t):
                lit = vocab.tok[t]
                slots[slot] = lit[lit.index(":") + 1:-1]
            else:
                raise ValueError(f"unexpected {vocab.tok.get(t, t)} in slot")
            t = nxt()
        actions.append(Action(intent, slots))
    if not actions:
        raise ValueError("empty plan")
    return actions


# --------------------------------------------------------------------------
# Dataset / collation
# --------------------------------------------------------------------------
class WireDataset(Dataset):
    """Each item: input_ids = utterance + gold label; labels supervise the
    label positions and -100 (ignored) over the utterance prefix."""

    def __init__(self, rows: list[Example], vocab: Vocab, tok, max_len: int):
        self.items = []
        for r in rows:
            label = encode_bpe(r.text, r.actions, vocab, tok)
            text_ids, _ = utt_ids(r.text, vocab, tok)
            if len(text_ids) == 0:
                continue
            if len(text_ids) + len(label) > max_len:
                continue                       # too long for the budget
            self.items.append((text_ids, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        utt, label = self.items[i]
        return utt, label


def collate_fn(batch, max_len: int):
    utts, labels = zip(*batch)
    n = len(batch)
    b = max(len(u) + len(l) for u, l in zip(utts, labels))
    b = min(b, max_len)
    input_ids = torch.zeros((n, b), dtype=torch.long)
    lab = torch.full((n, b), NEG, dtype=torch.long)
    for i, (u, l) in enumerate(zip(utts, labels)):
        seq = u + l
        input_ids[i, :len(seq)] = torch.tensor(seq)
        # label[i] targets the next token; supervise the first label token
        # after the last utterance token through the final <eop>.
        lab[i, len(u):len(u) + len(l)] = torch.tensor(l)
    return input_ids, lab


# --------------------------------------------------------------------------
# FSM-constrained greedy decoder (mirrors serialize.FSM)
# --------------------------------------------------------------------------
def _logits_to_token(logits: torch.Tensor, legal: set[int]) -> int:
    mask = torch.full_like(logits, float("-inf"))
    ids = torch.tensor(sorted(legal), device=logits.device, dtype=torch.long)
    mask[ids] = 0.0
    return int((logits + mask).argmax().item())


def decode_utt(model, text: str, vocab: Vocab, fsm: FSM, tok,
               device: torch.device) -> Optional[list[Action]]:
    """Constrained greedy decode of one utterance into Actions."""
    ids = tok.encode(text, add_special_tokens=False)[:vocab.max_input]
    if not ids:
        return [Action("UNAVAILABLE", {})]
    inp = torch.tensor([ids], device=device)
    st = fsm.start(len(ids))
    gen: list[int] = []
    with torch.no_grad():
        logits = model(inp).logits[:, -1, :]
        while not st.done and len(gen) < 64:
            tok_id = _logits_to_token(logits[0], fsm.legal(st))
            gen.append(tok_id)
            st = fsm.step(st, tok_id)
            if st.done:
                break
            inp = torch.cat([inp, torch.tensor([[tok_id]], device=device)], dim=1)
            logits = model(inp).logits[:, -1, :]
    try:
        return decode_bpe(ids, gen, vocab, tok)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(model, rows, vocab, fsm, tok, device) -> dict:
    model.eval()
    em = total = 0
    for r in rows:
        gold = [a for a in r.actions]
        pred = decode_utt(model, r.text, vocab, fsm, tok, device)
        total += 1
        if actions_match(pred, gold):
            em += 1
    return {"em": em / max(1, total), "n": total}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune SmolLM2 teacher")
    ap.add_argument("--train", default="data/train_a.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="checkpoints/teacher")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap training rows for smoke tests")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --out")
    ap.add_argument("--num-workers", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    train_rows = load_rows(args.train, args.limit)
    val_rows = load_rows(args.val)
    log.info("train=%d val=%d", len(train_rows), len(val_rows))

    log.info("loading base model & tokenizer %s", args.model)
    base_tok = AutoTokenizer.from_pretrained(args.model)
    tok = build_pruned_tokenizer(base_tok, PRUNED_VOCAB)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    # First PRUNED_VOCAB ids keep their original meaning; specials occupy
    # PRUNED_VOCAB .. Vocab.size-1, exactly as serialize.Vocab lays them out.
    model.resize_token_embeddings(Vocab().size)
    model.config.pad_token_id = tok.pad_token_id
    model.to(device)

    vocab = Vocab()                      # base_size=4096, max_input=128
    fsm = FSM(vocab)
    train_ds = WireDataset(train_rows, vocab, tok, args.max_seq_len)
    log.info("encodable training examples: %d/%d",
             len(train_ds), len(train_rows))
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, args.max_seq_len),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    os.makedirs(args.out, exist_ok=True)
    best_pt = os.path.join(args.out, "best.pt")
    last_pt = os.path.join(args.out, "last.pt")

    start_epoch = 0
    global_step = 0
    best_em = 0.0
    if args.resume and os.path.exists(last_pt):
        ck = torch.load(last_pt, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]
        best_em = ck["best_em"]
        log.info("resumed at epoch %d (best_em=%.4f)", start_epoch, best_em)

    log.info(
        "training: lr=%.2e epochs=%d batch=%d grad_accum=%d max_seq=%d",
        args.lr, args.epochs, args.batch_size, args.grad_accum, args.max_seq_len,
    )

    model.train()
    optimizer.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        running = 0.0
        count = 0
        pbar = range(len(loader))
        try:
            from tqdm import tqdm
            pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}",
                        unit="batch")
        except ImportError:
            pbar = loader

        for step, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()

            running += out.loss.item() * input_ids.size(0)
            count += input_ids.size(0)

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                optimizer.zero_grad()

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{out.loss.item():.4f}")

        avg_loss = running / max(1, count)
        log.info("epoch %d train_loss=%.4f", epoch + 1, avg_loss)

        # Evaluate on validation EM, save best.
        metrics = evaluate(model, val_rows, vocab, fsm, tok, device)
        log.info("epoch %d val_em=%.4f (n=%d)", epoch + 1,
                 metrics["em"], metrics["n"])
        if metrics["em"] > best_em:
            best_em = metrics["em"]
            torch.save({"model": model.state_dict(), "em": best_em,
                        "epoch": epoch}, best_pt)
            log.info("new best val_em=%.4f saved to %s", best_em, best_pt)

        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "scheduler": scheduler.state_dict(), "epoch": epoch,
              "global_step": global_step, "best_em": best_em}
        torch.save(ck, last_pt)

    log.info("done. best val_em=%.4f", best_em)


if __name__ == "__main__":
    main()

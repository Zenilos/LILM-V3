#!/usr/bin/env python3
"""Driver for the V5 fp16 C kernel host test.

1. Builds the shared vocab + writes cases.bin (rows of int16 token ids, CLS
   prepended, [:39] truncation -- identical to roundtrip_v4).
2. Compiles firmware/tests/test_v5_host.c + firmware/src/v5_model.c.
3. Runs the C binary, reads out.bin (intent + tag ids per row).
4. Scores intent acc and slot value accuracy exactly like roundtrip_v4 and
   compares against the expected reference (59.3% / 39.8% on n=1500).
"""
import argparse, json, os, struct, subprocess, sys, tempfile, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/export_v5_fp16", help="export dir w/ model.bin+toc")
    ap.add_argument("--val", default=os.path.join(ROOT, "data/v4/val_balanced.jsonl"))
    ap.add_argument("--train", default=os.path.join(ROOT, "data/v4/train_balanced.jsonl"))
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()

    import mlx.core as mx
    from train_student import build_base_tok
    from v4_model import INTENTS, SLOT_LABELS, V4Model
    from v4_train import SLOT_FAMILY

    rows = [json.loads(l) for l in open(args.val)][: args.n]
    tr = [json.loads(l) for l in open(args.train)]
    bt = build_base_tok([" ".join(r["tokens"]) for r in tr], size=4096)
    w2i = bt.word2id
    vocab_size = len(w2i)

    with tempfile.TemporaryDirectory() as td:
        # ---- cases.bin ----
        with open(os.path.join(td, "cases.bin"), "wb") as f:
            f.write(struct.pack("<I", len(rows)))
            for r in rows:
                ids = [0] + [w2i.get(w, 0) for w in r["tokens"][:39]]
                f.write(struct.pack("<H", len(ids)))
                f.write(struct.pack("<%dh" % len(ids), *ids))

        # ---- compile C ----
        cc = os.environ.get("CC", "cc")
        incl = os.path.join(HERE, "..", "include")
        src = os.path.join(HERE, "..", "src", "v5_model.c")
        tst = os.path.join(HERE, "test_v5_host.c")
        binp = os.path.join(td, "test")
        subprocess.check_call([cc, "-O0", "-w", "-I", incl, src, tst,
                               "-o", binp, "-lm"])
        subprocess.check_call([binp, args.dir], cwd=td)

        # ---- read out.bin ----
        data = open(os.path.join(td, "out.bin"), "rb").read()
        ilogits = np.fromfile(os.path.join(td, "intent_logits.bin"),
                              dtype=np.float32)
        slotdata = np.fromfile(os.path.join(td, "slot_logits.bin"),
                               dtype=np.float32)

    # ---- numeric intent comparison vs MLX reference ----
    model = V4Model(vocab_size=vocab_size, d=192, n_layers=2, use_crf=True)
    params = {}
    npz = np.load(os.path.join(ROOT, "checkpoints", "v5crf_mask", "best.npz"),
                  allow_pickle=False)
    for k in npz.files:
        parts = k.split("."); d_ = params
        for i in range(len(parts) - 1):
            p, nxt = parts[i], parts[i + 1]
            if nxt.isdigit(): d_ = d_.setdefault(p, [])
            elif p.isdigit():
                idx = int(p)
                while len(d_) <= idx: d_.append({})
                d_ = d_[idx]
            else: d_ = d_.setdefault(p, {})
        d_[parts[-1]] = mx.array(npz[k])
    model.update(params)

    ref_argmax = []
    maxdiff_i = 0.0
    maxdiff_s = 0.0
    nmismatch = 0
    smismatch = 0
    sp = 0
    for i, r in enumerate(rows):
        n = 1 + len(r["tokens"][:39])
        c_i = ilogits[i * 8:(i + 1) * 8]
        c_s = slotdata[sp:sp + n * 9].reshape(n, 9); sp += n * 9
        ids = mx.array([0] + [w2i.get(w, 0) for w in r["tokens"][:39]])[None, :]
        mask = mx.ones_like(ids)
        it, sl = model(ids, mask)
        ref = np.array(it[0])
        ref_s = np.array(sl[0])
        ref_argmax.append(int(np.argmax(ref)))
        maxdiff_i = max(maxdiff_i, float(np.max(np.abs(ref - c_i))))
        maxdiff_s = max(maxdiff_s, float(np.max(np.abs(ref_s - c_s))))
        if int(np.argmax(ref)) != int(np.argmax(c_i)):
            nmismatch += 1
        if np.any(np.argmax(ref_s, 1) != np.argmax(c_s, 1)):
            smismatch += 1
    print(f"[intent logits] max|C - MLX| = {maxdiff_i:.3e}  argmax mismatches = {nmismatch}/{len(rows)}")
    print(f"[slot   logits] max|C - MLX| = {maxdiff_s:.3e}  rows w/ tag-argmax mismatch = {smismatch}/{len(rows)}")

    # ---- score (mirror roundtrip_v4) ----
    def decode(toks, labels):
        spans = collections.defaultdict(list); cur = None
        for j, tok in enumerate(toks):
            lab = SLOT_LABELS[int(labels[j + 1])]
            if lab == "O": cur = None; continue
            bio, fam = lab.split("-", 1)
            if bio == "B": spans[fam].append([tok]); cur = (fam, len(spans[fam]) - 1)
            elif bio == "I" and cur and cur[0] == fam: spans[fam][cur[1]].append(tok)
            else: cur = None
        return {f: [" ".join(s) for s in v] for f, v in spans.items()}

    INTENT_TO_ID = {i: k for k, i in enumerate(INTENTS)}

    pos = 0; ittot = 0; ic = 0
    fam_p = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        ittot += 1
        it_id = INTENT_TO_ID.get(r["intent"], 7)
        n = len([0] + [w2i.get(w, 0) for w in r["tokens"][:39]])
        intent = data[pos]; pos += 1
        m2 = data[pos]; pos += 1
        assert m2 == n, (m2, n)
        labels = list(data[pos:pos + n]); pos += n
        if intent == it_id: ic += 1
        dec = decode(r["tokens"][:39], labels)
        gold = collections.defaultdict(list)
        for key, tags in r.get("slots", {}).items():
            fam = SLOT_FAMILY.get(key.split("-", 1)[1])
            toks = r["tokens"][:40]; tt = tags.split(); run = []
            for j, t in enumerate(tt):
                if t != "O": run.append(toks[j])
                elif run: gold[fam].append(" ".join(run)); run = []
            if run: gold[fam].append(" ".join(run))
        for fam, gv in gold.items():
            pv = dec.get(fam, []); fam_p[fam][2] += len(gv)
            for g in gv:
                fam_p[fam][1] += 1
                if g in pv: fam_p[fam][0] += 1

    print(f"=== C KERNEL intent n={ittot} ===")
    print(f"intent_acc = {ic}/{ittot} ({100.0*ic/ittot:.1f}%)")
    for fam, (c, g, tot) in sorted(fam_p.items()):
        print(f"  {fam:<10} {c}/{g} ({100.0*c/g if g else 0.0:.1f}%)  [gold tokens]{tot}")


if __name__ == "__main__":
    main()

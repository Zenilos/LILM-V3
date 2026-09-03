"""Export a V4/V5 joint intent + slot checkpoint to the on-device binary.

Same packing contract as export.py (ternary 5-trits/byte linears + per-row
absmean fp16 scale, int8 embedding, fp16 norms/biases) but for the joint
`v4_model.V4Model` architecture (8 linear heads, CRF transition, `blocks.*`
naming — NOT the old generative ModelConfig). Produces:

  * model.tern      -- packed blob of sectioned tensors
  * manifest.json   -- byte offsets/shapes + model config + CRF table

On-device runtime: firmware/v5_model.c.

Usage:
    ~/p3.11/bin/python3 export_v4.py --model checkpoints/v5crf_qat/best.npz \
        --out build/export_v5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from v4_model import SLOT_LABELS, N_SLOT_CLASSES, INTENTS

_TRITS_PER_BYTE = 5
_BASE3 = np.array([1, 3, 9, 27, 81], dtype=np.int64)

# tensor names threaded with QAT are those we ternarize; everything else fp.
_TERN_SUFFIX = (".weight",)
_NORM_KEYS = {"attn_norm.weight", "ffn.norm.weight", "final_norm.weight"}
# CRF transition + structural log-mask stay fp16 (small additive matrix).
_CRF_TENSORS = {"crf.trans", "crf.log_mask"}


def ternary_channel(w: np.ndarray, t: float = 1.0) -> np.ndarray:
    scale = np.abs(w).mean(axis=-1, keepdims=True)
    wn = w / np.maximum(scale, 1e-9)
    th = 0.5 * t
    return np.where(wn > th, 1.0, np.where(wn < -th, -1.0, 0.0)).astype(np.int8)


def pack_trits(trits: np.ndarray) -> np.ndarray:
    codes = (trits.astype(np.int64) + 1)
    n = codes.size
    pad = (-n) % _TRITS_PER_BYTE
    codes = np.append(codes, np.zeros(pad, dtype=np.int64))
    codes = codes.reshape(-1, _TRITS_PER_BYTE)
    return (codes * _BASE3).sum(axis=1).astype(np.uint8)


def int8_rows(w: np.ndarray):
    scale = np.abs(w).max(axis=-1, keepdims=True) / 127.0 + 1e-9
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="build/export_v5")
    ap.add_argument("--ramp", type=float, default=1.0)
    a = ap.parse_args()

    data = np.load(a.model, allow_pickle=False)
    os.makedirs(a.out, exist_ok=True)

    # model config (the loader needs these to size activations/heads)
    cfg = {
        "d": 192, "n_layers": 2, "n_heads": 4, "head_dim": 32, "ffn": 384,
        "max_len": 64, "n_intent": len(INTENTS), "n_tag": N_SLOT_CLASSES,
        "vocab": int(data["embedding.weight"].shape[0]),
    }

    blob = bytearray()
    manifest = {"config": cfg, "tensors": {}, "order": [],
                "slot_labels": SLOT_LABELS, "intents": INTENTS}
    t_counts = {"tern": 0, "int8": 0, "fp16": 0}

    def add(name, raw, meta):
        meta["offset"] = len(blob)
        meta["nbytes"] = len(raw)
        blob.extend(raw)
        manifest["tensors"][name] = meta
        manifest["order"].append(name)
        t_counts[meta["kind"]] += 1

    for key in data.files:
        if key in ("cos", "sin"):
            continue  # RoPE recomputed on device
        arr = data[key]

        if key == "embedding.weight":
            q, scale = int8_rows(arr)
            add("embedding.weight.q", q.tobytes(),
                {"kind": "int8", "shape": list(q.shape)})
            add("embedding.weight.scale", scale.tobytes(),
                {"kind": "fp16", "shape": list(scale.shape)})
            continue

        if key in _CRF_TENSORS:
            add(key, arr.astype(np.float16).tobytes(),
                {"kind": "fp16", "shape": list(arr.shape)})
            continue

        if key.endswith(".weight") and arr.ndim == 2:
            # ternary linear: q/k/v/o, w1/w2/w3, intent_head, slot_head
            if key in _NORM_KEYS:
                add(key, arr.astype(np.float16).tobytes(),
                    {"kind": "fp16", "shape": list(arr.shape)})
                continue
            q = ternary_channel(arr, a.ramp)
            packed = pack_trits(q.reshape(-1))
            add(key, packed.tobytes(),
                {"kind": "tern", "shape": list(arr.shape),
                 "packed": f"{_TRITS_PER_BYTE}trits/byte"})
            scale = np.abs(arr).mean(axis=-1, keepdims=True).astype(np.float16)
            add(f"{key}.scale", scale.tobytes(),
                {"kind": "fp16", "shape": list(scale.shape)})
            continue

        if arr.ndim == 1:
            # biases + norm weights: fp16
            add(key, arr.astype(np.float16).tobytes(),
                {"kind": "fp16", "shape": list(arr.shape)})
            continue

        raise ValueError(f"unexpected shape {arr.shape} for {key}")

    with open(os.path.join(a.out, "model.tern"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total = len(blob)
    print(f"config: {cfg}")
    print(f"tensors: {manifest['order']}")
    for k in ("tern", "int8", "fp16"):
        print(f"  {k}: {t_counts[k]}")
    print(f"total bytes: {total:,} ({total/1e6:.2f} MB)")
    print(f"  -> {a.out}/model.tern + manifest.json")


if __name__ == "__main__":
    main()

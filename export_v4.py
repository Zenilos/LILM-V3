"""Export a V4/V5 joint intent + slot checkpoint to the on-device binary.

DEFAULT path is **plain fp16** (no quantization): the model is only ~706k
params / ~1.4 MB fp16, which fits comfortably in ESP32-S3 N8R8 flash+PSRAM.
Everything (linears, embedding, norms, biases, CRF) is stored fp16 and the
runtime (firmware/v5_model.c) does plain fp16 matmuls — simplest, no accuracy
loss, no QAT.

Optional `--mode ternary` emits the old ternary 5-trits/byte blob (7.9x smaller
~0.21 MB) if flash ever demands it, using the same contract as export.py. QAT
(`v4_train.py --t`) is only needed if that mode is used.

Produces model.bin + manifest.json (byte offsets/shapes + config + CRF table).

Usage:
    ~/p3.11/bin/python3 export_v4.py --model checkpoints/v5crf_mask/best.npz \
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

_NORM_KEYS = {"attn_norm.weight", "ffn.norm.weight", "final_norm.weight"}
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
    ap.add_argument("--mode", default="fp32", choices=["fp32", "fp16", "ternary"],
                    help="fp32 (default, lossless, 2.8 MB) | fp16 (1.4 MB, "
                         "small accuracy cost) | ternary (0.2 MB, needs QAT, "
                         "lossy)")
    ap.add_argument("--ramp", type=float, default=1.0)
    a = ap.parse_args()

    data = np.load(a.model, allow_pickle=False)
    os.makedirs(a.out, exist_ok=True)

    q = data["blocks.0.attn.q.weight"]
    d = int(data["embedding.weight"].shape[1])
    head_dim = 2 * int(data["cos"].shape[1])  # RoPE pairs = head_dim/2
    ffn = int(data["blocks.0.ffn.w1.weight"].shape[0])
    max_len = int(data["cos"].shape[0])
    n_heads = int(q.shape[0]) // head_dim
    n_layers = 1 + max(int(k.split(".")[1]) for k in data.files
                       if k.startswith("blocks.") and ".attn" in k)
    cfg = {
        "d": d, "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
        "ffn": ffn, "max_len": max_len, "n_intent": len(INTENTS),
        "n_tag": N_SLOT_CLASSES, "vocab": int(data["embedding.weight"].shape[0]),
    }
    kind = ("tern" if a.mode == "ternary" else "fp16")

    blob = bytearray()
    manifest = {"config": cfg, "mode": a.mode, "tensors": {}, "order": [],
                "slot_labels": SLOT_LABELS, "intents": INTENTS}
    t_counts = {"tern": 0, "int8": 0, "fp32": 0, "fp16": 0}

    def add(name, raw, meta):
        meta["offset"] = len(blob)
        meta["nbytes"] = len(raw)
        blob.extend(raw)
        manifest["tensors"][name] = meta
        manifest["order"].append(name)
        t_counts[meta["kind"]] += 1

    def add_fp16(name, arr):
        add(name, arr.astype(np.float16).tobytes(),
            {"kind": "fp16", "shape": list(arr.shape)})

    for key in data.files:
        if key in ("cos", "sin"):
            # CRITICAL: RoPE must be loaded verbatim from the checkpoint. The
            # checkpoint's cos/sin use different parameters than the reference
            # rope_freqs(), and recomputing on device would give wrong logits.
            arr = data[key]
            if a.mode == "fp32":
                add(key, arr.astype(np.float32).tobytes(),
                    {"kind": "fp32", "shape": list(arr.shape)})
            elif a.mode == "fp16":
                add_fp16(key, arr)
            else:
                add_fp16(key, arr)  # RoPE stays fp16 even in ternary mode
            continue
        arr = data[key]

        if a.mode == "fp32":
            add(key, arr.astype(np.float32).tobytes(),
                {"kind": "fp32", "shape": list(arr.shape)})
            continue
        if a.mode == "fp16":
            # everything fp16, simplest on-device path (no quantization)
            add_fp16(key, arr)
            continue

        # ---- ternary mode (lossy, needs QAT) ----
        if key == "embedding.weight":
            q, scale = int8_rows(arr)
            add("embedding.weight.q", q.tobytes(),
                {"kind": "int8", "shape": list(q.shape)})
            add("embedding.weight.scale", scale.tobytes(),
                {"kind": "fp16", "shape": list(scale.shape)})
            continue
        if key in _CRF_TENSORS or key in _NORM_KEYS:
            add_fp16(key, arr)
            continue
        if key.endswith(".weight") and arr.ndim == 2:
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
            add_fp16(key, arr)
            continue
        raise ValueError(f"unexpected shape {arr.shape} for {key}")

    with open(os.path.join(a.out, "model.bin"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total = len(blob)
    print(f"mode={a.mode} config: {cfg}")
    for k in ("tern", "int8", "fp32", "fp16"):
        print(f"  {k}: {t_counts[k]}")
    print(f"total bytes: {total:,} ({total/1e6:.2f} MB)")
    print(f"  -> {a.out}/model.bin + manifest.json")


if __name__ == "__main__":
    main()

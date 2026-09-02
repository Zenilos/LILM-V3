"""Export the trained student checkpoint to the on-device binary format.

Produces two files under `--out`:

  * `model.tern`   -- packed weights (single binary blob of sectioned tensors)
  * `manifest.json`-- byte layout + scales/norms the C runtime needs

Packing (ARCHITECTURE.md storage, ~3.6 MB total):
  * linear weights  -- ternary {-1,0,+1}, per-output-channel absmean scale,
    packed 5 trits/byte (~1.6 bit/weight)
  * embedding       -- int8, per-row scale (the lm_head is tied so it shares
    the embedding tensor and is not emitted twice)
  * RMSNorm weights -- fp16

The manifest records byte offsets/shapes so the ESP32 firmware can locate each
tensor. Ternarization mirrors train_student.ternary_ste at the converged ramp
value (default 1.0).

Usage:
    python export.py --model checkpoints/student/best.npz --out build/export
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from model import ModelConfig

_TRITS_PER_BYTE = 5
_BASE3 = np.array([1, 3, 9, 27, 81], dtype=np.int64)


def ternary_channel(w: np.ndarray, t: float = 1.0) -> np.ndarray:
    scale = np.abs(w).mean(axis=-1, keepdims=True)
    wn = w / np.maximum(scale, 1e-9)
    th = 0.5 * t
    return np.where(wn > th, 1.0, np.where(wn < -th, -1.0, 0.0)).astype(np.int8)


def pack_trits(trits: np.ndarray) -> np.ndarray:
    """Pack row-major trits {-1,0,+1} into uint8 bytes, 5 trits/byte.

    PACKING CONTRACT (the C firmware must decode this exact layout):
      * encode -1,0,+1 as base-3 digits 0,1,2
      * a byte at index b stores trits at flat positions 5b .. 5b+4:
          byte_b = SUM_{i=0..4} digit_{5b+i} * 3^i
      * the last byte may hold <5 trits (trailing zeros are padding; the
        element count is known from the tensor shape in manifest.json)
      * decode: digit_{5b+i} = (byte_b // 3^i) % 3
    """
    codes = (trits.astype(np.int64) + 1)  # -1,0,1 -> 0,1,2
    n = codes.size
    pad = (-n) % _TRITS_PER_BYTE
    codes = np.append(codes, np.zeros(pad, dtype=np.int64))
    codes = codes.reshape(-1, _TRITS_PER_BYTE)
    return (codes * _BASE3).sum(axis=1).astype(np.uint8)


def unpack_trits(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of pack_trits; reference decode for the C firmware and tests."""
    pb = packed.astype(np.int64)
    codes = np.empty(n, dtype=np.int64)
    for i in range(5):
        sel = np.arange(i, n, 5)
        codes[sel] = (pb[sel // 5] // _BASE3[i]) % 3
    return (codes - 1).astype(np.int8)


def roundtrip_selftest(w: np.ndarray, t: float = 1.0) -> bool:
    """Pack → unpack a random ternary tensor and require exact recovery."""
    rng = np.random.default_rng(0)
    wt = rng.normal(size=w.shape)
    q = ternary_channel(wt, t)
    rec = unpack_trits(pack_trits(q.reshape(-1)), q.size).reshape(q.shape)
    if not bool((rec == q).all()):
        return False
    q2, _ = int8_rows(np.float64(wt))
    s2 = np.abs(wt).max(axis=-1, keepdims=True) / 127.0 + 1e-9
    ref = np.clip(np.round(wt / s2), -127, 127).astype(np.int8)
    return bool((q2 == ref).all())


def int8_rows(w: np.ndarray):
    scale = np.abs(w).max(axis=-1, keepdims=True) / 127.0 + 1e-9
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="build/export")
    ap.add_argument("--ramp", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true",
                    help="round-trip pack/unpack a random tensor and exit "
                         "(locks the packing contract for the C firmware)")
    a = ap.parse_args()

    if a.selftest:
        if roundtrip_selftest(np.empty((16, 32))):
            print("selftest OK: ternary trit-packing + int8 round-trip exact")
            return
        raise SystemExit("selftest FAILED")

    data = np.load(a.model, allow_pickle=False)
    os.makedirs(a.out, exist_ok=True)

    # identify the tied embedding tensor (shape == vocab x d_model)
    cfg = ModelConfig()
    emb_key = "embedding.weight"
    if emb_key not in data.files:
        # fall back: any tensor whose width == d_model and rows == vocab
        cand = [k for k in data.files if data[k].shape == (cfg.vocab_size, cfg.d_model)]
        emb_key = cand[0] if cand else None
    if emb_key is None:
        raise SystemExit(f"could not locate embedding weight in {a.model}")

    blob = bytearray()
    manifest = {"config": vars(cfg), "tensors": {}, "order": [], "embedding": emb_key}
    t_counts = {"tern": 0, "int8": 0, "fp16": 0}

    def add(name, raw, meta):
        meta["offset"] = len(blob)
        meta["nbytes"] = len(raw)
        blob.extend(raw)
        manifest["tensors"][name] = meta
        manifest["order"].append(name)
        t_counts[meta["kind"]] += 1

    for key in data.files:
        # RoPE buffers are recomputed on device; not exported weights
        if key == "cos" or key == "sin" or key.startswith("cos.") or key.startswith("sin."):
            continue
        arr = data[key]

        if key == emb_key or (key.endswith(".weight")
                              and arr.shape[0] == cfg.vocab_size
                              and arr.shape[1] == cfg.d_model
                              and "layers" not in key
                              and "norm" not in key):
            # embedding: int8
            if key != emb_key:
                emb_key = key
                manifest["embedding"] = emb_key
            q, scale = int8_rows(arr)
            add(f"{key}.q", q.tobytes(),
                {"kind": "int8", "shape": list(q.shape)})
            add(f"{key}.scale", scale.tobytes(),
                {"kind": "fp16", "shape": list(scale.shape)})
        elif arr.ndim == 2:
            # linear: ternary
            q = ternary_channel(arr, a.ramp)
            packed = pack_trits(q.reshape(-1))
            add(key, packed.tobytes(),
                {"kind": "tern", "shape": list(arr.shape),
                 "packed": f"{_TRITS_PER_BYTE}trits/byte"})
            scale = np.abs(arr).mean(axis=-1, keepdims=True).astype(np.float16)
            add(f"{key}.scale", scale.tobytes(),
                {"kind": "fp16", "shape": list(scale.shape)})
        elif arr.ndim == 1:
            # norm / biases: fp16
            add(key, arr.astype(np.float16).tobytes(),
                {"kind": "fp16", "shape": list(arr.shape)})
        else:
            raise ValueError(f"unexpected shape {arr.shape} for {key}")

    with open(os.path.join(a.out, "model.tern"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total = len(blob)
    print(f"tensors: {manifest['order']}")
    for k in ("tern", "int8", "fp16"):
        print(f"  {k}: {t_counts[k]}")
    print(f"total weight bytes: {total:,} ({total/1e6:.2f} MB)")
    print(f"  -> {a.out}/model.tern + manifest.json")


if __name__ == "__main__":
    main()

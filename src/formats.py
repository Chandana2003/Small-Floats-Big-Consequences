"""
formats.py — tiny, readable “small-float” emulation + helpers.

We don't try to perfectly emulate hardware TF32/bfloat16/FP8 — we only
quantize mantissas to the right *order of magnitude* so we can study
accuracy/speed trade-offs on CPU/NumPy.

Key ideas, human-style:
- quantize(x, "fp8")     -> round x as if it had ~4 fraction bits.
- quantize(x, "bfloat16")-> ~7 fraction bits.
- quantize(x, "tf32")    -> ~10 fraction bits.
- quantize(x, "fp32")    -> no-op (we keep float32).
- split_fmt("fp8-fp32")  -> ("fp8", "fp32")  # inputs in fp8, accumulate in fp32
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

# Approximate fraction bits for each format.
_FBITS = {
    "fp32": 23,       # real IEEE-754 single precision
    "tf32": 10,       # NVIDIA TF32 uses 10 mantissa bits (approx)
    "bfloat16": 7,    # BF16 has 7 fraction bits
    "fp8": 4,         # toy FP8 (e4m3-ish) for our study
}

def _round_to_fbits(x: np.ndarray, fbits: int) -> np.ndarray:
    """Round x so it looks like it has `fbits` fraction bits.
    Trick: use frexp/ldexp to round mantissas to a grid of size 2^-fbits.
    """
    if fbits >= 23:  # fp32 path -> basically no rounding here
        return x.astype(np.float32, copy=False)
    # Work in float32 to avoid double doing us favors.
    xf = np.asarray(x, dtype=np.float32)
    m, e = np.frexp(xf)            # xf = m * 2**e, m in [0.5, 1)
    step = np.float32(2.0 ** (-fbits))
    m_q = np.round(m / step) * step
    # Guard: if rounding pushes m to 1.0, pull back within [0.5,1)
    m_q = np.where(m_q >= 1.0, np.float32(0.5), m_q)
    e = np.where(m_q == 0.5, e + 1, e)
    return np.ldexp(m_q, e).astype(np.float32)

def quantize(x: np.ndarray, fmt: str) -> np.ndarray:
    """Quantize array to our approximate small-float format."""
    fmt = fmt.lower()
    if fmt not in _FBITS:
        raise ValueError(f"Unknown format '{fmt}' (expected one of {list(_FBITS)})")
    return _round_to_fbits(x, _FBITS[fmt])

def split_fmt(fmt_pair: str) -> Tuple[str, str]:
    """'fp8-fp32' -> ('fp8','fp32')  (inputs format, accumulation mode)"""
    a, b = fmt_pair.split("-")
    return a.lower(), b.lower()

def join_fmt(inp: str, accum: str) -> str:
    return f"{inp}-{accum}"

def tol_for(fmt: str) -> float:
    """Default *sense-check* tolerances per format (used by plot aids)."""
    return {
        "fp32": 1e-6,
        "tf32": 1e-3,
        "bfloat16": 5e-3,
        "fp8": 2e-1,
    }[fmt.lower()]

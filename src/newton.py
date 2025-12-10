# src/newton.py
# Newton's method with small-float emulation.
# - Supports formats: fp32, tf32, bfloat16, fp8 (via formats.quantize)
# - Two accumulation modes:
#     * accum == "small": do updates in the small format (more error)
#     * accum == "fp32":  do math in fp32/float64, but quantize x per-iteration
#
# Logs to CSV with columns:
#   func, x0, iters, failed, fmt, accum
#
# "Failed" means we hit max_iters without |f(x)| <= tol.

from __future__ import annotations
import os
import csv
import math
import numpy as np
from .formats import quantize

# -------- functions we support (name -> (f, f'))
def _sinx_minus_x2(x: float) -> float:
    # "sinx_minus_x2" = sin(x) - x/2  (matching your earlier driver)
    return math.sin(x) - 0.5 * x

def _d_sinx_minus_x2(x: float) -> float:
    return math.cos(x) - 0.5

def _x3_minus_7(x: float) -> float:
    return x * x * x - 7.0

def _d_x3_minus_7(x: float) -> float:
    return 3.0 * x * x

_FUNS = {
    "sinx_minus_x2": (_sinx_minus_x2, _d_sinx_minus_x2),
    "x3_minus_7":    (_x3_minus_7,    _d_x3_minus_7),
}

def _ensure_header(path: str, header: list[str]) -> None:
    """Create file with header if it doesn't exist yet."""
    exists = os.path.exists(path)
    if not exists or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def _safe_div(a: float, b: float) -> float:
    # guard against zero derivative
    return a / b if b != 0.0 else float("inf")

def run_newton_and_log(
    out_csv: str,
    func_name: str,
    starts: list[float],
    fmt: str,
    accum: str,
    tol: float,
    max_iters: int,
) -> None:
    """
    Run Newton on each start in `starts` with the chosen small-float `fmt` and
    accumulation mode `accum` ("small" or "fp32"), and append results to CSV.
    """
    if func_name not in _FUNS:
        raise ValueError(f"Unknown function '{func_name}'. Options: {list(_FUNS.keys())}")

    f, df = _FUNS[func_name]
    _ensure_header(out_csv, ["func", "x0", "iters", "failed", "fmt", "accum"])

    for x0 in starts:
        iters = 0
        failed = 1  # mark success by setting to 0 when we converge

        # Start from x0 in the target format (quantize once up front)
        x = float(quantize(np.array(x0, dtype=np.float64), fmt))

        # Simple, readable loop with two modes
        for k in range(int(max_iters)):
            iters = k + 1

            # Evaluate f and f' at current x
            if accum == "small":
                # Quantize inputs and each value — this is the "harsh" setting
                xq = float(quantize(np.array(x), fmt))
                fq = float(quantize(np.array(f(xq)), fmt))
                dfq = float(quantize(np.array(df(xq)), fmt))
                # Update also in small format (we quantize the delta)
                delta = _safe_div(fq, dfq)
                delta = float(quantize(np.array(delta), fmt))
                x_new = float(quantize(np.array(xq - delta), fmt))
                f_val = abs(fq)
            else:
                # accum == "fp32": do math in high precision, but keep x quantized each step
                fq = f(x)
                dfq = df(x)
                delta = _safe_div(fq, dfq)  # done in float64
                x_new = x - delta           # float64 update
                # re-quantize state so the *stored* x still lives in the chosen fmt
                x_new = float(quantize(np.array(x_new), fmt))
                f_val = abs(f(x_new))       # check convergence in high precision

            x = x_new

            if f_val <= tol:
                failed = 0
                break

        # Append one line per start
        with open(out_csv, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([func_name, x0, iters, failed, fmt, accum])

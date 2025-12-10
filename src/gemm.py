# src/gemm.py
# Human-style comments for CS260:
# We build A and B with a target condition number (kappa), do one high-precision
# reference GEMM (float32), then simulate low-precision compute in two modes:
#   - accum='fp32'  : multiply in small fmt but accumulate in fp32
#   - accum='small' : also simulate "small-precision accumulation" by
#                     breaking the K dimension into blocks and quantizing
#                     after each partial sum (toy, but captures the idea).
#
# We log: n, kappa, fmt, accum, rel_error, gflops.

import os
import csv
import time
import math
import numpy as np

from .formats import quantize  # your quantizer for 'fp32','tf32','bfloat16','fp8'


# ---------- helpers ----------

def _ensure_csv_header(path, fieldnames):
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not exists:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def _append_row(path, row, fieldnames):
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)


def _ill_conditioned_matrix(n: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """
    Create an n×n matrix with approximately the requested condition number.
    We do an SVD-like construction: A = U * diag(s) * V^T with
    singular values spaced geometrically so that max(s)/min(s) ≈ kappa.
    """
    # Random orthogonals via QR
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n), dtype=np.float64))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n), dtype=np.float64))
    # Geometric spectrum from 1 to 1/kappa (so cond ≈ kappa)
    s = np.geomspace(1.0, 1.0 / float(kappa), num=n, dtype=np.float64)
    A = (Q1 * s) @ Q2.T   # Q1 * diag(s) * Q2^T  (broadcast multiplies rows of Q1)
    return A.astype(np.float32)


def _simulate_small_accum(Aq: np.ndarray, Bq: np.ndarray, fmt: str, block_k: int = 64) -> np.ndarray:
    """
    Simulate 'small' accumulation by chunking K and quantizing each partial result
    and the running sum. This is intentionally conservative vs. real hardware,
    but it's a simple teaching model:
       C = Q( Q(C + Q(A[:,k:k+bk]@B[k:k+bk,:])) )
    where Q is quantize(·, fmt).
    """
    n = Aq.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    K = Aq.shape[1]
    for k0 in range(0, K, block_k):
        k1 = min(k0 + block_k, K)
        # partial product in fp32 (matmul) but immediately quantized to 'fmt'
        partial = Aq[:, k0:k1] @ Bq[k0:k1, :]
        partial_q = quantize(partial, fmt)  # simulate limited accum precision
        C = quantize(C + partial_q, fmt)    # running sum also in small fmt
    return C.astype(np.float32)


# ---------- public API ----------

def run_gemm_and_log(out_csv: str, n: int, kappa: float, fmt: str, accum: str, rng: np.random.Generator) -> None:
    """
    Do one GEMM experiment and append a row to `out_csv`.

    Args
    ----
    out_csv : where to log (CSV with header if needed)
    n       : matrix size (n x n)
    kappa   : target condition number for A and B generators
    fmt     : 'fp32' | 'tf32' | 'bfloat16' | 'fp8' (the compute format)
    accum   : 'fp32' | 'small'  (accumulation mode)
    rng     : numpy Generator for reproducibility
    """

    # 1) Build A, B with the requested conditioning
    #    (we do both ill-conditioned — that keeps the product challenging)
    A = _ill_conditioned_matrix(n, kappa, rng)
    B = _ill_conditioned_matrix(n, kappa, rng)

    # 2) Reference result in fp32
    C_ref = A @ B  # fp32 on purpose

    # 3) Quantize inputs to the "small" format
    A_q = quantize(A, fmt)
    B_q = quantize(B, fmt)

    # 4) Compute C_hat according to accumulation mode
    t0 = time.perf_counter()
    if accum == "fp32":
        # Small-format inputs, but full-precision accumulator
        C_hat = (A_q @ B_q).astype(np.float32)
    elif accum == "small":
        # Simulate quantized accumulation via block-K summation
        # (block 64 is a reasonable compromise for speed vs. effect)
        C_hat = _simulate_small_accum(A_q, B_q, fmt, block_k=64)
    else:
        raise ValueError(f"Unknown accum mode: {accum}")
    t1 = time.perf_counter()

    # 5) Relative error
    num = np.linalg.norm(C_hat - C_ref, ord="fro")
    den = np.linalg.norm(C_ref,      ord="fro")
    rel_error = float(num / den) if den != 0.0 else 0.0

    # 6) Simple GFLOPs estimate: 2*n^3 flops / time
    gflops = (2.0 * n * n * n) / (t1 - t0) / 1e9 if (t1 - t0) > 0 else float("inf")

    # 7) Log
    fields = ["n", "kappa", "fmt", "accum", "rel_error", "gflops"]
    _ensure_csv_header(out_csv, fields)
    _append_row(
        out_csv,
        {
            "n": int(n),
            "kappa": f"{kappa:g}",
            "fmt": fmt,
            "accum": accum,
            "rel_error": f"{rel_error:.6e}",
            "gflops": f"{gflops:.3f}",
        },
        fields,
    )


# Make the module runnable for quick local smoke tests:
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out = "results/gemm.csv"
    for fmt in ["fp32", "tf32", "bfloat16", "fp8"]:
        for accum in ["fp32", "small"]:
            run_gemm_and_log(out, n=128, kappa=1e3, fmt=fmt, accum=accum, rng=rng)
    print("Wrote", out)

# src/cg.py
# Human notes (CS260):
# - This is a plain Conjugate Gradient (CG) solver on a symmetric
#   positive definite (SPD) system A x = b.
# - We build A as a 2-D 5-point Laplacian on an m×m grid (classic SPD).
# - "accum" controls the working precision:
#       * "fp32"  -> keep state in full float32
#       * "small" -> quantize vectors each iter to the chosen fmt
# - Optional jacobi preconditioner (M = diag(A)).
# - We log a compact summary to results/cg.csv and per-iteration
#   residuals to results/cg_residuals.csv (for your residual plots).
#
# New: `true_residual_every` (int or None)
#   If set, every N iterations we compute a true residual r_true = b - A x
#   in fp32 and log *that* value (helps diagnose drift).

from __future__ import annotations

import os
import csv
from typing import Optional, Tuple

import numpy as np

from .formats import quantize, tol_for  # reuse your quantizer + per-format tol


# ---------------- helpers: CSV ----------------

_SUMMARY_FIELDS = [
    "m", "fmt", "accum", "tol", "max_iters",
    "warmup_small", "restart_every", "precond",
    "iters", "converged"
]

_RESID_FIELDS = [
    "m", "fmt", "accum", "iter", "rel_resid", "tol", "kind"  # kind: "iter" or "true"
]


def _ensure_csv(path: str, fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _append_row(path: str, row: dict, fields: list[str]) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


# ---------------- problem builder ----------------

def _laplacian_2d(m: int) -> np.ndarray:
    """
    Build the SPD matrix A for a 2-D Poisson (5-point stencil) on an m×m grid.
    Size is n = m*m.
    """
    n = m * m
    A = np.zeros((n, n), dtype=np.float32)

    def idx(i: int, j: int) -> int:
        return i * m + j

    for i in range(m):
        for j in range(m):
            p = idx(i, j)
            A[p, p] = 4.0
            if i > 0:     A[p, idx(i - 1, j)] = -1.0
            if i < m - 1: A[p, idx(i + 1, j)] = -1.0
            if j > 0:     A[p, idx(i, j - 1)] = -1.0
            if j < m - 1: A[p, idx(i, j + 1)] = -1.0
    return A


def _apply_prec_jacobi(A: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Jacobi preconditioner: z = M^{-1} r with M = diag(A)."""
    Dinv = 1.0 / np.diag(A)
    return Dinv * r


# ---------------- CG core ----------------

def _maybe_q(x: np.ndarray, fmt: str, accum: str) -> np.ndarray:
    """Quantize helper for accum='small'."""
    if accum == "small":
        return quantize(x.astype(np.float32, copy=False), fmt)
    return x


def _cg_solve(
    A: np.ndarray,
    b: np.ndarray,
    tol: float,
    max_iters: int,
    fmt: str,
    accum: str,
    precond: str = "jacobi",
    warmup_small: int = 0,
    restart_every: int = 0,
    true_residual_every: Optional[int] = None,
) -> Tuple[np.ndarray, int, bool, list[Tuple[int, float]], list[Tuple[int, float]]]:
    """
    Returns: (x, iters, converged, iter_residuals, true_residuals)
      where residual lists contain (k, rel_resid).
    """
    n = b.size
    x = np.zeros(n, dtype=np.float32)

    # Warmup: run a few iterations fully in fp32 to "stabilize" small formats.
    warmup_left = warmup_small if accum == "small" else 0

    # Preconditioner
    use_prec = (precond.lower() == "jacobi")
    M_inv = None
    if use_prec:
        M_inv = 1.0 / np.diag(A)

    def precond_apply(r: np.ndarray) -> np.ndarray:
        if not use_prec:
            return r
        return (M_inv * r).astype(np.float32, copy=False)

    # CG init
    r = b - A @ x
    z = precond_apply(r)
    p = z.copy()
    rz_old = float(r @ z)

    norm_r0 = float(np.linalg.norm(r))
    if norm_r0 == 0.0:
        return x, 0, True, [(0, 0.0)], [(0, 0.0)]

    iter_resid = []
    true_resid = []

    for k in range(1, max_iters + 1):
        # Optionally restart (throw away direction, keep x)
        if restart_every and (k % restart_every == 0):
            r = b - A @ x
            z = precond_apply(r)
            p = z.copy()
            rz_old = float(r @ z)

        # Choose working precision this iteration
        # During warmup, force fp32 state
        this_accum = "fp32" if warmup_left > 0 else accum

        Ap = A @ _maybe_q(p, fmt, this_accum)
        alpha = rz_old / float(p @ Ap)
        x = x + alpha * _maybe_q(p, fmt, this_accum)

        if this_accum == "small":
            x = _maybe_q(x, fmt, this_accum)

        r = r - alpha * Ap

        # Residual tracking (iter-relative)
        rel = float(np.linalg.norm(r) / norm_r0)
        iter_resid.append((k, rel))

        if rel <= tol:
            return x, k, True, iter_resid, true_resid

        z = precond_apply(r)
        rz_new = float(r @ z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

        # true residual (fp32) every N iters
        if true_residual_every and (k % true_residual_every == 0):
            r_true = b - (A @ x)  # compute fresh in fp32
            rel_true = float(np.linalg.norm(r_true) / norm_r0)
            true_resid.append((k, rel_true))

        if warmup_left > 0:
            warmup_left -= 1

    return x, max_iters, False, iter_resid, true_resid


# ---------------- Public API ----------------

def run_cg_and_log(
    out_csv: str,
    m: int,
    fmt: str,
    accum: str,
    tol: Optional[float] = None,
    max_iters: int = 1000,
    precond: str = "jacobi",
    warmup_small: int = 0,
    restart_every: int = 0,
    seed: int = 42,
    true_residual_every: Optional[int] = None,  # <-- NEW, optional
) -> None:
    """
    Solve A x = b with CG on an m×m Laplacian.
    Logs a summary row to `out_csv` and per-iteration residuals to
    `results/cg_residuals.csv`. If `true_residual_every` is set,
    also logs "true" residuals computed in fp32.
    """
    rng = np.random.default_rng(seed)
    A = _laplacian_2d(int(m)).astype(np.float32)
    n = A.shape[0]
    # Random RHS; fixed seed makes runs reproducible.
    x_ref = rng.standard_normal(n, dtype=np.float32)
    b = A @ x_ref

    # If tol wasn’t provided, use recommended per-format tolerance.
    tol_use = float(tol if tol is not None else tol_for(fmt))

    # Solve
    x, iters, ok, iter_resid, true_resid = _cg_solve(
        A=A,
        b=b,
        tol=tol_use,
        max_iters=int(max_iters),
        fmt=fmt,
        accum=accum,
        precond=precond,
        warmup_small=int(warmup_small),
        restart_every=int(restart_every),
        true_residual_every=true_residual_every,
    )

    # Write summary
    _ensure_csv(out_csv, _SUMMARY_FIELDS)
    _append_row(
        out_csv,
        {
            "m": int(m),
            "fmt": fmt,
            "accum": accum,
            "tol": f"{tol_use:.6g}",
            "max_iters": int(max_iters),
            "warmup_small": int(warmup_small),
            "restart_every": int(restart_every),
            "precond": precond,
            "iters": int(iters),
            "converged": int(1 if ok else 0),
        },
        _SUMMARY_FIELDS,
    )

    # Write residuals (per-iteration + optional true)
    res_csv = "results/cg_residuals.csv"
    _ensure_csv(res_csv, _RESID_FIELDS)
    for k, rel in iter_resid:
        _append_row(
            res_csv,
            {
                "m": int(m),
                "fmt": fmt,
                "accum": accum,
                "iter": int(k),
                "rel_resid": f"{rel:.8g}",
                "tol": f"{tol_use:.8g}",
                "kind": "iter",
            },
            _RESID_FIELDS,
        )
    for k, rel in true_resid:
        _append_row(
            res_csv,
            {
                "m": int(m),
                "fmt": fmt,
                "accum": accum,
                "iter": int(k),
                "rel_resid": f"{rel:.8g}",
                "tol": f"{tol_use:.8g}",
                "kind": "true",
            },
            _RESID_FIELDS,
        )


# Local smoke (optional)
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_cg_and_log(
        out_csv="results/cg.csv",
        m=64,
        fmt="fp32",
        accum="small",
        tol=1e-6,
        max_iters=100,
        precond="jacobi",
        warmup_small=5,
        restart_every=10,
        seed=0,
        true_residual_every=10,
    )
    print("Wrote results/cg.csv and results/cg_residuals.csv")

# src/main.py
# Chandana-friendly orchestrator:
# - Reads config.yaml (your current schema)
# - Runs GEMM / CG / Newton sweeps
# - Logs CSVs into results/ and prints concise progress

import os
import yaml
import numpy as np

from .gemm import run_gemm_and_log
from .cg import run_cg_and_log
from .newton import run_newton_and_log
from .formats import tol_for  # per-format fallback tolerances


def _as_list(x):
    """Ensure a config value is always a list."""
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]


def main():
    os.makedirs("results", exist_ok=True)

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # -------------------------
    # Global seed (reproducible)
    # -------------------------
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    # =========================
    # GEMM sweep (matrix multiply)
    # =========================
    gemm_cfg    = cfg.get("gemm", {})
    sizes       = [int(n) for n in _as_list(gemm_cfg.get("sizes", [128, 256, 512, 1024]))]
    kappas_raw  = _as_list(gemm_cfg.get("kappas", [1e2, 1e3, 1e4, 1e5, 1e6]))
    kappas      = [float(k) for k in kappas_raw]
    gemm_fmts   = [str(x) for x in _as_list(gemm_cfg.get("formats", ["fp32", "tf32", "bfloat16", "fp8"]))]
    gemm_accum  = [str(x) for x in _as_list(gemm_cfg.get("accum_modes", ["fp32", "small"]))]

    print(f"[GEMM] sizes={sizes}, κ={kappas_raw}, formats={gemm_fmts}, accum={gemm_accum}")
    for n in sizes:
        for kappa in kappas:
            for fmt in gemm_fmts:
                for accum in gemm_accum:
                    print(f"  GEMM n={n} κ={kappa:g} fmt={fmt}-{accum}")
                    run_gemm_and_log(
                        out_csv="results/gemm.csv",
                        n=int(n),
                        kappa=float(kappa),
                        fmt=fmt,
                        accum=accum,
                        rng=rng
                    )

    # =========================
    # CG sweep (iterative solver)
    # =========================
    cg_cfg        = cfg.get("cg", {})
    m_vals        = [int(m) for m in _as_list(cg_cfg.get("m_values", [64, 128]))]
    cg_maxit      = int(cg_cfg.get("max_iters", 1000))
    cg_fmts       = [str(x) for x in _as_list(cg_cfg.get("formats", ["fp32", "tf32", "bfloat16", "fp8"]))]
    cg_accum      = [str(x) for x in _as_list(cg_cfg.get("accum_modes", ["small", "fp32"]))]
    precond       = str(cg_cfg.get("precond", "jacobi"))

    # global defaults (if per-format not provided)
    global_warmup_small  = int(cg_cfg.get("warmup_small", 0))
    global_restart_every = int(cg_cfg.get("restart_every", 0))
    true_residual_every  = int(cg_cfg.get("true_residual_every", 10))

    per_fmt_cfg = cg_cfg.get("per_format", {}) or {}

    print(f"[CG] m={m_vals}, precond={precond}, max_iters={cg_maxit}")
    for m in m_vals:
        for fmt in cg_fmts:
            # tolerance: per-format else fallback via tol_for(fmt)
            fmt_cfg = per_fmt_cfg.get(fmt, {})
            tol = float(fmt_cfg.get("tol", tol_for(fmt)))

            # warmup/restart defaults for small-accum
            warmup_small  = int(fmt_cfg.get("warmup_small", global_warmup_small))
            restart_small = int(fmt_cfg.get("restart_every", global_restart_every))

            for accum in cg_accum:
                # Build unique (warmup, restart) pairs:
                pairs = {(0, 0)}
                if accum == "small":
                    pairs.add((warmup_small, restart_small))

                for (warmup, restart) in sorted(pairs):
                    print(f"  CG m={m} fmt={fmt}-{accum} tol={tol:g} warmup={warmup} restart={restart}")
                    run_cg_and_log(
                        out_csv="results/cg.csv",
                        m=int(m),
                        fmt=fmt,
                        accum=accum,
                        tol=tol,
                        max_iters=cg_maxit,
                        precond=precond,
                        warmup_small=warmup,
                        restart_every=restart,
                        true_residual_every=true_residual_every,
                        seed=seed,
                    )

    # =========================
    # Newton sweep (1D roots)
    # =========================
    newton_cfg = cfg.get("newton", {})
    fn_list    = [str(x) for x in _as_list(newton_cfg.get("functions", ["sinx_minus_x2", "x3_minus_7"]))]
    starts     = [float(x) for x in _as_list(newton_cfg.get("starts", [-3, -1, 0.1, 1, 3]))]
    n_fmts     = [str(x) for x in _as_list(newton_cfg.get("formats", ["fp32", "tf32", "bfloat16", "fp8"]))]
    n_accum    = [str(x) for x in _as_list(newton_cfg.get("accum_modes", ["small", "fp32"]))]
    newton_max = int(newton_cfg.get("max_iters", 100))

    print(f"[Newton] functions={fn_list}, starts={starts}, formats={n_fmts}, accum={n_accum}")
    for fn in fn_list:
        for fmt in n_fmts:
            for accum in n_accum:
                run_newton_and_log(
                    out_csv="results/newton.csv",
                    func_name=fn,
                    starts=starts,
                    fmt=fmt,           # now supported by new newton.py
                    accum=accum,       # now supported by new newton.py
                    tol=tol_for(fmt),
                    max_iters=newton_max,
                )


if __name__ == "__main__":
    main()

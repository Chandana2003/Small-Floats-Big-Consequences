# Small-Floats-Big-Consequences
A practical look at speed vs. accuracy with FP32 / TF32 / BF16 / FP8 and FP32 accumulation.
README.md
# Small Floats, Big Consequences
**A practical look at speed vs. accuracy with FP32 / TF32 / BF16 / FP8 and FP32 accumulation.**  
Authors: Chandana Anand Rangappa, Ranjitha Narasimhamurthy (CS260, UCR)

## What this repo contains
- `src/formats.py` — quantization helpers and per-format tolerances
- `src/gemm.py` — GEMM experiments (relative error & GFLOP/s)
- `src/cg.py` — Conjugate Gradient (iterations, residuals; warm-up & restarts)
- `src/newton.py` — Newton’s method (iteration counts, failures)
- `src/main.py` — reads `config.yaml`, runs all sweeps, writes CSVs to `results/`
- `src/make_plots.py` — reads CSVs, saves figures to `plots/`
- `config.yaml` — single place to change sizes, κ, formats, tolerances, etc.
- `results/` — (created) CSV logs
- `plots/` — (created) PNG figures

## TL;DR (what we show)
- Keep **inputs small** (TF32/BF16/FP8) for speed, but **accumulate in FP32** for stability.  
- In GEMM, FP32 accumulation cuts error by ~10–100× vs all-small, often with comparable throughput.  
- In CG/Newton, FP32 accumulation avoids stalls and reduces iteration counts.

---

## Quick start

### 1) Create & activate an environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2) Reproduce results (CSV → PNG)
# Runs GEMM / CG / Newton according to config.yaml
python -m src.main

# Builds all figures from results/*.csv into plots/*.png
python -m src.make_plots

Outputs:

CSVs in results/ (e.g., gemm.csv, cg.csv, newton.csv, cg_residuals.csv)

Figures in plots/ (e.g., gemm_n{size}.png, gemm_perf_n{size}.png, cg_iters_m{m}.png, newton_*.png)

Configuration (edit config.yaml)

Key fields (example):

gemm:
  sizes: [128, 256, 512, 1024]
  kappas: [1e2, 1e3, 1e4, 1e5, 1e6]
  formats: [fp32, tf32, bfloat16, fp8]
  accum_modes: [fp32, small]

cg:
  m_values: [64, 128]
  formats: [fp32, tf32, bfloat16, fp8]
  accum_modes: [small, fp32]
  max_iters: 1000
  warmup_small: 0
  restart_every: 0
  true_residual_every: 10
  per_format:
    fp32:     { tol: 1.0e-6 }
    tf32:     { tol: 1.0e-3,  warmup_small: 20, restart_every: 40 }
    bfloat16: { tol: 5.0e-3,  warmup_small: 10, restart_every: 25 }
    fp8:      { tol: 2.0e-1,  warmup_small: 10, restart_every: 20 }

newton:
  functions: [sinx_minus_x2, x3_minus_7]
  starts: [-3, -1, 0.1, 1, 3]
  formats: [fp32, tf32, bfloat16, fp8]
  accum_modes: [small, fp32]
  max_iters: 100
  per_format:
    fp32:     { tol: 1.0e-6 }
    tf32:     { tol: 1.0e-3 }
    bfloat16: { tol: 5.0e-3 }
    fp8:      { tol: 2.0e-1 }

Typical figures (what to expect)

GEMM accuracy vs κ: FP32-accum curves lie well below all-small for each format (TF32/BF16/FP8).

GEMM throughput: small-input + FP32-accum bars are fastest or near fastest; FP32-FP32 is slowest.

CG iterations/residuals: all-small needs more iters or stalls; FP32-accum converges steadily.

Newton iterations: FP8-small often hits the cap on harder starts; TF32-FP32 mirrors FP32.

Reproducibility tips

Set seed: in config.yaml for stable runs.

CSV headers are fixed; if you append runs, figures pick them up automatically.

Use true_residual_every in CG for periodic high-precision checks.

Requirements
See requirements.txt. Tested with Python 3.10–3.12.
---

# requirements.txt

```txt
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
PyYAML>=6.0

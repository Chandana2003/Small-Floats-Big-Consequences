from pathlib import Path
import sys
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

import yaml

_CFG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

def load_config():
    with open(_CFG_PATH, "r") as f:
        return yaml.safe_load(f)

# --- helpers that keep old/new config styles working -------------------------

def get_gemm_cfg(cfg):
    g = cfg["gemm"]
    # support either:
    #  (A) fmts: ["fp32-fp32","fp8-small", ...]
    #  (B) formats: [fp32, tf32, ...] + accum_modes: [fp32, small]
    if "fmts" in g:
        fmts = g["fmts"]
    else:
        fmts = [f"{fmt}-{am}" for fmt in g["formats"] for am in g["accum_modes"]]
    return {
        "sizes": g["sizes"],
        "kappas": g["kappas"],
        "fmts": fmts
    }

def get_cg_cfg(cfg):
    c = cfg["cg"]
    fmts = [f"{fmt}-{am}" for fmt in c["formats"] for am in c["accum_modes"]]
    return {
        "grids": c.get("m_values", c.get("grids", [64])),
        "fmts": fmts,
        "max_iters": c["max_iters"],
        "precond": c.get("precond", "jacobi"),
        "true_residual_every": c.get("true_residual_every", 10),
        "per_format": c["per_format"],
        "warmup_small_default": c.get("warmup_small", 0),
        "restart_every_default": c.get("restart_every", 0),
    }

def get_newton_cfg(cfg):
    n = cfg["newton"]
    fmts = [f"{fmt}-{am}" for fmt in n["formats"] for am in n["accum_modes"]]
    return {
        "functions": n["functions"],
        "starts": n["starts"],
        "fmts": fmts,
        "max_iters": n["max_iters"],
        "per_format": n["per_format"],
    }

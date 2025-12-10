# src/make_plots.py
# Robust plotter for the project. Handles messy CSV rows/headers gracefully.

import os
import math
import csv
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"


# -----------------------
# Helpers: robust CSV read
# -----------------------

def _read_csv(path):
    """
    Read a CSV into a list of dicts, but be forgiving:
    - Skip rows where all values are empty/None
    - Strip whitespace on keys/values; keep missing fields as ""
    - Ignore extra columns; tolerate missing ones
    """
    rows = []
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return rows

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows

        # Normalize headers
        headers = [h.strip() if isinstance(h, str) else "" for h in reader.fieldnames]

        for r in reader:
            clean = {}
            empty_count = 0
            for k in headers:
                v = r.get(k, "")
                if v is None:
                    v = ""
                elif isinstance(v, str):
                    v = v.strip()
                else:
                    v = str(v).strip()

                if v == "":
                    empty_count += 1
                clean[k] = v

            # Skip totally empty rows
            if empty_count == len(headers):
                continue
            rows.append(clean)
    return rows


def _num(x, default=np.nan):
    """Coerce to float safely."""
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _pick_col(row, options, default=""):
    """
    Pick the first present column name among options.
    Return its value (string) or default.
    """
    for c in options:
        if c in row:
            return row[c]
    return default


def _cols_present(rows):
    """Set of all column names present across rows."""
    s = set()
    for r in rows:
        s.update(r.keys())
    return s


def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------
# GEMM plots
# -----------------------

def plot_gemm_errors():
    """
    Line plots (log-log): relative error vs kappa per format/accum, grouped by n.
    Handles columns named:
      n, kappa, (rel_error|error), (fmt|format), (accum|accum_mode)
    """
    path = os.path.join(RESULTS_DIR, "gemm.csv")
    data = _read_csv(path)
    if not data:
        print("[plot_gemm_errors] No data, skipping.")
        return

    _ensure_plots_dir()

    # Normalize rows
    norm = []
    for r in data:
        n        = _num(_pick_col(r, ["n"]))
        kappa    = _num(_pick_col(r, ["kappa"]))
        rel_err  = _num(_pick_col(r, ["rel_error", "error"]))
        fmt      = _pick_col(r, ["fmt", "format"])
        accum    = _pick_col(r, ["accum", "accum_mode"])

        if any(map(math.isnan, [n, kappa, rel_err])) or fmt == "" or accum == "":
            continue
        norm.append({"n": int(n), "kappa": float(kappa), "rel_err": float(rel_err),
                     "fmt": fmt, "accum": accum})

    if not norm:
        print("[plot_gemm_errors] Normalized data empty, skipping.")
        return

    # Group by n
    by_n = defaultdict(list)
    for r in norm:
        by_n[r["n"]].append(r)

    for n, rows in by_n.items():
        plt.figure(figsize=(10, 6))

        # Order formats/accum for a stable legend
        fmt_order   = ["fp32", "tf32", "bfloat16", "fp8"]
        accum_order = ["fp32", "small"]
        lines_seen  = set()

        for fmt in fmt_order:
            for accum in accum_order:
                series = [r for r in rows if r["fmt"] == fmt and r["accum"] == accum]
                if not series:
                    continue
                series = sorted(series, key=lambda x: x["kappa"])
                x = [s["kappa"] for s in series]
                y = [s["rel_err"] for s in series]
                label = f"{fmt}-{accum}"
                plt.plot(x, y, marker="o", label=label)
                lines_seen.add(label)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Condition number (kappa)")
        plt.ylabel("Relative error  ||Ĉ - C*|| / ||C*||")   # plain text (no LaTeX)
        plt.title(f"GEMM accuracy vs conditioning (n={n})")

        if lines_seen:
            plt.legend(ncol=2)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"gemm_n{n}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[plot_gemm_errors] Wrote {out}")


def plot_gemm_perf():
    """
    Bar chart: average GFLOP/s per fmt-accum for each n.
    Accepts GFLOP column named either 'gflops' or 'gflop_s' or 'gflop'.
    """
    path = os.path.join(RESULTS_DIR, "gemm.csv")
    data = _read_csv(path)
    if not data:
        print("[plot_gemm_perf] No data, skipping.")
        return

    _ensure_plots_dir()

    norm = []
    for r in data:
        n     = _num(_pick_col(r, ["n"]))
        fmt   = _pick_col(r, ["fmt", "format"])
        accum = _pick_col(r, ["accum", "accum_mode"])
        g     = _num(_pick_col(r, ["gflops", "gflop_s", "gflop"]))

        if math.isnan(n) or fmt == "" or accum == "" or math.isnan(g):
            continue
        norm.append({"n": int(n), "fmt": fmt, "accum": accum, "gflops": float(g)})

    if not norm:
        print("[plot_gemm_perf] Normalized data empty, skipping.")
        return

    by_n = defaultdict(list)
    for r in norm:
        by_n[r["n"]].append(r)

    for n, rows in by_n.items():
        # average per (fmt,accum)
        groups = defaultdict(list)
        for r in rows:
            groups[(r["fmt"], r["accum"])].append(r["gflops"])

        labels = []
        vals   = []
        for (fmt, accum), glist in sorted(groups.items()):
            labels.append(f"{fmt}-{accum}")
            vals.append(float(np.mean(glist)))

        plt.figure(figsize=(10, 6))
        plt.bar(labels, vals)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("GFLOP/s (avg over κ)")
        plt.title(f"GEMM throughput (n={n})")
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"gemm_perf_n{n}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[plot_gemm_perf] Wrote {out}")


# -----------------------
# CG plots
# -----------------------

def plot_cg_iters():
    """
    Bar chart: iterations to tolerance per (fmt-accum) for each m.
    If there are multiple runs per label, we take the best (min) iters among converged runs;
    if none converged, we take the min iters and annotate with '× tol' using the median final residual.
    Requires columns:
      m, iters, (converged|failed), (rel_final_resid|final_resid),
      (fmt|format), (accum|accum_mode)
    """
    path = os.path.join(RESULTS_DIR, "cg.csv")
    data = _read_csv(path)
    if not data:
        print("[plot_cg_iters] No data, skipping.")
        return

    _ensure_plots_dir()

    norm = []
    for r in data:
        m     = _num(_pick_col(r, ["m"]))
        iters = _num(_pick_col(r, ["iters"]))
        # 'converged' (0/1) or 'failed' (1/0)
        conv_raw = _pick_col(r, ["converged", "failed"])
        # If it's 'failed', flip to converged=0/1 consistently
        converged = None
        if conv_raw != "":
            try:
                c = int(float(conv_raw))
                # Heuristic: if column was 'failed', many logs use 1=failed
                # We detect name and invert if needed
                if "failed" in _cols_present([r]):
                    converged = 0 if c == 1 else 1
                else:
                    converged = c
            except Exception:
                pass

        rel_res = _num(_pick_col(r, ["rel_final_resid", "final_resid"]))
        fmt     = _pick_col(r, ["fmt", "format"])
        accum   = _pick_col(r, ["accum", "accum_mode"])

        if any(map(math.isnan, [m, iters])) or fmt == "" or accum == "":
            continue

        if converged is None:
            # default: assume not converged unless explicitly 1
            converged = 0

        norm.append({
            "m": int(m),
            "iters": float(iters),
            "converged": int(converged),
            "rel_res": rel_res if not math.isnan(rel_res) else np.nan,
            "label": f"{fmt}-{accum}"
        })

    if not norm:
        print("[plot_cg_iters] Normalized data empty, skipping.")
        return

    by_m = defaultdict(list)
    for r in norm:
        by_m[r["m"]].append(r)

    for m, rows in by_m.items():
        # aggregate per label
        agg = {}
        for lab, grp in group_by(rows, key=lambda z: z["label"]).items():
            conv_iters = [g["iters"] for g in grp if g["converged"] == 1]
            notc_iters = [g["iters"] for g in grp if g["converged"] == 0]
            notc_rel   = [g["rel_res"] for g in grp if g["converged"] == 0 and not math.isnan(g["rel_res"])]

            if conv_iters:
                val = float(min(conv_iters))
                ann = ""   # no annotation
                conv = 1
            else:
                val = float(min(notc_iters)) if notc_iters else np.nan
                # annotate with median rel residual if we have it
                if notc_rel:
                    ann_val = np.median(notc_rel)
                    ann = f"× {ann_val:.1e}"
                else:
                    ann = "×"
                conv = 0

            agg[lab] = (val, conv, ann)

        labels = sorted(agg.keys())
        heights = [agg[l][0] for l in labels]
        convs   = [agg[l][1] for l in labels]
        anns    = [agg[l][2] for l in labels]

        plt.figure(figsize=(12, 7))
        colors = ["tab:blue" if c == 1 else "tab:orange" for c in convs]
        bars = plt.bar(labels, heights, color=colors)

        for rect, ann, cflag in zip(bars, anns, convs):
            if cflag == 0 and isinstance(ann, str) and ann:
                plt.text(rect.get_x() + rect.get_width()/2.0,
                         rect.get_height(), ann,
                         ha="center", va="bottom", fontsize=9)

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Iterations (min over repeats)")
        plt.title(f"CG: iterations to tolerance per format (m={m}) — blue=converged, orange=not")
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"cg_iters_m{m}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[plot_cg_iters] Wrote {out}")


def plot_cg_residual_curves():
    """
    Optional line plot of residual histories if you logged them to results/cg_residuals.csv.
    Expected columns: run_id, iter, rel_resid, label (fmt-accum), m
    Skips gracefully if file not found.
    """
    path = os.path.join(RESULTS_DIR, "cg_residuals.csv")
    data = _read_csv(path)
    if not data:
        print("[plot_cg_residual_curves] No residual logs, skipping.")
        return

    _ensure_plots_dir()

    # Try to parse; skip rows missing required bits
    rows = []
    for r in data:
        run_id = _pick_col(r, ["run_id", "id"])
        it     = _num(_pick_col(r, ["iter", "iteration"]))
        rr     = _num(_pick_col(r, ["rel_resid", "resid", "rr"]))
        lab    = _pick_col(r, ["label"])
        m      = _num(_pick_col(r, ["m"]))

        if run_id == "" or lab == "" or math.isnan(it) or math.isnan(rr) or math.isnan(m):
            continue
        rows.append({"id": run_id, "iter": int(it), "rr": float(rr), "label": lab, "m": int(m)})

    if not rows:
        print("[plot_cg_residual_curves] No usable rows, skipping.")
        return

    # Plot one figure per m
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["m"]].append(r)

    for m, chunk in by_m.items():
        plt.figure(figsize=(12, 7))
        by_label = group_by(chunk, key=lambda z: z["label"])
        for lab, seq in by_label.items():
            seq = sorted(seq, key=lambda z: z["iter"])
            x = [s["iter"] for s in seq]
            y = [s["rr"] for s in seq]
            plt.plot(x, y, label=lab)

        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Relative residual ||r_k|| / ||r_0||")
        plt.title(f"CG residual curves (m={m})")
        plt.legend(ncol=2)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"cg_residual_m{m}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[plot_cg_residual_curves] Wrote {out}")


# -----------------------
# Newton plots
# -----------------------

def plot_newton():
    """
    Bar charts: iterations per starting point, color by fmt-accum, per function.
    Requires: func, x0, iters, failed, fmt, accum
    """
    path = os.path.join(RESULTS_DIR, "newton.csv")
    data = _read_csv(path)
    if not data:
        print("[plot_newton] No data, skipping.")
        return

    _ensure_plots_dir()

    rows = []
    for r in data:
        func   = _pick_col(r, ["func", "function", "fname"])
        x0     = _pick_col(r, ["x0", "start"])
        iters  = _num(_pick_col(r, ["iters", "iterations"]))
        failed = _pick_col(r, ["failed"])
        fmt    = _pick_col(r, ["fmt", "format"])
        accum  = _pick_col(r, ["accum", "accum_mode"])

        if func == "" or x0 == "" or fmt == "" or accum == "" or math.isnan(iters):
            continue

        try:
            x0f = float(x0)
        except Exception:
            # keep string label if not numeric
            x0f = x0

        try:
            fflag = int(float(failed))
        except Exception:
            fflag = 0

        rows.append({"func": func, "x0": x0f, "iters": float(iters),
                     "failed": fflag, "label": f"{fmt}-{accum}"})

    if not rows:
        print("[plot_newton] Normalized data empty, skipping.")
        return

    by_func = group_by(rows, key=lambda z: z["func"])
    for func, seq in by_func.items():
        # build bars per start, values per label (fmt-accum)
        starts = sorted(list({s["x0"] for s in seq}), key=lambda v: (isinstance(v, str), v))
        labels = sorted(list({s["label"] for s in seq}))
        mat = np.full((len(starts), len(labels)), np.nan)

        for i, x0 in enumerate(starts):
            for j, lab in enumerate(labels):
                candidates = [s for s in seq if s["x0"] == x0 and s["label"] == lab]
                if not candidates:
                    continue
                # prefer successful (failed==0) with min iters
                succ = [c for c in candidates if c["failed"] == 0]
                if succ:
                    mat[i, j] = min(c["iters"] for c in succ)
                else:
                    mat[i, j] = min(c["iters"] for c in candidates)

        # Plot grouped bars
        plt.figure(figsize=(12, 7))
        idx = np.arange(len(starts), dtype=float)
        width = 0.8 / max(1, len(labels))

        for j, lab in enumerate(labels):
            y = mat[:, j]
            plt.bar(idx + j * width, y, width=width, label=lab)

        plt.xticks(idx + (len(labels)-1) * width / 2.0, [str(s) for s in starts])
        plt.ylabel("Iterations (lower is better)")
        plt.title(f"Newton iterations by start — {func}")
        plt.legend(ncol=2)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"newton_{func}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[plot_newton] Wrote {out}")


# -----------------------
# Small utilities
# -----------------------

def group_by(seq, key):
    d = defaultdict(list)
    for x in seq:
        d[key(x)].append(x)
    return d


def plot_all():
    plot_gemm_errors()
    plot_gemm_perf()
    plot_cg_iters()
    plot_cg_residual_curves()
    plot_newton()


if __name__ == "__main__":
    plot_all()

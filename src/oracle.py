import numpy as np
import mpmath as mp

def set_oracle(bits=200):
    mp.mp.dps = int(bits * 0.30103)  # bits -> decimal digits approx.

def relative_error(x_hat: np.ndarray, x_star: np.ndarray) -> float:
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_star = np.asarray(x_star, dtype=np.float64)
    num = np.linalg.norm(x_hat - x_star)
    den = np.linalg.norm(x_star) + 1e-30
    return float(num / den)

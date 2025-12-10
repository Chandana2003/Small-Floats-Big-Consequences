import numpy as np
from .precision import quantize_bfloat16, quantize_tf32, quantize_fp8_like

def get_quantizer(fmt: str):
    if fmt == 'fp32':     return lambda a: np.asarray(a, dtype=np.float32)
    if fmt == 'bfloat16': return quantize_bfloat16
    if fmt == 'tf32':     return quantize_tf32
    if fmt == 'fp8':      return lambda a: quantize_fp8_like(a, 5, 2)
    raise ValueError(fmt)

def fadd(a, b, fmt, accum='small'):
    q = get_quantizer(fmt)
    if accum == 'fp32':
        return q(np.asarray(a, np.float32) + np.asarray(b, np.float32))
    else:
        return q(q(a) + q(b))

def fmul(a, b, fmt, accum='small'):
    q = get_quantizer(fmt)
    if accum == 'fp32':
        return q(np.asarray(a, np.float32) * np.asarray(b, np.float32))
    else:
        return q(q(a) * q(b))

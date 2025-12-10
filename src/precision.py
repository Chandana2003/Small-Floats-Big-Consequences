import numpy as np

def _quantize_mantissa(x32: np.ndarray, frac_bits: int):
    x32 = x32.astype(np.float32, copy=False)
    u = x32.view(np.uint32)
    sign_exp = u & 0xFF800000      # keep sign+exp (1+8 bits)
    mant = u & 0x007FFFFF          # 23-bit mantissa
    shift = 23 - frac_bits
    if shift <= 0:
        return x32
    round_bit = 1 << (shift - 1)
    mant_rounded = mant + round_bit

    # ties-to-even
    tie_mask = (mant & ((1 << shift) - 1)) == round_bit
    mant_rounded[tie_mask] = mant[tie_mask] + (round_bit - ((mant[tie_mask] >> shift) & 1))

    mant_q = (mant_rounded >> shift) << shift
    u_q = sign_exp | mant_q
    return u_q.view(np.float32)

def quantize_bfloat16(x):  return _quantize_mantissa(np.asarray(x, dtype=np.float32), 7)
def quantize_tf32(x):      return _quantize_mantissa(np.asarray(x, dtype=np.float32), 10)

def quantize_fp8_like(x, exp_bits=5, frac_bits=2):
    # Simple: only mantissa truncation (documented). Exponent clamping could be added later if needed.
    return _quantize_mantissa(np.asarray(x, dtype=np.float32), frac_bits)

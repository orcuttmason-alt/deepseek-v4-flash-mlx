"""QAT activation sims used by the indexer (and elsewhere): Hadamard rotate + fp4 act-quant.
Numpy implementations (deterministic; match the official fast_hadamard_transform + fp4_act_quant).
"""
import numpy as np
import ml_dtypes


def hadamard_rotate(x):
    """rotate_activation: normalized Walsh-Hadamard (natural/Sylvester order) along last dim,
    scaled by n**-0.5. official_model.py: hadamard_transform(x, scale=x.size(-1)**-0.5)."""
    a = np.array(x, dtype=np.float32, copy=True)
    n = a.shape[-1]
    assert (n & (n - 1)) == 0, "Hadamard needs power-of-2 last dim"
    h = 1
    while h < n:
        a = a.reshape(*a.shape[:-1], n // (2 * h), 2, h)
        x0, x1 = a[..., 0, :], a[..., 1, :]
        a = np.concatenate([x0 + x1, x0 - x1], axis=-1).reshape(*a.shape[:-3], n)
        h *= 2
    return a * (n ** -0.5)


def fp4_act_quant(x, block=32):
    """Blockwise fp4-e2m1 round-trip with ue8m0 pow2 scale (official fp4_act_quant inplace)."""
    a = np.array(x, dtype=np.float32, copy=True); sh = a.shape
    fp4_max = 6.0
    ab = a.reshape(*sh[:-1], sh[-1] // block, block)
    amax = np.maximum(np.abs(ab).max(-1, keepdims=True), 6 * (2.0 ** -126))
    s = 2.0 ** np.ceil(np.log2(amax / fp4_max))
    q = np.clip(ab / s, -fp4_max, fp4_max).astype(ml_dtypes.float4_e2m1fn).astype(np.float32) * s
    return q.reshape(sh)

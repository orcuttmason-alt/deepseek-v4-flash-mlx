"""Explicit dequantization in MLX — affine int4 (gs64) and mxfp4 (gs32).

Mirrors ref_torch/quant.py. These explicit unpackers are what the offload path uses; they
are validated to match mx.dequantize (convention/packing) and, on real weights, ds4.
See ref_torch/quant.py for the landmine notes (scale != (max-min)/15).
"""
import mlx.core as mx

E2M1_LUT = mx.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                     0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=mx.float32)


def _unpack_nibbles(wq_u32, in_features):
    """wq_u32:[out, in/8] uint32 -> q:[out, in] uint32, nibbles low-bits-first."""
    shifts = (mx.arange(8) * 4).astype(mx.uint32)
    q = mx.right_shift(wq_u32[..., :, None], shifts) & mx.array(0xF, mx.uint32)
    return q.reshape(*wq_u32.shape[:-1], in_features)


def dequant_affine(wq_u32, scale, bias, in_features, group_size=64):
    q = _unpack_nibbles(wq_u32, in_features).astype(mx.float32)
    rep = in_features // scale.shape[-1]
    sc = mx.repeat(scale, rep, axis=-1)
    bi = mx.repeat(bias, rep, axis=-1)
    return sc * q + bi


def dequant_mxfp4(wq_u32, scale_e8m0, in_features, group_size=32):
    code = _unpack_nibbles(wq_u32, in_features)
    vals = E2M1_LUT[code]
    rep = in_features // scale_e8m0.shape[-1]
    exp = mx.repeat(scale_e8m0.astype(mx.float32), rep, axis=-1) - 127.0
    return vals * mx.power(mx.array(2.0), exp)

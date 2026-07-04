"""Explicit dequantization — affine int4 (gs64) and mxfp4 (gs32). Torch reference.

THE LANDMINE (user-flagged, empirically confirmed): do NOT assume a quantizer convention.
- MLX affine stores per-group (scale, bias) with `w = scale*q + bias`, q = 4-bit code,
  nibbles packed low-bits-first in uint32. VERIFIED: this reconstructs mx.dequantize to 1e-7.
  BUT MLX's stored `scale` != (max-min)/15 (it optimizes the scale); (max-min)/15 is the
  *naive* formula and does NOT reproduce MLX's scale. Immaterial for dequant (we use the
  stored scale) but a trap if anything ever recomputes scale. We dequant from stored tensors.
- mxfp4 has NO bias: `w = e2m1_lut[code] * 2^(e8m0_scale - 127)`, gs=32, e8m0 scale is uint8.

These explicit unpackers (not mx.dequantize) are what the offload path will use, and what the
ds4 gate validates on real weights. The torch cross-check proves they match MLX's native
decode bit-for-convention; the ds4 gate (real weights, downstream) carries the real信息.
"""
import numpy as np
import torch

# e2m1 (1 sign, 2 exp, 1 mantissa) value table; codes 0-7 positive, 8-15 negative.
E2M1_LUT = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                         0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float32)


def _unpack_nibbles(wq_u32, in_features):
    """wq_u32:[out, in/8] uint32 -> q:[out, in] int, nibbles low-bits-first (k=i%8)."""
    shifts = torch.arange(8, dtype=torch.int64) * 4
    q = (wq_u32[..., :, None] >> shifts) & 0xF        # [out, in/8, 8]
    return q.reshape(*wq_u32.shape[:-1], in_features)


def dequant_affine(wq_u32, scale, bias, in_features, group_size=64):
    """w[o,i] = scale[o,i//gs]*q[o,i] + bias[o,i//gs]."""
    q = _unpack_nibbles(wq_u32.to(torch.int64), in_features).to(torch.float32)
    rep = in_features // scale.shape[-1]
    sc = scale.repeat_interleave(rep, dim=-1)
    bi = bias.repeat_interleave(rep, dim=-1)
    return sc * q + bi


def dequant_mxfp4(wq_u32, scale_e8m0, in_features, group_size=32):
    """w[o,i] = E2M1_LUT[code] * 2^(scale_byte[o,i//gs] - 127)."""
    code = _unpack_nibbles(wq_u32.to(torch.int64), in_features)
    vals = E2M1_LUT[code]
    rep = in_features // scale_e8m0.shape[-1]
    exp = scale_e8m0.to(torch.float32).repeat_interleave(rep, dim=-1) - 127.0
    return vals * torch.pow(torch.tensor(2.0), exp)

"""Compressor (attention KV time-axis compression) — MLX, PREFILL path.
Mirrors ref_torch/compressor.py. act_quant = official fp8-e4m3 + ue8m0 pow2-scale sim (also
used for AR-1: the kv fed to sparse_attn). Weights F16 in GGUF (no affine landmine).
"""
import numpy as np
import ml_dtypes
import mlx.core as mx
from v4mlx.norm_rope import rmsnorm, apply_rotary_emb, rope_cos_sin


def act_quant_sim(x, block=64):
    """Official act_quant inplace round-trip (blockwise fp8-e4m3, pow2 ue8m0 scale).
    Done in numpy (deterministic quant; validated == ds4 exactly). x: mx.array [..., n]."""
    a = np.array(x.astype(mx.float32)); sh = a.shape
    fp8_max = 448.0
    ab = a.reshape(*sh[:-1], sh[-1] // block, block)
    amax = np.maximum(np.abs(ab).max(-1, keepdims=True), 1e-4)
    s = 2.0 ** np.ceil(np.log2(amax / fp8_max))
    q = np.clip(ab / s, -fp8_max, fp8_max).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * s
    return mx.array(q.reshape(sh))


def overlap_transform(t, ratio, d, value):
    """t:[b,s,ratio,2d] -> [b,s,2*ratio,d]."""
    b, s = t.shape[0], t.shape[1]
    new = mx.full((b, s, 2 * ratio, d), value, dtype=t.dtype)
    new[:, :, ratio:] = t[:, :, :, d:]
    new[:, 1:, :ratio] = t[:, :-1, :, :d]
    return new


def compressor_prefill(x, wkv, wgate, ape, norm_w, head_dim=512, ratio=4, rope_head_dim=64,
                       base=160000.0, original_seq_len=65536, factor=16, beta_fast=32, beta_slow=1,
                       apply_sim=True, rotate=False):
    """x:[b,s,dim] -> compressed kv:[b, s//ratio, head_dim] (post-rope, optionally fp8-simmed)."""
    b, s, _ = x.shape
    d, rd = head_dim, rope_head_dim
    x = x.astype(mx.float32)
    kv = x @ wkv.T
    score = x @ wgate.T
    remainder = s % ratio
    cutoff = s - remainder
    if remainder > 0:
        kv = kv[:, :cutoff]
        score = score[:, :cutoff]
    g = cutoff // ratio
    kv = kv.reshape(b, g, ratio, -1)
    score = score.reshape(b, g, ratio, -1) + ape
    kv = overlap_transform(kv, ratio, d, 0.0)
    score = overlap_transform(score, ratio, d, -mx.inf)
    kv = (kv * mx.softmax(score, axis=2)).sum(axis=2)
    kv = rmsnorm(kv, norm_w)
    cos, sin = rope_cos_sin(rd, s, original_seq_len, base, factor, beta_fast, beta_slow)
    rows = mx.arange(0, cutoff, ratio)                       # compressed-row rope positions 0,4,8,..
    roped = apply_rotary_emb(kv[..., -rd:], cos[rows], sin[rows])
    kv = mx.concatenate([kv[..., :-rd], roped], axis=-1)
    if rotate:  # indexer compressor: Hadamard(all) then fp4(all), official rotate_activation+fp4_act_quant
        from v4mlx import qat
        kv = mx.array(qat.fp4_act_quant(qat.hadamard_rotate(np.array(kv.astype(mx.float32))), 32))
    elif apply_sim:
        nope = act_quant_sim(kv[..., :-rd], 64)
        kv = mx.concatenate([nope, kv[..., -rd:]], axis=-1)
    return kv

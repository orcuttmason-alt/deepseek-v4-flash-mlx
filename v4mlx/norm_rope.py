"""RMSNorm + RoPE/YaRN — MLX implementation of deepseek_v4.

Mirrors ref_torch/norm_rope.py. MLX complex support is limited, so RoPE is done with the
real-valued cos/sin rotation that is mathematically identical to the official complex
multiply on interleaved pairs:
    (x0,x1) -> (x0*cos - x1*sin,  x0*sin + x1*cos)     [inverse: sin -> -sin]
Cross-checked CPU-to-CPU against the torch (complex) reference in tests/.
"""
import math
import mlx.core as mx


def rmsnorm(x, weight, eps=1e-6):
    dtype = x.dtype
    x = x.astype(mx.float32)
    var = x.square().mean(-1, keepdims=True)
    x = x * mx.rsqrt(var + eps)
    return (weight * x).astype(dtype)


def precompute_freqs(dim, original_seq_len, base, factor, beta_fast, beta_slow):
    """Per-(dim/2) angular frequencies with YaRN; identical math to the torch reference."""
    def corr_dim(num_rot):
        return dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    if original_seq_len > 0:
        low = max(math.floor(corr_dim(beta_fast)), 0)
        high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        lin = (mx.arange(dim // 2).astype(mx.float32) - low) / (high - low)
        ramp = mx.clip(lin, 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_cos_sin(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """Returns cos,sin of shape [seqlen, dim/2] for the interleaved-pair rotation."""
    freqs = precompute_freqs(dim, original_seq_len, base, factor, beta_fast, beta_slow)
    t = mx.arange(seqlen).astype(mx.float32)
    ang = t[:, None] * freqs[None, :]      # outer -> [seqlen, dim/2]
    return mx.cos(ang), mx.sin(ang)


def apply_rotary_emb(x, cos, sin, inverse=False):
    """x: [..., seqlen, (heads,) dim]; rope applied on the last dim's interleaved pairs.
    cos,sin: [seqlen, dim/2]. Broadcasts over batch/heads."""
    *lead, d = x.shape
    xp = x.astype(mx.float32).reshape(*lead, d // 2, 2)
    x0, x1 = xp[..., 0], xp[..., 1]
    # broadcast cos/sin: insert head axis if x has one (shape [...,seq,heads,d])
    if x.ndim == 4:                         # [b, seq, heads, d]
        c = cos[None, :, None, :]; s = sin[None, :, None, :]
    else:                                   # [b, seq, d]
        c = cos[None, :, :]; s = sin[None, :, :]
    if inverse:
        s = -s
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    out = mx.stack([o0, o1], axis=-1).reshape(*lead, d)
    return out.astype(x.dtype)

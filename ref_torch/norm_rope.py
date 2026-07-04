"""RMSNorm + RoPE/YaRN — pure-torch fp32 transcription of the official reference.

Sources: official_model.py :: RMSNorm (183-196), precompute_freqs_cis (199-229),
apply_rotary_emb (232-244). Faithful, including the interleaved-pair complex convention.
"""
import math
import torch


def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm.forward (official_model.py:191-196). weight:[dim]."""
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x).to(dtype)


def precompute_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """The per-(dim/2) angular frequencies, with YaRN interpolation when original_seq_len>0.
    Transcribes precompute_freqs_cis (199-227) up to (but not including) the polar() step,
    so torch and MLX can share the frequency table and build complex/cos-sin respectively.
    Returns freqs:[dim/2]. The full angle table is outer(arange(seqlen), freqs)."""
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn, mx, d):
        if mn == mx:
            mx += 0.001
        lin = (torch.arange(d, dtype=torch.float32) - mn) / (mx - mn)
        return torch.clamp(lin, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = precompute_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow)
    t = torch.arange(seqlen)
    ang = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(ang), ang)   # complex64 [seqlen, dim/2]


def apply_rotary_emb(x, freqs_cis, inverse=False):
    """apply_rotary_emb (232-244). x last dim is the rope sub-vector (size dim); pairs are
    interleaved (x0,x1),(x2,x3),... treated as (real,imag). Returns rotated x (same shape)."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    fc = freqs_cis.conj() if inverse else freqs_cis
    if xc.ndim == 3:
        fc = fc.view(1, xc.size(1), xc.size(-1))
    else:
        fc = fc.view(1, xc.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * fc).flatten(-2).to(x.dtype)

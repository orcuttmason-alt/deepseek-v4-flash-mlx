"""Multi-head Latent Attention (MLA) — MLX, per ARCHITECTURE.md §7 + official_model.py Attention.

Component = the Q/KV/O projections + norms + rope (NOT the sparse_attn core, which is the next
fenced component). Sub-functions are exposed so each can be gated isolated against ds4 dumps.

Weights are passed in already-dequantized ([out, in]); the caller controls decode (trusted
gguf.quants.dequantize for the gate). Linear: y = x @ W.T  (W is [out, in]).
"""
import mlx.core as mx
from v4mlx.norm_rope import rmsnorm, apply_rotary_emb

N_HEADS, HEAD_DIM, ROPE_HD = 64, 512, 64
N_GROUPS, O_LORA = 8, 1024


def linear(x, W):
    """y = x @ W.T, W:[out,in]."""
    return x @ W.T


def q_down(x, wq_a):
    """x:[...,dim] -> q_lora:[...,q_lora_rank]."""
    return linear(x, wq_a)


def perhead_norm(q, eps=1e-6):
    """Per-head RMS WITHOUT a learnable weight: q *= rsqrt(mean(q^2,-1)+eps). q:[...,nheads,hd]."""
    return q * mx.rsqrt(q.square().mean(-1, keepdims=True) + eps)


def q_up(qln, wq_b):
    """q_lora_norm:[...,q_lora_rank] -> Qraw:[...,nheads,head_dim]."""
    q = linear(qln, wq_b)
    return q.reshape(*q.shape[:-1], N_HEADS, HEAD_DIM)


def kv_proj(x, wkv):
    """x:[...,dim] -> KVraw:[...,head_dim] (single MLA latent)."""
    return linear(x, wkv)


def o_grouped(o, wo_a, wo_b):
    """Grouped low-rank output projection (official_model.py:537-542).
    o (post-attn, post inverse-rope):[...,nheads*head_dim] -> x:[...,dim].
    wo_a:[out=n_groups*o_lora, in=nheads*head_dim/n_groups]; wo_b:[out=dim, in=n_groups*o_lora].
    """
    lead = o.shape[:-1]
    og = o.reshape(*lead, N_GROUPS, -1)                     # [...,8, 4096]
    wa = wo_a.reshape(N_GROUPS, O_LORA, -1)                 # [8, 1024, 4096]
    # einsum '...gd,grd->...gr' via a per-group matmul loop (8 groups) — AVOIDS the O(S)
    # [S,8,1024,4096] intermediate that blew past the 30 GB Metal buffer cap at S~220 in long-context
    # prefill. Verified rel 8.7e-4 vs the elementwise-sum form (fp16 accum-order; << 1e-2 forward tol).
    # CP-4 must-fix (2026-06-30).
    o = mx.stack([og[..., g, :] @ wa[g].T for g in range(N_GROUPS)], axis=-2)  # [...,8,1024]
    o = o.reshape(*lead, -1)                                # [...,8192]
    return linear(o, wo_b)                                  # [...,dim]

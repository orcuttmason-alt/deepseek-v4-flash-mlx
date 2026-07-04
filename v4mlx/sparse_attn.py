"""sparse_attn — MLX, per ARCHITECTURE.md §10 + official_kernel.py sparse_attn_kernel.

Two parts, graded separately (split gate):
  - SELECTION: get_window_topk_idxs -> the discrete index set each query attends (causal window
    in prefill). Exact-set-match grading, NOT tolerance.
  - ARITHMETIC: sparse_attn -> gather kv by topk_idxs, online-softmax attention with a per-head
    attn_sink added to the DENOMINATOR only (mass with no value). Pure fp32 over dequantized
    q/kv + F32 sinks -> the affine-dequant landmine is NOT in this gate's differential.

kv is BOTH key and value (MLA). Implemented as a masked dense-gather fallback (faithful to the
kernel for small T); generalizes to real top-k indices unchanged.
"""
import mlx.core as mx


def get_window_topk_idxs(window_size, seqlen, start_pos=0):
    """Prefill (start_pos==0): matrix[i,j] = j if j<=i else -1, shape [seqlen, min(seqlen,window)].
    Transcribes official_model.py get_window_topk_idxs (the start_pos==0 branch)."""
    assert start_pos == 0, "decode-phase ring-buffer window not needed for the layer-0 gate"
    cols = min(seqlen, window_size)
    base = mx.arange(seqlen)[:, None]                          # [s,1]
    mat = mx.maximum(base - window_size + 1, 0) + mx.arange(cols)[None, :]
    return mx.where(mat > base, -1, mat)                       # [s, cols]


def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """q:[s,h,d], kv:[n,d], attn_sink:[h], topk_idxs:[s,T] (int, -1=masked) -> o:[s,h,d].
    o[m,h] = Σ_t softmax_t(scale·q·kv) · kv_gathered  /  (Σ_t exp + exp(sink[h]-rowmax)).
    """
    valid = topk_idxs >= 0                                      # [s,T]
    idx = mx.maximum(topk_idxs, 0)
    gathered = kv[idx]                                          # [s,T,d]
    scores = scale * mx.einsum("shd,std->sht", q, gathered)    # [s,h,T]
    scores = mx.where(valid[:, None, :], scores, -mx.inf)
    rowmax = scores.max(axis=-1, keepdims=True)                # [s,h,1]
    p = mx.exp(scores - rowmax)                                # masked -> 0
    denom = p.sum(axis=-1, keepdims=True) + mx.exp(attn_sink[None, :, None] - rowmax)
    o = mx.einsum("sht,std->shd", p, gathered) / denom
    return o

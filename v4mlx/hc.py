"""Hyper-Connections (HC) — MLX implementation of deepseek_v4.

Mirrors ref_torch/hc.py exactly. Graded against ds4 dumps (oracle); cross-checked against
the torch transcription on synthetic fp32 input (quant-free) by tests/test_hc_crosscheck.py.

hc=4, mix_hc=24. mixes layout: [0:4]=pre, [4:8]=post, [8:24]=comb(4x4).
"""
import mlx.core as mx


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=4, sinkhorn_iters=20, eps=1e-6):
    """mixes:[...,24] hc_scale:[3] hc_base:[24] -> pre:[...,4] post:[...,4] comb:[...,4,4]."""
    lead = mixes.shape[:-1]
    pre = mx.sigmoid(mixes[..., 0:hc] * hc_scale[0] + hc_base[0:hc]) + eps
    post = 2.0 * mx.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.reshape(*lead, hc, hc)
    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def hc_pre(x, hc_fn, hc_scale, hc_base, hc=4, norm_eps=1e-6, sinkhorn_iters=20, hc_eps=1e-6):
    """x:[b,s,hc,dim] -> y:[b,s,dim], post:[b,s,hc], comb:[b,s,hc,hc]."""
    shape = x.shape
    xf = x.reshape(*shape[:2], -1).astype(mx.float32)
    rsqrt = mx.rsqrt(xf.square().mean(-1, keepdims=True) + norm_eps)
    mixes = (xf @ hc_fn.T) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, sinkhorn_iters, hc_eps)
    y = mx.sum(pre[..., None] * x.reshape(shape), axis=2)
    return y, post, comb


def hc_post(x, residual, post, comb):
    """x:[b,s,dim], residual:[b,s,hc,dim], post:[b,s,hc], comb:[b,s,hc,hc] -> [b,s,hc,dim].
    y[...,j,:] = post[...,j]*x + sum_i comb[...,i,j]*residual[...,i,:]
    """
    return post[..., None] * x[..., None, :] + mx.sum(
        comb[..., None] * residual[..., None, :], axis=2)


def hc_head(x, hc_fn, hc_scale, hc_base, hc=4, norm_eps=1e-6, hc_eps=1e-6):
    """Final reduce before the LM head. x:[b,s,hc,dim] -> [b,s,dim]. hc_fn:[hc, hc*dim]."""
    shape = x.shape
    xf = x.reshape(*shape[:2], -1).astype(mx.float32)
    rsqrt = mx.rsqrt(xf.square().mean(-1, keepdims=True) + norm_eps)
    mixes = (xf @ hc_fn.T) * rsqrt
    pre = mx.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    return mx.sum(pre[..., None] * x.reshape(shape), axis=2)

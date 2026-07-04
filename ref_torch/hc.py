"""Hyper-Connections (HC) — pure-torch fp32 transcription of the official reference.

Sources (exact line correspondence):
  official_kernel.py :: hc_split_sinkhorn_kernel   -> hc_split_sinkhorn()
  official_model.py  :: Block.hc_pre / Block.hc_post -> hc_pre() / hc_post()
  official_model.py  :: ParallelHead.hc_head        -> hc_head()

This is the Layer-A development reference. The *grading* oracle is ds4 (see oracle/),
but torch==mlx on synthetic fp32 input is the quant-free cross-check that proves the two
implementations agree before the ds4 gate.

hc=hc_mult=4, mix_hc=(2+hc)*hc=24. mixes layout: [0:4]=pre, [4:8]=post, [8:24]=comb(4x4).
"""
import torch
import torch.nn.functional as F


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=4, sinkhorn_iters=20, eps=1e-6):
    """mixes:[...,24] hc_scale:[3] hc_base:[24] -> pre:[...,4] post:[...,4] comb:[...,4,4].

    Transcribes hc_split_sinkhorn_kernel_ (official_kernel.py:377-425):
      pre[j]   = sigmoid(mixes[j]   * scale[0] + base[j]) + eps
      post[j]  = 2*sigmoid(mixes[j+4] * scale[1] + base[j+4])
      comb[j,k]= mixes[8 + j*4 + k] * scale[2] + base[8 + j*4 + k]
      comb = softmax(comb, -1) + eps            # row softmax
      comb = comb / (comb.sum(-2) + eps)        # col normalize
      repeat (iters-1) times: row-normalize then col-normalize
    """
    lead = mixes.shape[:-1]
    pre  = torch.sigmoid(mixes[..., 0:hc]      * hc_scale[0] + hc_base[0:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2*hc] * hc_scale[1] + hc_base[hc:2*hc])
    comb = mixes[..., 2*hc:] * hc_scale[2] + hc_base[2*hc:]
    comb = comb.reshape(*lead, hc, hc)
    # first iteration: row = softmax (kernel does explicit max-subtract; torch.softmax matches)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    # remaining (iters-1) iterations: plain row-normalize then col-normalize
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x, hc_fn, hc_scale, hc_base, hc=4, norm_eps=1e-6, sinkhorn_iters=20, hc_eps=1e-6):
    """x:[b,s,hc,dim] -> y:[b,s,dim], post:[b,s,hc], comb:[b,s,hc,hc].

    Block.hc_pre (official_model.py:673-681):
      xf = x.flatten(2).float()
      mixes = (xf @ hc_fn.T) * rsqrt(mean(xf^2,-1)+norm_eps)
      pre,post,comb = hc_split_sinkhorn(mixes, ...)
      y = sum_k pre[...,k] * x[...,k,:]
    """
    shape = x.shape
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(xf, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, sinkhorn_iters, hc_eps)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y, post, comb


def hc_post(x, residual, post, comb):
    """x(attn/ffn out):[b,s,dim], residual:[b,s,hc,dim], post:[b,s,hc], comb:[b,s,hc,hc]
    -> y:[b,s,hc,dim].   Block.hc_post (official_model.py:683-686):
      y[...,j,:] = post[...,j]*x + sum_i comb[...,i,j]*residual[...,i,:]
    """
    return post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)


def hc_head(x, hc_fn, hc_scale, hc_base, hc=4, norm_eps=1e-6, hc_eps=1e-6):
    """Final reduce before the LM head (ParallelHead.hc_head, official_model.py:728-735).
    x:[b,s,hc,dim] -> y:[b,s,dim]. No post/comb, no Sinkhorn. hc_fn:[hc, hc*dim], scale:[1].
    """
    shape = x.shape
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(xf, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    return torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)

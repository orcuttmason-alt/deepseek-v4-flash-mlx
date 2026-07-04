"""Compressor (attention KV time-axis compression) — torch transcription of the official
Compressor.forward PREFILL path (official_model.py:316-377, start_pos==0, overlap ratio==4).

Produces the compressed KV rows (pre fp8-sim). The official then act_quants the nope dims and
ropes the rope dims; here we return pre-sim post-rope kv so the gate can split rope (tight) vs
nope (quant-floor). Weights are F16 in the GGUF (no affine landmine).
"""
import torch
from ref_torch.norm_rope import rmsnorm, apply_rotary_emb, precompute_freqs_cis


def overlap_transform(tensor, ratio, d, value):
    """tensor:[b,s,ratio,2d] -> [b,s,2*ratio,d]. official_model.py:307-314."""
    b, s, _, _ = tensor.size()
    new = tensor.new_full((b, s, 2 * ratio, d), value)
    new[:, :, ratio:] = tensor[:, :, :, d:]            # normal half -> own slots
    new[:, 1:, :ratio] = tensor[:, :-1, :, :d]         # overlap half of previous group
    return new


def compressor_prefill(x, wkv, wgate, ape, norm_w, head_dim=512, ratio=4, rope_head_dim=64,
                       base=160000.0, original_seq_len=65536, factor=16, beta_fast=32, beta_slow=1):
    """x:[b,s,dim] -> (kv_compressed:[b, s//ratio, head_dim] post-rope pre-sim, state dict).
    wkv/wgate:[out=2*head_dim, in=dim]; ape:[ratio, 2*head_dim]; norm_w:[head_dim]."""
    b, s, _ = x.shape
    d, rd = head_dim, rope_head_dim
    x = x.float()
    kv = x @ wkv.T                         # [b,s,1024]
    score = x @ wgate.T                    # [b,s,1024]
    remainder = s % ratio
    cutoff = s - remainder
    offset = ratio                         # overlap
    state = {}
    if cutoff >= ratio:
        state["kv"] = kv[:, cutoff - ratio:cutoff].clone()
        state["score"] = (score[:, cutoff - ratio:cutoff] + ape).clone()
    if remainder > 0:
        kv, kv_rem = kv[:, :cutoff], kv[:, cutoff:]
        score = score[:, :cutoff]
        state["kv_rem"] = kv_rem
    kv = kv.unflatten(1, (-1, ratio))                      # [b, g, ratio, 1024]
    score = score.unflatten(1, (-1, ratio)) + ape          # [b, g, ratio, 1024]
    kv = overlap_transform(kv, ratio, d, 0)                # [b, g, 2*ratio, 512]
    score = overlap_transform(score, ratio, d, float("-inf"))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)            # [b, g, 512]
    kv = rmsnorm(kv, norm_w)
    freqs = precompute_freqs_cis(rd, s, original_seq_len, base, factor, beta_fast, beta_slow)[:cutoff:ratio]
    roped = apply_rotary_emb(kv[..., -rd:], freqs)
    kv = torch.cat([kv[..., :-rd], roped], dim=-1)
    return kv, state

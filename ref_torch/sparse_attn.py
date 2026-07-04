"""sparse_attn references — torch transcription of official_model.py get_window_topk_idxs +
official_kernel.py sparse_attn_kernel (dense-gather equivalent). Used for the SELECTION
exact-set-match (vs official) and an arithmetic cross-check.
"""
import torch
import torch.nn.functional as F


def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    """Faithful copy of official_model.py:254-265 (all branches)."""
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """q:[s,h,d], kv:[n,d], attn_sink:[h], topk_idxs:[s,T] -> o:[s,h,d]. Dense-gather equivalent
    of sparse_attn_kernel: online-softmax == plain softmax over the gathered set + sink in denom."""
    valid = topk_idxs >= 0
    idx = topk_idxs.clamp(min=0)
    gathered = kv[idx]                                   # [s,T,d]
    scores = scale * torch.einsum("shd,std->sht", q, gathered)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    rowmax = scores.max(-1, keepdim=True).values
    p = torch.exp(scores - rowmax)
    denom = p.sum(-1, keepdim=True) + torch.exp(attn_sink.view(1, -1, 1) - rowmax)
    return torch.einsum("sht,std->shd", p, gathered) / denom

"""STEP 20: pure-MLX fast decode engine (quantized 8-bit dense, preloaded per-layer weight structs,
no per-call cache lookups, minimal host round-trips). Serial S=1 decode only; PREFILL stays on the
oracle (state structures identical: winr/kvbuf/ring/comp hand off directly).
Dense weights quantized per the STEP-18 gate (8-bit affine g64; router gate_inp kept fp16 as insurance).
MoE experts stay mxfp4 via the OVL async gather; shared experts quantized.
Modes:
  MODE=xcheck — one decode step, per-layer hidden comparison vs the qdq8 oracle (wiring check).
  MODE=gate   — C2: qdq8-oracle ref free-run N, fast teacher-forced N (flips/ppl) + leak + speed.
  MODE=solo   — cold base: oracle prefill -> caches cleared -> fast decode N steps @CAP.
Gates committed in OVERNIGHT_LOG STEP 20."""
import sys, os, json, time, math, copy
import numpy as np, mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import step12_glue as E
from v4mlx import norm_rope as NR, mla as MLA
G = E.G; H = G["H"]; SA = G["SA"]; D = E.D
RD = 64; LIM = E.LIM
TOPK = [int(os.environ.get("TOPK", "6"))]   # mutable so a test can sweep routed-expert count (active-param trade)
COS0, SIN0 = G["ROPE"][0]; COS1, SIN1 = G["ROPE"][1]
RATIOS = G["ratios"]
Q8 = dict(transpose=True, group_size=64, bits=8)
DENSE_BITS = [int(os.environ.get("DENSE_BITS", "8"))]   # attention/proj/compressor
SHEXP_BITS = [int(os.environ.get("SHEXP_BITS", "8"))]   # shared expert (FFN, tolerates lower bits)
HEAD_BITS  = [int(os.environ.get("HEAD_BITS",  "8"))]   # lm_head

def _raw16(n, twod=True):
    t = E.TEN[n]; a = E.gq.dequantize(np.array(t.data), t.tensor_type).astype(np.float16)
    a = a.reshape(int(t.shape[-1]), int(t.shape[-2])) if twod and len(t.shape) == 2 else a
    m = mx.array(a); mx.eval(m); return m
def qz(m16, bits=8, gs=64):
    t = mx.quantize(m16, group_size=gs, bits=bits); mx.eval(*t); return (t[0], t[1], t[2], bits, gs)
def qmm(x, qw):
    return mx.quantized_matmul(x, qw[0], qw[1], qw[2], transpose=True, group_size=qw[4], bits=qw[3])

class FastLayer:
    __slots__ = ("r4","ropec","q_a","q_a_norm","q_b","kv","kv_a_norm","attn_norm","ffn_norm","hc_af","hc_as",
                 "hc_ab","hc_ff","hc_fs","hc_fb","sinks","wo_a","wo_b","gate_inp","bias","tid",
                 "c_kv","c_gate","c_ape","c_norm","sg","su","sd")
    def __init__(s, L):
        n = lambda k: f"blk.{L}.{k}"
        s.r4 = (RATIOS[L] == 4)          # ring/comp ONLY at ratio 4 (ratio-128 layers have neither)
        s.ropec = bool(RATIOS[L])           # compressed-rope table for ANY nonzero ratio (4 or 128)
        s.q_a = qz(_raw16(n("attn_q_a.weight")), DENSE_BITS[0]); s.q_b = qz(_raw16(n("attn_q_b.weight")), DENSE_BITS[0])
        s.kv = qz(_raw16(n("attn_kv.weight")), DENSE_BITS[0])
        s.q_a_norm = _raw16(n("attn_q_a_norm.weight")); s.kv_a_norm = _raw16(n("attn_kv_a_norm.weight"))
        s.attn_norm = _raw16(n("attn_norm.weight")); s.ffn_norm = _raw16(n("ffn_norm.weight"))
        s.hc_af = _raw16(n("hc_attn_fn.weight")); s.hc_as = _raw16(n("hc_attn_scale.weight")); s.hc_ab = _raw16(n("hc_attn_base.weight"))
        s.hc_ff = _raw16(n("hc_ffn_fn.weight")); s.hc_fs = _raw16(n("hc_ffn_scale.weight")); s.hc_fb = _raw16(n("hc_ffn_base.weight"))
        s.sinks = _raw16(n("attn_sinks.weight"), False)
        wo_a = _raw16(n("attn_output_a.weight")).reshape(MLA.N_GROUPS, MLA.O_LORA, -1)
        s.wo_a = [qz(mx.array(wo_a[g]), DENSE_BITS[0]) for g in range(MLA.N_GROUPS)]
        s.wo_b = qz(_raw16(n("attn_output_b.weight")), DENSE_BITS[0])
        s.gate_inp = _raw16(n("ffn_gate_inp.weight"))          # fp16 (router insurance, per STEP-18)
        s.bias = None if L < 3 else np.array(E.gq.dequantize(np.array(E.TEN[n("exp_probs_b.bias")].data), E.TEN[n("exp_probs_b.bias")].tensor_type).astype(np.float32))
        s.tid = E.TID(L) if L < 3 else None
        if s.r4:
            s.c_kv = qz(_raw16(n("attn_compressor_kv.weight")), DENSE_BITS[0]); s.c_gate = qz(_raw16(n("attn_compressor_gate.weight")), DENSE_BITS[0])
            s.c_ape = _raw16(n("attn_compressor_ape.weight")); s.c_norm = _raw16(n("attn_compressor_norm.weight"), False)
        s.sg = qz(_raw16(n("ffn_gate_shexp.weight")), SHEXP_BITS[0]); s.su = qz(_raw16(n("ffn_up_shexp.weight")), SHEXP_BITS[0]); s.sd = qz(_raw16(n("ffn_down_shexp.weight")), SHEXP_BITS[0])

def moe_ovl_q(w, L, sel, ww, x_mx):
    """OVL gather (async expert loads under shared compute) with QUANTIZED shared experts. x fp32 [4096]."""
    gids = [L * 256 + int(e) for e in sel]; K = len(gids)
    E.PENDING.clear(); sm = E.CACHE.access_batch(gids)
    futs = E.PREAD.load_batch_async(list(E.PENDING)) if E.PENDING else None
    xr = x_mx[None]
    g = mx.minimum(qmm(xr, w.sg), LIM)
    sh = qmm((g * mx.sigmoid(g)) * mx.clip(qmm(xr, w.su), -LIM, LIM), w.sd)[0]
    mx.eval(sh)
    if futs: E.PREAD.load_batch_join(futs, E.CACHE.buffers); mx.eval([E.CACHE.buffers[k] for k in range(6)])
    xt = x_mx.astype(mx.float16).reshape(1, 1, -1)
    rhs = mx.array([sm[g2] for g2 in gids], dtype=mx.uint32); lhs0 = mx.zeros(K, dtype=mx.uint32)
    Q = dict(transpose=True, group_size=32, bits=4, mode="mxfp4")
    gg = mx.gather_qmm(xt, E.CACHE.buffers[0], E.CACHE.buffers[1], lhs_indices=lhs0, rhs_indices=rhs, **Q)
    uu = mx.gather_qmm(xt, E.CACHE.buffers[2], E.CACHE.buffers[3], lhs_indices=lhs0, rhs_indices=rhs, **Q)
    gc = mx.minimum(gg, LIM); act = (gc * mx.sigmoid(gc)) * mx.clip(uu, -LIM, LIM)
    act = act * mx.array(np.asarray(ww).astype(np.float16)).reshape(K, 1, 1)
    dd = mx.gather_qmm(act, E.CACHE.buffers[4], E.CACHE.buffers[5], lhs_indices=mx.arange(K, dtype=mx.uint32), rhs_indices=rhs, **Q)
    return dd.sum(0)[0].astype(mx.float32) + sh

def layer_step(w, L, x4, st, p, tokens):
    cos, sin = (COS1, SIN1) if w.ropec else (COS0, SIN0)
    cp = cos[p:p + 1]; sp = sin[p:p + 1]
    y, post, comb = H.hc_pre(x4, w.hc_af, w.hc_as, w.hc_ab)
    xn = NR.rmsnorm(y, w.attn_norm)
    q = qmm(NR.rmsnorm(qmm(xn, w.q_a), w.q_a_norm), w.q_b)
    q = MLA.perhead_norm(q.reshape(1, 1, MLA.N_HEADS, MLA.HEAD_DIM))
    q = mx.concatenate([q[..., :-RD], NR.apply_rotary_emb(q[..., -RD:], cp, sp)], axis=-1)[0]
    kv = NR.rmsnorm(qmm(xn, w.kv), w.kv_a_norm)
    kv = mx.concatenate([kv[..., :-RD], NR.apply_rotary_emb(kv[..., -RD:], cp, sp)], axis=-1)
    kvcur = mx.concatenate([E.mx_aq(kv[..., :-RD], 64), kv[..., -RD:]], axis=-1)[0]
    if getattr(st, "winr", None) is None:
        st.winr = np.zeros((128, 512), np.float32)
        for pp in range(max(0, len(st.win) - 128), len(st.win)): st.winr[pp % 128] = st.win[pp]
    st.winr[p % 128] = np.array(kvcur[0].astype(mx.float32))
    if w.r4:
        an = xn[0, 0]
        kvp = qmm(an[None], w.c_kv)[0]; scp = qmm(an[None], w.c_gate)[0] + w.c_ape[p % 4]
        st.kvs[4 + p % 4] = kvp; st.scs[4 + p % 4] = scp; mx.eval(st.kvs, st.scs)
        if (p + 1) % 4 == 0:
            kve = mx.concatenate([st.kvs[:4, :512], st.kvs[4:, 512:]], 0); sce = mx.concatenate([st.scs[:4, :512], st.scs[4:, 512:]], 0)
            wsm = mx.softmax(sce, axis=0); row = (kve * wsm).sum(0); row = w.c_norm * (row * mx.rsqrt(mx.mean(row ** 2) + 1e-6))
            pe = p + 1 - 4; roped = NR.apply_rotary_emb(row[-RD:][None, None], E.COSC[pe:pe + 1], E.SINC[pe:pe + 1])[0, 0]
            nope = E.mx_aq(row[:-RD][None], 64)[0]; emit = mx.concatenate([nope, roped]); mx.eval(emit)
            st.comp.append(np.array(emit)); st.kvs[:4] = st.kvs[4:]; st.scs[:4] = st.scs[4:]; mx.eval(st.kvs, st.scs)
    # attention (bounded kvbuf, 12a layout)
    wlo = max(0, p - 127); nw = p + 1 - wlo
    if getattr(st, "kvbuf", None) is None:
        st.kvbuf = np.zeros((128 + 256, 512), np.float32); st.ccount = 0
    if 128 + len(st.comp) > st.kvbuf.shape[0]:
        nb = np.zeros((max(128 + len(st.comp), st.kvbuf.shape[0] * 2), 512), np.float32); nb[:st.kvbuf.shape[0]] = st.kvbuf; st.kvbuf = nb
    while st.ccount < len(st.comp): st.kvbuf[128 + st.ccount] = st.comp[st.ccount]; st.ccount += 1
    nvis = (p + 1) // 4 if w.r4 else 0
    nc = min(len(st.comp), nvis)
    st.kvbuf[:nw] = st.winr[[pp % 128 for pp in range(wlo, p + 1)]]
    topk = np.concatenate([np.arange(nw, dtype=np.int32), 128 + np.arange(nc, dtype=np.int32)])
    o = SA.sparse_attn(q[0][None], mx.array(st.kvbuf[:128 + nc]), w.sinks, mx.array(topk[None]), 512 ** -0.5)[0][None][None]
    o = mx.concatenate([o[..., :-RD], NR.apply_rotary_emb(o[..., -RD:], cp, sp, inverse=True)], axis=-1)
    og = o.reshape(1, 1, MLA.N_GROUPS, -1)
    ob = mx.concatenate([qmm(og[..., g, :], w.wo_a[g]) for g in range(MLA.N_GROUPS)], axis=-1)
    h = H.hc_post(qmm(ob, w.wo_b), x4, post, comb)
    y2, post2, comb2 = H.hc_pre(h, w.hc_ff, w.hc_fs, w.hc_fb)
    xn2 = NR.rmsnorm(y2, w.ffn_norm)[0]
    scm = xn2 @ w.gate_inp.T
    sc = np.array(mx.sqrt(mx.log1p(mx.exp(-mx.abs(scm))) + mx.maximum(scm, 0)).astype(mx.float32))
    sel = w.tid[tokens[p]] if w.tid is not None else np.argsort(-(sc[0] + w.bias))[:TOPK[0]]
    ww = sc[0][sel]; ww = ww / ww.sum() * 1.5
    moe = moe_ovl_q(w, L, sel, ww, xn2[0])
    return H.hc_post(mx.stack([moe])[None], h, post2, comb2)

class V4Fast:
    def __init__(s):
        t0 = time.perf_counter(); s.layers = []
        for L in range(43):
            s.layers.append(FastLayer(L))
            if L % 8 == 0: print(f"  [fast] layer {L} loaded ({mx.get_active_memory()/1e9:.1f}GB)", flush=True)
        s.OUTQ = qz(G["OUT"].astype(mx.float16), HEAD_BITS[0])
        s.OHF = G["OHF"].astype(mx.float16); s.OHS = G["OHS"]; s.OHB = G["OHB"]; s.ONW = G["ONW"]
        print(f"[fast] {time.perf_counter()-t0:.0f}s load, dense {mx.get_active_memory()/1e9:.1f}GB mx-active", flush=True)
    def head(s, x4):
        f = H.hc_head(x4, s.OHF, s.OHS, s.OHB); f = NR.rmsnorm(f, s.ONW)
        return np.array(qmm(f, s.OUTQ)[0][0].astype(mx.float32))
    def step(s, states, tok, p, tokens):
        x4 = mx.array(np.repeat(E.embed[int(tok)][None, None, None, :], 4, axis=2))
        for L in range(43):
            x4 = layer_step(s.layers[L], L, x4, states[L], p, tokens)
        return x4

def clone_states(states):
    out = []
    for st in states:
        c = E.LS(); c.win = [r.copy() for r in st.win]; c.comp = [r.copy() for r in st.comp]
        c.kvs = mx.array(np.array(st.kvs)) if st.kvs is not None else None
        c.scs = mx.array(np.array(st.scs)) if st.scs is not None else None
        if getattr(st, "winr", None) is not None: c.winr = st.winr.copy()
        if getattr(st, "kvbuf", None) is not None: c.kvbuf = st.kvbuf.copy(); c.ccount = st.ccount
        out.append(c)
    return out

if __name__ == "__main__":
    mode = os.environ.get("MODE", "xcheck")
    E.dense8_switch()   # oracle reference = qdq8 (same weight VALUES as the fast engine)
    if mode == "xcheck":
        print("[xcheck] oracle qdq8 prefill...", flush=True)
        E.reset_cache(); states, pend = E.prefill(E.lf_b); mx.clear_cache()
        fast = V4Fast()
        stA = clone_states(states); stB = clone_states(states)
        fed = list(E.toks) + [int(pend)]; toks_np = np.array(fed, dtype=np.int64); p = len(E.toks)
        G["NP"] = len(E.toks); G["tokens"] = toks_np
        x4o = mx.array(np.repeat(E.embed[int(pend)][None, None, None, :], 4, axis=2))
        x4f = x4o
        for L in range(43):
            x4o = E.lf_b(x4o, L, stA[L], [p], decode_pos=p); mx.eval(x4o)
            x4f = layer_step(fast.layers[L], L, x4f, stB[L], p, toks_np); mx.eval(x4f)
            a = np.array(x4o[0, 0].astype(mx.float32)); b = np.array(x4f[0, 0].astype(mx.float32))
            rel = float(np.max(np.abs(a - b)) / (np.abs(a).max() + 1e-9))
            print(f"  L{L:2d} rel {rel:.2e} {'OK' if rel < 3e-2 else '<<< WIRING?'}", flush=True)
        lo = E.head_logits(x4o)[0]; lf_ = fast.head(x4f)
        print(f"[xcheck] head: oracle argmax {int(np.argmax(lo))} fast argmax {int(np.argmax(lf_))} "
              f"max|dlogit| {np.abs(lo-lf_).max():.3f}", flush=True)
    elif mode in ("gate", "solo"):
        N = int(os.environ.get("N", "150"))
        if mode == "gate":
            print(f"[gate] qdq8-oracle ref free-run {N} @cap={E.CAP}", flush=True)
            ref_toks, ref_per, _, ref_nll, ref_lgs, first = E.run_free(E.lf_b, N, "ref")
            n = len(ref_toks); pplR = math.exp(ref_nll / n)
        E.reset_cache(); states, pend = E.prefill(E.lf_b); mx.clear_cache()
        E.Wmx_single.cache_clear(); E.Wf16.cache_clear(); import gc; gc.collect(); mx.clear_cache()
        print(f"[fast] oracle caches cleared; mx-active {mx.get_active_memory()/1e9:.1f}GB", flush=True)
        fast = V4Fast()
        fed = list(E.toks); p = len(E.toks); flips = []; nllF = 0.0; per = []; memtr = []
        cur = int(pend) if mode == "solo" else int(first)
        M = N if mode == "solo" else n
        out = []
        from tokenizers import Tokenizer
        TOK = Tokenizer.from_file(os.path.join(os.path.dirname(D), "mlx-ckpt", "tokenizer.json"))
        for k in range(M):
            fed.append(cur); toks_np = np.array(fed, dtype=np.int64)
            t0 = time.perf_counter()
            x4 = fast.step(states, cur, p, toks_np)
            lg = fast.head(x4).astype(np.float64); per.append(time.perf_counter() - t0)
            t = int(np.argmax(lg)); out.append(t)
            if mode == "gate":
                bt = int(ref_toks[k])
                mvx = float(lg.max()); nllF += -(float(lg[bt]) - (float(np.log(np.exp(lg - mvx).sum())) + mvx))
                if t != bt:
                    rlg = ref_lgs[k].astype(np.float64); rank = int((lg > lg[bt]).sum()) + 1
                    flips.append(dict(k=k, b=bt, btxt=TOK.decode([bt]), m=t, mtxt=TOK.decode([t]), rank=rank,
                                      margin_b=float(rlg[bt] - rlg[t]), margin_m=float(lg[t] - lg[bt])))
                    print(f"  FLIP k={k} ref={TOK.decode([bt])!r} fast={TOK.decode([t])!r} rank={rank}", flush=True)
                cur = bt
            else:
                cur = t
            p += 1
            if k % 64 == 0: mx.clear_cache()
            memtr.append((mx.get_active_memory() / 1e9, E.swap_mb()))
            if k % 25 == 0:
                print(f"  [fast k={k}] {per[-1]*1000:.0f}ms mx {memtr[-1][0]:.2f}GB swap {memtr[-1][1]:.0f}MB", flush=True)
                if E.free_mb() < 1500: print("  LOW-FREE GUARD", flush=True); break
        ms = float(np.mean(per[3:]) * 1000); acts = [a for a, _ in memtr]
        print("\n" + "=" * 70)
        print(f"FAST {mode}: {ms:.0f} ms/tok = {1000/ms:.3f} tok/s over {len(out)} steps  "
              f"mx {acts[3]:.2f}->{acts[-1]:.2f}GB  hits {E.CACHE.hits} misses {E.CACHE.misses}")
        if mode == "gate":
            pplF = math.exp(nllF / len(out)); mech = sum(1 for f in flips if f['rank'] > 3)
            print(f"C2: flips={len(flips)}/{len(out)} mech-LB={mech} ppl-ratio={pplF/pplR:.4f} (<=1.02)  "
                  f"ref was {np.mean(ref_per[3:])*1000:.0f}ms/tok")
            for f in flips[:12]: print(f"  k={f['k']} ref={f['btxt']!r} fast={f['mtxt']!r} rank={f['rank']} mB={f['margin_b']:.4f}")
            json.dump(dict(ms=ms, flips=flips, ppl_ratio=pplF / pplR, mech=mech), open(os.path.join(D, "fast_gate.json"), "w"))
        else:
            json.dump(dict(ms=ms, n=len(out)), open(os.path.join(D, "fast_solo.json"), "w"))
    print(f"final swap={E.swap_mb():.0f}MB"); E.PREAD.close()

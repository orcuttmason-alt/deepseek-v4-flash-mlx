"""Batched-MoE prefill: replace the serial per-position MoE loop (62% of prefill) with ONE batched MLX
gather_qmm over all chunk positions. Attention (3%) and the already-batched proj/router (35%) unchanged.
Validation: prefill the SAME prompt through the serial engine (lf_b) and the batched engine (lf_pf), and
compare the final layer-42 hidden + per-position head argmax — if they match, the batched MoE is correct
(decode continuation is then guaranteed identical). Also times both.
Run: CAP=1664 N=128 ./v4-env/bin/python oracle/v4_prefill.py"""
import sys, os, json, time
import numpy as np, mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import step12_glue as E
G = E.G; mx_ = mx

PF_CHUNK = int(os.environ.get("PF_CHUNK", "96"))   # sub-batch size: bounds gather intermediates + expert union per burst

def _moe_gather_sub(L, sels, wws, X, LIMv):
    """batched routed+shared gather for a sub-chunk of positions. X: mx [s,4096] fp32."""
    s = len(sels); K = len(sels[0])
    flat_gids = [L * 256 + int(e) for sel in sels for e in sel]
    E.PENDING.clear(); sm = E.CACHE.access_batch(flat_gids)
    if E.PENDING: E.PREAD.load_batch(list(E.PENDING), E.CACHE.buffers); mx.eval([E.CACHE.buffers[k] for k in range(6)])
    sg = E.Wmx_single(f"blk.{L}.ffn_gate_shexp.weight"); su = E.Wmx_single(f"blk.{L}.ffn_up_shexp.weight"); sd = E.Wmx_single(f"blk.{L}.ffn_down_shexp.weight")
    g = mx.minimum(X @ sg.T, LIMv)
    sh = ((g * mx.sigmoid(g)) * mx.clip(X @ su.T, -LIMv, LIMv)) @ sd.T   # [s,4096]
    lhs = mx.array(np.repeat(np.arange(s, dtype=np.uint32), K))
    rhs = mx.array([sm[L * 256 + int(e)] for sel in sels for e in sel], dtype=mx.uint32)
    Xf = X.astype(mx.float16).reshape(s, 1, 4096)
    Q = dict(transpose=True, group_size=32, bits=4, mode="mxfp4")
    gg = mx.gather_qmm(Xf, E.CACHE.buffers[0], E.CACHE.buffers[1], lhs_indices=lhs, rhs_indices=rhs, **Q)
    uu = mx.gather_qmm(Xf, E.CACHE.buffers[2], E.CACHE.buffers[3], lhs_indices=lhs, rhs_indices=rhs, **Q)
    gc = mx.minimum(gg, LIMv); act = (gc * mx.sigmoid(gc)) * mx.clip(uu, -LIMv, LIMv)
    wflat = mx.array(np.concatenate([np.asarray(w, np.float16) for w in wws]).reshape(s * K, 1, 1))
    act = act * wflat
    dd = mx.gather_qmm(act, E.CACHE.buffers[4], E.CACHE.buffers[5], lhs_indices=mx.arange(s * K, dtype=mx.uint32), rhs_indices=rhs, **Q)
    return (dd.reshape(s, K, 4096).sum(1) + sh).astype(mx.float32)      # [s,4096]

def MOE_GATHER_BATCH(L, sels, wws, xn2_np, LIMv):
    """sels: list[S] of 6-expert ids; wws: list[S] of 6 weights; xn2_np: [S,4096] np -> mx [S,4096].
    Sub-chunked at PF_CHUNK so the gather intermediates + per-burst expert union stay bounded (safe for
    arbitrarily long prompts + keeps swap low at production cap)."""
    S = len(sels); X = mx.array(xn2_np)
    if S <= PF_CHUNK:
        return _moe_gather_sub(L, sels, wws, X, LIMv)
    outs = []
    for c in range(0, S, PF_CHUNK):
        sub = _moe_gather_sub(L, sels[c:c + PF_CHUNK], wws[c:c + PF_CHUNK], X[c:c + PF_CHUNK], LIMv)
        mx.eval(sub); outs.append(sub)
    return mx.concatenate(outs, axis=0)                                # [S,4096]
G["MOE_GATHER_BATCH"] = MOE_GATHER_BATCH

# edit: replace the serial prefill MoE loop in MOE_NEW_MX with the batched call
PF_OLD = '''    moe = np.zeros((S, 4096), np.float32)  # PREFILL (S>1): numpy, UNCHANGED from baseline
    for i in range(S):
        sel = tid[tokens[positions[i]]] if hashl else np.argsort(-(sc[i] + bias))[:6]
        w = sc[i][sel]; w = w / w.sum() * 1.5; x = xn2[i]
        moe[i] = MOE_GATHER_NP(L, sel, w, x, LIM)
    return H.hc_post(mx.array(moe[None]), h, post2, comb2)'''
PF_NEW = '''    sels = [ (tid[tokens[positions[i]]] if hashl else np.argsort(-(sc[i] + bias))[:6]) for i in range(S) ]
    wws = []
    for i in range(S):
        wv = sc[i][sels[i]]; wws.append(wv / wv.sum() * 1.5)
    moe = MOE_GATHER_BATCH(L, sels, wws, xn2, LIM)     # [S,4096] mx, batched gather
    return H.hc_post(moe[None], h, post2, comb2)'''

def build_prefill():
    edits = [(E.MOE_OLD, E.MOE_NEW_MX.replace(PF_OLD, PF_NEW))] + [
        (E.AQ_OLD, E.AQ_NEW), (E.INIT_OLD, E.INIT_NEW), (E.RING_OLD, E.RING_NEW),
        (E.B_OLD, E.B_NEW), (E.OG_OLD, E.OG_NEW)] + E.A_EDITS + E.B_EDITS
    return E.build("lf_pf", edits)

if __name__ == "__main__":
    N = int(os.environ.get("N", "128"))
    E.dense8_switch()
    lf_b = E.lf_b
    lf_pf = build_prefill()
    base = json.load(open(os.path.join(E.D, "code_prompt_tokens.json")))
    toks = (base * ((N // len(base)) + 1))[:N]
    def prefill(lf, label):
        E.toks[:] = toks; E.reset_cache()
        G["NP"] = len(toks); G["tokens"] = np.array(toks, dtype=np.int64)
        st = [E.LS() for _ in range(43)]
        x4 = mx.array(np.repeat(E.embed[np.array(toks)][:, None, :], 4, axis=1)[None])
        t0 = time.perf_counter()
        for L in range(43):
            x4 = lf(x4, L, st[L], list(range(len(toks)))); mx.eval(x4)
        dt = time.perf_counter() - t0
        lg = E.head_logits(x4)
        mx.clear_cache()
        print(f"  {label}: {dt:.1f}s = {N/dt:.1f} tok/s prefill", flush=True)
        return np.array(x4[0].astype(mx.float32)), lg, dt
    print(f"batched-MoE prefill validation, N={N} @cap={E.CAP}", flush=True)
    hb, lgb, tb = prefill(lf_b, "serial (lf_b)")
    hp, lgp, tp = prefill(lf_pf, "batched (lf_pf)")
    # correctness: final hidden rel-error + head argmax match per position
    rel = float(np.max(np.abs(hb - hp)) / (np.abs(hb).max() + 1e-9))
    amb = np.array([int(np.argmax(lgb[i])) for i in range(N)])
    amp = np.array([int(np.argmax(lgp[i])) for i in range(N)])
    match = int((amb == amp).sum())
    print("\n" + "=" * 60)
    print(f"CORRECTNESS: final-hidden global-rel {rel:.2e}  |  head argmax match {match}/{N}")
    print(f"SPEED: serial {N/tb:.1f} -> batched {N/tp:.1f} tok/s  = {tb/tp:.2f}x prefill")
    ok = rel < 3e-2 and match >= N - 2
    print(f"VERDICT: {'PASS (batched MoE correct + faster)' if ok and tp < tb else ('correct but not faster' if ok else 'FAIL — divergence')}")
    json.dump(dict(rel=rel, match=match, N=N, serial_tps=N/tb, batched_tps=N/tp, speedup=tb/tp),
              open(os.path.join(E.D, "prefill_batch.json"), "w"))
    print(f"final swap={E.swap_mb():.0f}MB"); E.PREAD.close()

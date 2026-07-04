"""STEP 12: FFN-side glue harvest. Three engines in one process:
  lf_ref = adopted STEP-11 engine (step10 edit set, MLX-shared decode).
  lf_a   = ref + BIT-IDENTICAL CPU fixes: KV window ring + persistent kvbuf (bounded upload,
           index-remapped), st.win growth frozen post-prefill, tid table lru-cached.
  lf_b   = lf_a + router-on-device (sc in MLX, 256 floats to host; device xn2 into MOE gather).
Driver: ref free-runs N steps (reference stream + self-nll + full logits kept for margins);
lf_a teacher-forced on the stream (flips MUST be 0); lf_b teacher-forced (C2 metrics).
Gates committed in OVERNIGHT_LOG STEP 12. cap=384, swap-watched (run_step12_guarded.sh)."""
import sys, os, json, re, time, subprocess, math
import numpy as np, mlx.core as mx, ml_dtypes
from functools import lru_cache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from offload_cache_v4 import TwoTierExpertCache, RECORD_BYTES
from pread_loader import PreadLoader
from v4mlx import norm_rope as NR, mla as MLA
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=os.path.join(ROOT,"dumps"); CK=os.path.join(ROOT,"mlx-ckpt")
DE=os.path.join(ROOT,"oracle","decode_engine.py"); RD=64; LIM=10.0; CAP=int(os.environ.get("CAP","384")); NG,OL=8,1024
def swap_mb():
    o=subprocess.check_output(["sysctl","-n","vm.swapusage"]).decode(); m=re.search(r"used\s*=\s*([\d.]+)M",o); return float(m.group(1)) if m else 0
def free_mb():
    o=subprocess.check_output(["vm_stat"]).decode(); g=lambda k:int(re.search(k+r":\s+(\d+)",o).group(1))
    return (g("Pages free")+g("Pages inactive")+g("Pages speculative"))*16384/1e6
BASE0=swap_mb(); assert BASE0<=3200, f"baseline {BASE0:.0f}MB>3200"
mx.set_memory_limit(int(38e9))
src=open(DE).read(); cut=src.index("# ---- PREFILL")
m=re.search(r"OHF = mx\.array.*?return np\.array\(\(f @ OUT\.T\)\[0\]\.astype\(mx\.float32\)\)",src,re.S)
G={"__file__":DE,"__name__":"de_defs"}; exec(compile(src[:cut]+"\nimport copy\n"+m.group(0)+"\n","de_defs","exec"),G)
TEN,gq=G["TEN"],G["gq"]; G["W"].cache_clear()
@lru_cache(maxsize=None)
def Wf16(n,twod=True):
    t=TEN[n]; a=gq.dequantize(np.array(t.data),t.tensor_type).astype(np.float16)
    a=a.reshape(int(t.shape[-1]),int(t.shape[-2])) if twod and len(t.shape)==2 else a
    if DENSE8["on"] and a.ndim==2 and a.size>=1_000_000 and "ffn_gate_inp" not in n:
        a=np.array(_qdq8(mx.array(a)).astype(mx.float16))
    return a
G["W"]=Wf16
DENSE8=dict(on=False)  # STEP-18: when on, big 2D dense weights are 8-bit-valued (qdq sim; router gate excluded)
def _qdq8(a_mx):
    q,s,b=mx.quantize(a_mx.astype(mx.float16),group_size=64,bits=8)
    return mx.dequantize(q,s,b,group_size=64,bits=8).astype(mx.float16)
def _d8_applies(n,a):
    return DENSE8["on"] and getattr(a,"ndim",len(getattr(a,"shape",())))==2 and a.size>=1_000_000 and "ffn_gate_inp" not in n
@lru_cache(maxsize=None)
def Wmx_single(n,twod=True):
    t=TEN[n]; a=gq.dequantize(np.array(t.data),t.tensor_type).astype(np.float16)
    a=a.reshape(int(t.shape[-1]),int(t.shape[-2])) if twod and len(t.shape)==2 else a
    mm=mx.array(a)
    if _d8_applies(n,mm): mm=_qdq8(mm)
    mx.eval(mm); del a; return mm
@lru_cache(maxsize=None)
def TID(L): return np.array(TEN[f"blk.{L}.ffn_gate_tid2eid.weight"].data).reshape(129280,-1)
mlx_idx=json.load(open(os.path.join(CK,"model.safetensors.index.json")))["weight_map"]
PREAD=PreadLoader(CK, mlx_idx, workers=8)
CACHE=TwoTierExpertCache(cap=CAP,pinned_keys=[],skip_load=False,loader=lambda g,s,b:PENDING.append((g,s)),alloc_buffers=True)
PENDING=[]
def mx_fp8_e4m3fn(v):
    sign=mx.sign(v); a=mx.abs(v); e=mx.floor(mx.log2(mx.maximum(a,1e-30)))
    ulp=mx.where(a<2.0**-6, mx.array(2.0**-9), mx.power(mx.array(2.0),e-3))
    q=mx.minimum(mx.round(a/ulp)*ulp,448.0); return mx.where(a==0,mx.array(0.0),sign*q)
def mx_aq(x,b=64):
    sh=x.shape; xb=x.reshape(*sh[:-1],sh[-1]//b,b); am=mx.maximum(mx.abs(xb).max(-1,keepdims=True),1e-4)
    s=mx.power(mx.array(2.0),mx.ceil(mx.log2(am/448.0))); return (mx_fp8_e4m3fn(mx.clip(xb/s,-448,448))*s).reshape(sh)
COSC,SINC=G["ROPE"][1]
def o_grouped_fast(o, wo_a, wo_b):
    lead=o.shape[:-1]; S=o.shape[1]
    if S==1:
        bs=int(np.prod(lead)); og=o.reshape(bs,NG,-1); wa=wo_a.reshape(NG,OL,-1)
        ob=mx.matmul(og.transpose(1,0,2), wa.transpose(0,2,1))
        o2=ob.transpose(1,0,2).reshape(*lead,-1)
        return o2 @ wo_b.T
    return MLA.o_grouped(o, wo_a, wo_b)
def _gather(L, sel, w, xt):
    gids=[L*256+int(e) for e in sel]; K=len(gids); PENDING.clear(); sm=CACHE.access_batch(gids)
    if PENDING: PREAD.load_batch(list(PENDING),CACHE.buffers); mx.eval([CACHE.buffers[k] for k in range(6)])
    rhs=mx.array([sm[g] for g in gids],dtype=mx.uint32); lhs0=mx.zeros(K,dtype=mx.uint32); Q=dict(transpose=True,group_size=32,bits=4,mode="mxfp4")
    gg=mx.gather_qmm(xt,CACHE.buffers[0],CACHE.buffers[1],lhs_indices=lhs0,rhs_indices=rhs,**Q)
    uu=mx.gather_qmm(xt,CACHE.buffers[2],CACHE.buffers[3],lhs_indices=lhs0,rhs_indices=rhs,**Q)
    def mx_silu(z): return z*mx.sigmoid(z)
    act=mx_silu(mx.minimum(gg,LIM))*mx.clip(uu,-LIM,LIM); act=act*mx.array(w.astype(np.float16)).reshape(K,1,1)
    dd=mx.gather_qmm(act,CACHE.buffers[4],CACHE.buffers[5],lhs_indices=mx.arange(K,dtype=mx.uint32),rhs_indices=rhs,**Q)
    return dd.sum(0)[0].astype(mx.float32)
def MOE_GATHER_NP(L, sel, w, x, LIMv):
    xt=mx.array(x.astype(np.float16)).reshape(1,1,-1); routed=np.array(_gather(L,sel,w,xt))
    sg,su,sd=Wf16(f"blk.{L}.ffn_gate_shexp.weight"),Wf16(f"blk.{L}.ffn_up_shexp.weight"),Wf16(f"blk.{L}.ffn_down_shexp.weight")
    g=np.minimum(sg@x,LIMv); u=np.clip(su@x,-LIMv,LIMv); return routed+sd@((g/(1+np.exp(-g)))*u)
def MOE_GATHER_MX(L, sel, w, x_in, LIMv):
    x_mx=x_in if isinstance(x_in,mx.array) else mx.array(x_in)
    routed=_gather(L, sel, w, x_mx.astype(mx.float16).reshape(1,1,-1))
    sg=Wmx_single(f"blk.{L}.ffn_gate_shexp.weight"); su=Wmx_single(f"blk.{L}.ffn_up_shexp.weight"); sd=Wmx_single(f"blk.{L}.ffn_down_shexp.weight")
    sh=sd@(( (lambda z: z*mx.sigmoid(z))(mx.minimum(sg@x_mx, LIMv)) )*mx.clip(su@x_mx, -LIMv, LIMv))
    return routed+sh
for k,v in dict(MOE_GATHER_NP=MOE_GATHER_NP,MOE_GATHER_MX=MOE_GATHER_MX,CACHE=CACHE,mx_aq=mx_aq,Wmx_single=Wmx_single,COSC=COSC,SINC=SINC,NR=NR,MLA=MLA,o_grouped_fast=o_grouped_fast,TID=TID).items(): G[k]=v
# ---- edit blocks ----
MOE_OLD='''    moe = np.zeros((S, 4096), np.float32)
    for i in range(S):
        sel = tid[tokens[positions[i]]] if hashl else np.argsort(-(sc[i] + bias))[:6]
        w = sc[i][sel]; w = w / w.sum() * 1.5; x = xn2[i]; acc = np.zeros(4096, np.float32)
        for j, ee in enumerate(sel):
            gg = np.minimum(EXP(f"blk.{L}.ffn_gate_exps.weight", ee) @ x, LIM); uu = np.clip(EXP(f"blk.{L}.ffn_up_exps.weight", ee) @ x, -LIM, LIM)
            acc += EXP(f"blk.{L}.ffn_down_exps.weight", ee) @ (w[j] * (silu(gg) * uu))
        moe[i] = acc + sd @ (silu(np.minimum(sg @ x, LIM)) * np.clip(su @ x, -LIM, LIM))
    return H.hc_post(mx.array(moe[None]), h, post2, comb2)'''
MOE_NEW_MX='''    if S == 1:  # DECODE-ONLY MLX-shared (S=1 -> no S=16 prefill promotion graph)
        sel = tid[tokens[positions[0]]] if hashl else np.argsort(-(sc[0] + bias))[:6]
        w = sc[0][sel]; w = w / w.sum() * 1.5
        return H.hc_post(mx.stack([MOE_GATHER_OVL(L, sel, w, xn2[0], LIM)])[None], h, post2, comb2)
    moe = np.zeros((S, 4096), np.float32)  # PREFILL (S>1): numpy, UNCHANGED from baseline
    for i in range(S):
        sel = tid[tokens[positions[i]]] if hashl else np.argsort(-(sc[i] + bias))[:6]
        w = sc[i][sel]; w = w / w.sum() * 1.5; x = xn2[i]
        moe[i] = MOE_GATHER_NP(L, sel, w, x, LIM)
    return H.hc_post(mx.array(moe[None]), h, post2, comb2)'''
AQ_OLD='    kvcur = mx.concatenate([act_quant_sim(kv[..., :-RD], 64), kv[..., -RD:]], axis=-1)[0]  # [S,512] AR-1'
AQ_NEW='    kvcur = mx.concatenate([mx_aq(kv[..., :-RD], 64), kv[..., -RD:]], axis=-1)[0]'
INIT_OLD='            if rem: st.kvs[4:4 + rem] = kvf[cut:]; st.scs[4:4 + rem] = scf[cut:] + ape[:rem]'
INIT_NEW='            if rem: st.kvs[4:4 + rem] = kvf[cut:]; st.scs[4:4 + rem] = scf[cut:] + ape[:rem]\n            st.kvs = mx.array(st.kvs); st.scs = mx.array(st.scs)'
RING_OLD='''        else:  # decode step: ring update + maybe emit
            an_np = np.array(xn[0, 0].astype(mx.float32)); wkv = W("blk.%d.attn_compressor_kv.weight" % L); wg = W("blk.%d.attn_compressor_gate.weight" % L); ape = W("blk.%d.attn_compressor_ape.weight" % L); nw = W("blk.%d.attn_compressor_norm.weight" % L, False)
            kvp = an_np @ wkv.T; scp = an_np @ wg.T + ape[decode_pos % 4]
            st.kvs[4 + decode_pos % 4] = kvp; st.scs[4 + decode_pos % 4] = scp
            if (decode_pos + 1) % 4 == 0:
                kve = np.concatenate([st.kvs[:4, :512], st.kvs[4:, 512:]], 0); sce = np.concatenate([st.scs[:4, :512], st.scs[4:, 512:]], 0)
                e = sce - sce.max(0, keepdims=True); w = np.exp(e) / np.exp(e).sum(0, keepdims=True)
                row = (kve * w).sum(0); row = nw * (row / np.sqrt((row ** 2).mean() + 1e-6))
                fc = torch.polar(torch.ones(1, 32), torch.tensor((decode_pos + 1 - 4) * freqsC)[None]); row[-RD:] = rope_t(torch.tensor(row[-RD:])[None, None], fc)[0, 0].numpy()
                row[:-RD] = aq(row[:-RD], 64); st.comp.append(row); st.kvs[:4] = st.kvs[4:]; st.scs[:4] = st.scs[4:]'''
RING_NEW='''        else:  # decode step: ring update + maybe emit (MLX PORT)
            an = xn[0, 0]; wkv = Wmx_single("blk.%d.attn_compressor_kv.weight" % L); wg = Wmx_single("blk.%d.attn_compressor_gate.weight" % L); apem = Wmx_single("blk.%d.attn_compressor_ape.weight" % L); nwm = Wmx_single("blk.%d.attn_compressor_norm.weight" % L, False)
            kvp = an @ wkv.T; scp = an @ wg.T + apem[decode_pos % 4]
            st.kvs[4 + decode_pos % 4] = kvp; st.scs[4 + decode_pos % 4] = scp; mx.eval(st.kvs, st.scs)
            if (decode_pos + 1) % 4 == 0:
                kve = mx.concatenate([st.kvs[:4, :512], st.kvs[4:, 512:]], 0); sce = mx.concatenate([st.scs[:4, :512], st.scs[4:, 512:]], 0)
                w = mx.softmax(sce, axis=0); row = (kve * w).sum(0); row = nwm * (row * mx.rsqrt(mx.mean(row ** 2) + 1e-6))
                p = decode_pos + 1 - 4; roped = NR.apply_rotary_emb(row[-RD:][None, None], COSC[p:p+1], SINC[p:p+1])[0, 0]
                nope = mx_aq(row[:-RD][None], 64)[0]; emit = mx.concatenate([nope, roped]); mx.eval(emit)
                st.comp.append(np.array(emit)); st.kvs[:4] = st.kvs[4:]; st.scs[:4] = st.scs[4:]; mx.eval(st.kvs, st.scs)'''
B_OLD='B = lambda n, t=True: mx.array(W(f"blk.{L}.{n}", t))'
B_NEW='B = lambda n, t=True: Wmx_single(f"blk.{L}.{n}", t)'
OG_OLD='H.hc_post(MLA.o_grouped(o.reshape(1, S, -1), B("attn_output_a.weight"), B("attn_output_b.weight")), x4, post, comb)'
OG_NEW='H.hc_post(o_grouped_fast(o.reshape(1, S, -1), B("attn_output_a.weight"), B("attn_output_b.weight")), x4, post, comb)'
# --- 12a: KV window ring + bounded kvbuf upload + frozen st.win (BIT-IDENTICAL by construction) ---
WIN_OLD='    for i in range(S): st.win.append(np.array(kvcur[i].astype(mx.float32)))'
WIN_NEW='''    if S == 1 and decode_pos is not None:
        if getattr(st, "winr", None) is None:
            st.winr = np.zeros((128, 512), np.float32)
            for pp in range(max(0, len(st.win) - 128), len(st.win)): st.winr[pp % 128] = st.win[pp]
        st.winr[positions[0] % 128] = np.array(kvcur[0].astype(mx.float32))
    else:
        for i in range(S): st.win.append(np.array(kvcur[i].astype(mx.float32)))'''
ATTN_OLD='''    win = np.stack(st.win); ncw = win.shape[0]
    kvcat = np.concatenate([win, np.stack(st.comp)], 0) if st.comp else win
    o_all = []
    for i in range(S):
        gp = positions[i]; wlo = max(0, gp - 127); widx = list(range(wlo, gp + 1))
        nvis = (gp + 1) // 4 if r == 4 else 0   # CAUSAL compressed rows visible to position gp
        cidx = [ncw + j for j in range(min(len(st.comp), nvis))]
        topk = np.array(widx + cidx, np.int32)
        o = SA.sparse_attn(q[i][None], mx.array(kvcat), B("attn_sinks.weight", False), mx.array(topk[None]), 512 ** -0.5)[0]
        o_all.append(o)
    o = mx.stack(o_all)[None]  # [1,S,64,512]'''
ATTN_NEW='''    if S == 1 and decode_pos is not None:  # bounded upload: last-128 window + visible comp rows (identical gathered rows)
        gp = positions[0]; wlo = max(0, gp - 127); nw = gp + 1 - wlo
        if getattr(st, "kvbuf", None) is None:
            st.kvbuf = np.zeros((128 + 256, 512), np.float32); st.ccount = 0
        if 128 + len(st.comp) > st.kvbuf.shape[0]:
            nb = np.zeros((max(128 + len(st.comp), st.kvbuf.shape[0] * 2), 512), np.float32); nb[:st.kvbuf.shape[0]] = st.kvbuf; st.kvbuf = nb
        while st.ccount < len(st.comp): st.kvbuf[128 + st.ccount] = st.comp[st.ccount]; st.ccount += 1
        nvis = (gp + 1) // 4 if r == 4 else 0
        nc = min(len(st.comp), nvis)
        st.kvbuf[:nw] = st.winr[[p % 128 for p in range(wlo, gp + 1)]]
        topk = np.concatenate([np.arange(nw, dtype=np.int32), 128 + np.arange(nc, dtype=np.int32)])
        o = SA.sparse_attn(q[0][None], mx.array(st.kvbuf[:128 + nc]), B("attn_sinks.weight", False), mx.array(topk[None]), 512 ** -0.5)[0][None][None]
    else:
        win = np.stack(st.win); ncw = win.shape[0]
        kvcat = np.concatenate([win, np.stack(st.comp)], 0) if st.comp else win
        o_all = []
        for i in range(S):
            gp = positions[i]; wlo = max(0, gp - 127); widx = list(range(wlo, gp + 1))
            nvis = (gp + 1) // 4 if r == 4 else 0   # CAUSAL compressed rows visible to position gp
            cidx = [ncw + j for j in range(min(len(st.comp), nvis))]
            topk = np.array(widx + cidx, np.int32)
            o = SA.sparse_attn(q[i][None], mx.array(kvcat), B("attn_sinks.weight", False), mx.array(topk[None]), 512 ** -0.5)[0]
            o_all.append(o)
        o = mx.stack(o_all)[None]  # [1,S,64,512]'''
TID_OLD='    tid = np.array(TEN[f"blk.{L}.ffn_gate_tid2eid.weight"].data).reshape(129280, -1) if hashl else None'
TID_NEW='    tid = TID(L) if hashl else None'
# --- 12b: router score on device (sc in MLX; only 256 floats to host; device xn2 into MOE) ---
Y2_OLD='''    xn2 = np.array(NR.rmsnorm(y2, B("ffn_norm.weight"))[0].astype(mx.float32))
    sc = ssp(xn2 @ W(f"blk.{L}.ffn_gate_inp.weight").T); hashl = L < 3'''
Y2_NEW='''    hashl = L < 3
    if S == 1 and decode_pos is not None:  # ROUTER ON DEVICE (12b)
        xn2 = NR.rmsnorm(y2, B("ffn_norm.weight"))[0]
        scm = xn2 @ B("ffn_gate_inp.weight").T
        sc = np.array(mx.sqrt(mx.log1p(mx.exp(-mx.abs(scm))) + mx.maximum(scm, 0)).astype(mx.float32))
    else:
        xn2 = np.array(NR.rmsnorm(y2, B("ffn_norm.weight"))[0].astype(mx.float32))
        sc = ssp(xn2 @ W(f"blk.{L}.ffn_gate_inp.weight").T)'''
BASE_EDITS=[(MOE_OLD,MOE_NEW_MX),(AQ_OLD,AQ_NEW),(INIT_OLD,INIT_NEW),(RING_OLD,RING_NEW),(B_OLD,B_NEW),(OG_OLD,OG_NEW)]
A_EDITS=[(WIN_OLD,WIN_NEW),(ATTN_OLD,ATTN_NEW),(TID_OLD,TID_NEW)]
B_EDITS=[(Y2_OLD,Y2_NEW)]
def build(name, edits):
    s=src
    for a,b in edits:
        assert a in s, f"{name}: edit anchor missing: {a[:60]!r}"; s=s.replace(a,b)
    gi=s.index("def layer_fwd"); gj=s.index("# ---- PREFILL")
    exec(compile(s[gi:gj].replace("def layer_fwd(",f"def {name}(",1),name,"exec"),G); return G[name]
lf_ref=build("lf_ref", BASE_EDITS)
lf_a  =build("lf_a",   BASE_EDITS+A_EDITS)
lf_b  =build("lf_b",   BASE_EDITS+A_EDITS+B_EDITS)
head_logits=G["head_logits"]; LS=G["LS"]; embed=G["embed"]
from tokenizers import Tokenizer
TOK=Tokenizer.from_file(os.path.join(CK,"tokenizer.json"))
def txt(i): return TOK.decode([int(i)], skip_special_tokens=False)
toks=json.loads(open(os.path.join(D,"code_prompt_tokens.json")).read())[:16]
OUTJ=os.path.join(D,"step12_glue.json")
def lse(v):
    mvx=float(v.max()); return float(np.log(np.exp(v-mvx).sum())+mvx)
def reset_cache():
    for k in range(6): CACHE.buffers[k]=mx.zeros_like(CACHE.buffers[k])
    CACHE.slot_of.clear(); CACHE.lru.clear(); CACHE.free=list(range(CACHE.cap)); CACHE.loads=CACHE.hits=CACHE.misses=0; mx.eval(CACHE.buffers)
def prefill(lf):
    st=[LS() for _ in range(43)]; G["NP"]=len(toks); G["tokens"]=np.array(list(toks),dtype=np.int64)
    x4=mx.array(np.repeat(embed[np.array(toks)][:,None,:],4,axis=1)[None])
    for L in range(43): x4=lf(x4,L,st[L],list(range(len(toks)))); mx.eval(x4)
    return st,int(np.argmax(head_logits(x4)[len(toks)-1]))
def run_free(lf, N, tag):
    """free-run: returns (tokens, per-ms, mem-trace, self-nll, full fp32 logits per step, prompt-argmax)"""
    reset_cache(); st,pend=prefill(lf); mx.clear_cache(); pend0=pend
    fed=list(toks); pos=len(toks); out=[]; per=[]; memtr=[]; nll=0.0; lgs=[]
    for k in range(N):
        t0=time.perf_counter(); fed.append(pend); G["tokens"]=np.array(fed,dtype=np.int64)
        xd=mx.array(np.repeat(embed[int(pend)][None,None,None,:],4,axis=2))
        for L in range(43): xd=lf(xd,L,st[L],[pos],decode_pos=pos); mx.eval(xd)
        lg=head_logits(xd)[0].astype(np.float64); per.append(time.perf_counter()-t0)
        t=int(np.argmax(lg)); nll+=-(float(lg[t])-lse(lg)); lgs.append(lg.astype(np.float32))
        out.append(t); pend=t; pos+=1; mx.clear_cache()
        act=mx.get_active_memory()/1e9; sw=swap_mb(); memtr.append((act,sw))
        if k%25==0: print(f"  [{tag} k={k}] {per[-1]*1000:.0f}ms mx {act:.2f}GB swap {sw:.0f}MB",flush=True)
        if sw>BASE0+1500: print(f"  {tag} SWAP GUARD @ {k}",flush=True); break
    return out,per,memtr,nll,lgs,pend0
def run_forced(lf, stream, first_tok, N, tag, ref_lgs):
    """teacher-forced on ref stream; per-step compare to ref token; returns flips, nll-on-ref, ms, mem.
    step k: engine has consumed prompt + stream[:k]; feed = first_tok (prompt argmax) at k=0 else stream[k-1]."""
    reset_cache(); st,pf_argmax=prefill(lf); mx.clear_cache()
    if pf_argmax!=first_tok: print(f"  [{tag}] NOTE: prefill argmax {pf_argmax} != ref {first_tok} (prefill path should be shared!)",flush=True)
    fed=list(toks); pos=len(toks); flips=[]; per=[]; memtr=[]; nll=0.0
    for k in range(N):
        tok_in = stream[k-1] if k>0 else first_tok
        fed.append(tok_in); G["tokens"]=np.array(fed,dtype=np.int64)
        t0=time.perf_counter()
        xd=mx.array(np.repeat(embed[int(tok_in)][None,None,None,:],4,axis=2))
        for L in range(43): xd=lf(xd,L,st[L],[pos],decode_pos=pos); mx.eval(xd)
        lg=head_logits(xd)[0].astype(np.float64); per.append(time.perf_counter()-t0)
        mt=int(np.argmax(lg)); bt=int(stream[k])
        nll+=-(float(lg[bt])-lse(lg))
        if mt!=bt:
            rlg=ref_lgs[k].astype(np.float64); rank=int((lg>lg[bt]).sum())+1
            flips.append(dict(k=k,b=bt,btxt=txt(bt),m=mt,mtxt=txt(mt),rank=rank,
                margin_b=float(rlg[bt]-rlg[mt]),margin_m=float(lg[mt]-lg[bt])))
            print(f"  [{tag}] FLIP k={k} ref={txt(bt)!r} var={txt(mt)!r} rank={rank} mB={rlg[bt]-rlg[mt]:.4f} mM={lg[mt]-lg[bt]:.4f}",flush=True)
        pos+=1; mx.clear_cache()
        act=mx.get_active_memory()/1e9; sw=swap_mb(); memtr.append((act,sw))
        if k%25==0: print(f"  [{tag} k={k}] {per[-1]*1000:.0f}ms flips={len(flips)} mx {act:.2f}GB swap {sw:.0f}MB",flush=True)
        if sw>BASE0+1500: print(f"  {tag} SWAP GUARD @ {k}",flush=True); break
    return flips,nll,per,memtr
def memstats(memtr):
    acts=[a for a,_ in memtr]
    if len(acts)<5: return float('nan'),float('nan')
    xs=np.arange(len(acts))[3:]; ys=np.array(acts)[3:]
    return float(np.polyfit(xs,ys,1)[0]*1000), acts[-1]-acts[min(3,len(acts)-1)]
HOB_THR=float(os.environ.get("HOB_THR","0.15")); DEG=set(); HOB=dict(deg=0,rescue=0,full=0)
def _gather_hob(L, sel, w, xt):
    """share-gated 2-bit degradation SIM (HOBBIT variant): missed experts whose routing share (<w>/1.5)
    is < HOB_THR are degraded in-slot to 2-bit fidelity (quant2->dequant->requant-mxfp4). A later access
    at share >= THR of a degraded slot forces a full-precision RELOAD (rescue). Quality sim only —
    measures the C2 question; byte savings computed from deg/rescue/full counters."""
    gids=[L*256+int(e) for e in sel]; share=np.asarray(w,np.float32)/1.5
    PENDING.clear(); sm=CACHE.access_batch(gids)
    fresh={g for g,_ in PENDING}
    if PENDING: PREAD.load_batch(list(PENDING),CACHE.buffers); mx.eval([CACHE.buffers[k] for k in range(6)])
    rescue=[(g,sm[g]) for j,g in enumerate(gids) if g in DEG and g not in fresh and share[j]>=HOB_THR]
    if rescue:
        PREAD.load_batch(rescue,CACHE.buffers); mx.eval([CACHE.buffers[k] for k in range(6)])
        for g,_ in rescue: DEG.discard(g); HOB["rescue"]+=1
    for j,g in enumerate(gids):
        if g in fresh:
            if share[j]<HOB_THR:
                s=sm[g]
                for pj in range(3):
                    deq=mx.dequantize(CACHE.buffers[2*pj][s],CACHE.buffers[2*pj+1][s],None,group_size=32,bits=4,mode="mxfp4").astype(mx.float16)
                    q2,s2,b2=mx.quantize(deq,group_size=32,bits=2)
                    d2=mx.dequantize(q2,s2,b2,group_size=32,bits=2)
                    rq,rs=mx.quantize(d2,group_size=32,bits=4,mode="mxfp4")
                    CACHE.buffers[2*pj][s]=rq; CACHE.buffers[2*pj+1][s]=rs
                mx.eval([CACHE.buffers[k] for k in range(6)]); DEG.add(g); HOB["deg"]+=1
            else: HOB["full"]+=1
    rhs=mx.array([sm[g] for g in gids],dtype=mx.uint32); lhs0=mx.zeros(len(gids),dtype=mx.uint32); Q=dict(transpose=True,group_size=32,bits=4,mode="mxfp4")
    gg=mx.gather_qmm(xt,CACHE.buffers[0],CACHE.buffers[1],lhs_indices=lhs0,rhs_indices=rhs,**Q)
    uu=mx.gather_qmm(xt,CACHE.buffers[2],CACHE.buffers[3],lhs_indices=lhs0,rhs_indices=rhs,**Q)
    gc=mx.minimum(gg,LIM); act=(gc*mx.sigmoid(gc))*mx.clip(uu,-LIM,LIM); act=act*mx.array(np.asarray(w).astype(np.float16)).reshape(len(gids),1,1)
    dd=mx.gather_qmm(act,CACHE.buffers[4],CACHE.buffers[5],lhs_indices=mx.arange(len(gids),dtype=mx.uint32),rhs_indices=rhs,**Q)
    return dd.sum(0)[0].astype(mx.float32)
def MOE_GATHER_HOB(L, sel, w, x_in, LIMv):
    x_mx=x_in if isinstance(x_in,mx.array) else mx.array(x_in)
    routed=_gather_hob(L, sel, w, x_mx.astype(mx.float16).reshape(1,1,-1))
    sg=Wmx_single(f"blk.{L}.ffn_gate_shexp.weight"); su=Wmx_single(f"blk.{L}.ffn_up_shexp.weight"); sd=Wmx_single(f"blk.{L}.ffn_down_shexp.weight")
    sh=sd@(( (lambda z: z*mx.sigmoid(z))(mx.minimum(sg@x_mx, LIMv)) )*mx.clip(su@x_mx, -LIMv, LIMv))
    return routed+sh
G["MOE_GATHER_HOB"]=MOE_GATHER_HOB
def head_top_dev(x4_):
    """device-side head: returns (argmax_id, top1-top2 margin) pulling 3 scalars instead of 517KB."""
    f=G["H"].hc_head(x4_,G["OHF"],G["OHS"],G["OHB"]); f=NR.rmsnorm(f,G["ONW"])
    lg=(f @ G["OUT"].T)[0][0]
    i1=mx.argmax(lg); m1=lg[i1]
    lg2=mx.where(mx.arange(lg.shape[0])==i1, mx.array(-np.inf), lg)
    m2=mx.max(lg2); mx.eval(i1,m1,m2)
    return int(i1), float(m1-m2)
def run_speed(lf, N, tag, clear=True, layer_eval=True, dev_head=False):
    """decode-speed variant runner: returns (tokens, per-step ms)."""
    reset_cache(); st,pend=prefill(lf); mx.clear_cache()
    fed=list(toks); pos=len(toks); out=[]; per=[]
    for k in range(N):
        t0=time.perf_counter(); fed.append(pend); G["tokens"]=np.array(fed,dtype=np.int64)
        xd=mx.array(np.repeat(embed[int(pend)][None,None,None,:],4,axis=2))
        for L in range(43):
            xd=lf(xd,L,st[L],[pos],decode_pos=pos)
            if layer_eval: mx.eval(xd)
        if dev_head: pend,_=head_top_dev(xd)
        else: pend=int(np.argmax(head_logits(xd)[0]))
        out.append(pend); pos+=1; per.append(time.perf_counter()-t0)
        if clear: mx.clear_cache()
        if k%20==0:
            sw=swap_mb()
            print(f"  [{tag} k={k}] {per[-1]*1000:.0f}ms mx {mx.get_active_memory()/1e9:.2f}GB swap {sw:.0f}MB",flush=True)
            if sw>BASE0+1500: print(f"  {tag} SWAP GUARD",flush=True); break
    return out,per
def MOE_GATHER_OVL(L, sel, w, x_in, LIMv):
    """MOE_GATHER_MX with async load / shared-compute overlap: preads run in threads while the GPU
    computes the shared expert (forced via eval), then join + gather. Same values, same summation order
    -> bit-identical expected (gate A0 flips==0)."""
    x_mx=x_in if isinstance(x_in,mx.array) else mx.array(x_in)
    gids=[L*256+int(e) for e in sel]; K=len(gids); PENDING.clear(); sm=CACHE.access_batch(gids)
    futs=PREAD.load_batch_async(list(PENDING)) if PENDING else None
    sg=Wmx_single(f"blk.{L}.ffn_gate_shexp.weight"); su=Wmx_single(f"blk.{L}.ffn_up_shexp.weight"); sd=Wmx_single(f"blk.{L}.ffn_down_shexp.weight")
    sh=sd@(( (lambda z: z*mx.sigmoid(z))(mx.minimum(sg@x_mx, LIMv)) )*mx.clip(su@x_mx, -LIMv, LIMv))
    mx.eval(sh)  # GPU busy on shared while the I/O threads read
    if futs: PREAD.load_batch_join(futs,CACHE.buffers); mx.eval([CACHE.buffers[k] for k in range(6)])
    xt=x_mx.astype(mx.float16).reshape(1,1,-1)
    rhs=mx.array([sm[g] for g in gids],dtype=mx.uint32); lhs0=mx.zeros(K,dtype=mx.uint32); Q=dict(transpose=True,group_size=32,bits=4,mode="mxfp4")
    gg=mx.gather_qmm(xt,CACHE.buffers[0],CACHE.buffers[1],lhs_indices=lhs0,rhs_indices=rhs,**Q)
    uu=mx.gather_qmm(xt,CACHE.buffers[2],CACHE.buffers[3],lhs_indices=lhs0,rhs_indices=rhs,**Q)
    gc2=mx.minimum(gg,LIM); act=(gc2*mx.sigmoid(gc2))*mx.clip(uu,-LIM,LIM); act=act*mx.array(np.asarray(w).astype(np.float16)).reshape(K,1,1)
    dd=mx.gather_qmm(act,CACHE.buffers[4],CACHE.buffers[5],lhs_indices=mx.arange(K,dtype=mx.uint32),rhs_indices=rhs,**Q)
    return dd.sum(0)[0].astype(mx.float32)+sh
G["MOE_GATHER_OVL"]=MOE_GATHER_OVL
def dense8_switch():
    """flip the engine to 8-bit-valued dense weights: clear both weight caches (rebuild lazily under the
    flag) and qdq/cast the module-level head tensors (OUT is fp32 2.1GB today — an exec-order artifact —
    becomes qdq8-in-fp16 1.06GB)."""
    DENSE8["on"]=True; Wmx_single.cache_clear(); Wf16.cache_clear()
    nq=0
    for nm in ("OUT","OHF","OHS","OHB","ONW"):
        a=G[nm]
        if getattr(a,"ndim",0)==2 and a.size>=1_000_000: G[nm]=_qdq8(a); nq+=1
        else: G[nm]=a.astype(mx.float16)
    mx.eval([G[nm] for nm in ("OUT","OHF","OHS","OHB","ONW")]); mx.clear_cache()
    print(f"[dense8] switched: head tensors qdq8={nq}, weight caches cleared (rebuild under flag)",flush=True)
if __name__=="__main__":
    if os.environ.get("MODE")=="dense8":  # STEP 18: 8-bit dense quality gate under the C2 bar
        N=int(os.environ.get("N","150"))
        print(f"DENSE8 GATE: cap={CAP} N={N} swap={BASE0:.0f}MB",flush=True)
        ref_toks,ref_per,_,ref_nll,ref_lgs,first=run_free(lf_b,N,"ref")
        n=len(ref_toks); pplR=math.exp(ref_nll/n)
        dense8_switch()
        fl,nll8,per8,mem8=run_forced(lf_b,ref_toks,first,n,"d8",ref_lgs)
        ppl8=math.exp(nll8/n); mech=sum(1 for f in fl if f['rank']>3)
        acts=[a for a,_ in mem8]
        print("\n"+"="*70)
        print(f"DENSE8 GATE: flips={len(fl)}/{n} mech-LB={mech} ppl-ratio={ppl8/pplR:.4f} (<=1.02)")
        print(f"mx-active d8 final {acts[-1]:.2f}GB (head fp32->qdq8fp16 saves ~1GB even in sim)")
        for f in fl: print(f"  flip k={f['k']} ref={f['btxt']!r} d8={f['mtxt']!r} rank={f['rank']} mB={f['margin_b']:.4f} mM={f['margin_m']:.4f}")
        verdict='PASS' if (mech==0 and ppl8/pplR<=1.02) else 'FAIL'
        print(f"VERDICT: {verdict} (mech-LB==0 and ppl<=1.02; text-review flips)")
        json.dump(dict(flips=fl,ppl_ratio=ppl8/pplR,mech=mech,verdict=verdict),open(os.path.join(D,"dense8_gate.json"),"w"))
        print(f"final swap={swap_mb():.0f}MB (base {BASE0:.0f})"); PREAD.close(); sys.exit(0)
    if os.environ.get("MODE")=="overlap":  # TASK 9: async load / shared-compute overlap (bit-identical reorder)
        N=int(os.environ.get("N","100"))
        lf_o=build("lf_ovl",[(MOE_OLD,MOE_NEW_MX.replace("MOE_GATHER_MX(","MOE_GATHER_OVL(")),(AQ_OLD,AQ_NEW),(INIT_OLD,INIT_NEW),(RING_OLD,RING_NEW),(B_OLD,B_NEW),(OG_OLD,OG_NEW)]+A_EDITS+B_EDITS)
        print(f"OVERLAP GATE: cap={CAP} N={N} (variant runs FIRST = colder page cache, bias against it)",flush=True)
        out_o,per_o=run_speed(lf_o,N,"ovl",clear=False,layer_eval=False)
        out_b,per_b=run_speed(lf_b,N,"ref",clear=False,layer_eval=False)
        m=min(len(out_o),len(out_b)); flips=sum(1 for a,b in zip(out_o[:m],out_b[:m]) if a!=b)
        mo=np.mean(per_o[3:])*1000; mb=np.mean(per_b[3:])*1000
        print("\n"+"="*70)
        print(f"OVERLAP: {mo:.0f} ms/tok vs ref {mb:.0f} ms/tok = {(mb-mo)/mb*100:+.1f}% (variant ran colder)")
        print(f"flips={flips}/{m} (MUST be 0) -> {'PASS' if flips==0 and mo<=mb else ('EXACT but not faster' if flips==0 else 'FAIL — NOT bit-identical, investigate')}")
        json.dump(dict(ms_ovl=mo,ms_ref=mb,flips=flips),open(os.path.join(D,"overlap_gate.json"),"w"))
        print(f"final swap={swap_mb():.0f}MB (base {BASE0:.0f})"); PREAD.close(); sys.exit(0)
    if os.environ.get("MODE")=="ideas":  # IDEA ROUND: A/B micro-variants on the deployed engine
        N=int(os.environ.get("N","60"))
        print(f"IDEAS: cap={CAP} N={N} swap={BASE0:.0f}MB free={free_mb():.0f}MB",flush=True)
        variants=[("baseline",dict()),("no_clear",dict(clear=False)),("no_layer_eval",dict(layer_eval=False)),
                  ("dev_head",dict(dev_head=True)),("combo",dict(clear=False,layer_eval=False,dev_head=True))]
        if os.environ.get("WIRED"): mx.set_wired_limit(int(float(os.environ["WIRED"])*1e9)); variants.append(("wired_combo",dict(clear=False,layer_eval=False,dev_head=True)))
        res={}; ref_stream=None
        for tag,kw in variants:
            out,per=run_speed(lf_b,N,tag,**kw)
            ms=float(np.mean(per[3:])*1000); res[tag]=ms
            if ref_stream is None: ref_stream=out
            same=sum(1 for a,b in zip(ref_stream,out) if a==b)
            print(f"{tag}: {ms:.0f} ms/tok = {1000/ms:.3f} tok/s  stream-match {same}/{min(len(out),len(ref_stream))}",flush=True)
        base=res["baseline"]
        print("\n== IDEAS SUMMARY (vs baseline, same-process page-cache-warm; order effects possible) ==")
        for tag,ms in res.items(): print(f"  {tag:15s} {ms:6.0f} ms/tok  {(base-ms)/base*100:+.1f}%")
        json.dump(res,open(os.path.join(D,"ideas_speed.json"),"w")); PREAD.close(); sys.exit(0)
    if os.environ.get("MODE")=="hobbit":  # share-gated 2-bit degradation QUALITY gate (HOBBIT variant of the rejected static cold-tail)
        N=int(os.environ.get("N","150"))
        lf_hob=build("lf_hob",[(MOE_OLD,MOE_NEW_MX.replace("MOE_GATHER_OVL(","MOE_GATHER_HOB(")),(AQ_OLD,AQ_NEW),(INIT_OLD,INIT_NEW),(RING_OLD,RING_NEW),(B_OLD,B_NEW),(OG_OLD,OG_NEW)]+A_EDITS+B_EDITS)
        print(f"HOBBIT SIM: thr={HOB_THR} cap={CAP} N={N} swap={BASE0:.0f}MB",flush=True)
        ref_toks,ref_per,_,ref_nll,ref_lgs,first=run_free(lf_b,N,"ref")
        DEG.clear(); HOB.update(deg=0,rescue=0,full=0)
        fh,nllh,ph,memh=run_forced(lf_hob,ref_toks,first,len(ref_toks),"hob",ref_lgs)
        n=len(ref_toks); pplR=math.exp(ref_nll/n); pplH=math.exp(nllh/n)
        tot=HOB["deg"]+HOB["full"]+HOB["rescue"]
        save=(HOB["deg"]*0.5-HOB["rescue"]*1.0)/max(1,tot)
        mech=sum(1 for f in fh if f['rank']>3)
        print("\n"+"="*70)
        print(f"HOBBIT GATE (thr={HOB_THR}): flips={len(fh)}/{n} mech-LB={mech} ppl-ratio={pplH/pplR:.4f} (<=1.02)")
        print(f"loads: degraded-2bit {HOB['deg']} full {HOB['full']} rescue-reloads {HOB['rescue']} "
              f"-> modeled miss-byte saving {save:.1%} (deg x0.5 minus rescue extra)")
        for f in fh: print(f"  flip k={f['k']} ref={f['btxt']!r} hob={f['mtxt']!r} rank={f['rank']} mB={f['margin_b']:.4f}")
        verdict='PASS' if (mech==0 and pplH/pplR<=1.02) else 'FAIL'
        print(f"VERDICT: {verdict} (mech-LB==0 and ppl<=1.02; text-review any flips)")
        json.dump(dict(thr=HOB_THR,flips=fh,ppl_ratio=pplH/pplR,hob=HOB,save=save,verdict=verdict),
                  open(os.path.join(D,"hobbit_gate.json"),"w"))
        print(f"final swap={swap_mb():.0f}MB (base {BASE0:.0f})"); PREAD.close(); sys.exit(0)
    if os.environ.get("MODE")=="base":  # STEP 13: clean cold-base measurement of the DEPLOYED engine (lf_b)
        N=int(os.environ.get("N","100"))
        print(f"STEP13 COLD BASE: deployed lf_b, cap={CAP}, N={N}, swap={BASE0:.0f}MB free={free_mb():.0f}MB FAST={os.environ.get('FAST','0')} REAP={os.environ.get('REAP','0')}",flush=True)
        lfm=lf_b
        if float(os.environ.get("REAP","0"))>0:  # STEP-17 routing prune: bottom-X% by saliency masked at selection
            frac=float(os.environ["REAP"]); sal=np.load(os.path.join(D,"reap_saliency.npz"))["sal"].reshape(43,256)
            pm=np.zeros((43,256),np.float32)
            for L in range(3,43): pm[L,np.argsort(sal[L])[:int(256*frac)]]=-np.inf
            G["PMASK"]=pm
            lfm=build("lf_reap_base",BASE_EDITS+A_EDITS+B_EDITS+[('    bias = None if hashl else W(f"blk.{L}.exp_probs_b.bias", False)','    bias = None if hashl else W(f"blk.{L}.exp_probs_b.bias", False) + PMASK[L]')])
        if os.environ.get("FAST")=="1":  # adopted ideas winners: no per-token clear_cache, no per-layer driver eval
            out,per=run_speed(lfm,N,"base13",clear=False,layer_eval=False)
            memtr=[(mx.get_active_memory()/1e9,swap_mb())]
        else:
            out,per,memtr,nll,lgs,_=run_free(lf_b,N,"base13")
        ms=np.mean(per[3:])*1000
        acts=[a for a,_ in memtr]
        print(f"\nCOLD BASE @cap={CAP}: {ms:.0f} ms/tok = {1000/ms:.3f} tok/s over {len(out)} steps "
              f"(p50 {np.median(per[3:])*1000:.0f}ms)  mx-active final {acts[-1]:.2f}GB  "
              f"cache hits {CACHE.hits} misses {CACHE.misses} (hit {CACHE.hits/max(1,CACHE.hits+CACHE.misses):.3f})")
        json.dump(dict(cap=CAP,ms=ms,toks=1000/ms,n=len(out),hit=CACHE.hits/max(1,CACHE.hits+CACHE.misses)),
                  open(os.path.join(D,"step13_base.json"),"w"))
        print(f"final swap={swap_mb():.0f}MB (base {BASE0:.0f})"); PREAD.close(); sys.exit(0)
    N=150
    print(f"STEP12 start swap={BASE0:.0f}MB free={free_mb():.0f}MB cap={CAP} N={N}",flush=True)
    print("\n--- PHASE 1: lf_ref free-run (reference stream) ---",flush=True)
    ref_toks,ref_per,ref_mem,ref_nll,ref_lgs,stream_first=run_free(lf_ref,N,"ref")
    n=len(ref_toks)
    print(f"ref: {n} steps, steady {np.mean(ref_per[3:])*1000:.0f}ms/tok",flush=True)
    print("\n--- PHASE 2: lf_a teacher-forced (BIT-IDENTITY gate: flips MUST be 0) ---",flush=True)
    fa,nlla,pa,mema=run_forced(lf_a,ref_toks,stream_first,n,"12a",ref_lgs)
    print("\n--- PHASE 3: lf_b teacher-forced (C2 gate) ---",flush=True)
    fb,nllb,pb,memb=run_forced(lf_b,ref_toks,stream_first,n,"12b",ref_lgs)
    pplR=math.exp(ref_nll/n); ppla=math.exp(nlla/n); pplb=math.exp(nllb/n)
    sa,da=memstats(mema); sb,db=memstats(memb)
    msR=np.mean(ref_per[3:])*1000; msA=np.mean(pa[3:])*1000; msB=np.mean(pb[3:])*1000
    print("\n"+"="*70)
    print(f"REF : {msR:.0f} ms/tok  self-ppl {pplR:.4f}")
    print(f"12a : {msA:.0f} ms/tok ({(msR-msA)/msR*+100:+.1f}%)  flips={len(fa)} (MUST be 0)  ppl-ratio {ppla/pplR:.4f}  slope {sa:+.1f}MB/step drift {da:+.2f}GB")
    print(f"12b : {msB:.0f} ms/tok ({(msR-msB)/msR*+100:+.1f}%)  flips={len(fb)}  ppl-ratio {pplb/pplR:.4f}  mech-LB={sum(1 for f in fb if f['rank']>3)}  slope {sb:+.1f}MB/step drift {db:+.2f}GB")
    ga = len(fa)==0 and abs(sa)<5 and da<0.5 and msA<=msR*1.01
    gb = pplb/pplR<=1.02 and sum(1 for f in fb if f['rank']>3)==0 and abs(sb)<5 and db<0.5 and msB<msR
    print(f"GATE 12a: {'PASS' if ga else 'FAIL'}   GATE 12b (pending text review of flips): {'PASS-mech' if gb else 'FAIL'}")
    for f in fb: print(f"  12b flip k={f['k']} ref={f['btxt']!r} var={f['mtxt']!r} rank={f['rank']} mB={f['margin_b']:.4f} mM={f['margin_m']:.4f}")
    json.dump(dict(n=n,msR=msR,msA=msA,msB=msB,flipsA=fa,flipsB=fb,pplR=pplR,ppla=ppla,pplb=pplb,
        slopeA=sa,driftA=da,slopeB=sb,driftB=db,gateA=bool(ga),gateB_mech=bool(gb)),open(OUTJ,"w"))
    print(f"final swap={swap_mb():.0f}MB (base {BASE0:.0f})"); PREAD.close()

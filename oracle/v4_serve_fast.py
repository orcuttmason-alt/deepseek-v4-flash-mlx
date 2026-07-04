"""V4-Flash persistent daemon on the STEP-20 FAST engine. DEFAULT CAP=1664 (STEP-22 sweep: +4.7%
over 1280; 30.3GB, comfortable for the serial engine).
Boot prefill = oracle (state structures shared); ingest + generation = fast serial steps.
Usage:  CAP=1280 ./v4-env/bin/python oracle/v4_serve_fast.py
REPL: one JSON per line: {"prompt": "...", "max_tokens": 200} | {"cmd": "reset"} | {"cmd": "stats"}"""
import sys, os, json, time
import numpy as np, mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CAP", "1664")
import step12_glue as E
import v4_fast as F
import v4_prefill as VP
LF_PREFILL = [None]   # batched-MoE prefill engine (2x faster prompt ingest); built lazily
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlx-ckpt", "tokenizer.json"))

HOTF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dumps", "hot_gids.json")
def seed_cache():
    """warm-boot: batch-load the previously-hot experts so turn 1 runs warm (cold first turn was ~1.5 tok/s)."""
    if not os.path.exists(HOTF): return 0
    gids = json.load(open(HOTF))[-E.CACHE.cap:]
    t0 = time.perf_counter(); n = 0
    for c in range(0, len(gids), 64):
        E.PENDING.clear(); E.CACHE.access_batch(gids[c:c + 64])
        if E.PENDING:
            E.PREAD.load_batch(list(E.PENDING), E.CACHE.buffers); n += len(E.PENDING)
            mx.eval([E.CACHE.buffers[k] for k in range(6)])
    print(f"[seed] {n} experts warmed in {time.perf_counter()-t0:.1f}s", flush=True)
    return n
def save_hot():
    try: json.dump([int(g) for g in E.CACHE.lru.values()], open(HOTF, "w"))
    except Exception: pass

# DeepSeek-V4 chat framing (from mlx-ckpt/chat_template.jinja, non-thinking mode)
BOS, USER, ASST, THINK, EOS = 0, 128803, 128804, 128822, 1

class Conv:
    """Chat-framed multi-turn state. Turns use the checkpoint's own <|User|>/<|Assistant|></think>
    tokens so the raw daemon actually follows instructions; generation stops on EOS; compaction drops
    oldest WHOLE turns (never mid-sentence), keeping the BOS/system prefix."""
    def __init__(s, fast):
        s.fast = fast; s.reset()
    def reset(s):
        s.states = None; s.fed = []; s.P = -1; s.NX = None; s.turn_starts = []; s.prefix_len = 0
        E.reset_cache()
    def _prefill(s, ids):
        E.dense8_switch()
        if LF_PREFILL[0] is None:
            LF_PREFILL[0] = VP.build_prefill() if os.environ.get("PF_BATCH","1")!="0" else E.lf_b  # batched prompt digestion (~2x); PF_BATCH=0 to disable
        G = E.G; G["NP"] = len(ids); G["tokens"] = np.array(ids, dtype=np.int64)
        s.states = [E.LS() for _ in range(43)]
        x4 = mx.array(np.repeat(E.embed[np.array(ids)][:, None, :], 4, axis=1)[None])
        for L in range(43):
            x4 = LF_PREFILL[0](x4, L, s.states[L], list(range(len(ids)))); mx.eval(x4)
        s.fed = list(ids); s.P = len(ids) - 1
        s.NX = int(np.argmax(E.head_logits(x4)[len(ids) - 1]))
        E.Wmx_single.cache_clear(); E.Wf16.cache_clear(); mx.clear_cache(); seed_cache()
    def _ingest(s, ids):
        for t in ids:
            s.fed.append(int(t)); x4 = s.fast.step(s.states, int(t), s.P + 1, np.array(s.fed, dtype=np.int64))
            s.NX = int(np.argmax(s.fast.head(x4))); s.P += 1
    def _generate(s, maxn, quiet=False):
        out = []
        for n in range(maxn):
            tok = s.NX
            if tok == EOS: break
            s.fed.append(tok); out.append(tok)
            if not quiet: print(TOK.decode([tok], skip_special_tokens=True), end="", flush=True)
            x4 = s.fast.step(s.states, tok, s.P + 1, np.array(s.fed, dtype=np.int64))
            s.NX = int(np.argmax(s.fast.head(x4))); s.P += 1
            if n % 64 == 0: mx.clear_cache()
        return out
    def turn(s, user_ids, maxn, system=None):
        if system is None and s.states is None:
            system = "You are a helpful assistant. Always respond in English, regardless of the language of the question."  # DS_ENGLISH: V4-Flash is Chinese-trained
        frame = [USER] + list(user_ids) + [ASST, THINK]
        if s.states is None:
            prefix = [BOS] + (TOK.encode(system).ids if system else [])
            s.prefix_len = len(prefix)
            s._prefill(prefix + frame)
        else:
            s.turn_starts.append(len(s.fed))   # this [USER] begins a new turn
            s._ingest(frame)
        if not s.turn_starts or s.turn_starts[-1] < s.prefix_len:
            s.turn_starts.append(s.prefix_len)
        out = s._generate(maxn)
        s.fed.append(EOS)                       # close the assistant turn (history marker)
        return out
    def compact(s, max_ctx):
        """drop oldest whole turns until the retained context is under half the window; keep prefix."""
        target = max_ctx // 2
        starts = [i for i in s.turn_starts if i >= s.prefix_len]
        if len(starts) <= 1: return False
        k = 0
        while k < len(starts) - 1 and (len(s.fed) - starts[k]) + s.prefix_len > target:
            k += 1
        keep_ids = s.fed[:s.prefix_len] + s.fed[starts[k]:]
        s.reset(); s._prefill(keep_ids)
        return True

def main():
    print("[serve-fast] loading fast engine...", flush=True)
    fast = F.V4Fast()
    conv = Conv(fast)
    print(f"[serve-fast] ready. cap={E.CAP} mx-active {mx.get_active_memory()/1e9:.1f}GB. JSON per line.", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: req = json.loads(line)
        except json.JSONDecodeError: print("[err] bad json", flush=True); continue
        if req.get("cmd") == "reset": conv.reset(); print("[reset]", flush=True); continue
        if req.get("cmd") == "stats":
            print(json.dumps(dict(cache=E.CACHE.stats(), pos=conv.P, mx_gb=mx.get_active_memory()/1e9)), flush=True); continue
        ids = TOK.encode(req["prompt"]).ids
        MAX_CTX = int(os.environ.get("MAX_CTX", "32768"))  # rope tables allow 65536; comp-row uploads slow decode past ~32k
        maxtok = int(req.get("max_tokens", 200))
        need = conv.P + 1 + len(ids) + 4 + maxtok
        if need > MAX_CTX and conv.states is not None:
            print(f"[ctx] {need} > {MAX_CTX} — compacting (dropping oldest whole turns)...", flush=True)
            if conv.compact(MAX_CTX): print(f"[ctx] compacted to {conv.P + 1} tokens", flush=True)
            else: print("[ctx] single huge turn — cannot compact; send {\"cmd\":\"reset\"}", flush=True)
        elif need > MAX_CTX * 3 // 4:
            print(f"[ctx] note: {need}/{MAX_CTX} — turn-drop compaction triggers at the cap", flush=True)
        t0 = time.perf_counter()
        first = conv.states is None
        out = conv.turn(ids, maxtok)
        dt = time.perf_counter() - t0; save_hot()
        print(f"\n[done] {'boot+' if first else ''}gen {len(out)} in {dt:.1f}s = {len(out)/max(dt,1e-9):.2f} tok/s", flush=True)

if __name__ == "__main__":
    main()

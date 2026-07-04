"""OpenAI-compatible HTTP shim for the DeepSeek V4-Flash FAST engine, so lazyllama (and any OpenAI
client) can talk to it. The daemon (v4_serve_fast.py) is stdin-only; this wraps the same engine + Conv
in an HTTP server on :18091.

Endpoints:
  GET  /v1/models              -> {"data":[{"id":"deepseek-v4-flash",...}]}
  POST /v1/chat/completions    -> SSE stream (OpenAI delta format), or one JSON if stream=false

Design notes:
- ONE engine, ONE request at a time (a lock serializes; DeepSeek is ~30GB, no batching).
- APPEND-ONLY KV REUSE: lazyllama resends the full history each turn. If the new request extends the
  committed conversation, only the new user turn is serial-ingested (fast, bounded to the new turn);
  otherwise the session is rebuilt (serial replay — slow but rare: edits / new chats). This is the
  memory-safe path (serial ingest; batched block-ingest is the swap-unsafe path we closed on 48GB).
- Greedy decode (argmax). lazyllama's temperature/top_p are advisory and ignored — deterministic output
  keeps the append-only reuse consistent (the echoed assistant text always matches what we generated).
- Chat-framed via the checkpoint's own <|User|>/<|Assistant|></think> tokens; stops on EOS.
- No tool-calling (the fast engine has no tool parser) — the lazyllama entry sets tools:false.
Launch via start-deepseek-lazyllama.sh (frees the coder first). Rollback: don't select it in lazyllama.
"""
import sys, os, json, time, threading, subprocess, re
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("DS_PORT", "18091"))
MODEL_ID = "deepseek-v4-flash"

def _swap_mb():
    o = subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode()
    m = re.search(r"used\s*=\s*([\d.]+)M", o); return float(m.group(1)) if m else 0

# free the coder + wait for swap to drain BEFORE importing the engine (step12_glue asserts swap<=3200 at import)
subprocess.run(["pkill", "-f", "pld_server.py"], capture_output=True)
subprocess.run(["pkill", "-f", "mlx_lm.server"], capture_output=True)
print("[ds-http] freed coder; waiting for swap to drain...", flush=True)
for _ in range(120):
    if _swap_mb() <= 3000: break
    time.sleep(2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mlx.core as mx
import v4_serve_fast as VS   # reuses Conv + engine + TOK + framing constants
F, E, TOK = VS.F, VS.E, VS.TOK
BOS, USER, ASST, THINK, EOS = VS.BOS, VS.USER, VS.ASST, VS.THINK, VS.EOS

LOCK = threading.Lock()
FAST = None
CONV = None
COMMITTED = []      # [(role, text)] currently resident in KV (excludes system)
SYSTEM = None

def boot_engine():
    global FAST, CONV
    print("[ds-http] loading fast engine...", flush=True)
    FAST = F.V4Fast()
    CONV = VS.Conv(FAST)
    print(f"[ds-http] ready on :{PORT}, mx-active {mx.get_active_memory()/1e9:.1f}GB", flush=True)

def _frame_user(text): return [USER] + TOK.encode(text).ids + [ASST, THINK]

PENDING_PF = [None]   # deferred prefill tokens: prefilled together with the last user turn (>=4 tok, avoids the sub-ratio compressor/ring edge case)

def _framed(prior):
    ids = []
    for role, text in prior:
        ids += _frame_user(text) if role == "user" else (TOK.encode(text).ids + [EOS])
    return ids

def _rebuild(system_text, prior):
    """reset; DEFER the prefill (prefix + prior) so _stream_gen prefills it together with the last user
    turn in one call — never a lone [BOS], which under-fills the ratio-4 compressor + ring."""
    global SYSTEM, COMMITTED
    CONV.reset()
    prefix = [BOS] + (TOK.encode(system_text).ids if system_text else [])
    PENDING_PF[0] = prefix + _framed(prior)
    SYSTEM = system_text; COMMITTED = list(prior)

def _ensure_context(system_text, prior):
    """append-only reuse: extend committed if the request is a clean continuation, else rebuild."""
    global COMMITTED
    reuse = (CONV.states is not None and SYSTEM == system_text
             and len(COMMITTED) <= len(prior)
             and all(COMMITTED[i] == prior[i] for i in range(len(COMMITTED))))
    if not reuse:
        _rebuild(system_text, prior); return "rebuild"
    PENDING_PF[0] = None
    for role, text in prior[len(COMMITTED):]:   # newly-completed turns since last request
        if role == "user": CONV._ingest(_frame_user(text))
        else: CONV._ingest(TOK.encode(text).ids + [EOS])
    COMMITTED = list(prior); return "reuse"

def _stream_gen(last_user, maxn, max_ctx):
    """ingest the final user turn, then yield generated token strings; commit on completion."""
    global COMMITTED
    if PENDING_PF[0] is not None:               # rebuild: prefill prefix+prior+this-user in ONE call (>=4 tok)
        full = PENDING_PF[0] + _frame_user(last_user); PENDING_PF[0] = None
        CONV._prefill(full)
        CONV.turn_starts = [len(full) - len(_frame_user(last_user))]
    else:                                       # reuse: ingest just the new user turn
        need = CONV.P + 1 + len(_frame_user(last_user)) + maxn
        if need > max_ctx and len(COMMITTED) >= 2:
            CONV.compact(max_ctx); COMMITTED[:] = COMMITTED[-2:]
        CONV.turn_starts.append(len(CONV.fed))
        CONV._ingest(_frame_user(last_user))
    out = []
    for n in range(maxn):
        tok = CONV.NX
        if tok == EOS: break
        CONV.fed.append(tok); out.append(tok)
        piece = TOK.decode([tok], skip_special_tokens=True)
        if piece: yield piece
        x4 = FAST.step(CONV.states, tok, CONV.P + 1, np.array(CONV.fed, dtype=np.int64))
        CONV.NX = int(np.argmax(FAST.head(x4))); CONV.P += 1
        if n % 64 == 0: mx.clear_cache()
    CONV.fed.append(EOS)
    COMMITTED = COMMITTED + [("user", last_user), ("assistant", TOK.decode(out, skip_special_tokens=True))]

DEFAULT_SYSTEM = "You are a helpful assistant. Always respond in English, regardless of the language of the question."
def _split(messages):
    system_text = next((m.get("content") for m in messages if m.get("role") == "system"), None)
    if not system_text: system_text = DEFAULT_SYSTEM   # English-only (V4-Flash is Chinese-trained; steer to English)
    turns = [(m["role"], m.get("content") or "") for m in messages if m.get("role") in ("user", "assistant")]
    return system_text, turns

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _hdr(self, ctype, chunked=False):
        self.send_response(200); self.send_header("Content-Type", ctype)
        if chunked: self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def _sse(self, s):
        chunk = f"data: {s}\n\n".encode()
        self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n"); self.wfile.flush()
    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_response(404); self.end_headers(); return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        messages = req.get("messages", [])
        maxtok = int(req.get("max_tokens", 512))
        stream = req.get("stream", True)
        max_ctx = int(os.environ.get("MAX_CTX", "32768"))
        system_text, turns = _split(messages)
        if not turns or turns[-1][0] != "user":
            self.send_response(400); self.end_headers(); return
        prior, last_user = turns[:-1], turns[-1][1]
        with LOCK:
            t0 = time.time()
            mode = _ensure_context(system_text, prior)
            created = int(time.time()); rid = f"chatcmpl-ds-{created}"
            if stream:
                self._hdr("text/event-stream", chunked=True)
                self._sse(json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_ID, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}))
                for piece in _stream_gen(last_user, maxtok, max_ctx):
                    self._sse(json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                        "model": MODEL_ID, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}))
                self._sse(json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                    "model": MODEL_ID, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
                self._sse("[DONE]"); self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
                print(f"[ds-http] {mode} turn done in {time.time()-t0:.1f}s", flush=True)
            else:
                text = "".join(_stream_gen(last_user, maxtok, max_ctx))
                body = json.dumps({"id": rid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

def main():
    boot_engine()
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[ds-http] serving OpenAI API on http://127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()

if __name__ == "__main__":
    main()

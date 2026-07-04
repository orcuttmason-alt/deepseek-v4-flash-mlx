"""One-time: extract the DENSE (non-expert) weights from the 153GB GGUF, dequantize to F16, and save a
~15GB companion so the engine no longer needs the GGUF (experts already come from the mlx-ckpt mxfp4).
After this + the loader repoint (decode_engine reads the companion when present) + revalidation, the
153GB GGUF can be deleted -> single-source, ~140GB freed, 8-bit dense quality preserved.
Writes: mlx-ckpt/dense_fp16.safetensors + mlx-ckpt/dense_shapes.json (the GGUF ne-shapes W() reshapes to)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mlx.core as mx
import gguf.quants as gq
from gguf import GGUFReader
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "mlx-ckpt")
G = os.path.join(ROOT, "gguf", open(os.path.join(ROOT, "gguf/TARGET.txt")).read().strip())
EXP = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")   # these come from mlx-ckpt (mxfp4) — skip
print(f"reading {os.path.basename(G)} ...", flush=True)
R = GGUFReader(G)
out = {}; meta = {}; t0 = time.time(); n = 0; tot = 0
for t in R.tensors:
    if any(e in t.name for e in EXP): continue
    try:
        arr = gq.dequantize(np.array(t.data), t.tensor_type).astype(np.float16)   # float weights -> F16
        raw = False
    except NotImplementedError:
        arr = np.array(t.data)                     # integer tables (tid2eid) -> stored raw, read via .data
        raw = True
    out[t.name] = mx.array(arr)
    meta[t.name] = {"shape": [int(x) for x in t.shape], "raw": raw, "qtype": int(t.tensor_type)}
    n += 1; tot += arr.nbytes
    if n % 200 == 0: print(f"  {n} tensors, {tot/1e9:.1f}GB, {time.time()-t0:.0f}s", flush=True)
print(f"dequantized {n} dense tensors ({tot/1e9:.1f}GB). saving...", flush=True)
mx.eval(list(out.values()))
mx.save_safetensors(os.path.join(CK, "dense_fp16.safetensors"), out)
json.dump(meta, open(os.path.join(CK, "dense_shapes.json"), "w"))
print(f"WROTE mlx-ckpt/dense_fp16.safetensors ({tot/1e9:.1f}GB) + dense_shapes.json ({n} tensors) in {time.time()-t0:.0f}s", flush=True)

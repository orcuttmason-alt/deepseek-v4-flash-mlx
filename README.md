# DeepSeek V4-Flash on 48 GB — MLX SSD-offload engine

Run **DeepSeek V4-Flash** (a ~100 B-class sparse MoE, 256 experts, ~13 B active per token) on an
Apple-Silicon Mac with **48 GB** of unified memory — a model that normally "needs 128 GB+".

It works by keeping the model on SSD and **streaming only the active experts** for each token into a
device-side LRU cache in unified memory. Dense/attention weights stay resident (8-bit); the 256 routed
experts (mxfp4) live on disk and are pulled on demand. So RAM holds a working set, not the whole model.

Stock `llama.cpp` / `mlx-lm` can't even load this architecture yet (`unknown model architecture:
'deepseek4'`). This is a from-scratch MLX inference engine built specifically to run it on constrained RAM.

**Performance:** ~4.5–5 tok/s decode on a 48 GB M-series Mac. That's not *fast* in absolute terms —
decode is fundamentally bottlenecked by SSD bandwidth (experts stream every token). The point is that a
model this size runs coherently *at all* on a laptop that can't hold it.

---

## Requirements

- **Apple Silicon Mac, 48 GB+ unified memory** (developed on M5 Pro / 48 GB).
- ~155 GB free disk for the weights.
- Python 3.12, `pip install -r requirements.txt` (MLX, gguf, torch, tokenizers, safetensors).

## Weights setup

Two pieces, both derived from DeepSeek's MIT-licensed model:

1. **Experts + tokenizer** — the public MLX checkpoint:
   ```bash
   hf download mlx-community/DeepSeek-V4-Flash-4bit --local-dir mlx-ckpt
   ```
   (Experts are used as mxfp4 and streamed from here.)

2. **8-bit dense companion** — the dense/attention weights at higher precision than the uniform-4-bit
   checkpoint (this is what preserves quality). Get any **DeepSeek V4-Flash GGUF that keeps dense at
   F16/Q8** — search HuggingFace for `DeepSeek-V4-Flash GGUF`; the right files have `Q8Attn` / `F16HC`
   in the name (the experts' quant doesn't matter, they're not used from the GGUF). Put the `.gguf` in
   `gguf/`, write its filename into `gguf/TARGET.txt`, then build the ~15 GB companion once:
   ```bash
   python oracle/build_dense_companion.py    # -> mlx-ckpt/dense_fp16.safetensors
   ```
   After that the GGUF can be deleted — the engine reads dense from the companion (bit-identical). If the
   companion is absent it falls back to reading the GGUF directly.

   *Simpler but slightly lower quality:* skip the companion and the engine will use the checkpoint's own
   4-bit dense weights (perplexity ~+2%, no semantic errors in testing).

## Run it

```bash
# stdin JSON REPL:
./scripts/serve.sh
{"prompt": "Write a Python function to reverse a linked list.", "max_tokens": 300}
{"cmd": "reset"}

# or an OpenAI-compatible HTTP server on :18091:
./scripts/serve-http.sh
curl localhost:18091/v1/chat/completions -d '{"model":"deepseek-v4-flash","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

Knobs (env vars): `CAP=1664` (expert-cache size), `MAX_CTX=32768` (context window; rope tables allow
65536), `PF_BATCH=1` (batched prompt digestion), `TOPK=6` (routed experts — `TOPK=5` is ~+10% for a mild
quality trade). DeepSeek V4-Flash is Chinese-trained; the daemon injects an "always respond in English"
system prompt by default.

## How it works

- **Expert offloading** (`offload_cache_v4.py`, `pread_loader.py`): a fixed-size device LRU over mxfp4
  expert records, streamed via parallel `pread` with async load / compute overlap. Only the ~6 experts
  per token per layer that actually route are read.
- **Quantized dense** (`v4_fast.py`): attention/shared-expert/head weights quantized to 8-bit at load
  (quality-gated), kept resident. `quantized_matmul` on the decode path.
- **Batched prefill** (`v4_prefill.py`): the per-position expert gather is batched into one call — ~2×
  faster prompt digestion.
- **Serve layer** (`v4_serve_fast.py`, `v4_http.py`): chat-framed multi-turn with append-only KV reuse,
  turn-drop compaction, and EOS-stopping.

## Honest caveats

- **~5 tok/s, byte-bound.** Decode speed is set by SSD expert streaming, not compute. It won't feel fast.
- **Lossless speculative decoding is built** (`v4_fast_spec.py` in the full project) and proven bit-exact,
  but the verify step's expert union doesn't fit alongside the model in 48 GB — it's a ~64 GB unlock.
- **Prefill/decode are the same ~200 ms/token** because every prompt position also pays for its experts.

## Credits & license

Engine code: MIT (this repo). The model is **DeepSeek V4-Flash** (© DeepSeek, MIT); expert checkpoint by
`mlx-community`. Built on [MLX](https://github.com/ml-explore/mlx). This is research-grade software shared
as-is — expect rough edges.

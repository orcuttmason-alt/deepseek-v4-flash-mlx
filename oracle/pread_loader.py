"""Phase-3 lever-1: pread-based expert loader. Reads each expert's CONTIGUOUS byte range directly
(os.pread over parsed safetensors offsets) instead of safetensors get_slice, and can read a miss-batch's
tensors in PARALLEL (thread pool; os.pread releases the GIL). FAITHFULNESS: reads the identical bytes
(expert e is a contiguous row-major block), only faster. mx.array conversion + buffer scatter stay on the
main thread (MLX single-threaded); only the IO (pread->bytes) is parallelized."""
import os, json, struct
import numpy as np
import mlx.core as mx
from concurrent.futures import ThreadPoolExecutor

_ST_DTYPE = {"U32": np.uint32, "U8": np.uint8, "I32": np.int32, "I8": np.int8}
_MP = ["gate_proj", "up_proj", "down_proj"]

class PreadLoader:
    def __init__(self, ck, mlx_idx, workers=8):
        self.ck = ck; self.mlx_idx = mlx_idx
        self.fds = {}          # shard -> fd
        self.hdr = {}          # shard -> (data_start, header dict)
        self.meta = {}         # tensor name -> (abs_base_offset, expert_bytes, np_dtype, per_expert_shape)
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def _fd(self, shard):
        if shard not in self.fds:
            self.fds[shard] = os.open(os.path.join(self.ck, shard), os.O_RDONLY)
        return self.fds[shard]

    def _shard_hdr(self, shard):
        if shard not in self.hdr:
            with open(os.path.join(self.ck, shard), "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(hlen))
            self.hdr[shard] = (8 + hlen, h)
        return self.hdr[shard]

    def _tmeta(self, name):
        if name not in self.meta:
            shard = self.mlx_idx[name]; data_start, h = self._shard_hdr(shard)
            m = h[name]; o0, o1 = m["data_offsets"]; a = m["shape"][0]
            ebytes = (o1 - o0) // a
            self.meta[name] = (data_start + o0, ebytes, _ST_DTYPE[m["dtype"]], tuple(m["shape"][1:]))
        return self.meta[name]

    def read_tensor(self, name, e):
        """pread expert e's contiguous block of `name` -> numpy array of per-expert shape."""
        base, ebytes, dt, shp = self._tmeta(name)
        fd = self._fd(self.mlx_idx[name]); off = base + e * ebytes; buf = bytearray(ebytes); got = 0
        while got < ebytes:
            n = os.preadv(fd, [memoryview(buf)[got:]], off + got) if hasattr(os, "preadv") else None
            if n is None:
                chunk = os.pread(fd, ebytes - got, off + got); buf[got:got + len(chunk)] = chunk; got += len(chunk)
            else:
                got += n
        return np.frombuffer(buf, dtype=dt).reshape(shp)  # zero-copy view of the pread buffer (bytes(buf) was a full extra memcpy per record)

    def _names(self, L, e):
        out = []
        for p in _MP:
            b = f"model.layers.{L}.ffn.switch_mlp.{p}"
            out.append((b + ".weight", e)); out.append((b + ".scales", e))
        return out

    # single-expert loader signature compatible with TwoTierExpertCache (gid, slot, buffers)
    def load(self, gid, slot, buffers):
        L, e = gid // 256, gid % 256
        for j, (name, ee) in enumerate(self._names(L, e)):
            buffers[j][slot] = mx.array(self.read_tensor(name, ee))

    # parallel batch load: read all tensors of a miss-set concurrently, then scatter on main thread
    def load_batch(self, jobs, buffers):
        """jobs = list of (gid, slot). Parallel-read all their tensors, write into buffers. Caller evals."""
        tasks = []
        for (gid, slot) in jobs:
            L, e = gid // 256, gid % 256
            for j, (name, ee) in enumerate(self._names(L, e)):
                tasks.append((slot, j, name, ee))
        if self.pool is None:
            for (slot, j, name, ee) in tasks:
                buffers[j][slot] = mx.array(self.read_tensor(name, ee))
            return
        def rd(t):
            slot, j, name, ee = t; return (slot, j, self.read_tensor(name, ee))
        for (slot, j, arr) in self.pool.map(rd, tasks):
            buffers[j][slot] = mx.array(arr)     # mx on main thread (MLX single-threaded)

    # async variant: submit the reads and return futures; caller overlaps GPU work, then joins.
    # values/ordering identical to load_batch (same tensors into the same slots; mx on main thread).
    def load_batch_async(self, jobs):
        futs = []
        for (gid, slot) in jobs:
            L, e = gid // 256, gid % 256
            for j, (name, ee) in enumerate(self._names(L, e)):
                if self.pool is None:
                    futs.append((slot, j, None, self.read_tensor(name, ee)))
                else:
                    futs.append((slot, j, self.pool.submit(self.read_tensor, name, ee), None))
        return futs
    def load_batch_join(self, futs, buffers):
        if os.environ.get("LOADSTACK") == "1":
            # one stacked H2D copy + device fancy-index scatter per buffer (vs 18 small copies/batch)
            groups = {}
            for (slot, j, f, arr) in futs:
                groups.setdefault(j, []).append((slot, f.result() if f is not None else arr))
            for j, items in groups.items():
                slots = mx.array(np.array([sl for sl, _ in items], dtype=np.uint32))
                buffers[j][slots] = mx.array(np.stack([a for _, a in items]))
            return
        for (slot, j, f, arr) in futs:
            buffers[j][slot] = mx.array(f.result() if f is not None else arr)

    def close(self):
        if self.pool: self.pool.shutdown(wait=False)
        for fd in self.fds.values(): os.close(fd)

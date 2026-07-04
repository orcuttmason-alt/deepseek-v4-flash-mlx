"""CP-2 — V4-Flash two-tier expert cache (port of Qwen GlobalExpertCacheV3 + pinned tier).

Two tiers:
  - PINNED (never evicted, excluded from LRU): non-expert weights + all 43 shared experts.
    These are normal resident model tensors; the cache only TRACKS them for residency-correctness
    accounting. Shared experts fire every token -> permanent residents (the free win).
  - LRU (routed experts only): cap slots of V4 mxfp4 buffers, slot-indexed for gather_qmm.

V4 routed expert record (mxfp4, group_size 32, bits 4 -> scales only, NO biases):
  gate_w (2048,512)u32 + gate_s (2048,128)u8 + up_w (2048,512)u32 + up_s (2048,128)u8
  + down_w (4096,256)u32 + down_s (4096,64)u8  = 13,369,344 B = 13.369 MB/expert.

gid scheme: routed global id = layer * 256 + expert_id (0..255). Shared experts are NOT given
routed gids and never enter the LRU (structural guarantee they are never evicted).

MISS_CHUNK bounds the host transient: 96 * 13.369 MB = 1.28 GB <= 1.3 GB for any caller (incl. a
warm-seed batching across all layers) -> the swap-bomb class is unrepresentable.

CP-2 uses skip_load=True (buffers allocated, residency logic exercised) — byte-correctness of the
loaded experts is CP-6's gate, not CP-2's.
"""
import os
from collections import OrderedDict
import numpy as np
import mlx.core as mx

N_LAYERS = 43
N_ROUTED = 256
# per-slot routed mxfp4 buffer field layout: (name, shape, mlx dtype, bytes)
ROUTED_FIELDS = [
    ("gate_w", (2048, 512), mx.uint32),
    ("gate_s", (2048, 128), mx.uint8),
    ("up_w",   (2048, 512), mx.uint32),
    ("up_s",   (2048, 128), mx.uint8),
    ("down_w", (4096, 256), mx.uint32),
    ("down_s", (4096, 64),  mx.uint8),
]
def _itembytes(dt): return 4 if dt == mx.uint32 else 1
RECORD_BYTES = sum(int(np.prod(shp)) * _itembytes(dt) for _, shp, dt in ROUTED_FIELDS)  # 13,369,344
MISS_CHUNK = 96  # 96 * 13.369MB = 1.283 GB <= 1.3 GB host-transient bound


class TwoTierExpertCache:
    def __init__(self, cap, pinned_keys, skip_load=True, loader=None, alloc_buffers=True):
        """cap: LRU capacity in routed experts.
        pinned_keys: iterable of identifiers for the permanently-resident tensors (non-expert
           weights + the 43 shared experts) — TRACKED here for residency accounting; their storage
           is the normal resident model, never the LRU.
        loader: optional fn(gid, slot, buffers) to fill a slot from the store (CP-6). skip_load -> noop.
        alloc_buffers: False -> accounting-only (residency/hit tracking) with no ~cap*record GPU
           allocation (the fp32-numpy base engine can't hold cap>~96 fp32; the quantized cap=1280
           residency + gather_qmm buffers are CP-6).
        """
        self.cap = cap
        self.skip_load = skip_load
        self.loader = loader
        self.rs = RECORD_BYTES
        # PINNED tier (set; never evicted, never in LRU)
        self.pinned = set(pinned_keys)
        # LRU tier buffers (routed only)
        if alloc_buffers:
            self.buffers = [mx.zeros((cap,) + shp, dtype=dt) for _, shp, dt in ROUTED_FIELDS]
            mx.eval(self.buffers)
        else:
            self.buffers = None
        self.slot_of = {}            # routed gid -> slot
        self.lru = OrderedDict()     # slot -> gid (LRU order)
        self.free = list(range(cap))
        self.hits = self.misses = self.loads = self.evictions = 0
        self.evicted_routed_only = True  # tripwire: set False if a non-routed key is ever evicted

    @staticmethod
    def routed_gid(layer, expert):
        assert 0 <= expert < N_ROUTED and 0 <= layer < N_LAYERS
        return layer * N_ROUTED + expert

    @staticmethod
    def is_routed_gid(gid):
        return 0 <= gid < N_LAYERS * N_ROUTED

    def _assert_routed(self, gid):
        # structural guard: only routed experts may enter the LRU; a pinned/shared key here is a bug.
        if not self.is_routed_gid(gid) or gid in self.pinned:
            raise AssertionError(f"non-routed/pinned key {gid} entered the LRU path")

    def access_batch(self, gids):
        """Resolve a layer-call's routed experts: assign slots, read misses in <=MISS_CHUNK
        sub-batches (host transient bounded), return {gid: slot}."""
        slot_of = {}
        misses = []
        for g in gids:
            self._assert_routed(g)
            s = self.slot_of.get(g)
            if s is not None:
                self.hits += 1
                self.lru.move_to_end(s)
            else:
                self.misses += 1
                if self.free:
                    s = self.free.pop()
                else:
                    s, old = self.lru.popitem(last=False)
                    del self.slot_of[old]
                    self.evictions += 1
                    if not self.is_routed_gid(old) or old in self.pinned:
                        self.evicted_routed_only = False  # must never happen
                self.slot_of[g] = s
                self.lru[s] = g
                misses.append((g, s))
            slot_of[g] = s
        self.loads += len(misses)
        if misses and not self.skip_load and self.loader is not None:
            for c in range(0, len(misses), MISS_CHUNK):
                for g, s in misses[c:c + MISS_CHUNK]:
                    self.loader(g, s, self.buffers)
                mx.eval(self.buffers)
        return slot_of

    def host_transient_bound_bytes(self):
        return MISS_CHUNK * self.rs

    def buffer_bytes(self):
        return self.cap * self.rs

    def stats(self):
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / n if n else float("nan"),
                "loads": self.loads, "evictions": self.evictions,
                "resident_routed": len(self.slot_of), "cap": self.cap,
                "pinned": len(self.pinned), "evicted_routed_only": self.evicted_routed_only}

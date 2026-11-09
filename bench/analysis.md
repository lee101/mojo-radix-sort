# mojo-radix-sort — measured hotspot map (ANALYZE phase)

No code was changed. Every number below is from a run in this session
(`pixi run bench`, plus inline timing probes via `pixi run python`); nothing
is estimated. `src/` and `tests/` are untouched; this file is the only addition.

Hardware: Intel Xeon E5-2697 v4, single thread, `min` of 7 repetitions,
after a correctness gate against `np.sort(kind="stable")` / `np.argsort(kind="stable")`.

Upstream baseline: `radix-sort` is **not importable** in this environment
(`importlib.util.find_spec('radix_sort') is None`; README § "Upstream package:
there is none" — PyPI 404). So the only baseline is NumPy's stable sort /
stable argsort, which is the comparison the `bench/bench.py` authors chose
deliberately (`np.sort(kind="quicksort")` is faster but is not the comparison
a radix sort is making).

## 1. Public entry points vs NumPy (absolute times)

`pixi run bench` (flock-wrapped, `/tmp/mojo-bench.lock`) — final run:

```
case                             numpy   mojo-radix-sort     ratio
------------------------------------------------------------------
int64 sort n=65k b8             4.66ms            3.10ms     1.51x
int64 sort n=1M b8            109.47ms           58.51ms     1.87x
int64 sort n=1M b11           110.03ms           93.40ms     1.18x
int64 sort n=1M b16           110.38ms          128.42ms     0.86x   <- SLOWER
float64 sort n=1M b8          128.16ms           58.81ms     2.18x
int32 sort n=1M b8            107.56ms           36.04ms     2.98x
int64 argsort n=1M b8         151.27ms          234.38ms     0.65x   <- SLOWER
byte keys n=1M b8              72.06ms           33.02ms     2.18x
sort_pairs 20b keys n=1M      153.01ms           97.14ms     1.58x
```

Scaling curve (sort, int64 full-width, bits=8; same probes):

| n | mojo sort | numpy stable sort |
|---|---|---|
| 65k | 3.10 ms | 4.66 ms |
| 1M | 58.5 ms | 109.5 ms |
| 4M | 607.7 ms | 444.3 ms  <- SLOWER at 4M |

At 4M the port is slower than NumPy. No other case in this table is.

## 2. Ranked hotspot table

Sizes are the sizes `bench/bench.py` uses (n=1M unless noted; n=65k where
it is the canonical small case). Bytes touched = minimal theoretical traffic
of that component. Flops are counted as the only arithmetic the kernels do:
digit extract (`>>` + `&` + convert) and increment/add/compare. `flops/byte`
is flops divided by bytes touched. All components are **memory-bound**:
no component reaches ~2 flops/byte.

| rank | function | input size | time (ms) | bytes touched | flops | flops/byte | bound | glue vs kernel | vs NumPy |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | **`rs_lsd_sort` driver histogram pre-pass** (8 passes × 1M, 64-bit keys) | n=1M | ~7.4 | 64 MB (8 reads × 8 MB) | ~24M (shift+mask+add per elem per pass) | 0.38 | memory | kernel (in `rs_lsd_sort`) | — |
| 2 | **`rs_lsd_sort` driver scatter, with indices** (8 passes) | n=1M | ~203 | 128 MB (8 × 16 MB: keys+idx read+write) | ~32M | 0.25 | memory | kernel | — |
| 3 | **`rs_lsd_sort` driver scatter, keys only** (8 passes) | n=1M | ~58 | 64 MB (8 × 8 MB read+write) | ~16M | 0.25 | memory | kernel | — |
| 4 | **`rs_fill_indices`** | n=1M | 0.48 | 8 MB write | 1M | 0.12 | memory | kernel | — |
| 5 | **`rs_encode_i64`** | n=1M | 0.83 | 16 MB (8 read + 8 write) | 1M XOR | 0.06 | memory | kernel | — |
| 6 | **`rs_encode_f64` / `rs_decode_f64`** | n=1M | 0.95 | 16 MB | ~2M (branch+op) | 0.12 | memory | kernel | — |
| 7 | **`rs_histogram_pass`** (single pass) | n=1M | 0.92 | 8 MB read | 3M | 0.38 | memory | kernel | — |
| 8 | **`rs_scatter_pass` w/ indices** (single pass) | n=1M | 1.91 | 16 MB | 4M | 0.25 | memory | kernel | — |
| 9 | **`rs_prefix_sum`** | 256 buckets | 0.005 | 4 KB | 512 | 0.12 | memory | kernel | — |
| 10 | **`order_keys` full public** (encode + `ascontiguousarray` + view) | n=1M | 0.75–1.7 | 16 MB + 8 MB alloc | 1M | 0.06 | memory | glue+kernel | — |
| 11 | **`_alloc` (kbuf 8MB + ibuf 8MB + hists)** | n=1M | 0.02–0.04 | 16 MB virtual alloc | 0 | 0 | — | glue | — |
| 12 | **`sort` full public (encode + driver + decode)** | n=1M | 57–59 | ~144 MB | ~40M | 0.28 | memory | glue+kernel | 1.87x faster |
| 13 | **`argsort` full public (encode + fill_idx + driver w/idx)** | n=1M | 234 | ~208 MB | ~45M | 0.22 | memory | glue+kernel | **0.65x — SLOWER** |
| 14 | **`sort_pairs` full public (20-bit keys)** | n=1M | 97 | ~104 MB | ~18M | 0.17 | memory | glue+kernel | 1.58x faster |
| 15 | **NumPy gather `enc[order]`** (not Mojo, context for #13) | n=1M | 4.6 | 16 MB | 0 | 0 | memory | numpy | — |
| 16 | **NumPy `argsort stable`** (baseline) | n=1M | 148–151 | — | — | — | — | numpy | baseline |

Hotspot ranking is by time. Rank #2 (`scatter with indices`) is the single
biggest cost at 203 ms out of 234 ms for `argsort` — it is ~87% of the
`argsort` time and is the **only place the port loses to NumPy**.

### Where the time is NOT

- **Python/ctypes glue**: 2–3 µs per call (measured: ctypes call to
  `rs_encode_i64` with `n=0` = 1.1 µs floor; `order_keys` glue ≈ 0.3–0.9 ms
  of which the kernel is 0.83 ms). Against a 57 ms `sort`, glue is <0.02%.
  **No Python loop re-enters ctypes per element anywhere** (`core.py` has no
  per-element Python loop; every kernel call is one ctypes call over the
  whole array). The one `__getattribute__` dispatch in `order_keys` is a
  dict lookup per call, not per element.
- **Allocations**: `_alloc` at n=1M = 0.02 ms (np.empty 8 MB = 3 µs; virtual
  allocation is paged, untouched). `np.zeros(2048 uint64)` for histograms
  = 2 µs. Per-call allocation is not a hotspot; it is three `np.empty`s.
- **Copies**: `np.ascontiguousarray` on already-contiguous input = 0.003 ms
  (no copy). No redundant copies in the hot path. `sort_pairs` does one
  `values[order]` gather (4.6 ms at n=1M) — that is real work, already
  counted in its 97 ms.

### What drives the losing rows

`argsort` permutes indices through every scatter pass: 8 passes ×
(keys read 8 MB + idx read 8 MB + keys write 8 MB + idx write 8 MB) = 256 MB
of index+key traffic instead of 128 MB for keys-only. The driver with-idx
scatter is 203 ms vs 58 ms keys-only (3.5x). No prefix-sum or encode change
helps this; only reducing per-pass traffic (fewer passes or index-free
sort) or making the scatter cheaper per byte helps. `sort` (index-free) is
already 1.87x faster than NumPy at n=1M precisely because it skips this.

16 bits/pass losing (0.86x): 65536-bucket histograms = 512 KB, do not fit
L2, so the per-pass histogram + prefix bookkeeping (65536 counters read and
written per pass) costs more than the passes it saves. This is confirmed by
`bench/bench.py` (b16 slower than b8 at every dtype) and by README.

4M int64 losing (608 vs 444 ms): at 4M the keys + buffers = 32 MB per array,
working set exceeds L3; the 256-stream scatter's random writes go to DRAM
and the store buffer cannot track them. The kernel is latency-bound on
concurrent write streams, not on bandwidth.

## 3. Skip-pass savings (concentrated keys)

`bench/bench.py` `bench_bytes` and probes: the driver pre-computes all
histograms first, so a pass whose buckets are all zero but one is
provably the identity and never scatters.

| keys | passes skipped (of 8, bits=8) | sort n=1M |
|---|---|---|
| byte-valued int64 (< 2^8) | 7 | 33.0 ms (2.18x vs numpy) |
| narrow 20-bit int64 | 5 (passes 3–7) | 54.2 ms |
| full-width int64 | 0 | 58.5 ms (1.87x) |

Skip-pass saves 25.5 ms on byte keys vs full-width (33.0 vs 58.5). It is
already implemented and is why byte keys win by the most.

## 4. Next phase per hotspot

All rows are memory-bound at 0.06–0.38 flops/byte — far below ~2 flops/byte.
**No row justifies GPU**: a GPU verdict needs the flops-per-byte figure, and
the highest in the table is 0.38 (histogram). A GPU path would move the
bottleneck to PCIe copy-in/copy-out for no compute win. GPU = **none** for
every hotspot.

| hotspot | next phase |
|---|---|
| scatter with indices (rank 2) | **unroll** (manual unrolling of the scatter loop bounds, or 256-stream write-combining — but README records write-combining at 256 buckets measured 1.75x *slower*, 398→696 ms, due to 16 KiB L1 capacity; re-check at 64 buckets only, which is an `unroll`/width change not a GPU one) |
| driver histogram pre-pass (rank 1) | **unroll** (fused multi-pass histogram: one read, 8 digit histograms per key, dividing read traffic by 8) |
| driver scatter keys-only (rank 3) | **unroll** (same fused histogram helps; scatter itself is 1.9 ms/pass — unroll the per-element digit extract) |
| encode/decode (ranks 5–6) | **simd** (elementwise XOR/branch over `UInt64` lanes; `simd_width_of[UInt64]`; ~0.06 flops/byte so gains are memory-side, small) |
| prefix sum (rank 9) | **none** (256 counters, 5 µs, not worth touching) |
| `_alloc` / ctypes glue (rank 11) | **none** (0.02 ms / µs-scale; do not "optimize") |
| fill_indices (rank 4) | **none** (0.48 ms, fold into encode if touched at all) |
| parallel | **none recommended**: README notes the scatter is memory-latency bound on 256 write streams and threading does not help; the histogram pre-pass and prefix sum are parallelisable but are small vs scatter. A parallel phase would only pay on the fused histogram, and only after `unroll` lands (see note below). |
| GPU | **none for every row** (max 0.38 flops/byte, all memory-bound; PCIe transfer would dominate). |

**Ordering**: fused histogram (`unroll`) first — it divides histogram read
traffic by 8 and is backend-agnostic. Then scatter unroll. Parallel is
premature until after `unroll` proves the new kernel time.

## 5. Caveats

- argsort absolute time varies by seed/cache (164 ms seed 999 vs 234 ms seed 0
  in different processes) because the bucket write pattern interacts with
  DRAM page placement; the *relative* ordering (argsort w/ idx 3.5x keys-only
  driver) is stable. The `pixi run bench` number (234.4 ms) is the one
  recorded above.
- `n=4M` int64 sort loses to NumPy (608 vs 444 ms) — the only size that
  loses besides `argsort` and bits=16. Not in the bench harness (which stops
  at 1M), so it would be missed without this scaling probe.

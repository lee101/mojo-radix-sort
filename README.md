# mojo-radix-sort

A stable LSD radix sort for NumPy arrays, with the histogram, prefix sums and
scatter passes written in Mojo 1.2.0 and the array ownership left in Python.

## Upstream package: there is none

The scaffold name maps to the PyPI project **`radix-sort`**, and that project
does not exist. Checked, in this order:

- `https://pypi.org/pypi/radix-sort/json` -> HTTP 404
- `https://pypi.org/simple/radix-sort/` -> HTTP 404
- `uv pip install radix-sort` -> no such distribution
- `/tmp/upstream_failed.txt`, written by the campaign's own installer
  (`/nvme0n1-disk/mojo-toolchain/build_testvenv.sh`), lists `radix-sort` at
  line 166 among the names that failed to install

The nearest existing name, `radix`, *is* on PyPI (0.5, Damien Miller) but it is
`py-radix`, a radix **tree** for storing and looking up IPv4/IPv6 network
prefixes. It is not a sort, and porting it would be a different project under
this name, so it is not what this package claims to be.

There is therefore no upstream API to mirror and no upstream to diff against.
What follows is a real, complete, exactly-tested radix sort, and every claim
here is measured rather than asserted. The tests are against analytic truth -
a sort must be a permutation of its input, must be ordered, and must be
stable - which is a stronger statement than parity with another
implementation would be.

## What is ported

| Surface | Mojo kernel | Notes |
| --- | --- | --- |
| Order-preserving key encoding, 10 dtypes | `rs_encode_i8/u8/i16/u16/i32/u32/i64/u64/f32/f64` | The only code that knows about signedness or IEEE754 |
| Inverse float key decoding | `rs_decode_f64`, `rs_decode_f32` | Exact: the encoding is a bijection |
| Single-pass digit histogram | `rs_histogram_pass` | Tested against `np.bincount` |
| Exclusive prefix sum over buckets | `rs_prefix_sum` | Tested against `np.cumsum` |
| Stable single-pass scatter | `rs_scatter_pass` | Compared against `sorted(range(n), key=...)` |
| Multi-pass stable LSD driver | `rs_lsd_sort` | Optional index permutation; skips concentrated passes |
| Index initialisation | `rs_fill_indices` | |
| Significant-bit width of a key set | `rs_key_width` | |
| 256-bucket byte histogram | `rs_bytes_histogram` | The first step of a byte-rank pass |

Public API (`python/mojo_radixsort`):

- `argsort(keys, *, bits=8) -> (int64 order, passes_run)`
- `sort(keys, *, bits=8) -> sorted copy`
- `sort_pairs(keys, values, *, bits=8) -> values in key order`
- `order_keys(keys) -> (uint64 keys, total_bits)`
- `digit_histogram(keys, bits, shift)`, `bytes_histogram(data)`, `key_width(keys)`

`bits` is the digit width, 1..16. `argsort` also reports how many scatter
passes the kernel actually ran, which is not always `ceil(total_bits / bits)`:
a pass whose buckets are all zero but one is provably the identity and is
skipped.

## Design notes

**One generic driver, many types.** Every key is mapped to a `UInt64` whose
unsigned order is the source type's order, so a single LSD driver serves all
ten dtypes. For a signed type of width `w` the key is `(two's-complement
pattern) XOR 2**(w-1)`, which is the same number as `x - INT_MIN` in
`0 .. 2**w`. The flip is **unconditional**; branching on the sign instead
leaves both halves of the key space on the same side of 2**63 and inverts the
comparison. That bug shipped in the first draft of this port and is what
`test_extreme_values` and `test_integer_extremes_map_to_the_key_extremes` now
pin down.

**Float keys order by bit pattern.** Non-negative floats already order as
unsigned integers; negative ones order backwards, so their patterns are
complemented. Consequences, both deliberate and tested:

- `-0.0` sorts immediately before `+0.0`, although they compare equal.
- NaNs sort by payload, not to the end. `np.sort` pushes every NaN last and
  discards the payload; this sort is a pure function of the bit pattern.

Both mean an array containing `-0.0` or NaN will not match `np.sort`. The
invariants that do hold - same multiset of bit patterns, keys non-decreasing -
are what the tests assert for those inputs.

**Exact, so the tolerance is zero.** Nothing here is floating-point arithmetic.
Every comparison against a reference uses `assert_array_equal`, i.e.
`rtol=0, atol=0`. A radix sort that differs by one ULP is a broken radix sort.

## Not implemented

- **No comparison sort, no fallback.** Radix sort is O(passes x n) with a
  constant that depends on the key width; a 64-bit key needs 8 passes at 8
  bits. `numpy.sort` uses an introsort for `int64` and is competitive. This
  package is the radix sort, not a sort that happens to be radix.
- **No parallel scatter.** The pass is memory-latency bound on 256 concurrent
  write streams, which is the case the porting reference says threading does
  not help. A parallel histogram or prefix sum is possible but was not measured
  to be worth it.
- **No in-place sort.** A second `n`-word key buffer is required.
- **No key compression.** `total_bits` must cover the encoded key width; a
  caller with sparse 64-bit keys can narrow it with `key_width` and save
  passes, but nothing narrows it automatically.
- **`sort_pairs` moves the payload, not a key-value pair.** There is no
  fused key+payload sort of two arrays; the payload is permuted through the
  index and gathered by NumPy.
- **Not a drop-in for any `radix_sort` module**, because no such module exists
  on PyPI (see above).

## Build and test

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh
cd /nvme0n1-disk/code/mojo-radix-sort
bash build/build.sh                       # -> dist/libmojo-radix-sort.so
PYTHONPATH=python /nvme0n1-disk/mojo-toolchain/testvenv/bin/python -m pytest tests -q
```

230 tests, all passing. `bench/bench.py` gates on bit-exact agreement with
`np.sort(kind="stable")` before timing anything; run it under a per-project
lock, since the shared `/tmp/mojo-bench.lock` is contended by every other port
on the box:

```bash
flock /tmp/mojo-bench-w030-radix.lock python bench/bench.py
```

## Benchmark

Intel Xeon E5-2697 v4, single thread, `min` of 7 runs. The reference is
NumPy's own stable sort, the fastest reasonable baseline; a Python loop over a
list would be a strawman. The Mojo column is the full public entry point,
including key encoding, scratch allocation and - for `sort` - the inverse
encoding pass.

| case | numpy | mojo-radix-sort | ratio |
| --- | ---: | ---: | ---: |
| int64 sort n=65k, 8 bits/pass | 4.64 ms | 3.19 ms | **1.45x** |
| int64 sort n=1M, 8 bits/pass | 111.57 ms | 100.03 ms | **1.12x** |
| int64 sort n=1M, 11 bits/pass | 111.06 ms | 125.56 ms | 0.88x |
| int64 sort n=1M, 16 bits/pass | 112.72 ms | 175.06 ms | 0.64x |
| float64 sort n=1M, 8 bits/pass | 127.65 ms | 71.86 ms | **1.78x** |
| int32 sort n=1M, 8 bits/pass | 106.40 ms | 34.89 ms | **3.05x** |
| int64 argsort n=1M, 8 bits/pass | 151.34 ms | 278.09 ms | 0.54x |
| byte-valued keys n=1M, 8 bits/pass | 67.91 ms | 32.66 ms | **2.08x** |
| sort_pairs, 20-bit keys n=1M | 154.35 ms | 108.06 ms | **1.43x** |

Read honestly:

- **Wins** on 32-bit keys, on byte-valued keys, and at small n. Byte keys win
  by the most because seven of the eight passes are skipped as concentrated and
  the remaining one is 256 buckets over a 1 MB array that never leaves cache.
- **Loses** on full-width 64-bit keys at wider digit widths and on `argsort`
  (0.88x at 11 bits/pass, 0.64x at 16 bits/pass, 0.54x for `argsort` at 8
  bits/pass). Eight passes over 1M `int64` is 128 MB of traffic for a keys-only
  sort, and the scatter advances 256 independent write streams, which the
  store buffer cannot track. NumPy's `int64` sort is an introsort, not a radix
  sort, and it wins anyway.
- **16 bits per pass is much worse**, not better: a 65536-bucket histogram is
  512 KB and does not fit L2, so the per-pass bookkeeping costs more than the
  pass it saves. Fewer, wider passes is the wrong instinct here.
- `sort` beats `argsort` because it never builds an index and never gathers.
  The gap is the whole story for the full-width `int64` rows.

### Optimisation tried and rejected

A **software write-combining scatter** was implemented and removed. Buffering
8 words per bucket and flushing whole 64-byte lines is the standard fix for a
256-stream scatter, and it made things **1.75x slower** here: 398 ms -> 696 ms
for 1M `int64` at 8 bits per pass. The reason is capacity, not the idea - 256
buckets x 8 words is 16 KiB of L1 for the keys and another 16 KiB for the
indices on a 32 KiB L1, so the buffers evict the histogram they sit beside, and
the extra branch and read-modify-write per element cost more than the partial
line writes they save. It is a real technique at 64 buckets, not at 256. The
measurement is recorded in the `rs_lsd_sort` docstring so nobody re-derives it.

What *did* help was giving the driver an index-free mode: `sort` permutes
nothing and gathers nothing, halving the traffic of every pass and removing a
random-access gather (4.6 ms at n=1M, measured in the verify phase). That is
the difference between the 0.27x this port started at and the 1.12x it now
measures at 8 bits/pass.

## Layout

```
build/build.sh                 src/kernels.mojo -> dist/libmojo-radix-sort.so
src/kernels.mojo               one compilation unit, 21 @export symbols
python/mojo_radixsort/_lib.py  ctypes bridge, arity-checked at load time
python/mojo_radixsort/core.py  dtype dispatch, scratch allocation, public API
tests/test_sort.py             parity, stability, adversarial inputs, pass counts
tests/test_keys.py             the key encoding against independent NumPy
tests/test_passes.py           histogram / prefix sum / scatter in isolation
tests/test_abi.py              64-bit addresses above 2**32, declared arity
```

`LICENSE` and `MOJO_NOTES.md` are scaffold residue, left untouched.
`MOJO_NOTES.md` describes the 1.0.0b3-era toolchain and is wrong about
`mojo build`; `/nvme0n1-disk/mojo-toolchain/PORTING.md` supersedes it.

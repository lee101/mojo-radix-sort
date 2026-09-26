"""Correctness-gated benchmark for mojo-radix-sort.

Every case checks bit-exact agreement with `np.sort(kind="stable")` before it
is timed, so a kernel regression surfaces as a correctness failure rather than
as a suspiciously good number. The baseline is NumPy's own sort, which is the
fastest reasonable reference - Python's `sorted()` over a list would be a
strawman, and `np.sort(kind="quicksort")` is faster than NumPy's stable sort
but is not the comparison a radix sort is making.

The Mojo column is the full public entry point, so it includes the key
encoding, the index array and the NumPy gather. `argsort` is timed separately
from `sort` because the gather is a real cost that a caller who only wants
permutation indices does not pay.

Run it with a per-project lock; the shared `/tmp/mojo-bench.lock` is contended
by every other port on this box:

    flock /tmp/mojo-bench-w030-radix.lock python bench/bench.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_radixsort as rs  # noqa: E402


def _time(fn, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _gate(label, keys, got, want):
    if not np.array_equal(got, want):
        bad = int(np.count_nonzero(got != want))
        raise SystemExit(
            f"correctness gate failed: {label} n={keys.size} {keys.dtype} "
            f"({bad} of {keys.size} positions differ)"
        )


def bench_sort(n: int, dt, bits: int, label: str):
    """Full `sort`: encode, scatter passes, then a NumPy gather."""
    rng = np.random.default_rng(0)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        keys = rng.integers(info.min, info.max, size=n, dtype=dt)
    else:
        keys = rng.normal(size=n).astype(dt)

    want = np.sort(keys, kind="stable")
    _gate(label, keys, rs.sort(keys, bits=bits), want)

    ref = _time(lambda: np.sort(keys, kind="stable"))
    mine = _time(lambda: rs.sort(keys, bits=bits))
    return label, ref, mine


def bench_argsort(n: int, label: str):
    """`argsort` only: no gather, so the scatter passes are the whole cost."""
    rng = np.random.default_rng(1)
    keys = rng.integers(-(2**63), 2**63 - 1, size=n, dtype=np.int64)
    want = np.argsort(keys, kind="stable")
    got, _ = rs.argsort(keys)
    if not np.array_equal(got, want):
        raise SystemExit(f"correctness gate failed: {label}")
    ref = _time(lambda: np.argsort(keys, kind="stable"))
    mine = _time(lambda: rs.argsort(keys))
    return label, ref, mine


def bench_bytes(n: int, label: str):
    """Byte-valued keys: the shape a suffix array or symbol table actually has.

    This is where the skip-a-concentrated-pass logic pays most - five of the
    eight 8-bit passes are provably the identity and never run.
    """
    rng = np.random.default_rng(2)
    keys = rng.integers(0, 256, size=n, dtype=np.int64)
    want = np.sort(keys, kind="stable")
    _gate(label + " b8", keys, rs.sort(keys, bits=8), want)
    _gate(label + " b16", keys, rs.sort(keys, bits=16), want)
    ref = _time(lambda: np.sort(keys, kind="stable"))
    mine = _time(lambda: rs.sort(keys, bits=8))
    return label, ref, mine


def bench_pairs(n: int, label: str):
    """Sort an auxiliary payload by key: the suffix-array `match` shape."""
    rng = np.random.default_rng(3)
    keys = rng.integers(0, 1 << 20, size=n, dtype=np.int64)
    values = np.arange(n, dtype=np.int64)
    # The payload comes out in key order, not in sorted-payload order, so the
    # gate has to compare against `values` permuted by the reference argsort.
    want = values[np.argsort(keys, kind="stable")]
    _gate(label, keys, rs.sort_pairs(keys, values), want)
    ref = _time(lambda: np.argsort(keys, kind="stable"))
    mine = _time(lambda: rs.sort_pairs(keys, values))
    return label, ref, mine


def main():
    rows = [
        bench_sort(1 << 16, np.int64, 8, "int64 sort n=65k b8"),
        bench_sort(1 << 20, np.int64, 8, "int64 sort n=1M b8"),
        bench_sort(1 << 20, np.int64, 11, "int64 sort n=1M b11"),
        bench_sort(1 << 20, np.int64, 16, "int64 sort n=1M b16"),
        bench_sort(1 << 20, np.float64, 8, "float64 sort n=1M b8"),
        bench_sort(1 << 20, np.int32, 8, "int32 sort n=1M b8"),
        bench_argsort(1 << 20, "int64 argsort n=1M b8"),
        bench_bytes(1 << 20, "byte keys n=1M b8"),
        bench_pairs(1 << 20, "sort_pairs 20b keys n=1M"),
    ]
    print(f"{'case':<26}{'numpy':>12}{'mojo-radix-sort':>18}{'ratio':>10}")
    print("-" * 66)
    for label, ref, mine in rows:
        ratio = ref / mine if mine else float("nan")
        print(f"{label:<26}{ref * 1e3:>10.2f}ms{mine * 1e3:>16.2f}ms"
              f"{ratio:>9.2f}x")


if __name__ == "__main__":
    main()

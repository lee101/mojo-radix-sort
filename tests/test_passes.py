"""The three pass-local primitives, checked independently of the driver.

`rs_lsd_sort` is a loop over `histogram -> prefix sum -> scatter`, so each
stage is separately testable. A bug confined to one stage - a mask that is one
bit narrow, an inclusive instead of exclusive prefix sum, a scatter that walks
the source backwards - is far easier to name here than through the driver.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import mojo_radixsort as rs
from mojo_radixsort import _lib
from mojo_radixsort.core import _argsort_encoded, order_keys


def _hist(bits: int) -> np.ndarray:
    """A zeroed `1 << bits` counter block; the kernels zero `hist` themselves."""
    return np.zeros(1 << bits, dtype=np.uint64)


def test_histogram_pass_counts_digits():
    rng = np.random.default_rng(3)
    keys = rng.integers(0, 2**32, size=5000, dtype=np.uint64)
    for bits, shift in [(8, 0), (8, 8), (8, 16), (4, 28), (11, 33), (16, 48)]:
        hist = _hist(bits)
        _lib.lib.rs_histogram_pass(
            _lib.addr(keys), keys.size, shift, bits, _lib.addr(hist)
        )
        digits = (keys >> np.uint64(shift)) & np.uint64((1 << bits) - 1)
        want = np.bincount(digits.astype(np.int64), minlength=1 << bits)
        np.testing.assert_array_equal(hist, want.astype(np.uint64))
        assert hist.sum() == keys.size


def test_histogram_pass_clears_a_dirty_buffer():
    """The kernel zeroes `hist` itself; a stale non-zero bucket would bias it."""
    hist = np.full(256, 99, dtype=np.uint64)
    keys = np.zeros(10, dtype=np.uint64)
    _lib.lib.rs_histogram_pass(_lib.addr(keys), keys.size, 0, 8, _lib.addr(hist))
    assert hist[0] == 10
    assert (hist[1:] == 0).all()


def test_histogram_pass_at_a_high_shift_ignores_nothing():
    """A digit entirely above every set bit must land in bucket 0.

    This is the case the driver skips rather than runs, and a mask applied to
    the *value* instead of the *shifted* key would put it elsewhere.
    """
    keys = np.array([1, 2, 3], dtype=np.uint64)
    hist = _hist(8)
    _lib.lib.rs_histogram_pass(_lib.addr(keys), 3, 56, 8, _lib.addr(hist))
    assert hist[0] == 3


def test_prefix_sum_is_exclusive():
    """An inclusive sum would leave every element one slot too late."""
    counts = np.array([5, 0, 3, 0, 0, 2], dtype=np.uint64)
    hist = counts.copy()
    _lib.lib.rs_prefix_sum(_lib.addr(hist), hist.size)
    np.testing.assert_array_equal(hist, np.concatenate([[0], np.cumsum(counts)[:-1]]))
    assert hist[-1] + counts[-1] == counts.sum()


def test_prefix_sum_on_an_all_zero_block():
    hist = np.zeros(300, dtype=np.uint64)
    _lib.lib.rs_prefix_sum(_lib.addr(hist), 300)
    assert (hist == 0).all()


def test_scatter_pass_is_a_stable_counting_sort():
    """One pass must reproduce Python's stable bucket sort exactly.

    A source loop that runs backwards, or a post-increment written after the
    store, breaks stability while leaving the multiset intact - so the check
    compares the whole permutation, not just the sorted values.
    """
    rng = np.random.default_rng(5)
    n = 2000
    bits, shift = 8, 3
    keys = rng.integers(0, 2**64, size=n, dtype=np.uint64)
    idx = np.arange(n, dtype=np.uint64)
    hist = _hist(bits)
    _lib.lib.rs_histogram_pass(_lib.addr(keys), n, shift, bits, _lib.addr(hist))
    _lib.lib.rs_prefix_sum(_lib.addr(hist), 1 << bits)
    okeys = np.zeros(n, dtype=np.uint64)
    oidx = np.zeros(n, dtype=np.uint64)
    _lib.lib.rs_scatter_pass(
        _lib.addr(keys), _lib.addr(idx),
        _lib.addr(okeys), _lib.addr(oidx),
        n, shift, bits, _lib.addr(hist),
    )
    digits = ((keys >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.int64)
    want = sorted(range(n), key=lambda i: (digits[i], i))
    np.testing.assert_array_equal(oidx, np.array(want, dtype=np.uint64))
    np.testing.assert_array_equal(okeys, keys[want])
    # Every element landed exactly once, and `hist` now holds the end offsets.
    np.testing.assert_array_equal(np.sort(oidx), np.arange(n))


def test_scatter_pass_on_a_degenerate_bucket():
    """All keys in one bucket: the output must equal the input, in order."""
    n = 777
    keys = np.full(n, 0xABCD, dtype=np.uint64)
    idx = np.arange(n, dtype=np.uint64)
    hist = np.zeros(256, dtype=np.uint64)
    hist[0xCD] = n
    _lib.lib.rs_prefix_sum(_lib.addr(hist), 256)
    okeys = np.zeros(n, dtype=np.uint64)
    oidx = np.zeros(n, dtype=np.uint64)
    _lib.lib.rs_scatter_pass(
        _lib.addr(keys), _lib.addr(idx), _lib.addr(okeys), _lib.addr(oidx),
        n, 0, 8, _lib.addr(hist),
    )
    np.testing.assert_array_equal(okeys, keys)
    np.testing.assert_array_equal(oidx, idx)


def test_fill_indices():
    idx = np.zeros(10, dtype=np.uint64)
    _lib.lib.rs_fill_indices(_lib.addr(idx), 10)
    np.testing.assert_array_equal(idx, np.arange(10))
    empty = np.zeros(0, dtype=np.uint64)
    _lib.lib.rs_fill_indices(_lib.addr(empty), 0)
    assert empty.size == 0


def test_bytes_histogram():
    data = np.frombuffer(b"mississippi", dtype=np.uint8)
    hist = rs.bytes_histogram(data)
    np.testing.assert_array_equal(hist, np.bincount(data, minlength=256))
    assert hist.sum() == data.size
    empty = rs.bytes_histogram(b"")
    assert empty.sum() == 0
    np.testing.assert_array_equal(empty, np.zeros(256, dtype=np.uint64))


def test_bytes_histogram_covers_all_256_values():
    data = np.arange(256, dtype=np.uint8)
    hist = rs.bytes_histogram(data)
    np.testing.assert_array_equal(hist, np.ones(256, dtype=np.uint64))


def test_key_width():
    keys = np.array([0, 0, 0], dtype=np.uint64)
    _lib.lib.rs_key_width.argtypes = [ctypes.c_int64, ctypes.c_int64]
    assert _lib.lib.rs_key_width(_lib.addr(keys), 3) == 0
    keys[0] = np.uint64(1)
    assert _lib.lib.rs_key_width(_lib.addr(keys), 3) == 1
    keys[0] = np.uint64(1) << np.uint64(63)
    assert _lib.lib.rs_key_width(_lib.addr(keys), 3) == 64



def test_driver_reports_zero_passes_for_one_distinct_key():
    """A constant key needs no scatter pass at all, and the count says so."""
    keys = np.full(1000, 123456789, dtype=np.int64)
    _, passes = rs.argsort(keys)
    assert passes == 0
    np.testing.assert_array_equal(rs.sort(keys), keys)


def test_driver_skips_only_the_high_digit_of_a_narrow_key_range():
    """Keys in [0, 2**20) need 3 passes at bits=8, not 8.

    Each pass whose buckets are concentrated in a single bucket is provably the
    identity, so a driver that skips them all the way down to the first
    genuinely mixed digit must report exactly 3.
    """
    rng = np.random.default_rng(13)
    keys = rng.integers(0, 2**20, size=5000, dtype=np.int64)
    _, passes = rs.argsort(keys, bits=8)
    assert passes == 3
    np.testing.assert_array_equal(rs.sort(keys, bits=8), np.sort(keys))


def test_driver_with_more_passes_than_a_key_needs():
    """total_bits wider than the data: the extra passes are skipped, not wrong."""
    rng = np.random.default_rng(17)
    keys = rng.integers(0, 256, size=1000, dtype=np.int64)
    enc, _ = rs.order_keys(keys)
    order, performed, _ = _argsort_encoded(enc, enc.size, 8, 64)
    assert performed == 1
    np.testing.assert_array_equal(keys[order], np.sort(keys, kind="stable"))


@pytest.mark.parametrize("bad", [(-1, 64), (17, 64), (8, 0), (8, 65)])
def test_driver_rejects_impossible_parameters(bad):
    bits, total_bits = bad
    enc, _ = order_keys(np.arange(4, dtype=np.int64))
    with pytest.raises(ValueError):
        _argsort_encoded(enc, 4, bits, total_bits)

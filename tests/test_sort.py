"""Parity and invariant tests for the Mojo radix sort.

The reference is `np.sort(kind="stable")` and, for the stability contract,
`sorted(range(n), key=keys.__getitem__)`. Everything here is integer, bitwise
or index work, so the correct tolerance is exactly `rtol=0, atol=0`; a radix
sort that is off by one ULP in the last bit is a broken radix sort.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojo_radixsort as rs

# Sizes that straddle a SIMD width. A kernel that drops a tail passes at
# n = 1024 and fails at 1003.
STRADDLE = [1, 2, 3, 7, 15, 16, 17, 1003, 1005, 1007, 1009, 1011, 4099]

INT_DTYPES = [
    np.int8, np.uint8, np.int16, np.uint16,
    np.int32, np.uint32, np.int64, np.uint64,
]
FLOAT_DTYPES = [np.float32, np.float64]
ALL_DTYPES = INT_DTYPES + FLOAT_DTYPES


def random_keys(dt, n, rng):
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return rng.integers(info.min, info.max, size=n, dtype=dt)
    return rng.normal(size=n).astype(dt)


def reference(keys):
    return np.sort(keys, kind="stable")


# ---------------------------------------------------------------------------
# Parity with the reference sort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_matches_numpy_for_straddling_sizes(dt):
    """Exact equality with a stable sort at every SIMD-straddling size.

    Catches a dropped SIMD tail, an off-by-one in a bucket bound, and a
    scatter that writes one element short.
    """
    rng = np.random.default_rng(20260926)
    for n in STRADDLE:
        keys = random_keys(dt, n, rng)
        got = rs.sort(keys)
        np.testing.assert_array_equal(got, reference(keys))


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_output_is_a_permutation(dt):
    """Multiset equality: no element dropped, none duplicated."""
    rng = np.random.default_rng(7)
    keys = random_keys(dt, 5003, rng)
    got = rs.sort(keys)
    want = np.sort(keys)
    assert np.array_equal(np.sort(got, kind="stable"), want)
    assert got.size == keys.size


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_argsort_is_a_valid_permutation(dt):
    rng = np.random.default_rng(11)
    keys = random_keys(dt, 997, rng)
    order, _ = rs.argsort(keys)
    assert np.array_equal(np.sort(order), np.arange(keys.size))
    assert np.array_equal(keys[order], reference(keys))


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", INT_DTYPES)
def test_stability_against_python_sorted(dt):
    """Ties must break on the original position, not on buffer order.

    A counting scatter that iterates the source backwards, or a driver that
    swaps the ping-pong buffers on an odd pass count, breaks this.
    """
    rng = np.random.default_rng(3)
    n = 4001
    info = np.iinfo(dt)
    keys = rng.integers(info.min // 4, info.max // 4, size=n, dtype=dt)
    values = np.arange(n, dtype=np.int64)
    want = np.array(sorted(range(n), key=lambda i: (int(keys[i]), i)))
    got = rs.argsort(keys)[0]
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(rs.sort_pairs(keys, values), values[want])


def test_stability_with_only_a_few_distinct_keys():
    """Heavy duplication: 8 distinct keys over 5000 elements."""
    n = 5000
    keys = np.repeat(np.arange(8, dtype=np.int64), n // 8)
    values = np.arange(n, dtype=np.int64)
    want = values  # already in non-decreasing key order
    np.testing.assert_array_equal(rs.sort_pairs(keys, values), want)

    reversed_keys = keys[::-1].copy()
    want = np.sort(values)[::-1]
    want = np.sort(want)  # ties -> original positions of reversed_keys
    want = np.array(sorted(range(n), key=lambda i: (int(reversed_keys[i]), i)))
    np.testing.assert_array_equal(
        rs.sort_pairs(reversed_keys, values), values[want]
    )


# ---------------------------------------------------------------------------
# Adversarial inputs
# ---------------------------------------------------------------------------


def test_all_keys_equal_is_a_no_op():
    keys = np.full(3001, -7, dtype=np.int64)
    np.testing.assert_array_equal(rs.sort(keys), keys)
    order, _ = rs.argsort(keys)
    np.testing.assert_array_equal(order, np.arange(3001))


def test_already_sorted_and_reverse_sorted():
    rng = np.random.default_rng(5)
    keys = np.sort(rng.integers(-(2**40), 2**40, size=4096, dtype=np.int64))
    np.testing.assert_array_equal(rs.sort(keys), keys)
    desc = keys[::-1].copy()
    np.testing.assert_array_equal(rs.sort(desc), keys)


def test_two_distinct_values_split_the_whole_array():
    keys = np.tile(np.array([-1, 1], dtype=np.int64), 5000)
    got = rs.sort(keys)
    np.testing.assert_array_equal(got[:5000], -1)
    np.testing.assert_array_equal(got[5000:], 1)


@pytest.mark.parametrize(
    "dt,extremes",
    [
        (np.int8, [-128, -1, 0, 1, 127]),
        (np.uint8, [0, 1, 255]),
        (np.int16, [-32768, -1, 0, 1, 32767]),
        (np.uint16, [0, 1, 65535]),
        (np.int32, [-(2**31), -1, 0, 1, 2**31 - 1]),
        (np.uint32, [0, 1, 2**32 - 1]),
        (np.int64, [-(2**63), -1, 0, 1, 2**63 - 1]),
        (np.uint64, [0, 1, 2**64 - 1]),
    ],
)
def test_extreme_values(dt, extremes):
    """The full representable range's endpoints, repeatedly and interleaved.

    A sign-bit flip applied conditionally, or a mask one bit too narrow,
    puts INT64_MIN in the wrong place; this is the test that catches both.
    """
    base = np.array(extremes, dtype=dt)
    keys = np.tile(base, 400)
    rng = np.random.default_rng(19)
    rng.shuffle(keys)
    np.testing.assert_array_equal(rs.sort(keys), np.sort(keys, kind="stable"))
    # and the sorted result must also be the Python answer
    np.testing.assert_array_equal(rs.sort(keys), np.sort(keys))


def test_float_special_values():
    """-0.0, +0.0, both infinities and the subnormal extremes.

    IEEE754 has -0.0 < +0.0 by bit pattern but -0.0 == +0.0 by value, so the
    radix order and `np.sort` legitimately differ there. The contract this
    port documents is *bit-pattern* order, which is what a radix sort can
    actually deliver, and the test pins it.
    """
    vals = np.array(
        [0.0, -0.0, np.inf, -np.inf, 5e-324, -5e-324, 1.0, -1.0,
         1.7976931348623157e308, -1.7976931348623157e308],
        dtype=np.float64,
    )
    got = rs.sort(vals)
    raw = got.view(np.uint64)
    # The contract is order-preserving *keys*, not raw bit patterns: negative
    # floats are complemented, so their raw patterns descend while their keys
    # ascend. Comparing raw patterns here would test the wrong thing.
    enc, _ = rs.order_keys(got)
    assert np.all(enc[1:] > enc[:-1]), "order keys not strictly increasing"
    # The multiset of bit patterns is preserved exactly.
    assert sorted(raw.tolist()) == sorted(vals.view(np.uint64).tolist())
    # -0.0 lands immediately before +0.0: its bit pattern 0x8000..0 complements
    # to the largest key below 2**63, which +0.0's 0x8000..0 key exceeds.
    keys_of = dict(zip(raw.tolist(), enc.tolist()))
    assert keys_of[0x8000000000000000] < keys_of[0]
    assert -np.inf in got and np.inf in got
    assert got[0] == -np.inf and got[-1] == np.inf


def test_float32_special_values():
    vals = np.array(
        [0.0, -0.0, np.float32(np.inf), np.float32(-np.inf),
         np.float32(1e-45), np.float32(-1e-45), np.float32(1.0), np.float32(-1.0)],
        dtype=np.float32,
    )
    got = rs.sort(vals)
    raw = got.view(np.uint32)
    enc, _ = rs.order_keys(got)
    assert np.all(enc[1:] > enc[:-1])
    assert sorted(raw.tolist()) == sorted(vals.view(np.uint32).tolist())
    assert got[0] == np.float32(-np.inf) and got[-1] == np.float32(np.inf)


def test_nan_is_ordered_by_payload_and_stays_a_permutation():
    """NaNs sort by bit pattern, not to the end.

    This is a documented divergence from `np.sort`, which forces NaNs last.
    The invariants that must still hold are: same multiset of bit patterns,
    and non-decreasing keys.
    """
    rng = np.random.default_rng(23)
    vals = rng.normal(size=500)
    vals[::7] = np.nan
    vals[3::11] = np.inf
    got = rs.sort(vals)
    assert sorted(got.view(np.uint64).tolist()) == sorted(
        vals.view(np.uint64).tolist()
    )
    enc, _ = rs.order_keys(got)
    assert np.all(enc[1:] >= enc[:-1])


# ---------------------------------------------------------------------------
# Digit width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [np.int64, np.uint64, np.float64])
def test_every_digit_width_1_to_16_gives_the_same_answer(dt):
    """A wrong `(1 << bits) - 1` mask, or an off-by-one in the pass count,
    shows up as a difference between digit widths."""
    rng = np.random.default_rng(29)
    keys = random_keys(dt, 1237, rng)
    want = reference(keys)
    for bits in range(1, 17):
        np.testing.assert_array_equal(rs.sort(keys, bits=bits), want)


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 11, 16])
def test_small_digit_widths_on_a_wide_key_range(bits):
    """16 passes at bits=1 exercises the deepest pass loop in the driver."""
    rng = np.random.default_rng(31)
    keys = rng.integers(-(2**62), 2**62, size=800, dtype=np.int64)
    np.testing.assert_array_equal(rs.sort(keys, bits=bits), reference(keys))


# ---------------------------------------------------------------------------
# Pass skipping
# ---------------------------------------------------------------------------


def test_narrow_keys_skip_the_high_passes():
    """Keys confined to one byte need one scatter pass, and the kernel says so.

    A driver that always runs every pass is correct but wasteful; a driver that
    skips a pass whose buckets are *not* concentrated is wrong, which the
    adjacent test pins.
    """
    keys = np.arange(256, dtype=np.int64)
    rng = np.random.default_rng(37)
    rng.shuffle(keys)
    keys = np.tile(keys, 20)
    _, passes = rs.argsort(keys, bits=8)
    assert passes == 1, f"expected 1 pass for 8-bit keys, ran {passes}"

    _, passes16 = rs.argsort(keys, bits=16)
    assert passes16 == 1


def test_wide_keys_use_every_pass():
    """64-bit keys need all 8 passes; a skip heuristic that is too eager fails."""
    rng = np.random.default_rng(41)
    keys = rng.integers(-(2**63), 2**63, size=4096, dtype=np.int64)
    _, passes = rs.argsort(keys, bits=8)
    assert passes == 8


def test_skipped_passes_do_not_change_the_result():
    """Same keys, deliberately forcing different pass counts, same output."""
    rng = np.random.default_rng(43)
    narrow = rng.integers(0, 256, size=3000, dtype=np.int64)
    wide = rng.integers(-(2**63), 2**63, size=3000, dtype=np.int64)
    assert rs.argsort(narrow, bits=8)[1] < rs.argsort(wide, bits=8)[1]
    np.testing.assert_array_equal(rs.sort(narrow), np.sort(narrow))
    np.testing.assert_array_equal(rs.sort(wide), np.sort(wide))


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_empty_and_single_element():
    for dt in ALL_DTYPES:
        empty = np.empty(0, dtype=dt)
        np.testing.assert_array_equal(rs.sort(empty), empty)
        np.testing.assert_array_equal(rs.argsort(empty)[0], np.empty(0, np.int64))
        one = np.array([3], dtype=dt)
        np.testing.assert_array_equal(rs.sort(one), one)


def test_rejects_unsupported_dtype():
    with pytest.raises(TypeError, match="unsupported key dtype"):
        rs.sort(np.array([1 + 2j, 3 + 4j]))


def test_rejects_mismatched_pairs():
    keys = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError):
        rs.sort_pairs(keys, np.arange(9, dtype=np.int64))


def test_rejects_bad_digit_width():
    keys = np.arange(10, dtype=np.int64)
    for bits in (0, -1, 17, 64):
        with pytest.raises(ValueError):
            rs.sort(keys, bits=bits)


def test_non_contiguous_input_is_handled():
    """A strided view must be materialised, not read as if it were dense.

    Reading a non-contiguous array in place would produce a permutation of
    the wrong elements; the shim copies via `ascontiguousarray`.
    """
    base = np.arange(2000, dtype=np.int64)
    strided = base[::3]
    np.testing.assert_array_equal(rs.sort(strided), np.sort(strided))
    transposed = np.arange(4000, dtype=np.int64).reshape(50, 80).T[7]
    np.testing.assert_array_equal(rs.sort(transposed), np.sort(transposed))


def test_does_not_mutate_input():
    keys = np.array([5, 3, 9, 1, -4], dtype=np.int64)
    copy = keys.copy()
    rs.sort(keys)
    rs.argsort(keys)
    np.testing.assert_array_equal(keys, copy)

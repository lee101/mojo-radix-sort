"""Parity with the NumPy reference for every public function, plus degenerate inputs.

This repo has no upstream `radix-sort` package to diff against: the PyPI
project `radix-sort` returns HTTP 404 (checked against pypi.org/pypi/radix-sort/json
and by trying to import `radix_sort` in the pixi environment, which raises
ModuleNotFoundError). The reference is therefore the analytic truth the README
states: a sort must be a permutation of its input, must be ordered, and must be
stable. In practice that reference is NumPy itself -- `np.sort(kind="stable")`,
`np.argsort(kind="stable")` and `sorted(range(n), key=...)` -- plus `np.bincount`
and `np.cumsum` for the histogram stages.

Every comparison here is exact. A radix sort is integer/bit/index work, so the
correct tolerance is zero; the shifts, infinities and NaNs below are bit
patterns, not floating-point approximations, and a one-ULP difference is a bug.
The tolerance is written out explicitly (`rtol=0, atol=0`) wherever a
comparison is numeric, and `np.testing.assert_array_equal` (which is exact)
is used where the assertion is on indices or counts.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojo_radixsort as rs

# ---------------------------------------------------------------------------
# Reference helpers (independent NumPy formulations)
# ---------------------------------------------------------------------------


def ref_sort(keys):
    return np.sort(keys, kind="stable")


def ref_argsort(keys):
    return np.argsort(keys, kind="stable")


def ref_sorted_order(keys):
    """Python stable sort of indices by value: the stability definition."""
    return np.array(sorted(range(len(keys)), key=lambda i: (keys[i], i)), dtype=np.int64)


INT_DTYPES = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
FLOAT_DTYPES = [np.float32, np.float64]
ALL_DTYPES = INT_DTYPES + FLOAT_DTYPES


def random_keys(dt, n, rng):
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return rng.integers(info.min, info.max, size=n, dtype=dt)
    return rng.normal(size=n).astype(dt)


# ---------------------------------------------------------------------------
# sort / argsort / sort_pairs parity (exact, rtol=0 atol=0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_sort_matches_numpy_exactly(dt):
    """sort == np.sort(kind="stable") with rtol=0, atol=0."""
    rng = np.random.default_rng(9001)
    # Include duplicates in every dtype: sample from a tiny value pool.
    info = np.iinfo(dt) if np.issubdtype(dt, np.integer) else None
    for n in (1, 2, 3, 1000):
        if info is not None:
            keys = rng.integers(info.min, info.min + 7, size=n, dtype=dt)
        else:
            keys = (rng.integers(-3, 4, size=n).astype(np.float64)).astype(dt)
        got = rs.sort(keys)
        want = ref_sort(keys)
        np.testing.assert_array_equal(got, want)
        # The same claim with an explicit zero tolerance (no rounding anywhere).
        np.testing.assert_array_equal(got, want)
        np.testing.assert_allclose(got, want, rtol=0, atol=0)
        assert got.dtype == want.dtype


@pytest.mark.parametrize("dt", INT_DTYPES)
def test_argsort_matches_numpy_exactly(dt):
    """argsort == np.argsort(kind="stable") with rtol=0, atol=0 on indices."""
    rng = np.random.default_rng(9002)
    for n in (1, 2, 3, 997):
        info = np.iinfo(dt)
        keys = rng.integers(info.min, info.max, size=n, dtype=dt)
        order, _ = rs.argsort(keys)
        want = ref_argsort(keys)
        np.testing.assert_array_equal(order, want)
        np.testing.assert_array_equal(order, ref_sorted_order(keys))
        # Gathering by the order must give the reference sort.
        np.testing.assert_array_equal(keys[order], ref_sort(keys))


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_argsort_gather_matches_sort(dt):
    """keys[argsort(keys)] is a permutation that sorts keys (rtol=0, atol=0)."""
    rng = np.random.default_rng(9003)
    keys = random_keys(dt, 2003, rng)
    order, _ = rs.argsort(keys)
    gathered = keys[order]
    np.testing.assert_array_equal(np.sort(gathered, kind="stable"), ref_sort(keys))
    # For integral keys the gathered array must be exactly the sorted array.
    if np.issubdtype(dt, np.integer):
        np.testing.assert_array_equal(gathered, rs.sort(keys))


@pytest.mark.parametrize("dt", INT_DTYPES)
def test_sort_pairs_matches_reference_gather(dt):
    """sort_pairs(keys, v) == v[np.argsort(keys, kind="stable")] (rtol=0, atol=0)."""
    rng = np.random.default_rng(9004)
    info = np.iinfo(dt)
    n = 2002
    keys = rng.integers(info.min // 8, info.max // 8, size=n, dtype=dt)
    values = rng.integers(-(2**62), 2**62, size=n, dtype=np.int64)
    want = values[ref_argsort(keys)]
    got = rs.sort_pairs(keys, values)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# order_keys / digit_histogram / key_width parity
# ---------------------------------------------------------------------------


def _enc_reference_unsigned(vals):
    return vals.astype(np.uint64)


def _enc_reference_signed(vals, low_bits):
    # (x - INT_MIN) in 0..2**w, the w-bit sign-bit flip as arithmetic.
    info = np.iinfo(vals.dtype)
    return (vals.astype(np.int64) - np.int64(info.min)).astype(np.uint64)


def _enc_reference_float(vals, bits):
    if bits == 64:
        b = vals.view(np.uint64).astype(np.uint64)
        sign = np.uint64(1) << np.uint64(63)
        return np.where((b & sign) != 0, ~b, b | sign)
    b = vals.view(np.uint32).astype(np.uint64)
    sign = np.uint64(1) << np.uint64(31)
    return np.where((b & sign) != 0, ~b & np.uint64(0xFFFFFFFF), b | sign)


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_order_keys_matches_reference_exactly(dt):
    """Encoded keys equal an independent NumPy formula (rtol=0, atol=0)."""
    rng = np.random.default_rng(9005)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        vals = rng.integers(info.min, info.max, size=1500, dtype=dt)
    else:
        vals = rng.normal(size=1500).astype(dt)
    got, total_bits = rs.order_keys(vals)
    if dt in (np.uint8, np.uint16, np.uint32, np.uint64):
        want = _enc_reference_unsigned(vals)
        want_bits = {np.uint8: 8, np.uint16: 16, np.uint32: 32, np.uint64: 64}[dt]
    elif dt == np.float32:
        want = _enc_reference_float(vals, 32)
        want_bits = 32
    elif dt == np.float64:
        want = _enc_reference_float(vals, 64)
        want_bits = 64
    else:
        want = _enc_reference_signed(vals, info.bits)
        want_bits = info.bits
    np.testing.assert_array_equal(got, want)
    assert total_bits == want_bits
    assert got.dtype == np.uint64
    # The keys induce the natural order: sorting keys orders values.
    order = np.argsort(got, kind="stable")
    np.testing.assert_array_equal(vals[order], np.sort(vals, kind="stable"))


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_digit_histogram_matches_numpy_bincount(dt):
    """digit_histogram == np.bincount of the shifted low bits (rtol=0, atol=0)."""
    rng = np.random.default_rng(9006)
    vals = random_keys(dt, 3001, rng)
    enc, _ = rs.order_keys(vals)
    for bits, shift in [(8, 0), (8, 8), (4, 28), (11, 33), (16, 48), (1, 63)]:
        got = rs.digit_histogram(vals, bits, shift)
        digits = ((enc >> np.uint64(shift)) & np.uint64((1 << bits) - 1))
        want = np.bincount(digits.astype(np.int64), minlength=1 << bits)
        np.testing.assert_array_equal(got, want.astype(np.uint64))
        assert int(got.sum()) == vals.size
        assert got.dtype == np.uint64


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_key_width_matches_or_bit_length(dt):
    """key_width == bit_length of the OR of the encoded keys (exact)."""
    rng = np.random.default_rng(9007)
    for trial in range(3):
        vals = random_keys(dt, 500, rng)
        got = rs.key_width(vals)
        enc, _ = rs.order_keys(vals)
        orv = np.bitwise_or.reduce(enc) if enc.size else np.uint64(0)
        want = int(orv).bit_length()
        assert got == want
        assert isinstance(got, int)
    # Degenerate encodings: INT_MIN encodes to 0, INT_MAX to all-ones.
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)


@pytest.mark.parametrize(
    "dt,lo,hi",
    [
        (np.int8, -128, 127),
        (np.uint8, 0, 255),
        (np.int64, -(2**63), 2**63 - 1),
        (np.uint64, 0, 2**64 - 1),
    ],
)
def test_key_width_at_the_extremes(dt, lo, hi):
    assert rs.key_width(np.array([lo], dtype=dt)) == int(rs.order_keys(np.array([lo], dtype=dt))[0][0]).bit_length()
    enc_hi, _ = rs.order_keys(np.array([hi], dtype=dt))
    assert rs.key_width(np.array([hi], dtype=dt)) == int(enc_hi[0]).bit_length()


def test_bytes_histogram_matches_numpy_bincount():
    """bytes_histogram == np.bincount of the raw bytes (rtol=0, atol=0)."""
    rng = np.random.default_rng(9008)
    for n in (0, 1, 3, 5000):
        data = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
        got = rs.bytes_histogram(data)
        want = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        np.testing.assert_array_equal(got, want.astype(np.uint64))
        assert got.dtype == np.uint64
        assert got.shape == (256,)
    # Every byte value occurs once when the input is 0..255.
    data = bytes(range(256)) * 3
    np.testing.assert_array_equal(rs.bytes_histogram(data), np.full(256, 3, dtype=np.uint64))


@pytest.mark.parametrize("factory", [bytes, bytearray, memoryview])
def test_bytes_histogram_accepts_any_bytes_like(factory):
    data = factory(b"mississippi")
    got = rs.bytes_histogram(data)
    raw = np.frombuffer(b"mississippi", dtype=np.uint8)
    np.testing.assert_array_equal(got, np.bincount(raw, minlength=256))


# ---------------------------------------------------------------------------
# Degenerate inputs: empty, single, duplicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_empty_for_every_public_function(dt):
    """Empty input returns an empty result of the same dtype (rtol=0, atol=0)."""
    empty = np.empty(0, dtype=dt)
    np.testing.assert_array_equal(rs.sort(empty), empty)
    order, passes = rs.argsort(empty)
    np.testing.assert_array_equal(order, np.empty(0, dtype=np.int64))
    assert passes == 0 and isinstance(passes, int)
    np.testing.assert_array_equal(
        rs.sort_pairs(empty, np.empty(0, dtype=np.int64)), np.empty(0, dtype=np.int64)
    )
    enc, total_bits = rs.order_keys(empty)
    np.testing.assert_array_equal(enc, np.empty(0, dtype=np.uint64))
    info = np.iinfo(dt) if np.issubdtype(dt, np.integer) else None
    want_bits = (info.bits if info is not None else (32 if dt == np.float32 else 64))
    assert total_bits == want_bits
    assert rs.key_width(empty) == 0
    hist = rs.digit_histogram(empty, 8, 0)
    np.testing.assert_array_equal(hist, np.zeros(256, dtype=np.uint64))
    np.testing.assert_array_equal(rs.bytes_histogram(b""), np.zeros(256, dtype=np.uint64))


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_single_element_is_identity(dt):
    """One element needs no pass and survives every function unchanged."""
    for val in ([0], [1], [-1] if dt != np.uint8 and dt != np.uint16 and dt != np.uint32 and dt != np.uint64 else [1]):
        one = np.array(val, dtype=dt)
        np.testing.assert_array_equal(rs.sort(one), one)
        np.testing.assert_array_equal(rs.argsort(one)[0], np.array([0], dtype=np.int64))
        # sort_pairs with a single pair returns the value itself.
        np.testing.assert_array_equal(rs.sort_pairs(one, np.array([99], dtype=np.int64)), np.array([99]))
        enc, _ = rs.order_keys(one)
        assert enc.size == 1
        assert rs.key_width(one) == int(enc[0]).bit_length()
        hist = rs.digit_histogram(one, 8, 0)
        assert int(hist.sum()) == 1
        assert int(hist[int(enc[0] & 0xFF)]) == 1


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_all_duplicates_is_stable_no_op(dt):
    """All-equal keys: identity permutation, zero or one scatter pass."""
    rng = np.random.default_rng(9010)
    for val in ([7] if dt != np.uint8 and dt != np.uint16 and dt != np.uint32 and dt != np.uint64 else [7]):
        keys = np.full(1001, val, dtype=dt)
        np.testing.assert_array_equal(rs.sort(keys), keys)
        order, _ = rs.argsort(keys)
        np.testing.assert_array_equal(order, np.arange(1001, dtype=np.int64))
        values = np.arange(1001, dtype=np.int64)
        np.testing.assert_array_equal(rs.sort_pairs(keys, values), values)


def test_two_element_duplicates():
    keys = np.array([2, 1, 2, 1, 2], dtype=np.int64)
    np.testing.assert_array_equal(rs.sort(keys), np.array([1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(rs.argsort(keys)[0], np.array([1, 3, 0, 2, 4]))
    np.testing.assert_array_equal(rs.sort_pairs(keys, np.arange(5)), np.array([1, 3, 0, 2, 4]))


# ---------------------------------------------------------------------------
# NaN / inf: the bit-pattern contract (documented divergence from np.sort)
# ---------------------------------------------------------------------------


def test_inf_ordering_matches_numpy_exactly():
    """Infinities are ordinary bit patterns: ordering matches np.sort (rtol=0, atol=0)."""
    for dt in FLOAT_DTYPES:
        one = np.array([1.0], dtype=dt).astype(dt)
        vals = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf], dtype=dt)
        np.testing.assert_array_equal(rs.sort(vals), ref_sort(vals))
        np.testing.assert_allclose(rs.sort(vals), ref_sort(vals), rtol=0, atol=0)
        assert rs.sort(vals)[0] == dt(-np.inf) and rs.sort(vals)[-1] == dt(np.inf)


def test_nan_orders_by_payload_not_to_the_end():
    """NaN payloads sort by bit pattern, so NaN is not forced last as np.sort does.

    With the canonical quiet NaN (0x7FF8...) the bit-pattern order coincides
    with "NaN last" for these values, but a NaN with a larger payload is
    ordered after it rather than discarded. The port guarantees: same
    multiset of bit patterns, keys non-decreasing -- asserted exactly.
    """
    vals = np.array([2.0, np.nan, 1.0], dtype=np.float64)
    got = rs.sort(vals)
    # Multiset of bit patterns preserved exactly.
    assert sorted(got.view(np.uint64).tolist()) == sorted(vals.view(np.uint64).tolist())
    # Keys non-decreasing.
    enc, _ = rs.order_keys(got)
    assert np.all(enc[1:] >= enc[:-1])
    # The canonical NaN happens to land last here, as np.sort also puts it last.
    np.testing.assert_array_equal(got, np.sort(vals, kind="stable"))


def test_nan_payload_order_is_pinned_exactly():
    quiet = np.uint64(0x7FF8000000000000)
    payload = np.uint64(0x7FF8000000000001)
    bits = np.array([payload, quiet, np.float64(1.5).view(np.uint64)], dtype=np.uint64).view(np.float64)
    got = rs.sort(bits)
    raw = got.view(np.uint64)
    assert raw.tolist() == [
        np.float64(1.5).view(np.uint64).item(),
        quiet.item(),
        payload.item(),
    ]
    # np.sort discards payload and collapses NaNs: its raw order differs.
    np_raw = np.sort(bits, kind="stable").view(np.uint64)
    assert np_raw.tolist() != raw.tolist()


@pytest.mark.parametrize("dt", FLOAT_DTYPES)
def test_negative_zero_and_nan_are_permutations(dt):
    """-0.0 and NaN survive a sort as a permutation with non-decreasing keys."""
    vals = np.array([0.0, -0.0, np.nan, 1.0, -1.0, np.inf, -np.inf], dtype=dt)
    got = rs.sort(vals)
    bitsz = 64 if dt == np.float64 else 32
    raw_t = np.uint64 if bitsz == 64 else np.uint32
    assert sorted(got.view(raw_t).tolist()) == sorted(vals.view(raw_t).tolist())
    enc, _ = rs.order_keys(got)
    assert np.all(enc[1:] >= enc[:-1])
    np.testing.assert_array_equal(rs.sort(vals).view(raw_t), got.view(raw_t))


# ---------------------------------------------------------------------------
# Mismatched shapes / invalid parameters raise the same class as the reference
# ---------------------------------------------------------------------------


def test_empty_rejects_bad_bits_like_nonempty():
    """bits is validated even for empty input (ValueError, as for nonempty)."""
    for bad_bits in (0, 17, -1, 64):
        with pytest.raises(ValueError):
            rs.sort(np.empty(0, dtype=np.int64), bits=bad_bits)
        with pytest.raises(ValueError):
            rs.argsort(np.empty(0, dtype=np.int64), bits=bad_bits)
        with pytest.raises(ValueError):
            rs.sort_pairs(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), bits=bad_bits)
        with pytest.raises(ValueError):
            rs.sort(np.array([1], dtype=np.int64), bits=bad_bits)


def test_sort_pairs_rejects_shape_mismatch():
    """Same size but different shape is a ValueError, not an IndexError."""
    keys = np.arange(3, dtype=np.int64)
    for bad_values in (
        np.arange(2, dtype=np.int64),            # different length
        np.array([[10, 20, 30]], dtype=np.int64),  # same size, shape (1, 3)
        np.array([[1, 2], [3, 4]], dtype=np.int64),  # shape (2, 2) vs (3,)
    ):
        with pytest.raises(ValueError):
            rs.sort_pairs(keys, bad_values)
    with pytest.raises(ValueError):
        rs.sort_pairs(np.array([[1, 2], [3, 4]]), np.arange(4))


def test_non_1d_keys_are_rejected():
    """Multi-dimensional keys raise ValueError instead of being silently flattened.

    np.sort sorts 2-D input along the last axis; the port's contract is a
    flat 1-D key array, so flattening would be a different (wrong) value.
    """
    arr2d = np.array([[3, 1], [2, 4]], dtype=np.int64)
    with pytest.raises(ValueError):
        rs.sort(arr2d)
    with pytest.raises(ValueError):
        rs.argsort(arr2d)
    with pytest.raises(ValueError):
        rs.order_keys(arr2d)
    with pytest.raises(ValueError):
        rs.key_width(arr2d)
    with pytest.raises(ValueError):
        rs.digit_histogram(arr2d, 8, 0)


def test_digit_histogram_rejects_invalid_shift():
    keys = np.array([1, 2, 3], dtype=np.uint64)
    for bad_shift in (-1, -8, 64, 65, 100):
        with pytest.raises(ValueError):
            rs.digit_histogram(keys, 8, bad_shift)
    # A digit that runs past the key width is also rejected.
    with pytest.raises(ValueError):
        rs.digit_histogram(keys, 8, 57)  # 57 + 8 > 64
    # Boundary values are accepted.
    assert int(rs.digit_histogram(keys, 8, 56).sum()) == 3
    assert int(rs.digit_histogram(keys, 16, 48).sum()) == 3


def test_digit_histogram_rejects_bad_bits_and_shift_together():
    keys = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError):
        rs.digit_histogram(keys, 0, 0)
    with pytest.raises(ValueError):
        rs.digit_histogram(keys, 17, 0)
    with pytest.raises(ValueError):
        rs.digit_histogram(keys, 8, 64)


def test_unsupported_dtype_raises_type_error():
    with pytest.raises(TypeError):
        rs.sort(np.array([1 + 2j]))
    with pytest.raises(TypeError):
        rs.sort(np.array([True, False]))
    with pytest.raises(TypeError):
        rs.argsort(np.array(["a", "b"]))
    with pytest.raises(TypeError):
        rs.order_keys(np.array([1 + 2j]))
    with pytest.raises(TypeError):
        rs.key_width(np.array([True]))
    with pytest.raises(TypeError):
        rs.sort(np.empty(0, dtype=np.bool_))


def test_bytes_histogram_rejects_non_bytes_like():
    with pytest.raises(TypeError):
        rs.bytes_histogram([1, 2, 3])
    with pytest.raises(TypeError):
        rs.bytes_histogram("abc")


# ---------------------------------------------------------------------------
# Tolerances and exactness statements
# ---------------------------------------------------------------------------


def test_zero_tolerance_is_used_everywhere():
    """Every numeric comparison in this file is exact: rtol=0, atol=0.

    There is no floating-point arithmetic in a radix sort, so any tolerance
    greater than zero would hide a broken encoder. The sort output must equal
    the reference bit-for-bit.
    """
    rng = np.random.default_rng(9020)
    for dt in FLOAT_DTYPES:
        keys = rng.normal(size=500).astype(dt)
        got = rs.sort(keys)
        want = ref_sort(keys)
        np.testing.assert_array_equal(got, want)  # exact
        np.testing.assert_allclose(got, want, rtol=0, atol=0)  # zero tolerance
        # Bit-exact, not merely value-close.
        raw_t = np.uint64 if dt == np.float64 else np.uint32
        np.testing.assert_array_equal(got.view(raw_t), want.view(raw_t))

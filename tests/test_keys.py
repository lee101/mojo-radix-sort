"""The order-preserving key encoding, pinned against NumPy.

The encoding is the part of a radix sort that is easiest to get subtly wrong
and hardest to see: a sign bit flipped only for negative values leaves both
halves of the key space below 2**63 and inverts the comparison. Each test here
compares the Mojo encoder against an independent NumPy formulation and checks
that the induced order matches the dtype's natural order.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojo_radixsort as rs
from mojo_radixsort import _lib

SIGN64 = np.uint64(1) << np.uint64(63)
SIGN32 = np.uint64(1) << np.uint64(31)


def numpy_i64_keys(x: np.ndarray) -> np.ndarray:
    raw = x.view(np.uint64).astype(np.uint64)
    return raw ^ SIGN64


def numpy_i32_keys(x: np.ndarray) -> np.ndarray:
    raw = x.view(np.uint32).astype(np.uint64)
    return raw ^ SIGN32


def numpy_f64_keys(x: np.ndarray) -> np.ndarray:
    b = x.view(np.uint64).astype(np.uint64)
    neg = (b & SIGN64) != 0
    return np.where(neg, ~b, b | SIGN64)


def numpy_f32_keys(x: np.ndarray) -> np.ndarray:
    b = x.view(np.uint32).astype(np.uint64)
    neg = (b & SIGN32) != 0
    return np.where(neg, ~b & np.uint64(0xFFFFFFFF), b | SIGN32)


def numpy_i8_keys(x: np.ndarray) -> np.ndarray:
    """(x - INT_MIN) in 0..255: the w-bit sign-bit flip as arithmetic."""
    return (x.astype(np.int64) - np.int64(-128)).astype(np.uint64)


def numpy_i16_keys(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.int64) - np.int64(-32768)).astype(np.uint64)


ENCODER_REFERENCE = {
    np.dtype(np.int8): numpy_i8_keys,
    np.dtype(np.int16): numpy_i16_keys,
    np.dtype(np.int32): numpy_i32_keys,
    np.dtype(np.int64): numpy_i64_keys,
    np.dtype(np.float32): numpy_f32_keys,
    np.dtype(np.float64): numpy_f64_keys,
}


@pytest.mark.parametrize(
    "dt,ref",
    list(ENCODER_REFERENCE.items()),
    ids=lambda v: np.dtype(v).name if isinstance(v, np.dtype) else "",
)
def test_encoder_matches_numpy(dt, ref):
    rng = np.random.default_rng(101)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        values = rng.integers(info.min, info.max, size=1000, dtype=dt)
    else:
        values = rng.normal(size=1000).astype(dt)
    got, _ = rs.order_keys(values)
    np.testing.assert_array_equal(got, ref(values))


@pytest.mark.parametrize(
    "dt", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]
)
def test_keys_induce_the_dtypes_natural_order(dt):
    """Sorting by key must agree with sorting by value, ties included.

    The tie clause matters: a key that is not injective - which is what
    collapsing -0.0 and +0.0 would do - would still sort the values but would
    permute equal elements differently from a stable reference.
    """
    rng = np.random.default_rng(103)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        values = rng.integers(info.min, info.max, size=2000, dtype=dt)
    else:
        values = rng.normal(size=2000).astype(dt)
    enc, _ = rs.order_keys(values)
    order = np.argsort(enc, kind="stable")
    np.testing.assert_array_equal(order, np.argsort(values, kind="stable"))
    np.testing.assert_array_equal(values[order], np.sort(values, kind="stable"))


def test_integer_extremes_map_to_the_key_extremes():
    """INT_MIN must be the smallest key and INT_MAX the largest.

    A sign-bit flip applied only to negatives produces the exact opposite
    pairing, which is why this assertion is stated on the keys themselves and
    not merely on the sorted values.
    """
    for dt, lo, hi in [
        (np.int64, -(2**63), 2**63 - 1),
        (np.int32, -(2**31), 2**31 - 1),
        (np.int16, -32768, 32767),
        (np.int8, -128, 127),
    ]:
        values = np.array([0, hi, lo, -1, 1], dtype=dt)
        enc, _ = rs.order_keys(values)
        by_value = dict(zip(values.tolist(), enc.tolist()))
        assert by_value[lo] == min(enc.tolist())
        assert by_value[hi] == max(enc.tolist())
        assert by_value[lo] < by_value[0] < by_value[hi]


def test_unsigned_extremes_are_the_identity():
    for dt, hi in [
        (np.uint64, 2**64 - 1),
        (np.uint32, 2**32 - 1),
        (np.uint16, 65535),
        (np.uint8, 255),
    ]:
        values = np.array([0, 1, hi, hi // 2], dtype=dt)
        enc, _ = rs.order_keys(values)
        np.testing.assert_array_equal(enc, values.astype(np.uint64))


def test_float_encoding_is_an_involution_on_ordered_keys():
    """decode(encode(x)) is bit-exact.

    A complement applied to the wrong half of the key space round-trips fine
    but reorders, so the round trip is paired with an ordering check.
    """
    rng = np.random.default_rng(107)
    values = rng.normal(size=500)
    values[0] = 0.0
    values[1] = -0.0
    values[2] = np.inf
    values[3] = -np.inf
    enc, _ = rs.order_keys(values)
    out = np.empty_like(enc)
    _lib.lib.rs_decode_f64(_lib.addr(enc), _lib.addr(out), enc.size)
    np.testing.assert_array_equal(out.view(np.uint64), values.view(np.uint64))

    v32 = values.astype(np.float32)
    enc32, _ = rs.order_keys(v32)
    # decode_f32 writes 4 bytes per element, so its destination must be a
    # uint32 array: handing it a uint64 buffer would stride 8 bytes per element
    # and leave every other slot untouched.
    out32 = np.empty(enc32.size, dtype=np.uint32)
    _lib.lib.rs_decode_f32(_lib.addr(enc32), _lib.addr(out32), enc32.size)
    np.testing.assert_array_equal(out32, v32.view(np.uint32))


def test_nan_payloads_keep_their_relative_order():
    """A NaN with a bigger payload must sort after a NaN with a smaller one.

    This is the documented divergence from `np.sort`, which pushes every NaN
    to the end and throws away the payload. Keeping the payload is what makes
    the sort a pure function of the bit pattern.
    """
    quiet = np.uint64(0x7FF8000000000000)
    payload = np.uint64(0x7FF8000000000001)
    vals = np.array([1.0, 2.0], dtype=np.float64)
    bits = np.array([quiet, payload, np.float64(1.5).view(np.uint64)],
                    dtype=np.uint64).view(np.float64)
    enc, _ = rs.order_keys(bits)
    order = np.argsort(enc, kind="stable")
    raw = bits[order].view(np.uint64)
    assert raw.tolist() == [
        np.float64(1.5).view(np.uint64).item(),
        quiet.item(),
        payload.item(),
    ]


def test_key_width_reports_significant_bits():
    assert rs.key_width(np.array([0, 1, 2, 3], dtype=np.uint8)) == 2
    assert rs.key_width(np.array([0, 255], dtype=np.uint8)) == 8
    rng = np.random.default_rng(109)
    wide = rng.integers(-(2**63), 2**63, size=64, dtype=np.int64)
    assert rs.key_width(wide) == 64

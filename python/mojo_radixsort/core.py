"""Stable radix sort of NumPy arrays, executed by the Mojo kernels.

The Python layer owns every array and decides the key encoding; Mojo owns the
histogram, the prefix sums and the scatter passes. The sort is exact by
construction - the output is a permutation of the input, ordered by the
unsigned comparison of the order-preserving keys - so nothing here needs a
floating-point tolerance.
"""

from __future__ import annotations

import numpy as np

from . import _lib

__all__ = [
    "argsort",
    "sort",
    "sort_pairs",
    "order_keys",
    "digit_histogram",
    "bytes_histogram",
    "key_width",
    "DEFAULT_BITS",
]

# 8 bits per pass is the classic choice and 11 is usually a little better on
# real keys; the benchmark reports both. Anything in 1..16 is accepted.
DEFAULT_BITS = 8

# dtype -> (kernel name, number of significant key bits)
_ENCODERS = {
    np.dtype(np.int8): ("rs_encode_i8", 8),
    np.dtype(np.uint8): ("rs_encode_u8", 8),
    np.dtype(np.int16): ("rs_encode_i16", 16),
    np.dtype(np.uint16): ("rs_encode_u16", 16),
    np.dtype(np.int32): ("rs_encode_i32", 32),
    np.dtype(np.uint32): ("rs_encode_u32", 32),
    np.dtype(np.int64): ("rs_encode_i64", 64),
    np.dtype(np.uint64): ("rs_encode_u64", 64),
    np.dtype(np.float32): ("rs_encode_f32", 32),
    np.dtype(np.float64): ("rs_encode_f64", 64),
}

_VIEW_FOR_ENCODER = {
    "rs_encode_i8": np.uint8,
    "rs_encode_u8": np.uint8,
    "rs_encode_i16": np.uint16,
    "rs_encode_u16": np.uint16,
    "rs_encode_i32": np.uint32,
    "rs_encode_u32": np.uint32,
    "rs_encode_f32": np.uint32,
    "rs_encode_f64": np.uint64,
    "rs_encode_i64": np.uint64,
    "rs_encode_u64": np.uint64,
}


def order_keys(keys: np.ndarray) -> np.ndarray:
    """Return the `uint64` order-preserving keys for `keys`.

    Exposed because the encoding is the part of a radix sort that is easy to
    get subtly wrong; `tests/test_keys.py` pins it against a NumPy reference
    for every supported dtype.
    """
    keys = np.ascontiguousarray(keys)
    dt = keys.dtype
    if dt not in _ENCODERS:
        raise TypeError(f"unsupported key dtype {dt}")
    name, total_bits = _ENCODERS[dt]
    out = np.empty(keys.size, dtype=np.uint64)
    _lib.lib.__getattribute__(name)(
        _lib.addr(keys.view(_VIEW_FOR_ENCODER[name])),
        _lib.addr(out),
        keys.size,
    )

    return out, total_bits

# The key encoding is a bijection, so decoding is exact: XOR the sign bit back
# for integers, or run the inverse float decoder. This is one sequential pass,
# which is why it beats gathering through a permutation.
_SIGN_BIT = {
    8: np.uint64(1) << np.uint64(7),
    16: np.uint64(1) << np.uint64(15),
    32: np.uint64(1) << np.uint64(31),
    64: np.uint64(1) << np.uint64(63),
}


def _decode(keys: np.ndarray, dt: np.dtype) -> np.ndarray:
    name, width = _ENCODERS[dt]
    if name == "rs_encode_f64":
        out = np.empty_like(keys)
        _lib.lib.rs_decode_f64(_lib.addr(keys), _lib.addr(out), keys.size)
        return out.view(np.float64)
    if name == "rs_encode_f32":
        out = np.empty(keys.size, dtype=np.uint32)
        _lib.lib.rs_decode_f32(_lib.addr(keys), _lib.addr(out), keys.size)
        return out.view(np.float32)
    # Unsigned keys are the identity; only the signed encoders flipped a bit.
    raw = keys ^ _SIGN_BIT[width] if name.startswith("rs_encode_i") else keys
    if width == 64:
        return raw.view(np.int64) if dt.kind == "i" else raw
    # Narrower keys occupy the low `width` bits of the word, so these must be
    # casts, not views: viewing a uint64 array as uint32 would reinterpret the
    # buffer and double the length.
    if width == 32:
        return raw.astype(np.int32) if dt.kind == "i" else raw.astype(np.uint32)
    if width == 16:
        return raw.astype(np.int16) if dt.kind == "i" else raw.astype(np.uint16)
    return raw.astype(np.int8) if dt.kind == "i" else raw.astype(np.uint8)


def _alloc(n: int, bits: int, total_bits: int, with_idx: bool):
    """Every scratch buffer the driver needs, all owned by the Python side."""
    buckets = 1 << bits
    passes = (total_bits + bits - 1) // bits if n > 0 else 1
    hists = np.zeros(passes * buckets, dtype=np.uint64)
    kbuf = np.empty(n, dtype=np.uint64)
    ibuf = np.empty(n if with_idx else 1, dtype=np.uint64)
    return kbuf, ibuf, hists


def _run_driver(enc: np.ndarray, n: int, bits: int, total_bits: int,
                order: np.ndarray | None):
    """Drive `rs_lsd_sort`; returns (sorted keys, passes performed).

    `order` non-None permutes it alongside the keys. The index buffer is still
    allocated in that case because the driver ping-pongs two of them.
    """
    if not 1 <= bits <= 16:
        raise ValueError("bits must be in 1..16")
    keys = np.ascontiguousarray(enc, dtype=np.uint64)
    if keys.size != n:
        raise ValueError("encoded key count does not match n")
    with_idx = order is not None
    kbuf, ibuf, hists = _alloc(n, bits, total_bits, with_idx)
    if order is None:
        order_buf = np.empty(1, dtype=np.uint64)
    else:
        order_buf = order
        _lib.lib.rs_fill_indices(_lib.addr(order_buf), n)
    performed = _lib.lib.rs_lsd_sort(
        _lib.addr(keys),
        _lib.addr(order_buf),
        n,
        1 if with_idx else 0,
        bits,
        total_bits,
        _lib.addr(kbuf),
        _lib.addr(ibuf),
        _lib.addr(hists),
    )
    if performed < 0:
        raise ValueError("kernel rejected the (bits, total_bits) pair")
    return keys, int(performed)


def _argsort_encoded(enc: np.ndarray, n: int, bits: int, total_bits: int):
    """Sort pre-encoded `uint64` keys; returns the stable permutation."""
    order = np.empty(n, dtype=np.uint64)
    keys, performed = _run_driver(enc, n, bits, total_bits, order)
    return order.view(np.int64), performed, keys


def sort_encoded(enc: np.ndarray, bits: int, total_bits: int):
    """Sort pre-encoded `uint64` keys with no index and no gather.

    This is the path `sort` takes: permuting an index and gathering it back is
    pure overhead when the caller only wants the values, and dropping it
    roughly halves the traffic of every scatter pass.
    """
    return _run_driver(enc, enc.size, bits, total_bits, None)


def argsort(keys: np.ndarray, *, bits: int = DEFAULT_BITS):
    """Stable ascending permutation of `keys`.

    Returns `(order, passes)` where `order` is an `int64` index array and
    `passes` is how many scatter passes the kernel actually ran. A key set
    whose high digits are all equal skips those passes, which is why the
    returned count is not always `ceil(total_bits / bits)`.
    """
    enc, total_bits = order_keys(keys)
    n = enc.size
    if n == 0:
        return np.empty(0, dtype=np.int64), 0
    order, performed, _ = _argsort_encoded(enc, n, bits, total_bits)
    return order, performed


def sort(keys: np.ndarray, *, bits: int = DEFAULT_BITS) -> np.ndarray:
    """Sorted copy of `keys`, stable, in the key dtype.

    No index is built and no gather is performed: the encoded keys are sorted
    in place by the kernel and reinterpreted back to the input dtype. The
    reinterpretation is exact, because the key encoding is a bijection.
    """
    keys = np.ascontiguousarray(keys)
    enc, total_bits = order_keys(keys)
    if enc.size == 0:
        return keys.copy()
    sorted_keys, _ = sort_encoded(enc, bits, total_bits)
    return _decode(sorted_keys, keys.dtype)


def sort_pairs(keys: np.ndarray, values: np.ndarray, *,
               bits: int = DEFAULT_BITS) -> np.ndarray:
    """Sort `values` by `keys`, ties broken by original position."""
    keys = np.ascontiguousarray(keys)
    values = np.ascontiguousarray(values)
    if keys.size != values.size:
        raise ValueError("keys and values must be the same length")
    order, _ = argsort(keys, bits=bits)
    return values[order]


def digit_histogram(keys: np.ndarray, bits: int, shift: int) -> np.ndarray:
    """Counts of the `shift`-th `bits`-wide digit of pre-encoded `uint64` keys."""
    if not 1 <= bits <= 16:
        raise ValueError("bits must be in 1..16")
    enc, _ = order_keys(keys)
    hist = np.empty(1 << bits, dtype=np.uint64)
    _lib.lib.rs_histogram_pass(
        _lib.addr(enc), enc.size, shift, bits, _lib.addr(hist)
    )
    return hist


def bytes_histogram(data) -> np.ndarray:
    """256-bucket histogram of the raw bytes of `data`."""
    raw = np.frombuffer(data, dtype=np.uint8)
    raw = np.ascontiguousarray(raw)
    hist = np.empty(256, dtype=np.uint64)
    _lib.lib.rs_bytes_histogram(_lib.addr(raw), raw.size, _lib.addr(hist))
    return hist


def key_width(keys: np.ndarray) -> int:
    """Bit length of the OR of the encoded keys of `keys`."""
    enc, _ = order_keys(keys)
    return int(_lib.lib.rs_key_width(_lib.addr(enc), enc.size))

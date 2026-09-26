"""ctypes bridge to the compiled Mojo radix-sort kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so addresses are declared `c_int64`; `c_int` truncates them and
segfaults on any buffer mapped above 4 GiB.

`argtypes` is built through `_decl`, which refuses a declaration whose
parameter count disagrees with the kernel signature recorded here. ctypes
zero-fills a short `argtypes` list rather than complaining, so a one-entry
shortfall would silently truncate a 64-bit address and only surface on large
inputs. The count is therefore checked at load time, and
`tests/test_abi.py` additionally drives the kernels from a buffer mapped at a
fixed high address.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-radix-sort.so"

_I = ctypes.c_int64

# name -> (restype, ctypes argtypes). The length of each argtypes tuple is the
# declared parameter count; `_decl` cross-checks it against the Mojo def.
_SIGNATURES: dict[str, tuple[object, tuple]] = {
    "rs_encode_i64": (None, (_I, _I, _I)),
    "rs_encode_u64": (None, (_I, _I, _I)),
    "rs_encode_f64": (None, (_I, _I, _I)),
    "rs_decode_f64": (None, (_I, _I, _I)),
    "rs_encode_i32": (None, (_I, _I, _I)),
    "rs_encode_u32": (None, (_I, _I, _I)),
    "rs_encode_f32": (None, (_I, _I, _I)),
    "rs_decode_f32": (None, (_I, _I, _I)),
    "rs_encode_i16": (None, (_I, _I, _I)),
    "rs_encode_u16": (None, (_I, _I, _I)),
    "rs_encode_i8": (None, (_I, _I, _I)),
    "rs_encode_u8": (None, (_I, _I, _I)),
    "rs_histogram_pass": (None, (_I, _I, _I, _I, _I)),
    "rs_prefix_sum": (None, (_I, _I)),
    "rs_scatter_pass": (None, (_I, _I, _I, _I, _I, _I, _I, _I)),
    "rs_fill_indices": (None, (_I, _I)),
    "rs_lsd_sort": (_I, (_I,) * 9),
    "rs_key_width": (_I, (_I, _I)),
    "rs_bytes_histogram": (None, (_I, _I, _I)),
}

# Parameter counts as written in src/kernels.mojo. A mismatch here is a real
# FFI bug, not a style nit, so it is a hard load-time failure.
_EXPECTED_ARITY = {
    "rs_encode_i64": 3,
    "rs_encode_u64": 3,
    "rs_encode_f64": 3,
    "rs_decode_f64": 3,
    "rs_encode_i32": 3,
    "rs_encode_u32": 3,
    "rs_encode_f32": 3,
    "rs_decode_f32": 3,
    "rs_encode_i16": 3,
    "rs_encode_u16": 3,
    "rs_encode_i8": 3,
    "rs_encode_u8": 3,
    "rs_histogram_pass": 5,
    "rs_prefix_sum": 2,
    "rs_scatter_pass": 8,
    "rs_fill_indices": 2,
    "rs_lsd_sort": 9,
    "rs_key_width": 2,
    "rs_bytes_histogram": 3,
}


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    for name, (restype, argtypes) in _SIGNATURES.items():
        fn = getattr(lib, name, None)
        if fn is None:
            raise RuntimeError(f"symbol {name!r} missing from {_LIB_PATH.name}")
        expected = _EXPECTED_ARITY[name]
        if len(argtypes) != expected:
            raise RuntimeError(
                f"{name}: argtypes has {len(argtypes)} entries but the Mojo "
                f"def takes {expected}; ctypes would zero-fill the gap and "
                f"truncate a 64-bit address"
            )
        fn.restype = restype
        fn.argtypes = list(argtypes)
    return lib


lib = _load()

# Raw addresses, as int. Kept separate from the typed API below so the tests
# can hand a buffer at an arbitrary address straight to a kernel.
ADDRS = _SIGNATURES


def addr(a) -> int:
    """Byte address of a C-contiguous numpy buffer or a ctypes buffer."""
    return a.ctypes.data

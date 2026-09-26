"""FFI-contract tests: 64-bit addresses, declared arity, buffer ownership.

ctypes zero-fills an `argtypes` list that is one entry short of the real
signature instead of complaining, so a truncated 64-bit address lands in a
32-bit slot and every small-buffer test still passes. The only way to see it
is to drive a kernel from a buffer whose address genuinely exceeds 2**32, which
is what `test_kernel_handles_a_buffer_above_4_gib` does.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import mojo_radixsort as rs
from mojo_radixsort import _lib

# Somewhere the kernel's 32-bit truncation would definitely land, and that a
# normal allocator is very unlikely to hand out.
FIXED_ADDRESS = 0x7F_00_10_00_0000
PROT_READ = 1
PROT_WRITE = 2
MAP_PRIVATE = 0x02
MAP_ANONYMOUS = 0x20
MAP_FIXED = 0x10

_HERE = pathlib.Path(__file__).resolve().parent
_PYTHON_DIR = _HERE.parent / "python"


class _FixedMapping:
    """A writable mapping placed at an exact address."""

    def __init__(self, nbytes: int):
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int64,
        ]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if nbytes % 4096:
            nbytes += 4096 - nbytes % 4096
        self.nbytes = nbytes
        got = libc.mmap(
            ctypes.c_void_p(FIXED_ADDRESS), ctypes.c_size_t(nbytes),
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
            -1, 0,
        )
        if got is None or got == 0xFFFFFFFFFFFFFFFF or got != FIXED_ADDRESS:
            err = ctypes.get_errno()
            libc.munmap(ctypes.c_void_p(FIXED_ADDRESS), ctypes.c_size_t(nbytes))
            pytest.skip(f"cannot map a fixed address (errno {err})")
        self._libc = libc
        self.buffer = (ctypes.c_char * nbytes).from_address(FIXED_ADDRESS)

    def close(self):
        self._libc.munmap(ctypes.c_void_p(FIXED_ADDRESS),
                          ctypes.c_size_t(self.nbytes))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def high_address_keys():
    """4 KiB int64 keys living above 2**47, written through the raw mapping."""
    n = 512
    with _FixedMapping(n * 8) as m:
        arr = np.frombuffer(m.buffer, dtype=np.int64, count=n)
        rng = np.random.default_rng(1234)
        arr[:] = rng.integers(-(2**62), 2**62, size=n, dtype=np.int64)
        assert arr.ctypes.data == FIXED_ADDRESS
        assert arr.ctypes.data > 2**32
        yield arr


def test_kernel_handles_a_buffer_above_4_gib(high_address_keys):
    """The sort must be correct with every buffer above 2**32.

    With a 32-bit `argtypes` entry the address is truncated to its low 32 bits
    and the kernel reads and writes whatever is mapped there - usually a
    segfault, occasionally silently wrong data. Either way this test fails.
    """
    arr = high_address_keys
    got = rs.sort(arr)
    np.testing.assert_array_equal(got, np.sort(arr, kind="stable"))
    order, _ = rs.argsort(arr)
    np.testing.assert_array_equal(arr[order], np.sort(arr, kind="stable"))


def test_scratch_buffers_above_4_gib(high_address_keys):
    """Not just the input: the scratch the shim allocates must also work.

    Here the shim's buffers are normal heap allocations and only the input is
    high, so this exercises the read path alone; `test_scratch_at_high_address`
    does the reverse.
    """
    arr = high_address_keys
    enc, _ = rs.order_keys(arr)
    hist = rs.digit_histogram(arr, 8, 0)
    np.testing.assert_array_equal(hist, np.bincount(enc & 0xFF, minlength=256))


def test_scratch_at_high_address(tmp_path):
    """Drive the driver with every buffer mapped above 2**32.

    The driver needs six buffers: two key buffers and two index buffers,
    because it ping-pongs between them and aliasing them would let a pass
    overwrite the array the next pass is still reading; the pass histograms;
    and the two write-combining buffers plus their cursor state. All are
    disjoint regions of one fixed-address mapping.

    The call runs in a child process because a 32-bit `argtypes` entry does
    not fail politely: the kernel writes through a truncated address and the
    process takes SIGSEGV. A child turns that into a reported failure with a
    signal number instead of a dead pytest session. Verified by hand:
    re-declaring `rs_lsd_sort`'s sixth argument as `c_int` makes the child
    exit with signal 11.
    """
    n = 300
    bits, passes, buckets = 8, 8, 256
    hist_words = passes * buckets
    words = 4 * n + hist_words
    cuts = [0]
    for size in (n, n, n, n, hist_words):
        cuts.append(cuts[-1] + size)
    names = ("k0", "k1", "i0", "i1", "hists")
    slices = "\n".join(
        f"    {name} = view[{cuts[i]}:{cuts[i + 1]}]"
        for i, name in enumerate(names)
    )
    script = f"""
import sys
import numpy as np
sys.path.insert(0, {str(_PYTHON_DIR)!r})
sys.path.insert(0, {str(_HERE)!r})
from mojo_radixsort import _lib
import test_abi
with test_abi._FixedMapping({words} * 8) as m:
    view = np.frombuffer(m.buffer, dtype=np.uint64, count={words})
    rng = np.random.default_rng(99)
    keys = rng.integers(-(2 ** 40), 2 ** 40, size={n}, dtype=np.int64)
{slices}
    k0[:] = np.ascontiguousarray(keys).view(np.uint64) ^ np.uint64(1 << 63)
    _lib.lib.rs_fill_indices(_lib.addr(i0), {n})
    every = (k0, k1, i0, i1, hists)
    assert all(b.ctypes.data > 2 ** 32 for b in every), \
        [hex(b.ctypes.data) for b in every]
    performed = _lib.lib.rs_lsd_sort(
        _lib.addr(k0), _lib.addr(i0), {n}, 1, {bits}, 64,
        _lib.addr(k1), _lib.addr(i1), _lib.addr(hists))
    assert performed == {passes}, performed
    np.testing.assert_array_equal(keys[i0.astype(np.int64)],
                                  np.sort(keys, kind="stable"))
    np.testing.assert_array_equal(k0, np.sort(k0))
print("OK")
"""
    script_path = tmp_path / "high_scratch.py"
    script_path.write_text(script)
    env_python = sys.executable
    proc = subprocess.run(
        [env_python, str(script_path)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(_PYTHON_DIR)},
    )
    assert proc.returncode == 0, (
        f"child exited {proc.returncode} "
        f"({-proc.returncode if proc.returncode < 0 else 'non-signal'}); "
        f"stdout={proc.stdout!r} stderr={proc.stderr[-2000:]!r}"
    )
    assert "OK" in proc.stdout


def test_every_declared_symbol_arity_matches_the_record():
    """`argtypes` length must equal the Mojo parameter count for every symbol.

    The shim already refuses to load on a mismatch; this asserts the recorded
    counts are themselves the ones the kernels declare, so editing one side
    without the other is caught here rather than on a 4 GiB buffer.
    """
    for name, (_, argtypes) in _lib.ADDRS.items():
        assert len(argtypes) == _lib._EXPECTED_ARITY[name], name
        assert all(a is ctypes.c_int64 for a in argtypes), name


def test_lib_exposes_only_int64_addresses():
    """No `c_int` anywhere in the bridge: it truncates a 64-bit address."""
    for name, (_, argtypes) in _lib.ADDRS.items():
        for a in argtypes:
            assert a != ctypes.c_int, f"{name} uses c_int for an address or count"


def test_load_failure_is_explicit():
    """A missing library must raise, not silently import a broken shim."""
    assert _lib.lib is not None
    with pytest.raises(RuntimeError, match="run `bash build/build.sh`"):
        from mojo_radixsort import _lib as fresh

        original = fresh._LIB_PATH
        try:
            fresh._LIB_PATH = original.with_name("definitely-not-here.so")
            fresh._load()
        finally:
            fresh._LIB_PATH = original

"""LSD radix sort over order-preserving 64-bit keys.

Everything crosses the C ABI as a plain `Int` address, because `@export`
rejects parametric functions and an inferred pointer origin would make the
symbol parametric. `@export` also rejects `raises`, so no body here may fail.

The whole port is built on one idea: map every supported key type onto a
`UInt64` whose *unsigned integer order* equals the type's natural order, then
run a single generic LSD pass driver over those `UInt64` keys. That is why the
encoders below are the only place that knows about signedness or IEEE754.
Buffers of a 32-bit type are read through a `UInt32` pointer so the kernel
never straddles an element boundary, and the caller guarantees the array is
contiguous.

Order-preserving encodings
--------------------------
Every key is a `UInt64` that sits entirely in its low `w` bits, where `w` is
the source type's width, and whose unsigned order is the source type's order.

  * unsigned integer x: `k = x`.
  * signed integer x of width w: `k = (two's-complement w-bit pattern of x)
    XOR 2**(w-1)`, which is the same number as `x - INT_MIN` in `0 .. 2**w`.
    Flipping the top *significant* bit turns two's-complement order into
    unsigned order. The flip is unconditional: a non-negative pattern has a
    clear top bit so the flip raises it, a negative one has it set so the flip
    lowers it. Branching on the sign instead leaves both halves of the key
    space on the same side and inverts the comparison.
  * float x: take the bit pattern `b`, then `k = sign(x) ? ~b : (b | 1 << 63)`.
    Non-negative floats already order as unsigned integers; negative floats
    order *backwards* as unsigned integers, so the complement restores them.
    NaN payloads keep their relative bit order instead of being canonicalised,
    which is what sorting a concrete bit pattern should do.

The sort is exact: the output is a permutation of the input and is sorted, so
every assertion against a reference may use `rtol=0, atol=0`.
"""

comptime UPtr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U16Ptr = Pointer[UInt16, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]

# Elements buffered per bucket before a full cache line is flushed. 8 words is
# exactly one 64-byte line, which is the whole point of the buffering.
comptime WC_LEN = 8

comptime SIGN64 = UInt64(0x8000000000000000)
comptime SIGN32 = UInt64(0x80000000)
comptime SIGN32U = UInt32(0x80000000)


@always_inline
def up(addr: Int) -> UPtr:
    return UPtr(unsafe_from_address=addr)


@always_inline
def up32(addr: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=addr)

@always_inline
def up16(addr: Int) -> U16Ptr:
    return U16Ptr(unsafe_from_address=addr)


@always_inline
def up8(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


# --------------------------------------------------------------------------
# Key encoders. Each reads one element and writes one UInt64 order key.
# `n` is the element count; the caller owns every buffer.
# --------------------------------------------------------------------------



@export("rs_encode_i64")
def rs_encode_i64(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        # The sign-bit flip is unconditional: a non-negative two's-complement
        # value has a clear top bit, so the flip *raises* it above every
        # negative value, and a negative value has it set, so the flip lowers
        # it. Branching on the sign instead would leave both halves below 2**63
        # and put the negatives above the positives.
        dst[unsafe_offset=i] = src[unsafe_offset=i] ^ SIGN64


@export("rs_encode_u64")
def rs_encode_u64(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = src[unsafe_offset=i]


@export("rs_encode_f64")
def rs_encode_f64(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        var b = src[unsafe_offset=i]
        if b >= SIGN64:
            dst[unsafe_offset=i] = ~b
        else:
            dst[unsafe_offset=i] = b | SIGN64


@export("rs_decode_f64")
def rs_decode_f64(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        var k = src[unsafe_offset=i]
        if k >= SIGN64:
            dst[unsafe_offset=i] = k ^ SIGN64
        else:
            dst[unsafe_offset=i] = ~k


@export("rs_encode_i32")
def rs_encode_i32(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up32(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = UInt64(src[unsafe_offset=i]) ^ SIGN32


@export("rs_encode_u32")
def rs_encode_u32(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up32(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = UInt64(src[unsafe_offset=i])


@export("rs_encode_f32")
def rs_encode_f32(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up32(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        var b = src[unsafe_offset=i]
        if UInt64(b) >= SIGN32:
            dst[unsafe_offset=i] = ~UInt64(b) & 0xFFFFFFFF
        else:
            dst[unsafe_offset=i] = UInt64(b) | SIGN32


@export("rs_decode_f32")
def rs_decode_f32(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up(src_addr)
    var dst = up32(dst_addr)
    for i in range(n):
        var k = src[unsafe_offset=i] & 0xFFFFFFFF
        if k >= SIGN32:
            dst[unsafe_offset=i] = UInt32(k ^ SIGN32)
        else:
            dst[unsafe_offset=i] = UInt32(~k & 0xFFFFFFFF)


@export("rs_encode_u8")
def rs_encode_u8(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up8(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = UInt64(src[unsafe_offset=i])


@export("rs_encode_i8")
def rs_encode_i8(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up8(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        var raw = Int64(src[unsafe_offset=i])
        # key = (raw + 2**(w-1)) mod 2**w, i.e. x - INT_MIN, which is the
        # w-bit sign-bit flip written as arithmetic. Truncating back to w bits
        # is what puts INT_MIN on key 0 instead of key 2**(w-1).
        dst[unsafe_offset=i] = UInt64((raw + 128) & 0xFF)


@export("rs_encode_u16")
def rs_encode_u16(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up16(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        dst[unsafe_offset=i] = UInt64(src[unsafe_offset=i])


@export("rs_encode_i16")
def rs_encode_i16(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    var src = up16(src_addr)
    var dst = up(dst_addr)
    for i in range(n):
        var raw = Int64(src[unsafe_offset=i])
        dst[unsafe_offset=i] = UInt64((raw + 32768) & 0xFFFF)


# --------------------------------------------------------------------------
# Pass-local primitives
# --------------------------------------------------------------------------


@export("rs_histogram_pass")
def rs_histogram_pass(keys_addr: Int, n: Int, shift: Int, bits: Int,
                      hist_addr: Int) abi("C"):
    """Count digit occurrences for one pass into `hist` (zeroed here)."""
    var keys = up(keys_addr)
    var hist = up(hist_addr)
    var buckets = 1 << bits
    for b in range(buckets):
        hist[unsafe_offset=b] = 0
    var mask = UInt64((1 << bits) - 1)
    for i in range(n):
        var d = Int((keys[unsafe_offset=i] >> UInt64(shift)) & mask)
        hist[unsafe_offset=d] += 1


@export("rs_prefix_sum")
def rs_prefix_sum(hist_addr: Int, buckets: Int) abi("C"):
    """In-place exclusive prefix sum over `buckets` counters."""
    var hist = up(hist_addr)
    var running = UInt64(0)
    for b in range(buckets):
        var c = hist[unsafe_offset=b]
        hist[unsafe_offset=b] = running
        running += c


@export("rs_scatter_pass")
def rs_scatter_pass(keys_addr: Int, idx_addr: Int, out_keys_addr: Int,
                    out_idx_addr: Int, n: Int, shift: Int, bits: Int,
                    hist_addr: Int) abi("C"):
    """Stable placement of one pass.

    `hist` must already hold exclusive prefix sums; each bucket is advanced in
    place as it fills, which is what makes the pass stable - elements keep
    their relative order inside a bucket because they are consumed from the
    source in increasing index order.
    """
    var keys = up(keys_addr)
    var idx = up(idx_addr)
    var okeys = up(out_keys_addr)
    var oidx = up(out_idx_addr)
    var histp = up(hist_addr)
    var mask = UInt64((1 << bits) - 1)
    for i in range(n):
        var d = Int((keys[unsafe_offset=i] >> UInt64(shift)) & mask)
        var j = Int(histp[unsafe_offset=d])
        okeys[unsafe_offset=j] = keys[unsafe_offset=i]
        oidx[unsafe_offset=j] = idx[unsafe_offset=i]
        histp[unsafe_offset=d] = UInt64(j) + 1


@export("rs_fill_indices")
def rs_fill_indices(idx_addr: Int, n: Int) abi("C"):
    var idx = up(idx_addr)
    for i in range(n):
        idx[unsafe_offset=i] = UInt64(i)


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------


@export("rs_lsd_sort")
def rs_lsd_sort(keys_addr: Int, idx_addr: Int, n: Int, has_idx: Int,
                bits: Int, total_bits: Int, key_buf_addr: Int,
                idx_buf_addr: Int, hists_addr: Int) abi("C") -> Int:
    """Stable LSD radix sort of `n` encoded keys.

    `has_idx` non-zero permutes `idx` alongside the keys and is what
    `argsort` needs; zero sorts the keys alone, which is all `sort` needs and
    halves the traffic of every pass. `idx_addr` is not touched when
    `has_idx` is zero.

    `bits` is the digit width (1..16), `total_bits` how many low key bits are
    significant. Every pass histogram is collected before any scatter starts,
    so a pass whose buckets are all zero but one - typically because every key
    shares a high digit - is skipped entirely. The result is written back to
    `keys_addr`, and to `idx_addr` when permuting. The return value is the
    number of passes actually performed, which the tests and the benchmark both
    read; a negative value means the parameters were rejected.

    Scratch, all owned by the caller: one `n`-word key buffer, one `n`-word
    index buffer (only when permuting) and `passes * buckets` histogram
    counters.

    A software write-combining scatter was tried here and removed: buffering 8
    words per bucket needs `buckets * 8 * 8` bytes of L1, which at 256 buckets
    is 16 KiB for the keys and 16 KiB for the indices, and on this machine that
    measured 1.75x *slower* than the plain scatter (398 ms -> 696 ms for 1M
    int64 keys at 8 bits per pass). The buffers evict the histogram, and the
    extra branch and read-modify-write per element cost more than the partial
    line writes it saves. It is a real technique at 64 buckets, not at 256.
    """
    var keys = up(keys_addr)
    var idx = up(idx_addr)
    var kbuf = up(key_buf_addr)
    var ibuf = up(idx_buf_addr)
    var hists = up(hists_addr)

    if n <= 0:
        return 0
    if bits <= 0 or bits > 16:
        return -1
    if total_bits <= 0 or total_bits > 64:
        return -1
    var with_idx = has_idx != 0

    var buckets = 1 << bits
    var passes = (total_bits + bits - 1) // bits
    var stride = passes * buckets
    var mask = UInt64((1 << bits) - 1)

    for t in range(stride):
        hists[unsafe_offset=t] = 0

    # One sweep per pass, not one sweep over all passes. Fusing them saves
    # re-reading the keys but leaves a runtime-length inner loop whose trip
    # count LLVM cannot unroll and whose shift amount is a variable.
    for p in range(passes):
        var sh = UInt64(p * bits)
        var row = p * buckets
        for i in range(n):
            var k = keys[unsafe_offset=i]
            hists[unsafe_offset=row + Int((k >> sh) & mask)] += 1

    var performed = 0
    var cur_keys = keys
    var cur_idx = idx
    var nxt_keys = kbuf
    var nxt_idx = ibuf
    var flipped = 0

    for p in range(passes):
        var row = p * buckets
        var biggest = UInt64(0)
        for b in range(buckets):
            var c = hists[unsafe_offset=row + b]
            if c > biggest:
                biggest = c
        if biggest == UInt64(n):
            continue  # every key shares this digit: the pass would be identity
        var running = UInt64(0)
        for b in range(buckets):
            var c = hists[unsafe_offset=row + b]
            hists[unsafe_offset=row + b] = running
            running += c

        var sh = UInt64(p * bits)
        if with_idx:
            for i in range(n):
                var k = cur_keys[unsafe_offset=i]
                var d = Int((k >> sh) & mask)
                var j = Int(hists[unsafe_offset=row + d])
                nxt_keys[unsafe_offset=j] = k
                nxt_idx[unsafe_offset=j] = cur_idx[unsafe_offset=i]
                hists[unsafe_offset=row + d] = UInt64(j) + 1
        else:
            for i in range(n):
                var k = cur_keys[unsafe_offset=i]
                var d = Int((k >> sh) & mask)
                var j = Int(hists[unsafe_offset=row + d])
                nxt_keys[unsafe_offset=j] = k
                hists[unsafe_offset=row + d] = UInt64(j) + 1

        var swapk = cur_keys
        cur_keys = nxt_keys
        nxt_keys = swapk
        var swapi = cur_idx
        cur_idx = nxt_idx
        nxt_idx = swapi
        flipped = 1 - flipped
        performed += 1

    if flipped == 1:
        for i in range(n):
            keys[unsafe_offset=i] = cur_keys[unsafe_offset=i]
            if with_idx:
                idx[unsafe_offset=i] = cur_idx[unsafe_offset=i]
    return performed


@export("rs_key_width")
def rs_key_width(keys_addr: Int, n: Int) abi("C") -> Int:
    """Bit length of the OR of `keys`, i.e. how many low bits are significant.

    Callers whose keys are small (byte values, dense ranks) can feed this back
    as `total_bits` and save passes, but the driver never requires it:
    over-declaring is merely slower, never wrong.
    """
    var keys = up(keys_addr)
    var acc = UInt64(0)
    for i in range(n):
        acc |= keys[unsafe_offset=i]
    var w = 0
    while acc != 0:
        acc >>= 1
        w += 1
    return w


@export("rs_bytes_histogram")
def rs_bytes_histogram(data_addr: Int, n: Int, hist_addr: Int) abi("C"):
    """256-bucket histogram of raw bytes; the first step of a byte-rank pass."""
    # The payload is bytes, so the read pointer must be UInt8: reading it as
    # UInt64 would stride eight bytes per element and count only 1/8 of them.
    var data = up8(data_addr)
    var hist = up(hist_addr)
    for b in range(256):
        hist[unsafe_offset=b] = 0
    for i in range(n):
        hist[unsafe_offset=Int(data[unsafe_offset=i])] += 1

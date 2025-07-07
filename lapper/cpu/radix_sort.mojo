"""Most significant byte radix sort.

# Attribution:
- https://github.com/mzaks/mojo-sort/blob/main/radix_sorting/radix_sorting.mojo
"""
from memory import memset_zero, memcpy, stack_allocation
from memory.unsafe import bitcast


alias last_bit_8 = 1 << 7
alias last_bit_16 = 1 << 15
alias last_bit_32 = 1 << 31
alias last_bit_64 = 1 << 63


@always_inline
fn _get_index[
    D: DType, place: UInt
](vector: List[SIMD[D, 1]], v_index: UInt) -> UInt:
    # fmt: off
    @parameter
    if D == DType.int8:
        return UInt((bitcast[DType.uint8, 1](vector[v_index]) ^ last_bit_8) >> place) & 255
    elif D == DType.int16:
        return UInt((bitcast[DType.uint16, 1](vector[v_index]) ^ last_bit_16) >> place) & 255
    elif D == DType.float16:
        var f = bitcast[DType.uint16, 1](vector[v_index])
        var mask = bitcast[DType.uint16, 1](-bitcast[DType.int16, 1](f >> 15) | last_bit_16)
        return UInt((f ^ mask) >> place) & 255
    elif D == DType.int32:
        return UInt((bitcast[DType.uint32, 1](vector[v_index]) ^ last_bit_32) >> place) & 255
    elif D == DType.float32:
        var f = bitcast[DType.uint32, 1](vector[v_index])
        var mask = bitcast[DType.uint32, 1](-bitcast[DType.int32, 1](f >> 31) | last_bit_32)
        return UInt((f ^ mask) >> place) & 255
    elif D == DType.int64:
        return UInt((bitcast[DType.uint64, 1](vector[v_index]) ^ last_bit_64) >> place) & 255
    elif D == DType.float64:
        var f = bitcast[DType.uint64, 1](vector[v_index])
        var mask = bitcast[DType.uint64, 1](-bitcast[DType.int64, 1](f >> 63) | last_bit_64)
        return UInt((f ^ mask) >> place) & 255
    else:
        return UInt(vector[v_index] >> place) & 255
    # fmt: on


@always_inline
fn _counting_sort[
    D: DType, CD: DType, place: UInt
](mut vector: List[SIMD[D, 1]]):
    var size = len(vector)
    var output = List[SIMD[D, 1]](length=size, fill=0)

    var counts = stack_allocation[256, CD]()
    memset_zero(counts, 256)

    for i in range(size):
        var index = _get_index[D, place](vector, i)
        counts.offset(index).store(counts.offset(index).load() + 1)

    var count = counts.offset(0).load()
    for i in range(1, 256):
        count += counts.offset(i).load()
        counts.offset(i).store(count)

    var i = size - 1
    while i >= 0:
        var index = _get_index[D, place](vector, i)
        output[Int(counts.offset(index).load() - 1)] = vector[i]
        counts.offset(index).store(counts.offset(index).load() - 1)
        i -= 1
    vector = output


@always_inline
fn _radix_sort[D: DType, CD: DType](mut vector: List[SIMD[D, 1]]):
    constrained[D is not DType.invalid, "D must be a valid DType."]()

    @parameter
    fn call_counting_sort[index: Int]():
        _counting_sort[D, CD, index * 8](vector)

    # fmt: off
    alias unroll_factor: UInt = (
        1  if D.bitwidth() == 8 else
        2  if D.bitwidth() == 16 else
        4  if D.bitwidth() == 32 else
        8  if D.bitwidth() == 64 else
        16 if D.bitwidth() == 128 else
        32 if D.bitwidth() == 256 else
        32
    )
    # fmt: on

    @parameter
    for i in range(0, unroll_factor):
        call_counting_sort[i]()


@always_inline
fn radix_sort[D: DType](mut vector: List[SIMD[D, 1]]):
    constrained[D is not DType.invalid, "D must be a valid DType."]()
    _radix_sort[D, DType.uint32](vector)

    # NOTE: I hoped that the code below would make the algorithm faster but it made it slower
    # let size = len(vector)
    # alias m8 = max_or_inf[DType.uint8]().to_int()
    # alias m16 = max_or_inf[DType.uint16]().to_int()
    # alias m32 = max_or_inf[DType.uint32]().to_int()

    # if size <= m16:
    #     if size > m8:
    #         return _radix_sort[D, DType.uint16](vector)
    #     return _radix_sort[D, DType.uint8](vector)
    # if size <= m32:
    #     return _radix_sort[D, DType.uint32](vector)
    # return _radix_sort[D, DType.uint64](vector)


from lapper.lapper import Interval


@always_inline
fn radix_sort_ivs(mut vector: List[Interval]):
    @parameter
    fn call_counting_sort[index: Int]():
        _counting_sort_ivs[DType.uint32, index * 8](vector)

    @parameter
    for i in range(0, 4):  # 4 for DType.uint32
        call_counting_sort[i]()


@always_inline
fn _counting_sort_ivs[CD: DType, place: UInt](mut vector: List[Interval]):
    @parameter
    @always_inline
    fn get_index[place: UInt](v: UInt32) -> UInt:
        return UInt(v >> place) & 255

    var size = len(vector)
    var output = List[Interval](unsafe_uninit_length=size)

    var counts = stack_allocation[256, CD]()
    memset_zero(counts, 256)

    for i in range(size):
        var index = get_index[place](vector[i].start)
        counts.offset(index).store(counts.offset(index).load() + 1)

    var count = counts.offset(0).load()
    for i in range(1, 256):
        count += counts.offset(i).load()
        counts.offset(i).store(count)

    var i = size - 1
    while i >= 0:
        var index = get_index[place](vector[i].start)
        output[Int(counts.offset(index).load() - 1)] = vector[i]
        counts.offset(index).store(counts.offset(index).load() - 1)
        i -= 1
    vector = output

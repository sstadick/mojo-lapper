"""Insertion-point searches over spans sorted in ascending order.

`naive_bsearch`, `offset_bsearch`, and `lower_bound` return the first position
whose value is at least the target. `upper_bound` returns the first position
strictly greater than the target. An empty span returns zero; a target beyond
the end returns the span length.
"""

from std.bit import count_leading_zeros
from std.sys import bit_width_of


def naive_bsearch[dtype: DType](values: Span[Scalar[dtype], _], value: Scalar[dtype]) -> UInt:
    """Return the first index with an element >= value, or len(values).

    The input must be sorted in ascending order. Duplicate targets return
    the first occurrence; empty input returns zero.
    """
    if len(values) == 0:
        return 0

    if values[0] >= value:
        return 0

    var high = UInt32(len(values))
    var low = UInt32(0)

    while high - low > 1:
        var mid = (high + low) // 2
        if values[mid] < value:
            low = mid
        else:
            high = mid

    return UInt(high)


@inline(.always)
def _lpow2[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    """Return the largest power of two <= value; value must be positive."""
    return Scalar[dtype](1) << (
        Scalar[dtype](bit_width_of[dtype]()) - count_leading_zeros(value) - 1
    )


def offset_bsearch[dtype: DType](values: Span[Scalar[dtype], _], value: Scalar[dtype]) -> UInt:
    """Return the first index with an element >= value, or len(values).

    The input must be sorted in ascending order. Duplicate targets return
    the first occurrence; empty input returns zero.
    """
    if len(values) == 0 or values[0] >= value:
        return 0

    var step = _lpow2(UInt32(len(values)))
    var offset = UInt32(len(values))
    while step > 0:
        if step <= offset:
            var idx = offset - step
            if values[idx] >= value:
                offset = idx
        step >>= 1
    return UInt(offset)


def lower_bound[dtype: DType](values: Span[Scalar[dtype], _], value: Scalar[dtype]) -> UInt:
    """Return the first index with an element >= value, or len(values).

    The input must be sorted in ascending order. Duplicate targets return
    the first occurrence; empty input returns zero.
    """
    # alias FETCH_OPTS = PrefetchOptions().for_read().high_locality().to_data_cache()

    # Handle edge cases
    if len(values) == 0 or values[0] >= value:
        return 0
    elif values[len(values) - 1] < value:
        return UInt(len(values))

    var cursor = UInt32(0)
    var length = UInt32(len(values))
    while length > 1:
        var half = length >> 1
        length -= half
        # Prefetch does not seem to help on M3
        # prefetch[FETCH_OPTS](
        #     values.unsafe_ptr().offset(cursor + (length >> 1) - 1)
        # )
        # prefetch[FETCH_OPTS](
        #     values.unsafe_ptr().offset(cursor + half + (length >> 1) - 1)
        # )

        # Should be equivalent
        # cursor += UInt(half) if values[cursor + half - 1] < value else 0
        cursor += UInt32(values[cursor + half - 1] < value) * half

    return UInt(cursor)


def upper_bound[dtype: DType](values: Span[Scalar[dtype], _], key: Scalar[dtype]) -> UInt:
    """Return the first index with an element > key, or len(values).

    The input must be sorted in ascending order. Empty input returns zero.
    """
    var left = 0
    var right = len(values)

    while left < right:
        # 0 <= left <= mid < right <= len(values). Use the proven bound
        # rather than rebuilding a checked Span access at every probe.
        var mid = left + ((right - left) >> 1)
        if values.unsafe_ptr()[unsafe_offset=mid] <= key:
            left = mid + 1
        else:
            right = mid

    return UInt(left)

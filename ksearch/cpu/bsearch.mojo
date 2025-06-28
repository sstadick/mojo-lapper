from bit import count_leading_zeros
from sys.intrinsics import prefetch

from ExtraMojo.math.ops import saturating_sub


fn naive_bsearch[
    dtype: DType
](values: Span[Scalar[dtype]], value: Scalar[dtype]) -> UInt:
    """Naive binary search implementation.

    Returns the index after the rightmost element that is less than the value.
    This is equivalent to the insertion point for the value to maintain sorted order.

    Args:
        values: A sorted span of values to search in.
        value: The value to search for.

    Returns:
        The index after the rightmost element < value, or 0 if all elements > value.
    """
    constrained[dtype is not DType.invalid, "dtype must be valid."]()

    if len(values) == 0:
        return 0

    if values[0] >= value:
        return 0

    var high = UInt(len(values))
    var low = UInt(0)

    while high - low > 1:
        var mid = (high + low) // 2
        if values[Int(mid)] < value:
            low = mid
        else:
            high = mid

    return high


@always_inline
fn _lpow2[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    """Find the largest power of 2 that is not greater than `value`.

    Args:
        value: The value to find find the largest power of two for.

    Returns:
        The largest power of two < value
    """
    constrained[dtype is not DType.invalid, "dtype must be valid."]()

    return 1 << (dtype.bitwidth() - count_leading_zeros(value) - 1)


fn offset_bsearch[
    dtype: DType
](values: Span[Scalar[dtype]], value: Scalar[dtype]) -> UInt:
    """
    Offset binary search implementation using power-of-two stepping.

    Finds the index of the first element in `values` that is greater than or equal to `value`.
    Returns `len(values)` if `value` is greater than all elements.

    Args:
        values: A sorted span of values in ascending order.
        value: The value to search for.

    Returns:
        The index of the first element ≥ `value`, or `len(values)` if all elements are < `value`.
    """
    constrained[dtype is not DType.invalid, "dtype must be valid."]()

    if len(values) == 0 or values[0] >= value:
        return 0

    var step = _lpow2(UInt64(len(values)))
    var offset = UInt64(len(values))
    while step > 0:
        if step <= offset:
            var idx = offset - step
            if values[idx] >= value:
                offset = idx
        step >>= 1
    return UInt(offset)


fn branchless_offset_bsearch[
    dtype: DType
](values: Span[Scalar[dtype]], value: Scalar[dtype]) -> UInt:
    """
    Offset binary search implementation using branchless power-of-two stepping.

    Finds the index of the first element in `values` that is greater than or equal to `value`.
    Returns `len(values)` if `value` is greater than all elements.

    Args:
        values: A sorted span of values in ascending order.
        value: The value to search for.

    Returns:
        The index of the first element ≥ `value`, or `len(values)` if all elements are < `value`.
    """
    constrained[dtype is not DType.invalid, "dtype must be valid."]()

    # if len(values) == 0 or values[0] >= value:
    #     return 0

    # var offset = UInt64(len(values))
    # var step = _lpow2(UInt64(len(values)))

    # while step > 0:
    #     var idx = offset - step
    #     var cmp = UInt64(Int(step <= offset and values[idx] >= value))
    #     offset = cmp * idx + ((1 - cmp) * offset)
    #     step >>= 1
    # return UInt(offset)
    var cursor = UInt(0)
    var length = len(values)
    while length > 1:
        var half = length >> 1
        length -= half
        # prefetch(values.unsafe_ptr().offset((length >> 1) - 1))
        # prefetch(values.unsafe_ptr().offset(half - 1 + (length >> 1) - 1))
        cursor += Int(values[cursor + half - 1] < value) * half

    return cursor

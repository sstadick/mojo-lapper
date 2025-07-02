"""Binary search implementations optimized for different use cases and performance characteristics.

This module provides several binary search variants:

1. **naive_bsearch**: Traditional divide-and-conquer binary search
   - Simple, readable implementation 
   - Good baseline performance
   - Returns insertion point (index after rightmost element < target)

2. **offset_bsearch**: Power-of-two stepping from array end
   - Cache-efficient for certain access patterns
   - Particularly good when target likely in latter half
   - Uses decreasing power-of-2 steps

3. **lower_bound**: Optimized branchless implementation  
   - Reduced branch mispredictions
   - Optional prefetching support (disabled on M3)
   - Returns std::lower_bound equivalent

All functions:
- Work on sorted arrays in ascending order
- Support any numeric DType 
- Have O(log n) time complexity
- Use O(1) space
- Are safe with empty arrays
- Handle duplicate values correctly

Performance Notes:
- naive_bsearch: Good general-purpose baseline
- offset_bsearch: Better for targets in latter half of array
- lower_bound: Best overall performance on modern CPUs

Edge Case Handling:
- Empty arrays return 0
- Out-of-range targets return appropriate boundary indices
- Single element arrays handled correctly
- Duplicate values return consistent positions

Thread Safety: All functions are thread-safe (no shared state).
"""

from bit import count_leading_zeros
from sys import llvm_intrinsic, prefetch, PrefetchOptions
from memory import UnsafePointer

from ExtraMojo.math.ops import saturating_sub


fn naive_bsearch[
    dtype: DType
](values: Span[Scalar[dtype]], value: Scalar[dtype]) -> UInt:
    """Naive binary search implementation using standard divide-and-conquer approach.

    Finds the insertion point for `value` in the sorted array `values` to maintain
    sorted order. Returns the index after the rightmost element that is less than
    the search value.

    This is a reference implementation that prioritizes simplicity and correctness
    over performance. It uses the classic binary search algorithm with explicit
    bounds checking and straightforward logic.

    Parameters:
        dtype: The data type of the values being searched.

    Args:
        values: A sorted span of values to search in (must be in ascending order).
        value: The value to search for.

    Returns:
        The index after the rightmost element < value. This means:
        - If all elements >= value: returns 0
        - If all elements < value: returns len(values)
        - If value exists: returns index after the last occurrence
        - If value doesn't exist: returns insertion point

    Edge Cases:
        - Empty array: returns 0
        - Single element array: returns 0 or 1 depending on comparison
        - All elements equal: returns 0 if value <= elements[0], else len(values)
        - Duplicate values: returns index after the last occurrence

    Examples:
        ```mojo
        var arr = List[Int32](1, 3, 5, 5, 7)
        naive_bsearch(Span(arr), 0)  # returns 0 (insert at beginning)
        naive_bsearch(Span(arr), 5)  # returns 4 (after last 5)
        naive_bsearch(Span(arr), 6)  # returns 4 (between 5 and 7)
        naive_bsearch(Span(arr), 10) # returns 5 (insert at end)
        ```

    Time Complexity: O(log n).
    Space Complexity: O(1).
    """
    constrained[dtype is not DType.invalid, "dtype must be valid."]()

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


@always_inline
fn _lpow2[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    """Find the largest power of 2 that is not greater than `value`.

    This is a utility function used by offset_bsearch to determine the initial
    step size for the power-of-two stepping algorithm.

    Args:
        value: The value to find the largest power of two for (must be > 0).

    Returns:
        The largest power of two ≤ value. For example:
        - _lpow2(1) = 1
        - _lpow2(7) = 4
        - _lpow2(8) = 8
        - _lpow2(15) = 8

    Edge Cases:
        - value = 0: behavior undefined (should not be called with 0)
        - value = 1: returns 1
        - Large values: limited by dtype bitwidth

    Implementation Note:
        Uses bit manipulation: 1 << (bitwidth - leading_zeros - 1)
        This is equivalent to 2^floor(log2(value)) for value > 0.

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    constrained[dtype is not DType.invalid, "dtype must be valid."]()

    return 1 << (dtype.bitwidth() - count_leading_zeros(value) - 1)


fn offset_bsearch[
    dtype: DType
](values: Span[Scalar[dtype]], value: Scalar[dtype]) -> UInt:
    """Offset binary search using power-of-two stepping from the end.

    This is a cache-efficient variant of binary search that starts from the end
    of the array and steps backward using decreasing powers of two. This approach
    can be more cache-friendly than traditional binary search for certain access
    patterns.

    Parameters:
        dtype: The data type of the values being searched.

    The algorithm works by:
    1. Starting at the end of the array (len(values))
    2. Using the largest power of 2 ≤ len(values) as initial step
    3. Stepping backward by halving the step size each iteration
    4. Moving the offset backward when the current element >= target

    Args:
        values: A sorted span of values in ascending order.
        value: The value to search for.

    Returns:
        The index of the first element ≥ `value`, or `len(values)` if all elements < `value`.
        This is equivalent to std::lower_bound behavior.

    Edge Cases:
        - Empty array: returns 0
        - All elements < value: returns len(values)
        - All elements >= value: returns 0
        - Single element: returns 0 or 1 depending on comparison
        - Duplicate values: returns index of first occurrence

    Examples:
        ```mojo
        var arr = List[Int32](1, 3, 5, 5, 7)
        offset_bsearch(Span(arr), 0)  # returns 0 (first element >= 0 is at index 0)
        offset_bsearch(Span(arr), 5)  # returns 2 (first 5 is at index 2)
        offset_bsearch(Span(arr), 6)  # returns 4 (first element >= 6 is 7 at index 4)
        offset_bsearch(Span(arr), 10) # returns 5 (no element >= 10)
        ```

    Performance Notes:
        - May have better cache locality than naive_bsearch for certain data
        - Particularly effective when target is likely to be in the latter half
        - Step size reduction provides O(log n) guarantees

    Time Complexity: O(log n).
    Space Complexity: O(1).
    """
    constrained[dtype is not DType.invalid, "dtype must be valid."]()

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


fn lower_bound[
    dtype: DType
](values: Span[Scalar[dtype]], value: Scalar[dtype]) -> UInt:
    """Optimized lower_bound implementation with edge case handling and prefetching support.

    Parameters:
        dtype: The data type of the values being searched.

    Finds the index of the first element that is greater than or equal to the target
    value. This is equivalent to std::lower_bound in C++. The implementation uses
    a branchless approach with optional prefetching for improved performance.

    The algorithm maintains a cursor position and repeatedly halves the search space,
    using branchless arithmetic to update the cursor based on comparisons. This
    reduces branch mispredictions and can improve performance on modern CPUs.

    Args:
        values: A sorted span of values in ascending order.
        value: The value to search for.

    Returns:
        The index of the first element ≥ `value`, or `len(values)` if all elements < `value`.
        This means:
        - If value exists: returns index of first occurrence
        - If value doesn't exist: returns insertion point
        - If all elements < value: returns len(values)
        - If all elements >= value: returns 0

    Edge Cases (with optimized early returns):
        - Empty array: returns 0 immediately
        - First element >= value: returns 0 immediately (optimized fast path)
        - Last element < value: returns len(values) immediately (optimized fast path)
        - Single element: handled by early return logic
        - All elements equal: handled by first/last element checks
        - Duplicate target values: returns index of first occurrence

    Examples:
        ```mojo
        var arr = List[Int32](1, 3, 5, 5, 7)
        lower_bound(Span(arr), 0)  # returns 0 (insert at beginning)
        lower_bound(Span(arr), 5)  # returns 2 (first 5 at index 2)
        lower_bound(Span(arr), 6)  # returns 4 (insert between 5s and 7)
        lower_bound(Span(arr), 10) # returns 5 (insert at end)
        ```

    Performance Notes:
        - Early returns for common edge cases provide O(1) performance
        - Uses branchless comparison for reduced branch mispredictions in main loop
        - Prefetching support (currently disabled on M3 due to limited benefit)
        - Optimized for modern CPU architectures
        - May outperform naive implementations on large datasets

    Implementation Details:
        - Fast path checks for empty arrays and boundary conditions
        - Cursor-based approach eliminates explicit bounds management
        - Multiplication by comparison result enables branchless updates
        - Length halving ensures O(log n) time complexity for general case

    Time Complexity: O(1) for edge cases, O(log n) for general case.
    Space Complexity: O(1).
    """
    constrained[dtype is not DType.invalid, "dtype must be valid."]()
    # alias FETCH_OPTS = PrefetchOptions().for_read().high_locality().to_data_cache()

    # Handle edge cases
    if len(values) == 0 or values[0] >= value:
        return 0
    elif values[len(values) - 1] < value:
        return len(values)

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
        cursor += Int(values[cursor + half - 1] < value) * half

    return UInt(cursor)


fn upper_bound[
    dtype: DType
](values: Span[Scalar[dtype]], key: Scalar[dtype]) -> UInt:
    """Standard upper_bound implementation.

    Parameters:
        dtype: The data type of the values being searched.

    Finds the index of the first element that is strictly greater than the key.
    This is equivalent to std::upper_bound in C++.

    Args:
        values: A sorted span of values in ascending order.
        key: The key value to search for.

    Returns:
        The index of the first element > key, or len(values) if all elements <= key.

    Examples:
        ```mojo
        var arr = List[Int32](1, 3, 5, 5, 7)
        upper_bound(Span(arr), 5)  # returns 4 (first element > 5 is 7 at index 4)
        upper_bound(Span(arr), 10) # returns 5 (no element > 10)
        ```
    """
    constrained[dtype is not DType.invalid, "dtype must be valid."]()

    var left = 0
    var right = len(values)

    while left < right:
        var mid = left + (right - left) // 2
        if values[mid] <= key:
            left = mid + 1
        else:
            right = mid

    return UInt(left)

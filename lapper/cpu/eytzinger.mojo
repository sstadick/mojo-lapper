"""Reusable Eytzinger search trees with lookup into the input's sorted order.

Search results index `layout`, not the original input. Check that a result
is nonzero and in bounds before reading `layout` or `lookup`.
"""

from std.bit import count_trailing_zeros, count_leading_zeros
from std.collections import Deque


@fieldwise_init
struct Eytzinger[dtype: DType](Copyable, Movable):
    """Own a search layout and its mapping to sorted input indices.

    Nonempty layouts use one-based indices; index zero is a sentinel, not
    a search result. Empty input produces two empty lists.
    """

    var layout: List[Scalar[Self.dtype]]
    """Searchable values in Eytzinger order, with a sentinel at index zero."""

    var lookup: List[Scalar[Self.dtype]]
    """Map a valid layout index to its zero-based sorted input index."""


# TODO, allow this to take mutable spans for the layout and lookup so it can write directly to hostbuffers
def eytzinger_with_lookup[
    dtype: DType
](original: Span[Scalar[dtype], _]) raises -> Eytzinger[dtype]:
    """Copy an ascending sorted span into an owned Eytzinger tree.

    Empty input returns empty lists. Otherwise each output list has
    len(original) + 1 elements, including its sentinel at index zero.

    Raises:
        Propagates errors encountered while constructing the result.

    ```mojo {doctest="eytzinger" global=true}
    from lapper.cpu.eytzinger import eytzinger_with_lookup, lower_bound
    from std.testing import assert_equal, assert_true

    def main() raises:
        var values: List[Int32] = [1, 3, 5, 7, 9, 11, 13]
        var tree = eytzinger_with_lookup(Span(values))
        var found = Int(lower_bound(Span(tree.layout), Int32(6)))
        assert_true(found > 0 and found < len(tree.layout))
        assert_equal(tree.layout[found], Int32(7))
        assert_equal(tree.lookup[found], Int32(3))
    ```
    """
    if len(original) == 0:
        var empty_output = List[Scalar[dtype]]()
        var empty_lookup = List[Scalar[dtype]]()
        return Eytzinger(empty_output^, empty_lookup^)

    # Initialize arrays with proper sentinel values at index 0
    var output = List[Scalar[dtype]](unsafe_uninit_length=len(original) + 1)
    var lookup = List[Scalar[dtype]](unsafe_uninit_length=len(original) + 1)

    # Set sentinel values at index 0
    # Both arrays use minimum value as sentinel for consistency
    # output[0] gets minimum value to ensure it's <= any search target
    # lookup[0] gets minimum value as invalid index sentinel (negative for signed, 0 for unsigned)
    comptime min_val = Scalar[dtype].MIN if dtype.is_signed() else Scalar[dtype](0)
    output[0] = min_val
    lookup[0] = min_val

    var stack = Deque[Int]()
    var index = 0

    # Start with root node
    stack.append(1)

    while len(stack) > 0:
        var k = stack.pop()

        if k < 0:  # Process this node
            k = -k
            output[k] = original[index]
            lookup[k] = Scalar[dtype](index)
            index += 1
        elif k <= len(original):
            # Push right child first
            if 2 * k + 1 <= len(original):
                stack.append(2 * k + 1)

            # Push current node for processing
            stack.append(-k)

            # Push left child
            if 2 * k <= len(original):
                stack.append(2 * k)

    return Eytzinger(output^, lookup^)


def lower_bound[dtype: DType](values: Span[Scalar[dtype], _], value: Scalar[dtype]) -> UInt:
    """Search a nonempty Eytzinger layout for the first element >= value.

    Use a layout returned by `eytzinger_with_lookup`, including its sentinel.
    The result is a layout index, not a sorted input index. Zero means no
    match beyond the end. Check `0 < result < len(values)` before indexing;
    use the tree's lookup table to recover a sorted input index.
    Do not pass an empty layout.
    """
    # alias FETCH_OPTS = PrefetchOptions().for_read().high_locality().to_data_cache()

    # Use fixed-iteration version like C++ for better edge case handling
    var iters = Int(32 - count_leading_zeros(UInt32(len(values) + 1)) - 1)
    var k: UInt32 = 1

    # Execute fixed number of iterations
    for _ in range(iters):
        # prefetch not doing anything
        # prefetch[FETCH_OPTS](values.unsafe_ptr().offset(k * 16))
        k = 2 * k + UInt32(values[k] < value)

    # Final comparison with predication (handle edge cases)
    # Compare against actual data size (len - 1), not total array size
    var n = len(values) - 1  # Actual number of elements (excluding sentinel)
    var loc = values[k] if Int(k) <= n else values[0]
    k = 2 * k + UInt32(loc < value)

    # TODO: add prefetch, gate as gpu only

    # Restore actual index using exact C++ bit manipulation
    var ffs_result = count_trailing_zeros(~k) + 1
    k >>= ffs_result
    return UInt(k)

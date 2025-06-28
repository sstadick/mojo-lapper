from testing import assert_equal
from ksearch.cpu.bsearch import (
    branchless_offset_bsearch,
    naive_bsearch,
    offset_bsearch,
    _lpow2,
)


############################
# Naive Binary
############################


def test_naive_bsearch_simple():
    """Test basic binary search functionality."""
    var int_values = List[Int32](1, 3, 5, 7, 9, 11, 13)
    var int_span = Span(int_values)

    var result = naive_bsearch(int_span, 5)
    assert_equal(result, 2)

    result = naive_bsearch(int_span, 1)
    assert_equal(result, 0)

    result = naive_bsearch(int_span, 13)
    assert_equal(result, 6)

    # Test values between elements
    result = naive_bsearch(int_span, 6)
    assert_equal(result, 3)

    result = naive_bsearch(int_span, 4)
    assert_equal(result, 2)

    # Test value smaller than all elements
    result = naive_bsearch(int_span, 0)
    assert_equal(result, 0)

    # Test value larger than all elements
    result = naive_bsearch(int_span, 15)
    assert_equal(result, 7)


def test_naive_bsearch_floats():
    """Test binary search with floating point values."""
    # Test with floats
    var float_values = List[Float32](1.5, 2.5, 3.5, 4.5, 5.5)
    var float_span = Span(float_values)

    # Test finding existing values - returns insertion point after the value
    var result = naive_bsearch(float_span, 3.5)
    assert_equal(result, 2)

    # Test value between elements
    result = naive_bsearch(float_span, 3.0)
    assert_equal(result, 2)


def test_naive_bsearch_edge_cases():
    """Test edge cases for binary search."""
    # Empty list
    var empty_values = List[Int32]()
    var empty_span = Span(empty_values)
    var result = naive_bsearch(empty_span, 5)
    assert_equal(result, 0)

    # Single element - found
    var single_values = List[Int32](42)
    var single_span = Span(single_values)
    result = naive_bsearch(single_span, 42)
    assert_equal(result, 0)
    # Single element - value smaller than element
    result = naive_bsearch(single_span, 10)
    assert_equal(result, 0)

    # Single element - value larger than element
    result = naive_bsearch(single_span, 50)
    assert_equal(result, 1)


############################
# Offset Binary
############################


def test_offset_bsearch_simple():
    """Test basic binary search functionality."""
    var int_values = List[Int32](1, 3, 5, 7, 9, 11, 13)
    var int_span = Span(int_values)

    var result = offset_bsearch(int_span, 5)
    assert_equal(result, 2)

    result = offset_bsearch(int_span, 1)
    assert_equal(result, 0)

    result = offset_bsearch(int_span, 13)
    assert_equal(result, 6)

    # Test values between elements
    result = offset_bsearch(int_span, 6)
    assert_equal(result, 3)

    result = offset_bsearch(int_span, 4)
    assert_equal(result, 2)

    # Test value smaller than all elements
    result = offset_bsearch(int_span, 0)
    assert_equal(result, 0)

    # Test value larger than all elements
    result = offset_bsearch(int_span, 15)
    assert_equal(result, 7)


def test_offset_bsearch_floats():
    """Test binary search with floating point values."""
    # Test with floats
    var float_values = List[Float32](1.5, 2.5, 3.5, 4.5, 5.5)
    var float_span = Span(float_values)

    # Test finding existing values - returns insertion point after the value
    var result = offset_bsearch(float_span, 3.5)
    assert_equal(result, 2)

    # Test value between elements
    result = offset_bsearch(float_span, 3.0)
    assert_equal(result, 2)


def test_offset_bsearch_edge_cases():
    """Test edge cases for binary search."""
    # Empty list
    var empty_values = List[Int32]()
    var empty_span = Span(empty_values)
    var result = offset_bsearch(empty_span, 5)
    assert_equal(result, 0)

    # Single element - found
    var single_values = List[Int32](42)
    var single_span = Span(single_values)
    result = offset_bsearch(single_span, 42)
    assert_equal(result, 0)
    # Single element - value smaller than element
    result = offset_bsearch(single_span, 10)
    assert_equal(result, 0)

    # Single element - value larger than element
    result = offset_bsearch(single_span, 50)
    assert_equal(result, 1)


############################
# Branchless Offset Binary
############################


def test_branchless_offset_bsearch_simple():
    """Test basic binary search functionality."""
    var int_values = List[Int32](1, 3, 5, 7, 9, 11, 13)
    var int_span = Span(int_values)

    var result = branchless_offset_bsearch(int_span, 5)
    assert_equal(result, 2)

    result = branchless_offset_bsearch(int_span, 1)
    assert_equal(result, 0)

    result = branchless_offset_bsearch(int_span, 13)
    assert_equal(result, 6)

    # Test values between elements
    result = branchless_offset_bsearch(int_span, 6)
    assert_equal(result, 3)

    result = branchless_offset_bsearch(int_span, 4)
    assert_equal(result, 2)

    # Test value smaller than all elements
    result = branchless_offset_bsearch(int_span, 0)
    assert_equal(result, 0)

    # Test value larger than all elements
    result = branchless_offset_bsearch(int_span, 15)
    assert_equal(result, 7)


def test_branchless_offset_bsearch_floats():
    """Test binary search with floating point values."""
    # Test with floats
    var float_values = List[Float32](1.5, 2.5, 3.5, 4.5, 5.5)
    var float_span = Span(float_values)

    # Test finding existing values - returns insertion point after the value
    var result = branchless_offset_bsearch(float_span, 3.5)
    assert_equal(result, 2)

    # Test value between elements
    result = branchless_offset_bsearch(float_span, 3.0)
    assert_equal(result, 2)


def test_branchless_offset_bsearch_edge_cases():
    """Test edge cases for binary search."""
    # Empty list
    var empty_values = List[Int32]()
    var empty_span = Span(empty_values)
    var result = branchless_offset_bsearch(empty_span, 5)
    assert_equal(result, 0)

    # Single element - found
    var single_values = List[Int32](42)
    var single_span = Span(single_values)
    result = branchless_offset_bsearch(single_span, 42)
    assert_equal(result, 0)
    # Single element - value smaller than element
    result = branchless_offset_bsearch(single_span, 10)
    assert_equal(result, 0)

    # Single element - value larger than element
    result = branchless_offset_bsearch(single_span, 50)
    assert_equal(result, 1)


def test_lpow2_simple():
    var value = UInt64(100)
    var output = _lpow2(value)
    assert_equal(output, 64)


def main():
    test_naive_bsearch_simple()
    test_naive_bsearch_floats()
    test_naive_bsearch_edge_cases()
    test_offset_bsearch_simple()
    test_offset_bsearch_floats()
    test_offset_bsearch_edge_cases()
    test_branchless_offset_bsearch_simple()
    test_branchless_offset_bsearch_floats()
    test_branchless_offset_bsearch_edge_cases()
    test_lpow2_simple()
    print("All binary search tests passed!")

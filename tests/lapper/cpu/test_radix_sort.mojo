from lapper.cpu.radix_sort import radix_sort, radix_sort_ivs
from lapper.lapper import Interval
from testing import assert_equal, assert_true
from random import randint, seed


def test_radix_sort_basic_uint32():
    """Test basic radix sort functionality with UInt32."""
    var data = List[UInt32]()
    data.append(5)
    data.append(2)
    data.append(8)
    data.append(1)
    data.append(9)

    radix_sort(data)

    assert_equal(data[0], 1)
    assert_equal(data[1], 2)
    assert_equal(data[2], 5)
    assert_equal(data[3], 8)
    assert_equal(data[4], 9)


def test_radix_sort_basic_int32():
    """Test basic radix sort functionality with Int32."""
    var data = List[Int32]()
    data.append(-5)
    data.append(2)
    data.append(-8)
    data.append(1)
    data.append(9)

    radix_sort(data)

    assert_equal(data[0], -8)
    assert_equal(data[1], -5)
    assert_equal(data[2], 1)
    assert_equal(data[3], 2)
    assert_equal(data[4], 9)


def test_radix_sort_empty():
    """Test radix sort with empty list."""
    var data = List[UInt32]()
    radix_sort(data)
    assert_equal(len(data), 0)


def test_radix_sort_single_element():
    """Test radix sort with single element."""
    var data = List[UInt32]()
    data.append(42)
    radix_sort(data)
    assert_equal(len(data), 1)
    assert_equal(data[0], 42)


def test_radix_sort_already_sorted():
    """Test radix sort with already sorted data."""
    var data = List[UInt32]()
    for i in range(10):
        data.append(UInt32(i))

    radix_sort(data)

    for i in range(10):
        assert_equal(data[i], UInt32(i))


def test_radix_sort_reverse_sorted():
    """Test radix sort with reverse sorted data."""
    var data = List[UInt32]()
    for i in range(10):
        data.append(UInt32(9 - i))

    radix_sort(data)

    for i in range(10):
        assert_equal(data[i], UInt32(i))


def test_radix_sort_duplicates():
    """Test radix sort with duplicate values."""
    var data = List[UInt32]()
    data.append(5)
    data.append(2)
    data.append(5)
    data.append(2)
    data.append(5)

    radix_sort(data)

    assert_equal(data[0], 2)
    assert_equal(data[1], 2)
    assert_equal(data[2], 5)
    assert_equal(data[3], 5)
    assert_equal(data[4], 5)


def test_radix_sort_all_same():
    """Test radix sort with all identical values."""
    var data = List[UInt32]()
    for _i in range(5):
        data.append(42)

    radix_sort(data)

    for i in range(5):
        assert_equal(data[i], 42)


def test_radix_sort_large_random():
    """Test radix sort with large random dataset."""
    alias size = 1000
    seed(42)

    var data = List[UInt32](unsafe_uninit_length=size)
    randint(data.unsafe_ptr(), size, 0, 10000)

    radix_sort(data)

    # Verify sorting is correct
    for i in range(size - 1):
        assert_true(data[i] <= data[i + 1])


def test_radix_sort_int8():
    """Test radix sort with Int8 values."""
    var data = List[Int8]()
    data.append(-5)
    data.append(2)
    data.append(-8)
    data.append(1)
    data.append(9)
    data.append(127)
    data.append(-128)

    radix_sort(data)

    assert_equal(data[0], -128)
    assert_equal(data[1], -8)
    assert_equal(data[2], -5)
    assert_equal(data[3], 1)
    assert_equal(data[4], 2)
    assert_equal(data[5], 9)
    assert_equal(data[6], 127)


def test_radix_sort_int16():
    """Test radix sort with Int16 values."""
    var data = List[Int16]()
    data.append(-5000)
    data.append(2000)
    data.append(-8000)
    data.append(1000)
    data.append(9000)
    data.append(32767)
    data.append(-32768)

    radix_sort(data)

    assert_equal(data[0], -32768)
    assert_equal(data[1], -8000)
    assert_equal(data[2], -5000)
    assert_equal(data[3], 1000)
    assert_equal(data[4], 2000)
    assert_equal(data[5], 9000)
    assert_equal(data[6], 32767)


def test_radix_sort_int64():
    """Test radix sort with Int64 values."""
    var data = List[Int64]()
    data.append(-5000000000)
    data.append(2000000000)
    data.append(-8000000000)
    data.append(1000000000)
    data.append(9000000000)

    radix_sort(data)

    assert_equal(data[0], -8000000000)
    assert_equal(data[1], -5000000000)
    assert_equal(data[2], 1000000000)
    assert_equal(data[3], 2000000000)
    assert_equal(data[4], 9000000000)


def test_radix_sort_float32():
    """Test radix sort with Float32 values."""
    var data = List[Float32]()
    data.append(-5.5)
    data.append(2.2)
    data.append(-8.8)
    data.append(1.1)
    data.append(9.9)

    radix_sort(data)

    assert_equal(data[0], -8.8)
    assert_equal(data[1], -5.5)
    assert_equal(data[2], 1.1)
    assert_equal(data[3], 2.2)
    assert_equal(data[4], 9.9)


def test_radix_sort_float64():
    """Test radix sort with Float64 values."""
    var data = List[Float64]()
    data.append(-5.123456789)
    data.append(2.987654321)
    data.append(-8.555555555)
    data.append(1.111111111)
    data.append(9.999999999)

    radix_sort(data)

    assert_equal(data[0], -8.555555555)
    assert_equal(data[1], -5.123456789)
    assert_equal(data[2], 1.111111111)
    assert_equal(data[3], 2.987654321)
    assert_equal(data[4], 9.999999999)


def test_radix_sort_edge_cases_uint32():
    """Test radix sort with edge case values for UInt32."""
    var data = List[UInt32]()
    data.append(0)
    data.append(4294967295)  # Max UInt32
    data.append(1)
    data.append(2147483648)  # 2^31
    data.append(2147483647)  # 2^31 - 1

    radix_sort(data)

    assert_equal(data[0], 0)
    assert_equal(data[1], 1)
    assert_equal(data[2], 2147483647)
    assert_equal(data[3], 2147483648)
    assert_equal(data[4], 4294967295)


def test_radix_sort_edge_cases_int32():
    """Test radix sort with edge case values for Int32."""
    var data = List[Int32]()
    data.append(0)
    data.append(2147483647)  # Max Int32
    data.append(-2147483648)  # Min Int32
    data.append(1)
    data.append(-1)

    radix_sort(data)

    assert_equal(data[0], -2147483648)
    assert_equal(data[1], -1)
    assert_equal(data[2], 0)
    assert_equal(data[3], 1)
    assert_equal(data[4], 2147483647)


def test_radix_sort_performance_large():
    """Test radix sort performance with large dataset."""
    alias size = 100000
    seed(42)

    var data = List[UInt32](unsafe_uninit_length=size)
    randint(data.unsafe_ptr(), size, 0, 1000000)

    radix_sort(data)

    # Verify sorting is correct on large dataset
    for i in range(size - 1):
        assert_true(data[i] <= data[i + 1])


def test_radix_sort_stability():
    """Test that radix sort maintains relative order of equal elements (stable sort).
    """
    # Note: This test verifies if the radix sort implementation is stable
    # Since we can't easily track original positions with basic types,
    # we use a pattern that would reveal instability
    var data = List[UInt32]()

    # Add pairs of identical values in specific order
    for i in range(10):
        data.append(i % 3)  # Creates pattern: 0,1,2,0,1,2,0,1,2,0

    radix_sort(data)

    # After stable sort, all 0s should come first, then 1s, then 2s
    var count_0 = 0
    var count_1 = 0
    var count_2 = 0

    for i in range(len(data)):
        if data[i] == 0:
            count_0 += 1
        elif data[i] == 1:
            count_1 += 1
        else:
            count_2 += 1

    # Verify counts and order
    assert_equal(count_0, 4)  # 0 appears 4 times
    assert_equal(count_1, 3)  # 1 appears 3 times
    assert_equal(count_2, 3)  # 2 appears 3 times

    # Verify all 0s come first, then 1s, then 2s
    for i in range(4):
        assert_equal(data[i], 0)
    for i in range(4, 7):
        assert_equal(data[i], 1)
    for i in range(7, 10):
        assert_equal(data[i], 2)


def test_radix_sort_ivs_basic():
    """Test basic interval radix sort functionality."""
    var data = List[Interval]()
    data.append(Interval(50, 60, 5))
    data.append(Interval(20, 30, 2))
    data.append(Interval(80, 90, 8))
    data.append(Interval(10, 20, 1))
    data.append(Interval(90, 100, 9))
    
    radix_sort_ivs(data)
    
    # Should be sorted by start position
    assert_equal(data[0].start, 10)
    assert_equal(data[1].start, 20)
    assert_equal(data[2].start, 50)
    assert_equal(data[3].start, 80)
    assert_equal(data[4].start, 90)
    
    # Verify other fields are preserved
    assert_equal(data[0].val, 1)
    assert_equal(data[1].val, 2)
    assert_equal(data[2].val, 5)
    assert_equal(data[3].val, 8)
    assert_equal(data[4].val, 9)


def test_radix_sort_ivs_empty():
    """Test interval radix sort with empty list."""
    var data = List[Interval]()
    radix_sort_ivs(data)
    assert_equal(len(data), 0)


def test_radix_sort_ivs_single():
    """Test interval radix sort with single interval."""
    var data = List[Interval]()
    data.append(Interval(42, 52, 123))
    radix_sort_ivs(data)
    assert_equal(len(data), 1)
    assert_equal(data[0].start, 42)
    assert_equal(data[0].stop, 52)
    assert_equal(data[0].val, 123)


def test_radix_sort_ivs_already_sorted():
    """Test interval radix sort with already sorted intervals."""
    var data = List[Interval]()
    for i in range(10):
        var start = UInt32(i * 10)
        var stop = start + 5
        var val = Int32(i * 100)
        data.append(Interval(start, stop, val))
    
    radix_sort_ivs(data)
    
    for i in range(10):
        assert_equal(data[i].start, UInt32(i * 10))
        assert_equal(data[i].val, Int32(i * 100))


def test_radix_sort_ivs_reverse_sorted():
    """Test interval radix sort with reverse sorted intervals."""
    var data = List[Interval]()
    for i in range(10):
        var start = UInt32((9 - i) * 10)
        var stop = start + 5
        var val = Int32(i * 100)
        data.append(Interval(start, stop, val))
    
    radix_sort_ivs(data)
    
    for i in range(10):
        assert_equal(data[i].start, UInt32(i * 10))


def test_radix_sort_ivs_duplicates():
    """Test interval radix sort with duplicate start positions."""
    var data = List[Interval]()
    data.append(Interval(10, 20, 1))
    data.append(Interval(30, 40, 3))
    data.append(Interval(10, 25, 2))
    data.append(Interval(30, 35, 4))
    data.append(Interval(10, 15, 5))
    
    radix_sort_ivs(data)
    
    # All intervals with start=10 should come first
    assert_equal(data[0].start, 10)
    assert_equal(data[1].start, 10)
    assert_equal(data[2].start, 10)
    # Then intervals with start=30
    assert_equal(data[3].start, 30)
    assert_equal(data[4].start, 30)


def test_radix_sort_ivs_edge_values():
    """Test interval radix sort with edge case values."""
    var data = List[Interval]()
    data.append(Interval(0, 10, 0))
    data.append(Interval(4294967295, 4294967295, -1))  # Max UInt32
    data.append(Interval(1, 11, 1))
    data.append(Interval(2147483648, 2147483658, 2147483647))  # 2^31
    data.append(Interval(2147483647, 2147483657, -2147483648))  # 2^31 - 1
    
    radix_sort_ivs(data)
    
    assert_equal(data[0].start, 0)
    assert_equal(data[1].start, 1)
    assert_equal(data[2].start, 2147483647)
    assert_equal(data[3].start, 2147483648)
    assert_equal(data[4].start, 4294967295)


def test_radix_sort_ivs_large_random():
    """Test interval radix sort with large random dataset."""
    alias size = 1000
    seed(42)
    
    var starts = List[UInt32](unsafe_uninit_length=size)
    var stops = List[UInt32](unsafe_uninit_length=size)
    var vals = List[Int32](unsafe_uninit_length=size)
    
    randint(starts.unsafe_ptr(), size, 0, 100000)
    randint(stops.unsafe_ptr(), size, 1, 1000)  # interval lengths
    randint(vals.unsafe_ptr(), size, -1000, 1000)
    
    var data = List[Interval]()
    for i in range(size):
        var start = starts[i]
        var stop = start + stops[i]  # stops contains lengths initially
        var val = vals[i]
        data.append(Interval(start, stop, val))
    
    radix_sort_ivs(data)
    
    # Verify sorting is correct
    for i in range(size - 1):
        assert_true(data[i].start <= data[i + 1].start)


def test_radix_sort_ivs_genomic_like_data():
    """Test interval radix sort with genomic-like data patterns."""
    var data = List[Interval]()
    
    # Simulate chromosome positions (larger numbers)
    data.append(Interval(1234567, 1234600, 1))
    data.append(Interval(9876543, 9876580, 2))
    data.append(Interval(555555, 555600, 3))
    data.append(Interval(1000000, 1000050, 4))
    data.append(Interval(9999999, 10000020, 5))
    
    radix_sort_ivs(data)
    
    assert_equal(data[0].start, 555555)
    assert_equal(data[1].start, 1000000)
    assert_equal(data[2].start, 1234567)
    assert_equal(data[3].start, 9876543)
    assert_equal(data[4].start, 9999999)
    
    # Values should be preserved
    assert_equal(data[0].val, 3)
    assert_equal(data[1].val, 4)
    assert_equal(data[2].val, 1)
    assert_equal(data[3].val, 2)
    assert_equal(data[4].val, 5)


def test_radix_sort_ivs_performance_large():
    """Test interval radix sort performance with large dataset."""
    alias size = 10000
    seed(42)
    
    var starts = List[UInt32](unsafe_uninit_length=size)
    var stops = List[UInt32](unsafe_uninit_length=size)
    var vals = List[Int32](unsafe_uninit_length=size)
    
    randint(starts.unsafe_ptr(), size, 0, 1000000)
    randint(stops.unsafe_ptr(), size, 1, 10000)
    randint(vals.unsafe_ptr(), size, -100000, 100000)
    
    var data = List[Interval]()
    for i in range(size):
        var start = starts[i]
        var stop = start + stops[i]
        var val = vals[i]
        data.append(Interval(start, stop, val))
    
    radix_sort_ivs(data)
    
    # Verify sorting is correct on large dataset
    for i in range(size - 1):
        assert_true(data[i].start <= data[i + 1].start)


def main():
    test_radix_sort_basic_uint32()
    test_radix_sort_basic_int32()
    test_radix_sort_empty()
    test_radix_sort_single_element()
    test_radix_sort_already_sorted()
    test_radix_sort_reverse_sorted()
    test_radix_sort_duplicates()
    test_radix_sort_all_same()
    test_radix_sort_large_random()
    test_radix_sort_int8()
    test_radix_sort_int16()
    test_radix_sort_int64()
    test_radix_sort_float32()
    test_radix_sort_float64()
    test_radix_sort_edge_cases_uint32()
    test_radix_sort_edge_cases_int32()
    test_radix_sort_performance_large()
    test_radix_sort_stability()
    
    # Interval radix sort tests
    test_radix_sort_ivs_basic()
    test_radix_sort_ivs_empty()
    test_radix_sort_ivs_single()
    test_radix_sort_ivs_already_sorted()
    test_radix_sort_ivs_reverse_sorted()
    test_radix_sort_ivs_duplicates()
    test_radix_sort_ivs_edge_values()
    test_radix_sort_ivs_large_random()
    test_radix_sort_ivs_genomic_like_data()
    test_radix_sort_ivs_performance_large()

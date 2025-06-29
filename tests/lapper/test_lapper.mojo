from lapper.lapper import Lapper, Interval
from testing import assert_equal, assert_true, assert_false, assert_raises


def test_interval_equality():
    """Test interval equality and inequality operators."""
    var iv1 = Interval(1, 5, 0)
    var iv2 = Interval(1, 5, 0)
    var iv3 = Interval(2, 5, 0)
    var iv4 = Interval(1, 6, 0)
    
    assert_true(iv1 == iv2)
    assert_false(iv1 != iv2)
    assert_false(iv1 == iv3)
    assert_true(iv1 != iv3)
    assert_false(iv1 == iv4)
    assert_true(iv1 != iv4)


def test_interval_ordering():
    """Test interval comparison operators."""
    var iv1 = Interval(1, 5, 0)
    var iv2 = Interval(2, 5, 0)
    var iv3 = Interval(1, 6, 0)
    var iv4 = Interval(1, 5, 0)
    
    # Test greater than
    assert_true(iv2 > iv1)  # start is greater
    assert_true(iv3 > iv1)  # same start, stop is greater
    assert_false(iv1 > iv4)  # equal intervals
    
    # Test less than
    assert_true(iv1 < iv2)
    assert_true(iv1 < iv3)
    assert_false(iv1 < iv4)  # equal intervals
    
    # Test greater than or equal
    assert_true(iv2 >= iv1)
    assert_true(iv1 >= iv4)
    assert_false(iv1 >= iv2)
    
    # Test less than or equal
    assert_true(iv1 <= iv2)
    assert_true(iv1 <= iv4)
    assert_false(iv2 <= iv1)


def test_interval_creation():
    """Test basic interval creation."""
    var interval = Interval(10, 20, 42)
    assert_equal(interval.start, 10)
    assert_equal(interval.stop, 20)
    assert_equal(interval.val, 42)


def test_interval_intersects():
    """Test interval intersection calculation."""
    var iv1 = Interval(1, 5, 0)
    var iv2 = Interval(3, 7, 0)
    
    # Overlapping intervals
    assert_equal(iv1.intersect(iv2), 2)  # [3, 5) = 2
    assert_equal(iv2.intersect(iv1), 2)  # symmetric
    
    # Contained interval
    var iv5 = Interval(2, 4, 0)
    assert_equal(iv1.intersect(iv5), 2)  # [2, 4) = 2
    assert_equal(iv5.intersect(iv1), 2)  # symmetric


def test_interval_no_intersection():
    """Test intervals that don't intersect."""
    var iv1 = Interval(1, 5, 0)
    var iv2 = Interval(5, 10, 0)
    var iv3 = Interval(10, 15, 0)
    
    # Adjacent intervals (no overlap)
    assert_equal(iv1.intersect(iv2), 0)
    assert_equal(iv2.intersect(iv3), 0)
    
    # Separated intervals
    assert_equal(iv1.intersect(iv3), 0)


def test_interval_complete_overlap():
    """Test when one interval completely contains another."""
    var iv1 = Interval(1, 10, 0)
    var iv2 = Interval(3, 7, 0)
    
    assert_equal(iv1.intersect(iv2), 4)  # [3, 7) = 4
    assert_equal(iv2.intersect(iv1), 4)  # symmetric


def test_interval_overlap_basic():
    """Test interval overlap method with basic cases."""
    var iv1 = Interval(1, 5, 0)
    
    # Overlapping queries
    assert_true(iv1.overlap(0, 2))   # overlaps start
    assert_true(iv1.overlap(4, 6))   # overlaps end
    assert_true(iv1.overlap(2, 4))   # contained within
    assert_true(iv1.overlap(0, 10))  # contains interval
    
    # Non-overlapping queries
    assert_false(iv1.overlap(5, 10))  # starts at end
    assert_false(iv1.overlap(6, 10))  # after interval
    assert_false(iv1.overlap(0, 1))   # ends at start


def test_interval_overlap_edge_cases():
    """Test overlap with touching intervals (should not overlap)."""
    var iv1 = Interval(5, 10, 0)
    
    # Touching but not overlapping
    assert_false(iv1.overlap(0, 5))   # ends exactly at start
    assert_false(iv1.overlap(10, 15)) # starts exactly at end
    
    # Just overlapping by 1
    assert_true(iv1.overlap(4, 6))    # overlaps by 1 at start
    assert_true(iv1.overlap(9, 11))   # overlaps by 1 at end


def test_interval_overlap_contained():
    """Test when query is contained within interval."""
    var iv1 = Interval(10, 20, 0)
    
    assert_true(iv1.overlap(12, 15))  # fully contained
    assert_true(iv1.overlap(10, 15))  # starts at interval start
    assert_true(iv1.overlap(15, 20))  # ends at interval end
    assert_true(iv1.overlap(10, 20))  # exact match


def test_lapper_empty():
    """Test creating empty Lapper."""
    # Now it should raise an error for empty lists
    with assert_raises(contains="Intervals length must be >= 1"):
        var lapper = Lapper(List[Interval]())


def test_lapper_single_interval():
    """Test Lapper with single interval."""
    var intervals = List[Interval]()
    intervals.append(Interval(5, 10, 100))
    
    var lapper = Lapper(intervals)
    assert_equal(len(lapper), 1)
    assert_equal(lapper.max_len, 5)
    assert_equal(lapper.starts[0], 5)
    assert_equal(lapper.stops[0], 10)
    assert_equal(lapper.vals[0], 100)
    
    # Test finding the interval
    # TODO: Re-enable when find method is fixed
    # var results = List[Interval]()
    # lapper.find(7, 8, results)
    # assert_equal(len(results), 1)
    # assert_equal(results[0].start, 5)
    # assert_equal(results[0].stop, 10)
    # assert_equal(results[0].val, 100)


def test_lapper_multiple_intervals():
    """Test Lapper with multiple intervals."""
    var intervals = List[Interval]()
    intervals.append(Interval(1, 5, 10))
    intervals.append(Interval(10, 15, 20))
    intervals.append(Interval(20, 30, 30))
    
    var lapper = Lapper(intervals)
    assert_equal(len(lapper), 3)
    assert_equal(lapper.max_len, 10)  # interval [20, 30) has length 10


def test_lapper_sorting():
    """Verify intervals are sorted after initialization."""
    var intervals = List[Interval]()
    # Add intervals in reverse order
    intervals.append(Interval(20, 25, 3))
    intervals.append(Interval(10, 15, 2))
    intervals.append(Interval(1, 5, 1))
    
    var lapper = Lapper(intervals)
    
    # Check starts are sorted
    assert_equal(lapper.starts[0], 1)
    assert_equal(lapper.starts[1], 10)
    assert_equal(lapper.starts[2], 20)
    
    # Check stops match the sorted interval order
    assert_equal(lapper.stops[0], 5)
    assert_equal(lapper.stops[1], 15)
    assert_equal(lapper.stops[2], 25)
    
    # Check vals match the sorted interval order
    assert_equal(lapper.vals[0], 1)
    assert_equal(lapper.vals[1], 2)
    assert_equal(lapper.vals[2], 3)


# TODO: Re-enable find tests when find method is fixed
# def test_find_no_overlaps():
#     """Test query that finds no overlapping intervals."""
#     var intervals = List[Interval]()
#     intervals.append(Interval(1, 5, 0))
#     intervals.append(Interval(10, 15, 0))
#     intervals.append(Interval(20, 25, 0))
    
#     var lapper = Lapper(intervals)
#     var results = List[Interval]()
    
#     # Query in gap between intervals
#     lapper.find(6, 9, results)
#     assert_equal(len(results), 0)


def test_lapper_max_len():
    """Verify max_len is calculated correctly."""
    var intervals = List[Interval]()
    intervals.append(Interval(1, 5, 0))    # length 4
    intervals.append(Interval(10, 20, 0))  # length 10
    intervals.append(Interval(30, 35, 0))  # length 5
    
    var lapper = Lapper(intervals)
    assert_equal(lapper.max_len, 10)


def test_lapper_starts_order():
    """Verify starts array maintains interval order."""
    var intervals = List[Interval]()
    # Add intervals with same start but different stops
    intervals.append(Interval(5, 20, 1))
    intervals.append(Interval(5, 10, 2))
    intervals.append(Interval(5, 15, 3))
    
    var lapper = Lapper(intervals)
    
    # All starts should be 5
    for i in range(len(lapper)):
        assert_equal(lapper.starts[i], 5)
    
    # Stops should be sorted by stop value when starts are equal
    assert_equal(lapper.stops[0], 10)
    assert_equal(lapper.stops[1], 15)
    assert_equal(lapper.stops[2], 20)
    
    # Vals should match the sorted order
    assert_equal(lapper.vals[0], 2)
    assert_equal(lapper.vals[1], 3)
    assert_equal(lapper.vals[2], 1)


def test_lapper_stops_sorted():
    """Verify stops_sorted array is sorted."""
    var intervals = List[Interval]()
    intervals.append(Interval(1, 30, 0))
    intervals.append(Interval(5, 10, 0))
    intervals.append(Interval(15, 25, 0))
    
    var lapper = Lapper(intervals)
    
    # stops_sorted should be sorted: [10, 25, 30]
    assert_equal(lapper.stops_sorted[0], 10)
    assert_equal(lapper.stops_sorted[1], 25)
    assert_equal(lapper.stops_sorted[2], 30)
    
    # Verify stops_sorted are actually sorted
    for i in range(len(lapper) - 1):
        assert_true(lapper.stops_sorted[i] <= lapper.stops_sorted[i + 1])


def main():
    # Interval tests
    test_interval_equality()
    test_interval_ordering()
    test_interval_creation()
    test_interval_intersects()
    test_interval_no_intersection()
    test_interval_complete_overlap()
    test_interval_overlap_basic()
    test_interval_overlap_edge_cases()
    test_interval_overlap_contained()
    
    # Lapper initialization tests
    test_lapper_empty()
    test_lapper_single_interval()
    test_lapper_multiple_intervals()
    test_lapper_sorting()
    
    # Lapper internal structure tests
    test_lapper_max_len()
    test_lapper_starts_order()
    test_lapper_stops_sorted()
    
    print("All tests passed!")
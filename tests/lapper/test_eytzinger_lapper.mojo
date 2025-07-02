from lapper.eytzinger_lapper import EzLapper, Interval
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
    assert_true(iv1.overlap(0, 2))  # overlaps start
    assert_true(iv1.overlap(4, 6))  # overlaps end
    assert_true(iv1.overlap(2, 4))  # contained within
    assert_true(iv1.overlap(0, 10))  # contains interval

    # Non-overlapping queries
    assert_false(iv1.overlap(5, 10))  # starts at end
    assert_false(iv1.overlap(6, 10))  # after interval
    assert_false(iv1.overlap(0, 1))  # ends at start


def test_interval_overlap_edge_cases():
    """Test overlap with touching intervals (should not overlap)."""
    var iv1 = Interval(5, 10, 0)

    # Touching but not overlapping
    assert_false(iv1.overlap(0, 5))  # ends exactly at start
    assert_false(iv1.overlap(10, 15))  # starts exactly at end

    # Just overlapping by 1
    assert_true(iv1.overlap(4, 6))  # overlaps by 1 at start
    assert_true(iv1.overlap(9, 11))  # overlaps by 1 at end


def test_interval_overlap_contained():
    """Test when query is contained within interval."""
    var iv1 = Interval(10, 20, 0)

    assert_true(iv1.overlap(12, 15))  # fully contained
    assert_true(iv1.overlap(10, 15))  # starts at interval start
    assert_true(iv1.overlap(15, 20))  # ends at interval end
    assert_true(iv1.overlap(10, 20))  # exact match


def test_ezlapper_empty():
    """Test creating empty EzLapper."""
    # Now it should raise an error for empty lists
    with assert_raises(contains="Intervals length must be >= 1"):
        var lapper = EzLapper(List[Interval]())


def test_ezlapper_single_interval():
    """Test EzLapper with single interval."""
    var intervals = List[Interval]()
    intervals.append(Interval(5, 10, 100))

    var lapper = EzLapper(intervals)
    assert_equal(len(lapper), 1)
    assert_equal(lapper.inner.max_len, 5)

    # Test counting overlapping intervals
    assert_equal(lapper.count(7, 8), 1)  # overlaps the interval
    assert_equal(lapper.count(0, 4), 0)  # before the interval
    assert_equal(lapper.count(11, 15), 0)  # after the interval


def test_ezlapper_multiple_intervals():
    """Test EzLapper with multiple intervals."""
    var intervals = List[Interval]()
    intervals.append(Interval(1, 5, 10))
    intervals.append(Interval(10, 15, 20))
    intervals.append(Interval(20, 30, 30))

    var lapper = EzLapper(intervals)
    assert_equal(len(lapper), 3)
    assert_equal(lapper.inner.max_len, 10)  # interval [20, 30) has length 10

    # Test counting with multiple intervals
    assert_equal(lapper.count(2, 4), 1)  # overlaps first interval
    assert_equal(lapper.count(12, 14), 1)  # overlaps second interval
    assert_equal(lapper.count(25, 28), 1)  # overlaps third interval
    assert_equal(
        lapper.count(6, 9), 0
    )  # overlaps none (gap between first and second)
    assert_equal(lapper.count(0, 35), 3)  # overlaps all intervals


def test_ezlapper_sorting():
    """Verify intervals are sorted after initialization."""
    var intervals = List[Interval]()
    # Add intervals in reverse order
    intervals.append(Interval(20, 25, 3))
    intervals.append(Interval(10, 15, 2))
    intervals.append(Interval(1, 5, 1))

    var lapper = EzLapper(intervals)

    # Check that the underlying data is correctly sorted
    # Note: We can't directly access the Eytzinger arrays, but we can verify
    # behavior through count operations that the sorting worked correctly
    assert_equal(lapper.count(2, 3), 1)  # should find interval [1, 5)
    assert_equal(lapper.count(11, 12), 1)  # should find interval [10, 15)
    assert_equal(lapper.count(21, 22), 1)  # should find interval [20, 25)


def test_ezlapper_max_len():
    """Verify max_len is calculated correctly."""
    var intervals = List[Interval]()
    intervals.append(Interval(1, 5, 0))  # length 4
    intervals.append(Interval(10, 20, 0))  # length 10
    intervals.append(Interval(30, 35, 0))  # length 5

    var lapper = EzLapper(intervals)
    assert_equal(lapper.inner.max_len, 10)


def test_count_no_overlaps():
    """Test count with no overlapping intervals."""
    var intervals = List[Interval]()
    intervals.append(Interval(5, 10, 0))
    intervals.append(Interval(15, 20, 0))
    intervals.append(Interval(25, 30, 0))

    var lapper = EzLapper(intervals)

    # Query in gaps between intervals
    assert_equal(lapper.count(0, 4), 0)  # before first interval
    assert_equal(lapper.count(11, 14), 0)  # between first and second
    assert_equal(lapper.count(21, 24), 0)  # between second and third
    assert_equal(lapper.count(31, 35), 0)  # after last interval


def test_count_single_overlap():
    """Test count with queries that overlap single intervals."""
    var intervals = List[Interval]()
    intervals.append(Interval(5, 10, 0))
    intervals.append(Interval(15, 20, 0))
    intervals.append(Interval(25, 30, 0))

    var lapper = EzLapper(intervals)

    # Each query should overlap exactly one interval
    assert_equal(lapper.count(7, 8), 1)  # overlaps first interval
    assert_equal(lapper.count(17, 18), 1)  # overlaps second interval
    assert_equal(lapper.count(27, 28), 1)  # overlaps third interval

    # Queries that touch boundaries
    assert_equal(lapper.count(4, 6), 1)  # overlaps start of first
    assert_equal(lapper.count(9, 11), 1)  # overlaps end of first
    assert_equal(lapper.count(14, 16), 1)  # overlaps start of second
    assert_equal(lapper.count(19, 21), 1)  # overlaps end of second


def test_count_multiple_overlaps():
    """Test count with queries that overlap multiple intervals."""
    var intervals = List[Interval]()
    intervals.append(Interval(5, 15, 0))
    intervals.append(Interval(10, 20, 0))
    intervals.append(Interval(12, 18, 0))
    intervals.append(Interval(25, 30, 0))

    var lapper = EzLapper(intervals)

    # Query overlapping first three intervals
    assert_equal(lapper.count(8, 17), 3)

    # Query overlapping only first two intervals
    assert_equal(lapper.count(7, 11), 2)

    # Query overlapping all intervals
    assert_equal(lapper.count(1, 35), 4)

    # Query overlapping last interval only
    assert_equal(lapper.count(26, 29), 1)


def test_count_contained_intervals():
    """Test count with intervals that are contained within the query."""
    var intervals = List[Interval]()
    intervals.append(Interval(10, 15, 0))
    intervals.append(Interval(20, 25, 0))
    intervals.append(Interval(30, 35, 0))

    var lapper = EzLapper(intervals)

    # Query that contains single intervals
    assert_equal(lapper.count(5, 18), 1)  # contains first interval
    assert_equal(lapper.count(18, 28), 1)  # contains second interval
    assert_equal(lapper.count(28, 38), 1)  # contains third interval

    # Query that contains multiple intervals
    assert_equal(lapper.count(5, 28), 2)  # contains first and second
    assert_equal(lapper.count(18, 38), 2)  # contains second and third
    assert_equal(lapper.count(5, 38), 3)  # contains all intervals


def test_count_exact_boundaries():
    """Test count with queries that exactly match interval boundaries."""
    var intervals = List[Interval]()
    intervals.append(Interval(10, 20, 0))
    intervals.append(Interval(30, 40, 0))

    var lapper = EzLapper(intervals)

    # Exact matches should count the interval
    assert_equal(lapper.count(10, 20), 1)  # exact match first interval
    assert_equal(lapper.count(30, 40), 1)  # exact match second interval

    # Touching boundaries (no overlap with strict inequalities)
    assert_equal(lapper.count(5, 10), 0)  # ends at start of first
    assert_equal(lapper.count(20, 25), 0)  # starts at end of first
    assert_equal(lapper.count(20, 30), 0)  # between intervals
    assert_equal(lapper.count(40, 45), 0)  # starts at end of second


def test_count_single_interval():
    """Test count with single interval."""
    var intervals = List[Interval]()
    intervals.append(Interval(10, 20, 42))

    var lapper = EzLapper(intervals)

    # Overlapping queries
    assert_equal(lapper.count(5, 15), 1)  # overlaps start
    assert_equal(lapper.count(15, 25), 1)  # overlaps end
    assert_equal(lapper.count(12, 18), 1)  # contained within
    assert_equal(lapper.count(5, 25), 1)  # contains interval
    assert_equal(lapper.count(10, 20), 1)  # exact match

    # Non-overlapping queries
    assert_equal(lapper.count(0, 10), 0)  # before interval
    assert_equal(lapper.count(20, 30), 0)  # after interval
    assert_equal(lapper.count(0, 5), 0)  # well before interval


def test_count_overlapping_intervals():
    """Test count with heavily overlapping intervals."""
    var intervals = List[Interval]()
    intervals.append(Interval(1, 10, 0))
    intervals.append(Interval(5, 15, 0))
    intervals.append(Interval(8, 12, 0))
    intervals.append(Interval(11, 20, 0))

    var lapper = EzLapper(intervals)

    # Query that hits all overlapping intervals
    assert_equal(lapper.count(9, 11), 3)  # overlaps first 3 intervals
    assert_equal(lapper.count(5, 12), 4)  # overlaps all 4 intervals
    assert_equal(
        lapper.count(11, 12), 3
    )  # overlaps intervals [5,15), [8,12), [11,20)
    assert_equal(lapper.count(1, 20), 4)  # overlaps all intervals

    # Edge cases within overlapping region
    assert_equal(lapper.count(6, 9), 3)  # overlaps first 3 intervals
    assert_equal(
        lapper.count(12, 19), 2
    )  # overlaps intervals [5,15) and [11,20)


def test_count_same_start_different_stops():
    """Test count with intervals that have same start but different stops."""
    var intervals = List[Interval]()
    intervals.append(Interval(10, 15, 1))
    intervals.append(Interval(10, 20, 2))
    intervals.append(Interval(10, 25, 3))

    var lapper = EzLapper(intervals)

    # Query that overlaps all intervals with same start
    assert_equal(lapper.count(5, 12), 3)  # overlaps all 3
    assert_equal(lapper.count(12, 17), 3)  # overlaps all 3
    assert_equal(lapper.count(17, 22), 2)  # overlaps last 2 (stops 20, 25)
    assert_equal(lapper.count(26, 30), 0)  # overlaps none


def test_count_adjacent_intervals():
    """Test count with adjacent (touching) intervals."""
    var intervals = List[Interval]()
    intervals.append(Interval(5, 10, 0))
    intervals.append(Interval(10, 15, 0))
    intervals.append(Interval(15, 20, 0))

    var lapper = EzLapper(intervals)

    # Queries spanning adjacent intervals
    assert_equal(lapper.count(8, 12), 2)  # spans first two intervals
    assert_equal(lapper.count(12, 17), 2)  # spans second two intervals
    assert_equal(lapper.count(8, 17), 3)  # spans all three intervals

    # Queries at exact boundaries (should not overlap)
    assert_equal(lapper.count(0, 5), 0)  # ends at start of first
    assert_equal(lapper.count(20, 25), 0)  # starts at end of last


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

    # EzLapper initialization tests
    test_ezlapper_empty()
    test_ezlapper_single_interval()
    test_ezlapper_multiple_intervals()
    test_ezlapper_sorting()

    # EzLapper internal structure tests
    test_ezlapper_max_len()

    # EzLapper count method tests
    test_count_no_overlaps()
    test_count_single_overlap()
    test_count_multiple_overlaps()
    test_count_contained_intervals()
    test_count_exact_boundaries()
    test_count_single_interval()
    test_count_overlapping_intervals()
    test_count_same_start_different_stops()
    test_count_adjacent_intervals()

    print("All EzLapper tests passed!")

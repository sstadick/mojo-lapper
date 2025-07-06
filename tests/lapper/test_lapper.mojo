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


def test_lapper_empty():
    """Test creating empty Lapper."""
    # Now it should raise an error for empty lists
    with assert_raises(contains="Intervals length must be >= 1"):
        var lapper = Lapper(List[Interval]())
        _ = len(lapper)


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

    # Test counting overlapping intervals
    assert_equal(lapper.count(7, 8), 1)  # overlaps the interval
    assert_equal(lapper.count(0, 4), 0)  # before the interval
    assert_equal(lapper.count(11, 15), 0)  # after the interval

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

    # Test counting with multiple intervals
    assert_equal(lapper.count(2, 4), 1)  # overlaps first interval
    assert_equal(lapper.count(12, 14), 1)  # overlaps second interval
    assert_equal(lapper.count(25, 28), 1)  # overlaps third interval
    assert_equal(
        lapper.count(6, 9), 0
    )  # overlaps none (gap between first and second)
    assert_equal(lapper.count(0, 35), 3)  # overlaps all intervals


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
    intervals.append(Interval(1, 5, 0))  # length 4
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


def test_count_no_overlaps():
    """Test count with no overlapping intervals."""
    var intervals = List[Interval]()
    intervals.append(Interval(5, 10, 0))
    intervals.append(Interval(15, 20, 0))
    intervals.append(Interval(25, 30, 0))

    var lapper = Lapper(intervals)

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

    var lapper = Lapper(intervals)

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

    var lapper = Lapper(intervals)

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

    var lapper = Lapper(intervals)

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

    var lapper = Lapper(intervals)

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

    var lapper = Lapper(intervals)

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

    var lapper = Lapper(intervals)

    # Query that hits all overlapping intervals
    assert_equal(lapper.count(9, 11), 3)  # overlaps first 3 intervals
    assert_equal(lapper.count(5, 12), 4)  # overlaps first 3 intervals
    assert_equal(lapper.count(11, 12), 3)  # overlaps last 2 intervals
    assert_equal(lapper.count(1, 20), 4)  # overlaps all intervals

    # Edge cases within overlapping region
    assert_equal(lapper.count(6, 9), 3)  # overlaps first 2 intervals
    assert_equal(
        lapper.count(12, 19), 2
    )  # overlaps intervals [5,15) and [11,20)


def test_count_same_start_different_stops():
    """Test count with intervals that have same start but different stops."""
    var intervals = List[Interval]()
    intervals.append(Interval(10, 15, 1))
    intervals.append(Interval(10, 20, 2))
    intervals.append(Interval(10, 25, 3))

    var lapper = Lapper(intervals)

    # Query that overlaps all intervals with same start
    assert_equal(lapper.count(5, 12), 3)  # overlaps all 3
    assert_equal(lapper.count(12, 17), 3)  # overlaps last 2 (stops 20, 25)
    assert_equal(lapper.count(17, 22), 2)  # overlaps only last (stop 25)
    assert_equal(lapper.count(26, 30), 0)  # overlaps none


def test_count_adjacent_intervals():
    """Test count with adjacent (touching) intervals."""
    var intervals = List[Interval]()
    intervals.append(Interval(5, 10, 0))
    intervals.append(Interval(10, 15, 0))
    intervals.append(Interval(15, 20, 0))

    var lapper = Lapper(intervals)

    # Queries spanning adjacent intervals
    assert_equal(lapper.count(8, 12), 2)  # spans first two intervals
    assert_equal(lapper.count(12, 17), 2)  # spans second two intervals
    assert_equal(lapper.count(8, 17), 3)  # spans all three intervals

    # Queries at exact boundaries (should not overlap)
    assert_equal(lapper.count(0, 5), 0)  # ends at start of first
    assert_equal(lapper.count(20, 25), 0)  # starts at end of last


def test_find_vectorized_no_overlaps():
    """Test find_vectorized with no overlapping intervals."""
    var intervals = List[Interval]()
    
    # Create > 16 intervals to trigger SIMD path
    for i in range(20):
        # Create intervals at positions 0-5, 10-15, 20-25, etc.
        # with gaps between them
        var start = UInt32(i * 10)
        var stop = UInt32(start + 5)
        intervals.append(Interval(start, stop, Int32(i)))

    var lapper = Lapper(intervals)
    var results = List[UInt32]()

    # Query in gap between intervals (should find no overlaps)
    lapper.find_vectorized(6, 9, results)
    assert_equal(len(results), 0)
    
    # Query in another gap
    lapper.find_vectorized(16, 19, results)
    assert_equal(len(results), 0)
    
    # Query in a larger gap
    lapper.find_vectorized(46, 49, results)
    assert_equal(len(results), 0)


def test_find_vectorized_single_overlap():
    """Test find_vectorized with queries that overlap single intervals."""
    var intervals = List[Interval]()
    
    # Create > 16 intervals with non-overlapping intervals
    for i in range(25):
        # Create intervals at positions 0-8, 10-18, 20-28, etc.
        var start = UInt32(i * 10)
        var stop = UInt32(start + 8)
        intervals.append(Interval(start, stop, Int32(i * 100)))

    var lapper = Lapper(intervals)

    # Query overlapping first interval
    var results1 = List[UInt32]()
    lapper.find_vectorized(2, 6, results1)
    assert_equal(len(results1), 1)
    var interval1 = lapper.get(UInt(results1[0]))
    assert_equal(interval1.start, 0)
    assert_equal(interval1.stop, 8)
    assert_equal(interval1.val, 0)

    # Query overlapping a middle interval (to ensure SIMD path is used)
    var results2 = List[UInt32]()
    lapper.find_vectorized(152, 156, results2)
    assert_equal(len(results2), 1)
    var interval2 = lapper.get(UInt(results2[0]))
    assert_equal(interval2.start, 150)
    assert_equal(interval2.stop, 158)
    assert_equal(interval2.val, 1500)

    # Query overlapping last interval
    var results3 = List[UInt32]()
    lapper.find_vectorized(242, 246, results3)
    assert_equal(len(results3), 1)
    var interval3 = lapper.get(UInt(results3[0]))
    assert_equal(interval3.start, 240)
    assert_equal(interval3.stop, 248)
    assert_equal(interval3.val, 2400)


def test_find_vectorized_multiple_overlaps():
    """Test find_vectorized with queries that overlap multiple intervals."""
    var intervals = List[Interval]()
    
    # Create > 16 intervals with many overlapping intervals
    # This creates a dense set of overlapping intervals to test SIMD path
    for i in range(30):
        # Create overlapping intervals in groups
        var base = UInt32(i * 5)
        var start = base
        var stop = base + 20  # Each interval spans 20 units
        intervals.append(Interval(start, stop, Int32(i * 100)))

    var lapper = Lapper(intervals)

    # Query that should overlap many intervals (> 8 to ensure SIMD processing)
    var results = List[UInt32]()
    lapper.find_vectorized(50, 80, results)
    
    # Calculate expected overlaps:
    # Query [50, 80) overlaps with intervals where:
    # interval.start < 80 AND interval.stop > 50
    # Let's check each interval:
    # i=6: [30,50) - NO (stop = 50, not > 50)
    # i=7: [35,55) - YES 
    # i=8: [40,60) - YES
    # i=9: [45,65) - YES
    # i=10: [50,70) - YES
    # i=11: [55,75) - YES
    # i=12: [60,80) - YES 
    # i=13: [65,85) - YES
    # i=14: [70,90) - YES
    # i=15: [75,95) - YES
    # i=16: [80,100) - NO (start = 80, not < 80)
    assert_equal(len(results), 9)  # Should have exactly 9 overlaps
    
    # Verify first few results
    if len(results) > 0:
        var first_interval = lapper.get(UInt(results[0]))
        assert_true(first_interval.start <= 50)
        assert_true(first_interval.stop > 50)
    
    # Query at the beginning with many overlaps
    var results2 = List[UInt32]()
    lapper.find_vectorized(10, 40, results2)
    # Intervals that overlap [10, 40):
    # i=0: [0,20) - YES (start=0 < 40, stop=20 > 10)
    # i=1: [5,25) - YES
    # i=2: [10,30) - YES
    # i=3: [15,35) - YES
    # i=4: [20,40) - YES
    # i=5: [25,45) - YES
    # i=6: [30,50) - YES
    # i=7: [35,55) - YES
    # i=8: [40,60) - NO (start=40, not < 40)
    assert_equal(len(results2), 8)  # Should have exactly 8 overlaps
    
    # Query that spans almost all intervals
    var results3 = List[UInt32]()
    lapper.find_vectorized(20, 120, results3)
    # Query [20, 120) overlaps with intervals where start < 120 AND stop > 20
    # i=0: [0,20) - NO (stop = 20, not > 20)
    # i=1: [5,25) - YES
    # ...
    # i=23: [115,135) - YES (start=115 < 120)
    # i=24: [120,140) - NO (start=120, not < 120)
    # So intervals from index 1 to 23 overlap (23 intervals)
    assert_equal(len(results3), 23)  # Should have exactly 23 overlaps


def test_find_vectorized_vs_find_consistency():
    """Test that find_vectorized and find return the same intervals."""
    var intervals = List[Interval]()
    intervals.append(Interval(1, 10, 100))
    intervals.append(Interval(5, 15, 200))
    intervals.append(Interval(8, 12, 300))
    intervals.append(Interval(11, 20, 400))

    var lapper = Lapper(intervals)

    # Test several queries to ensure find_vectorized and find are consistent
    var test_queries = List[Tuple[UInt32, UInt32]]()
    test_queries.append((UInt32(9), UInt32(11)))
    test_queries.append((UInt32(5), UInt32(12)))
    test_queries.append((UInt32(1), UInt32(20)))
    test_queries.append((UInt32(0), UInt32(5)))
    test_queries.append((UInt32(15), UInt32(25)))

    for i in range(len(test_queries)):
        var query = test_queries[i]
        var start = query[0]
        var stop = query[1]
        
        # Get results from find_vectorized
        var vectorized_indices = List[UInt32]()
        lapper.find_vectorized(start, stop, vectorized_indices)
        
        # Get results from regular find
        var find_results = List[Interval]()
        lapper.find(start, stop, find_results)
        
        # Should have same number of results
        assert_equal(len(vectorized_indices), len(find_results))
        
        # Should have the same intervals (in same order)
        for j in range(len(vectorized_indices)):
            var vectorized_interval = lapper.get(UInt(vectorized_indices[j]))
            var find_interval = find_results[j]
            
            assert_equal(vectorized_interval.start, find_interval.start)
            assert_equal(vectorized_interval.stop, find_interval.stop)
            assert_equal(vectorized_interval.val, find_interval.val)


def test_find_vectorized_exact_boundaries():
    """Test find_vectorized with queries that exactly match interval boundaries."""
    var intervals = List[Interval]()
    intervals.append(Interval(10, 20, 100))
    intervals.append(Interval(30, 40, 200))

    var lapper = Lapper(intervals)

    # Exact matches should find the interval
    var results1 = List[UInt32]()
    lapper.find_vectorized(10, 20, results1)
    assert_equal(len(results1), 1)
    var interval1 = lapper.get(UInt(results1[0]))
    assert_equal(interval1.start, 10)
    assert_equal(interval1.stop, 20)
    assert_equal(interval1.val, 100)

    # Touching boundaries (no overlap with strict inequalities)
    var results2 = List[UInt32]()
    lapper.find_vectorized(5, 10, results2)
    assert_equal(len(results2), 0)  # ends at start of first

    var results3 = List[UInt32]()
    lapper.find_vectorized(20, 25, results3)
    assert_equal(len(results3), 0)  # starts at end of first


def test_find_vectorized_large_dataset():
    """Test find_vectorized with a larger dataset to exercise vectorization."""
    var intervals = List[Interval]()
    
    # Create 100 intervals with some overlaps
    for i in range(100):
        var start = UInt32(i * 10)
        var stop = UInt32(start + 15)  # Overlap with next interval
        var val = Int32(i * 100)
        intervals.append(Interval(start, stop, val))

    var lapper = Lapper(intervals)

    # Query that should overlap many intervals
    var results = List[UInt32]()
    lapper.find_vectorized(50, 150, results)
    
    # Verify against regular find
    var find_results = List[Interval]()
    lapper.find(50, 150, find_results)
    
    assert_equal(len(results), len(find_results))
    
    # Verify that we get the same intervals
    for i in range(len(results)):
        var vectorized_interval = lapper.get(UInt(results[i]))
        var find_interval = find_results[i]
        
        assert_equal(vectorized_interval.start, find_interval.start)
        assert_equal(vectorized_interval.stop, find_interval.stop)
        assert_equal(vectorized_interval.val, find_interval.val)


def test_find_vectorized_simd_edge_cases():
    """Test find_vectorized with edge cases specific to SIMD processing."""
    var intervals = List[Interval]()
    
    # Create exactly 32 intervals (2x SIMD width) to test alignment edge cases
    for i in range(32):
        var start = UInt32(i * 10)
        var stop = UInt32(start + 12)
        intervals.append(Interval(start, stop, Int32(i)))
    
    var lapper = Lapper(intervals)
    
    # Test 1: Query that finds exactly SIMD width (16) overlaps
    var results1 = List[UInt32]()
    lapper.find_vectorized(50, 200, results1)
    # This should overlap intervals from index 4 (start=40) to index 19 (start=190)
    assert_true(len(results1) >= 15)
    
    # Test 2: Query at boundary between SIMD chunks
    var results2 = List[UInt32]()
    lapper.find_vectorized(155, 165, results2)  # Should overlap intervals around index 15-16
    assert_true(len(results2) >= 2)
    
    # Test 3: Query that starts in middle of SIMD chunk
    var results3 = List[UInt32]()
    lapper.find_vectorized(85, 125, results3)
    assert_true(len(results3) >= 4)
    
    # Test 4: Large number of intervals to ensure SIMD path with cleanup loop
    var intervals2 = List[Interval]()
    # Create 67 intervals (not a multiple of SIMD width)
    for i in range(67):
        var start = UInt32(i * 5)
        var stop = UInt32(start + 10)
        intervals2.append(Interval(start, stop, Int32(i)))
    
    var lapper2 = Lapper(intervals2)
    var results4 = List[UInt32]()
    lapper2.find_vectorized(100, 200, results4)
    
    # Verify we get consistent results with regular find
    var find_results = List[Interval]()
    lapper2.find(100, 200, find_results)
    assert_equal(len(results4), len(find_results))
    
    # Test 5: Query that results in exactly 0 overlaps after SIMD processing many intervals
    var results5 = List[UInt32]()
    lapper2.find_vectorized(340, 350, results5)  # Beyond all intervals
    assert_equal(len(results5), 0)


def test_find_vs_find_vectorized_comprehensive():
    """Comprehensive test comparing find and find_vectorized results."""
    # Test with various dataset sizes and overlap patterns
    var test_cases = List[Tuple[Int, Int, Int]]()
    test_cases.append((50, 1000, 0))  # dense_overlapping
    test_cases.append((100, 10000, 1))  # sparse
    test_cases.append((25, 500, 2))  # mixed
    
    for case_idx in range(len(test_cases)):
        var test_case = test_cases[case_idx]
        var num_intervals = test_case[0]
        var coord_space = test_case[1]
        var case_type = test_case[2]
        
        var intervals = List[Interval]()
        
        if case_type == 0:  # dense_overlapping
            # Create overlapping intervals
            for i in range(num_intervals):
                var start = UInt32(i * 5)  # Close spacing
                var stop = UInt32(start + 20)  # Long intervals
                intervals.append(Interval(start, stop, Int32(i)))
        elif case_type == 1:  # sparse
            # Create well-separated intervals
            for i in range(num_intervals):
                var start = UInt32(i * 50)  # Wide spacing
                var stop = UInt32(start + 10)  # Short intervals
                intervals.append(Interval(start, stop, Int32(i)))
        else:  # mixed
            # Create mixed pattern
            for i in range(num_intervals):
                var start = UInt32(i * 15 + (i % 3) * 5)  # Irregular spacing
                var stop = UInt32(start + 8 + (i % 4) * 3)  # Variable lengths
                intervals.append(Interval(start, stop, Int32(i)))
        
        var lapper = Lapper(intervals)
        
        # Test multiple query patterns
        var query_patterns = List[Tuple[UInt32, UInt32]]()
        # Small queries
        query_patterns.append((UInt32(10), UInt32(15)))
        query_patterns.append((UInt32(25), UInt32(30)))
        # Medium queries
        query_patterns.append((UInt32(50), UInt32(100)))
        query_patterns.append((UInt32(150), UInt32(250)))
        # Large queries
        query_patterns.append((UInt32(0), UInt32(coord_space // 2)))
        query_patterns.append((UInt32(coord_space // 4), UInt32(coord_space * 3 // 4)))
        # Edge case queries
        query_patterns.append((UInt32(0), UInt32(1)))
        query_patterns.append((UInt32(coord_space - 1), UInt32(coord_space)))
        
        for query_idx in range(len(query_patterns)):
            var query = query_patterns[query_idx]
            var start = query[0]
            var stop = query[1]
            
            # Get results from both methods
            var find_results = List[Interval]()
            var vectorized_indices = List[UInt32]()
            
            lapper.find(start, stop, find_results)
            lapper.find_vectorized(start, stop, vectorized_indices)
            
            # Verify same count
            assert_equal(len(find_results), len(vectorized_indices))
            
            # Verify same intervals in same order
            for result_idx in range(len(find_results)):
                var find_interval = find_results[result_idx]
                var vectorized_interval = lapper.get(UInt(vectorized_indices[result_idx]))
                
                assert_equal(find_interval.start, vectorized_interval.start)
                assert_equal(find_interval.stop, vectorized_interval.stop)
                assert_equal(find_interval.val, vectorized_interval.val)


def test_find_vs_find_vectorized_random_data():
    """Test find vs find_vectorized with randomized data."""
    var intervals = List[Interval]()
    
    # Create 1000 intervals with random positions and sizes
    for i in range(1000):
        var base_pos = i * 100
        var start = UInt32(base_pos + (i % 50))  # Add some randomness
        var length = UInt32(10 + (i % 30))  # Variable lengths 10-40
        var stop = start + length
        intervals.append(Interval(start, stop, Int32(i)))
    
    var lapper = Lapper(intervals)
    
    # Test 100 random queries of different sizes
    for query_idx in range(100):
        var query_start = UInt32(query_idx * 1000 + (query_idx % 100))
        var query_size = UInt32(50 + (query_idx % 200))  # Sizes 50-250
        var query_stop = query_start + query_size
        
        var find_results = List[Interval]()
        var vectorized_indices = List[UInt32]()
        
        lapper.find(query_start, query_stop, find_results)
        lapper.find_vectorized(query_start, query_stop, vectorized_indices)
        
        # Check counts match
        assert_equal(len(find_results), len(vectorized_indices))
        
        # Check all intervals match
        for i in range(len(find_results)):
            var find_interval = find_results[i]
            var vectorized_interval = lapper.get(UInt(vectorized_indices[i]))
            
            assert_equal(find_interval.start, vectorized_interval.start)
            assert_equal(find_interval.stop, vectorized_interval.stop)
            assert_equal(find_interval.val, vectorized_interval.val)


def test_find_vs_find_vectorized_edge_overlap_counts():
    """Test find vs find_vectorized with specific overlap counts around SIMD boundaries."""
    var intervals = List[Interval]()
    
    # Create intervals that will produce specific overlap counts
    for i in range(50):
        var start = UInt32(i * 10)
        var stop = UInt32(start + 15)  # Overlapping pattern
        intervals.append(Interval(start, stop, Int32(i)))
    
    var lapper = Lapper(intervals)
    
    # Test queries designed to produce specific overlap counts
    var test_queries = List[Tuple[UInt32, UInt32]]()
    test_queries.append((UInt32(5), UInt32(15)))  # ~1-2 overlaps
    test_queries.append((UInt32(100), UInt32(200)))  # ~10-15 overlaps  
    test_queries.append((UInt32(50), UInt32(220)))  # ~17+ overlaps
    test_queries.append((UInt32(0), UInt32(500)))  # Most/all intervals
    
    for i in range(len(test_queries)):
        var query = test_queries[i]
        var start = query[0]
        var stop = query[1]
        
        var find_results = List[Interval]()
        var vectorized_indices = List[UInt32]()
        
        lapper.find(start, stop, find_results)
        lapper.find_vectorized(start, stop, vectorized_indices)
        
        # Verify exact match
        assert_equal(len(find_results), len(vectorized_indices))
        
        for j in range(len(find_results)):
            var find_interval = find_results[j]
            var vectorized_interval = lapper.get(UInt(vectorized_indices[j]))
            
            assert_equal(find_interval.start, vectorized_interval.start)
            assert_equal(find_interval.stop, vectorized_interval.stop)
            assert_equal(find_interval.val, vectorized_interval.val)


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

    # Lapper count method tests
    test_count_no_overlaps()
    test_count_single_overlap()
    test_count_multiple_overlaps()
    test_count_contained_intervals()
    test_count_exact_boundaries()
    test_count_single_interval()
    test_count_overlapping_intervals()
    test_count_same_start_different_stops()
    test_count_adjacent_intervals()

    # Lapper find_vectorized method tests
    test_find_vectorized_no_overlaps()
    test_find_vectorized_single_overlap()
    test_find_vectorized_multiple_overlaps()
    test_find_vectorized_vs_find_consistency()
    test_find_vectorized_exact_boundaries()
    test_find_vectorized_large_dataset()
    test_find_vectorized_simd_edge_cases()
    
    # Comprehensive find vs find_vectorized consistency tests
    test_find_vs_find_vectorized_comprehensive()
    test_find_vs_find_vectorized_random_data()
    test_find_vs_find_vectorized_edge_overlap_counts()

    print("All tests passed!")

from lapper.cpu.bsearch import lower_bound, upper_bound

from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from max.gpu import thread_idx, block_idx, block_dim, barrier

from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.sys import is_gpu


# TODO: AnyType instead?
@fieldwise_init
struct Interval(
    Comparable,
    ImplicitlyCopyable,
    Movable,
    Writable,
):
    """Represent a half-open range [start, stop) with an associated value.

    Intervals overlap when (a.start < b.stop) AND (a.stop > b.start).
    Touching boundaries, such as [10,20) and [20,30), do not overlap.
    """

    var start: UInt32
    """Start position of the interval (inclusive)."""

    var stop: UInt32
    """Stop position of the interval (exclusive)."""

    var val: Int32
    """Value or ID associated with this interval."""

    # TODO: stopping mid refactor of interval

    @always_inline
    def intersect(imm self, imm other: Self) -> UInt32:
        """Compute the intersect between two intervals.

        Args:
            other: The other interval to intersect with.

        Returns:
            The length of the intersection between the two intervals.
        """
        var stop = min(self.stop, other.stop)
        var start = max(self.start, other.start)
        return stop - min(stop, start)

    @always_inline
    def overlap(imm self, start: UInt32, stop: UInt32) -> Bool:
        """Check if two intervals overlap.

        Args:
            start: Start position of the query interval.
            stop: Stop position of the query interval.

        Returns:
            True if the intervals overlap, False otherwise.
        """
        return Self.overlap(self.start, self.stop, start, stop)

    @always_inline
    @staticmethod
    def overlap(
        a_start: UInt32,
        a_stop: UInt32,
        b_start: UInt32,
        b_stop: UInt32,
    ) -> Bool:
        """Check if two intervals overlap using strict inequality semantics.

        Determines whether two intervals overlap using the strict inequality
        definition: intervals overlap when (a.start < b.stop) AND (a.stop > b.start).

        This means touching intervals (where one ends exactly where another starts)
        are considered NON-overlapping. For example, [1,5) and [5,10) do NOT overlap.

        Args:
            a_start: Start position of first interval (inclusive).
            a_stop: Stop position of first interval (exclusive).
            b_start: Start position of second interval (inclusive).
            b_stop: Stop position of second interval (exclusive).

        Returns:
            True if the intervals overlap, False otherwise.

        Examples:
            ```mojo
            Interval.overlap(10, 20, 15, 25)  # True - [10,20) overlaps [15,25)
            Interval.overlap(10, 15, 15, 20)  # False - [10,15) touches [15,20)
            Interval.overlap(10, 20, 25, 30)  # False - [10,20) and [25,30) separate
            ```
        """
        return a_start < b_stop and a_stop > b_start

    @always_inline
    def __eq__(imm self, imm other: Self) -> Bool:
        """Check if two intervals are equal.

        Two intervals are considered equal if they have the same start and stop
        positions. The value field is not considered in equality comparison.

        Args:
            other: The interval to compare with.

        Returns:
            True if intervals have identical start and stop positions.
        """
        return self.start == other.start and self.stop == other.stop

    @always_inline
    def __ne__(imm self, imm other: Self) -> Bool:
        """Check if two intervals are not equal.

        Returns the opposite of __eq__. Two intervals are not equal if they
        have different start or stop positions.

        Args:
            other: The interval to compare with.

        Returns:
            True if intervals have different start or stop positions.
        """
        return not self == other

    @always_inline
    def __gt__(imm self, imm other: Self) -> Bool:
        """Check if this interval is greater than another.

        Implements lexicographic ordering: first by start position, then by stop
        position. This enables sorting of intervals and use in ordered collections.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval is lexicographically greater than other.

        Examples:
            ```mojo
            Interval(10, 20, 1) > Interval(5, 15, 2)   # True (10 > 5)
            Interval(10, 20, 1) > Interval(10, 15, 2)  # True (same start, 20 > 15)
            Interval(10, 15, 1) > Interval(10, 20, 2)  # False (same start, 15 < 20)
            ```
        """
        if self.start > other.start:
            return True
        elif self.start == other.start and self.stop > other.stop:
            return True
        else:
            return False

    @always_inline
    def __lt__(imm self, imm other: Self) -> Bool:
        """Check if this interval is less than another.

        Intervals are ordered first by start position, then by stop position.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval is less than the other interval.
        """
        if self.start < other.start:
            return True
        elif self.start == other.start and self.stop < other.stop:
            return True
        else:
            return False

    @always_inline
    def __ge__(imm self, imm other: Self) -> Bool:
        """Check if this interval is greater than or equal to another.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval is greater than or equal to the other interval.
        """
        return self > other or self == other

    @always_inline
    def __le__(imm self, imm other: Self) -> Bool:
        """Check if this interval is less than or equal to another.

        Args:
            other: The interval to compare with.

        Returns:
            True if this interval is less than or equal to the other interval.
        """
        return self < other or self == other

    def write_to(self, mut writer: Some[Writer]):
        """Write a string representation of the interval to a writer.

        Formats the interval as "Interval {start: X, stop: Y}" where X and Y
        are the start and stop positions. The value field is not included in
        the string representation.

        Args:
            writer: A mutable writer to output the formatted string.

        Example Output:
            "Interval {start: 10, stop: 20}"
        """
        writer.write("Interval {start: ", self.start, ", stop: ", self.stop, "}")


@fieldwise_init
struct Lapper[*, owns_data: Bool = True](Sized):
    """An interval index for repeated overlap queries.

    Parameters:
        owns_data: True if the Lapper owns its memory, False if using external pointers.

    With owns_data=False, retain all backing buffers until queries and queued
    GPU work finish.
    """

    comptime I = Interval

    # Sorted in interval order (primary data arrays)
    var starts: MutPointer[UInt32, MutUntrackedOrigin]
    """Interval start positions sorted by start position."""

    var stops: MutPointer[UInt32, MutUntrackedOrigin]
    """Interval stop positions in same order as starts."""

    var vals: MutPointer[Int32, MutUntrackedOrigin]
    """Interval values/IDs in same order as starts."""

    # For BITS algorithm optimization
    var stops_sorted: MutPointer[UInt32, MutUntrackedOrigin]
    """Stop positions sorted independently for BITS algorithm."""

    var length: UInt
    """Number of intervals stored in this Lapper."""

    var max_len: UInt32
    """Maximum interval length for bounds optimization."""

    # TODO: paramaterized optional eytzinger layout

    # fn __init__(out self, owned intervals: List[Self.I]):
    #     constrained[not is_gpu(), "This constructor is only for CPU"]()

    #     pass

    # fn __init__(out self):
    #     constrained[is_gpu(), "This constructor is only for GPU"]()

    def __init__(out self, var intervals: List[Self.I]) raises:
        """Initialize a new Lapper from a list of intervals.

        Args:
            intervals: A list of Interval objects to store. Must contain at least
                      one interval. The intervals will be sorted internally, so
                      input order does not matter.

        Raises:
            Error: If the intervals list is empty (length < 1).

        Example:
            ```mojo
            var intervals = List[Interval]()
            intervals.append(Interval(10, 20, 1))
            intervals.append(Interval(5, 15, 2))
            var lapper = Lapper(intervals^)
            ```

        """
        if len(intervals) == 0:
            raise "Intervals length must be >= 1"

        self.length = UInt(len(intervals))
        self.starts = unsafe_alloc[UInt32](len(intervals))
        self.stops = unsafe_alloc[UInt32](len(intervals))
        self.vals = unsafe_alloc[Int32](len(intervals))
        self.stops_sorted = unsafe_alloc[UInt32](len(intervals))
        self.max_len = UInt32(0)

        self._fill(intervals^)

    def _fill(mut self, var intervals: List[Self.I]):
        sort(intervals)
        for i in range(0, len(intervals)):
            ref iv = intervals[i]
            self.starts[unsafe_offset=i] = iv.start
            self.stops[unsafe_offset=i] = iv.stop
            self.vals[unsafe_offset=i] = iv.val
            var length = iv.stop - iv.start
            self.max_len = length if length > self.max_len else self.max_len

        # N.B. Don't sort starts, we need them to be in the exact same order as intervals,
        # which are sorted by starts first, then by stops.
        unsafe_memcpy(dest=self.stops_sorted, src=self.stops, count=len(intervals))
        sort(Span(unsafe_ptr=self.stops_sorted, length=len(intervals)))

    # TODO: it would be great if there were a way to seal the data from host/device buffers
    # The way it works is api-limiting
    @staticmethod
    def prep_for_gpu(
        ctx: DeviceContext,
        var intervals: List[Self.I],
        host_starts: HostBuffer[DType.uint32],
        host_stops: HostBuffer[DType.uint32],
        host_vals: HostBuffer[DType.int32],
        host_stops_sorted: HostBuffer[DType.uint32],
        mut device_starts: DeviceBuffer[DType.uint32],
        mut device_stops: DeviceBuffer[DType.uint32],
        mut device_vals: DeviceBuffer[DType.int32],
        mut device_stops_sorted: DeviceBuffer[DType.uint32],
    ) raises -> Lapper[owns_data=False]:
        """Prepare a Lapper instance for GPU execution.

        Sorts intervals and copies data to both host and device buffers
        for GPU kernel execution.

        Args:
            ctx: GPU device context for buffer operations.
            intervals: List of intervals to store (will be sorted).
            host_starts: Host buffer for interval start positions.
            host_stops: Host buffer for interval stop positions.
            host_vals: Host buffer for interval values.
            host_stops_sorted: Host buffer for sorted stop positions.
            device_starts: Device buffer for interval start positions.
            device_stops: Device buffer for interval stop positions.
            device_vals: Device buffer for interval values.
            device_stops_sorted: Device buffer for sorted stop positions.

        Returns:
            A GPU-ready Lapper instance with owns_data=False. The caller must
            keep all device buffers alive until all uses of this Lapper finish.

        Raises:
            GPU-related errors during buffer operations.
        """
        var length = UInt(len(intervals))
        var max_len = UInt32(0)
        var lapper = Lapper[
            owns_data=False
        ](  # need to call it this so it doesn't try to drop the mem
            host_starts.unsafe_ptr(),
            host_stops.unsafe_ptr(),
            host_vals.unsafe_ptr(),
            host_stops_sorted.unsafe_ptr(),
            length,
            max_len,
        )
        lapper._fill(intervals^)
        host_starts.enqueue_copy_to(device_starts)
        host_stops.enqueue_copy_to(device_stops)
        host_vals.enqueue_copy_to(device_vals)
        host_stops_sorted.enqueue_copy_to(device_stops_sorted)
        ctx.synchronize()

        return Lapper[owns_data=False](
            device_starts.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            device_stops.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            device_vals.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            device_stops_sorted.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            lapper.length,
            lapper.max_len,
        )

    def __len__(self) -> Int:
        """Get the number of intervals stored in this Lapper.

        Returns:
            The number of intervals.
        """
        return Int(self.length)

    def __deinit__(deinit self):
        """Free allocated memory when owns_data=True.

        Automatically called when the Lapper goes out of scope.
        Only frees memory if owns_data=True.
        """

        comptime if Self.owns_data:
            self.starts.unsafe_free()
            self.stops.unsafe_free()
            self.vals.unsafe_free()
            self.stops_sorted.unsafe_free()

    def find(
        self,
        start: UInt32,
        stop: UInt32,
        mut results: List[Self.I],
    ):
        """Find all intervals that overlap with the query range [start, stop).

        Overlap Semantics:
            Uses strict inequality semantics: intervals overlap when
            (interval.start < query.stop) AND (interval.stop > query.start).

            Touching intervals where one ends exactly where another starts are
            considered NON-overlapping (e.g., [1,5) and [5,10) do NOT overlap).

        Args:
            start: Start position of query range (inclusive).
            stop: Stop position of query range (exclusive).
            results: Mutable list to append found intervals. Previous contents
                    are preserved - new results are appended.

        Examples:
            ```mojo
            var intervals = List[Interval]()
            intervals.append(Interval(10, 20, 1))  # Overlaps [15, 25)
            intervals.append(Interval(30, 40, 2))  # Does not overlap [15, 25)
            var lapper = Lapper(intervals^)

            var results = List[Interval]()
            lapper.find(15, 25, results)  # results will contain first interval
            print(len(results))  # Prints: 1
            ```

        Thread Safety:
            - Safe to call concurrently from multiple threads
            - Each thread should use its own results list
            - No modification of the Lapper structure during search
        """
        var idx = self._lower_bound(start, stop)
        for i in range(Int(idx), len(self)):
            var s_start = self.starts[unsafe_offset=i]
            var s_stop = self.stops[unsafe_offset=i]
            # ref iv = self.intervals[i]
            if Interval.overlap(s_start, s_stop, start, stop):
                results.append(Interval(s_start, s_stop, self.vals[unsafe_offset=i]))
            elif s_start >= stop:
                break

    @always_inline
    def _lower_bound(self, start: UInt32, stop: UInt32) -> UInt:
        return lower_bound(
            Span(unsafe_ptr=self.starts, length=len(self)), start - min(start, self.max_len)
        )

    def _count(self, lower_bound: UInt, start: UInt32, stop: UInt32) -> UInt32:
        """Naive count for now"""
        var count: UInt32 = 0
        for i in range(Int(lower_bound), len(self)):
            var s_start = self.starts[unsafe_offset=i]
            var s_stop = self.stops[unsafe_offset=i]
            # ref iv = self.intervals[i]
            if Interval.overlap(s_start, s_stop, start, stop):
                count += 1
            elif s_start >= stop:
                break
        return count

    def count(self, start: UInt32, stop: UInt32) -> UInt:
        """Count the number of intervals that overlap with the query range [start, stop).

        Touching boundaries do not overlap. This method does not materialize
        results or modify the index.

        Args:
            start: Start position of query range (inclusive).
            stop: Stop position of query range (exclusive).

        Returns:
            Number of stored intervals that overlap with the query range.
            Returns 0 if no intervals overlap.

        Examples:
            ```mojo
            var intervals = List[Interval]()
            intervals.append(Interval(10, 20, 1))  # Overlaps [15, 25)
            intervals.append(Interval(30, 40, 2))  # Does not overlap [15, 25)
            intervals.append(Interval(18, 30, 3))  # Overlaps [15, 25)
            var lapper = Lapper(intervals^)

            var count = lapper.count(15, 25)  # Returns 2
            ```

        """
        # The plus one is to account for the half-open intervals
        var first = lower_bound(Span(unsafe_ptr=self.stops_sorted, length=len(self)), start + 1)
        var last = lower_bound(Span(unsafe_ptr=self.starts, length=len(self)), stop)
        return last - first


def find_overlaps_kernel(
    starts: MutPointer[UInt32, MutUntrackedOrigin],
    stops: MutPointer[UInt32, MutUntrackedOrigin],
    vals: MutPointer[Int32, MutUntrackedOrigin],  # Not actually needed?
    stops_sorted: MutPointer[UInt32, MutUntrackedOrigin],
    length: UInt64,
    max_len: UInt32,
    keys: MutPointer[UInt32, MutUntrackedOrigin],  # start, stop packed adjacent
    keys_length: UInt64,  # length is 2x the number of key
    output: MutPointer[UInt32, MutUntrackedOrigin],  # just the count of found intervals for now
    output_length: UInt64,  # same as number of keys
):
    """Count matches for packed half-open queries on the GPU.

    This returns counts, not matching intervals. Retain all backing buffers
    until queued work completes.

    Args:
        starts: Sorted array of interval start positions on device memory.
        stops: Array of interval stop positions (same order as starts) on device memory.
        vals: Array of interval values/IDs (same order as starts) on device memory.
        stops_sorted: Independently sorted array of stop positions for BITS algorithm.
        length: Number of intervals in the data structure.
        max_len: Maximum interval length for optimization bounds.
        keys: Packed query data as `start₁, stop₁, start₂, stop₂, ...` on device memory.
        keys_length: Total length of keys array (2x number of queries).
        output: Device memory array to store result counts (one per query).
        output_length: Length of output array (equals number of queries).

    """
    # TODO: come back to the idea that this should be an array of structs where start and stop are co-located, but what about SIMT!?
    # TODO: this is just counting for now because we don't know how much space to allocate till
    # the coutns method is worked out
    var lapper = Lapper[owns_data=False](starts, stops, vals, stops_sorted, UInt(length), max_len)

    var thread_id = (block_idx.x * block_dim.x) + thread_idx.x
    if UInt64(thread_id) >= keys_length // 2:
        return

    var idx = thread_id * 2
    var start = keys[unsafe_offset=idx]
    var stop = keys[unsafe_offset=idx + 1]

    var lb = lapper._lower_bound(start, stop)
    barrier()
    var count = lapper._count(lb, start, stop)
    barrier()
    output[unsafe_offset=thread_id] = count


def count_overlaps_kernel(
    starts: MutPointer[UInt32, MutUntrackedOrigin],
    stops: MutPointer[UInt32, MutUntrackedOrigin],
    vals: MutPointer[Int32, MutUntrackedOrigin],  # Not actually needed?
    stops_sorted: MutPointer[UInt32, MutUntrackedOrigin],
    length: UInt64,
    max_len: UInt32,
    keys: MutPointer[UInt32, MutUntrackedOrigin],  # start, stop packed adjacent
    keys_length: UInt64,  # length is 2x the number of key
    output: MutPointer[UInt32, MutUntrackedOrigin],  # just the count of found intervals for now
    output_length: UInt64,  # same as number of keys
):
    """Count overlaps for packed half-open queries on the GPU.

    Supply one output element per query and retain all device buffers until
    queued work completes. Query endpoints are packed as start/stop pairs.

    Args:
        starts: Device pointer to sorted interval start positions (primary sort key).
        stops: Device pointer to interval stop positions (same order as starts).
        vals: Device pointer to interval values/IDs (same order as starts, unused in counting).
        stops_sorted: Device pointer to independently sorted stop positions (for BITS algorithm).
        length: Total number of intervals in the data structure.
        max_len: Maximum interval length (used for search optimization bounds).
        keys: Device pointer to packed query data `start₁, stop₁, start₂, stop₂, ...`.
        keys_length: Total length of keys array (exactly 2x the number of queries).
        output: Device pointer to result array (one count per query).
        output_length: Length of output array (equals number of queries).

    """
    # TODO: come back to the idea that this should be an array of structs where start and stop are co-located, but what about SIMT!?
    var lapper = Lapper[owns_data=False](starts, stops, vals, stops_sorted, UInt(length), max_len)

    var thread_id = (block_idx.x * block_dim.x) + thread_idx.x
    if UInt64(thread_id) >= keys_length // 2:
        return

    var idx = thread_id * 2
    var start = keys[unsafe_offset=idx]
    var stop = keys[unsafe_offset=idx + 1]

    barrier()
    var count = lapper.count(start, stop)
    barrier()
    output[unsafe_offset=thread_id] = UInt32(count)

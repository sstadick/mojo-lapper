from ExtraMojo.math.ops import saturating_sub

from lapper.cpu.bsearch import lower_bound

from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, warp, barrier

from memory import UnsafePointer, memcpy
from sys import is_gpu


# TODO: AnyType instead?
@fieldwise_init
struct Interval(
    Copyable,
    EqualityComparable,
    GreaterThanComparable,
    GreaterThanOrEqualComparable,
    LessThanComparable,
    LessThanOrEqualComparable,
    Movable,
    Writable,
):
    """Represent a range from [start, stop).

    Inclusive start, exclusive stop
    """

    var start: UInt32
    var stop: UInt32
    var val: Int32

    # TODO: stopping mid refactor of interval

    @always_inline
    fn intersect(read self, read other: Self) -> UInt32:
        """Compute the intersect between two intervals."""
        return saturating_sub(
            min(self.stop, other.stop), max(self.start, other.start)
        )

    @always_inline
    fn overlap(read self, start: UInt32, stop: UInt32) -> Bool:
        """Check if two intervals overlap."""
        return Self.overlap(self.start, self.stop, start, stop)

    @always_inline
    @staticmethod
    fn overlap(
        a_start: UInt32,
        a_stop: UInt32,
        b_start: UInt32,
        b_stop: UInt32,
    ) -> Bool:
        return a_start < b_stop and a_stop > b_start

    @always_inline
    fn __eq__(read self, read other: Self) -> Bool:
        return self.start == other.start and self.stop == other.stop

    @always_inline
    fn __ne__(read self, read other: Self) -> Bool:
        return not self == other

    @always_inline
    fn __gt__(read self, read other: Self) -> Bool:
        if self.start > other.start:
            return True
        elif self.start == other.start and self.stop > other.stop:
            return True
        else:
            return False

    @always_inline
    fn __lt__(read self, read other: Self) -> Bool:
        if self.start < other.start:
            return True
        elif self.start == other.start and self.stop < other.stop:
            return True
        else:
            return False

    @always_inline
    fn __ge__(read self, read other: Self) -> Bool:
        return self > other or self == other

    @always_inline
    fn __le__(read self, read other: Self) -> Bool:
        return self < other or self == other

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Interval {start: ", self.start, ", stop: ", self.stop, "}"
        )


@fieldwise_init
struct Lapper[*, data_location: String = "cpu"](Sized):
    alias I = Interval

    # Sorted in interval order
    var starts: UnsafePointer[UInt32]
    var stops: UnsafePointer[UInt32]
    var vals: UnsafePointer[Int32]

    # For BITS
    var stops_sorted: UnsafePointer[UInt32]

    var length: UInt

    var max_len: UInt32

    # TODO: paramaterized optional eytzinger layout

    # fn __init__(out self, owned intervals: List[Self.I]):
    #     constrained[not is_gpu(), "This constructor is only for CPU"]()

    #     pass

    # fn __init__(out self):
    #     constrained[is_gpu(), "This constructor is only for GPU"]()

    fn __init__(out self, owned intervals: List[Self.I]) raises:
        if len(intervals) == 0:
            raise "Intervals length must be >= 1"
        sort(intervals)

        self.length = len(intervals)
        self.starts = UnsafePointer[UInt32].alloc(len(intervals))
        self.stops = UnsafePointer[UInt32].alloc(len(intervals))
        self.vals = UnsafePointer[Int32].alloc(len(intervals))
        self.stops_sorted = UnsafePointer[UInt32].alloc(len(intervals))
        self.max_len = UInt32(0)

        self._fill(intervals)

    fn _fill(mut self, owned intervals: List[Self.I]):
        for i in range(0, len(intervals)):
            ref iv = intervals[i]
            self.starts[i] = iv.start
            self.stops[i] = iv.stop
            self.vals[i] = iv.val
            var length = iv.stop - iv.start
            self.max_len = length if length > self.max_len else self.max_len

        # N.B. Don't sort starts, we need them to be in the exact same order as intervals,
        # which are sorted by starts first, then by stops.
        memcpy(self.stops_sorted, self.stops, len(intervals))
        sort(Span(ptr=self.stops_sorted, length=len(intervals)))

    @staticmethod
    fn prep_for_gpu(
        ctx: DeviceContext, intervals: List[Self.I]
    ) raises -> Self[data_location="gpu"]:
        var length = len(intervals)
        var max_len = UInt32(0)
        var starts = ctx.enqueue_create_host_buffer[DType.uint32](length)
        var stops = ctx.enqueue_create_host_buffer[DType.uint32](length)
        var vals = ctx.enqueue_create_host_buffer[DType.int32](length)
        var stops_sorted = ctx.enqueue_create_host_buffer[DType.uint32](length)
        ctx.synchronize()
        lapper = Lapper(
            starts.unsafe_ptr(),
            stops.unsafe_ptr(),
            vals.unsafe_ptr(),
            stops_sorted.unsafe_ptr(),
            length,
            max_len,
        )
        lapper._fill(intervals)
        var starts_dev = ctx.enqueue_create_buffer[DType.uint32](length)
        var stops_dev = ctx.enqueue_create_buffer[DType.uint32](length)
        var vals_dev = ctx.enqueue_create_buffer[DType.int32](length)
        var stops_sorted_dev = ctx.enqueue_create_buffer[DType.uint32](length)
        starts.enqueue_copy_to(starts_dev)
        stops.enqueue_copy_to(stops_dev)
        vals.enqueue_copy_to(vals_dev)
        starts_sorted.enqueue_copy_to(starts_sorted_dev)
        ctx.synchronize()

        return Lapper[data_locaton="gpu"](
            starts_dev.unsafe_ptr(),
            stops_dev.unsafe_ptr(),
            vals_dev.unsafe_ptr(),
            stops_sorted_dev.unsafe_ptr(),
            lapper.length,
            lapper.max_len,
        )

    fn __len__(read self) -> Int:
        return self.length

    fn __del__(owned self):
        @parameter
        if data_location != "gpu":
            self.starts.free()
            self.stops.free()
            self.vals.free()
            self.stops_sorted.free()

    fn find(
        read self,
        start: UInt32,
        stop: UInt32,
        mut results: List[Self.I],
    ):
        var idx = self._lower_bound(start, stop)
        for i in range(idx, len(self)):
            var s_start = self.starts[i]
            var s_stop = self.stops[i]
            # ref iv = self.intervals[i]
            if Interval.overlap(s_start, s_stop, start, stop):
                results.append(Interval(s_start, s_stop, self.vals[i]))
            elif s_start >= stop:
                break

    @always_inline
    fn _lower_bound(read self, start: UInt32, stop: UInt32) -> UInt:
        return lower_bound(
            Span(self.starts, len(self)), saturating_sub(start, self.max_len)
        )

    fn _count(
        read self, lower_bound: UInt, start: UInt32, stop: UInt32
    ) -> UInt32:
        """Naive count for now"""
        var count: UInt32 = 0
        for i in range(lower_bound, len(self)):
            var s_start = self.starts[i]
            var s_stop = self.stops[i]
            # ref iv = self.intervals[i]
            if Interval.overlap(s_start, s_stop, start, stop):
                count += 1
            elif s_start >= stop:
                break
        return count

    @staticmethod
    fn find_overlaps_kernel(
        starts: UnsafePointer[UInt32],
        stops: UnsafePointer[UInt32],
        vals: UnsafePointer[Int32],  # Not actually needed?
        stops_sorted: UnsafePointer[UInt32],
        length: UInt,
        max_len: UInt32,
        keys: UnsafePointer[UInt32],  # start, stop packed adjacent
        keys_length: UInt,  # length is 2x the number of key
        output: UnsafePointer[
            UInt32
        ],  # just the count of found intervals for now
        output_length: UInt,  # same as number of keys
    ):
        # TODO: come back to the idea that this should be an array of structs where start and stop are co-located, but what about SIMT!?
        var lapper = Lapper[data_location="gpu"](
            starts, stops, vals, stops_sorted, length, max_len
        )

        var thread_id = (block_idx.x * block_dim.x) + thread_idx.x
        if thread_id >= keys_length // 2:
            return

        var idx = thread_id * 2
        var start = keys[idx]
        var stop = keys[idx + 1]

        var lb = lapper._lower_bound(start, stop)
        barrier()
        var count = lapper._count(lb, start, stop)
        barrier()
        output[thread_id] = count


# make sure starts list is in the same order as starts, exactly
# then use the starts list for the `lower_bound` search so eytzinger is allowed

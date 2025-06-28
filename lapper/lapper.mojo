from ExtraMojo.math.ops import saturating_sub
from lapper.cpu.bsearch import lower_bound as bs_lower_bound


# TODO: AnyType instead?
@fieldwise_init
struct Interval[dtype: DType, T: AnyTrivialRegType](
    Copyable,
    EqualityComparable,
    GreaterThanComparable,
    GreaterThanOrEqualComparable,
    LessThanComparable,
    LessThanOrEqualComparable,
    Movable,
):
    """Represent a range from [start, stop).

    Inclusive start, exclusive stop
    """

    var start: Scalar[dtype]
    var stop: Scalar[dtype]
    var val: T

    @always_inline
    fn intersect(read self, read other: Self) -> Scalar[dtype]:
        """Compute the intersect between two intervals."""
        return saturating_sub(
            min(self.stop, other.stop), max(self.start, other.start)
        )

    @always_inline
    fn overlap(read self, start: Scalar[dtype], stop: Scalar[dtype]) -> Bool:
        """Check if two intervals overlap."""
        return self.start < stop and self.stop > start

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
        elif self.stop > other.stop:
            return True
        else:
            return False

    @always_inline
    fn __lt__(read self, read other: Self) -> Bool:
        return not self > other

    @always_inline
    fn __ge__(read self, read other: Self) -> Bool:
        return self > other or self == other

    @always_inline
    fn __le__(read self, read other: Self) -> Bool:
        return self < other or self == other


struct Lapper[dtype: DType, T: AnyTrivialRegType]:
    alias I = Interval[dtype, T]
    var intervals: List[Self.I]
    var starts: List[Scalar[dtype]]
    var stops: List[Scalar[dtype]]
    var max_len: Scalar[dtype]

    # TODO: paramaterized optional eytzinger layout

    fn __init__(out self, owned intervals: List[Self.I]):
        sort(intervals)

        var starts = List[Scalar[dtype]](unsafe_uninit_length=len(intervals))
        var stops = List[Scalar[dtype]](unsafe_uninit_length=len(intervals))

        var max_len = Scalar[dtype](0)
        for i in range(0, len(intervals)):
            ref iv = intervals[i]
            starts.unsafe_set(i, iv.start)
            stops.unsafe_set(i, iv.stop)
            var length = iv.stop - iv.start
            max_len = length if length > max_len else max_len

        # N.B. Don't sort starts, we need them to be in the exact same order as intervals,
        # which are sorted by starts first, then by stops.
        sort(stops)

        self.intervals = intervals^
        self.starts = starts^
        self.stops = stops^
        self.max_len = max_len

    fn __len__(read self) -> Int:
        return len(self.intervals)

    fn find(
        read self,
        start: Scalar[dtype],
        stop: Scalar[dtype],
        mut results: List[Self.I],
    ):
        var idx = bs_lower_bound(
            self.starts, saturating_sub(start, self.max_len)
        )

        for i in range(idx, len(self)):
            rev iv = self.intervals[i]
            if iv.overlap(start, stop):
                results.append(iv)


# make sure starts list is in the same order as starts, exactly
# then use the starts list for the `lower_bound` search so eytzinger is allowed

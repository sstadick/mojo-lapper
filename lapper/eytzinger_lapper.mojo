from lapper.lapper import Lapper, Interval
from lapper.cpu.eytzinger import Eytzinger, eytzinger_with_lookup, lower_bound


struct EzLapper(Sized):
    var inner: Lapper
    var starts: Eytzinger[DType.uint32]
    var stops: Eytzinger[DType.uint32]

    fn __init__(out self, owned intervals: List[Interval]) raises:
        var inner = Lapper(intervals)
        var starts = eytzinger_with_lookup(Span(inner.starts, len(inner)))
        var stops = eytzinger_with_lookup(Span(inner.stops_sorted, len(inner)))
        self.inner = inner^
        self.starts = starts
        self.stops = stops

    fn __len__(read self) -> Int:
        return len(self.inner)

    fn count(read self, start: UInt32, stop: UInt32) -> UInt:
        # Find first interval whose stop > query.start (using stops_sorted)
        var first_idx = lower_bound(Span(self.stops.layout), start + 1)
        var first: UInt
        if first_idx == 0:
            # Special case: start+1 is larger than all stop values
            first = len(self)
        elif first_idx >= len(self.stops.lookup):
            # Out of bounds - no intervals can overlap
            first = len(self)
        else:
            first = UInt(self.stops.lookup[first_idx])

        # Find first interval whose start >= query.stop (using starts)
        var last_idx = lower_bound(Span(self.starts.layout), stop)
        var last: UInt
        if last_idx == 0:
            # Special case: stop is larger than all start values
            last = len(self)
        elif last_idx >= len(self.starts.lookup):
            # Out of bounds
            last = len(self)
        else:
            last = UInt(self.starts.lookup[last_idx])

        var num_cant_after = len(self) - last
        return UInt(len(self) - first - num_cant_after)

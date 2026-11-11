"""The 32-interval forward-query index used by rust-lapper 2.0.1.

Adapted from rust-lapper (MIT), Copyright (c) 2019 Seth M. Stadick.
"""

from std.memory import pack_bits
from std.sys import simd_width_of

comptime BLOCK_SIZE = 32
comptime _WIDTH = min(BLOCK_SIZE, simd_width_of[UInt32]())


struct BlockIndex(Movable):
    """CPU-owned end summaries; the primary start/stop arrays stay in Lapper."""

    var max_tree: List[UInt32]
    var min_ends: List[UInt32]
    var prefix_max_ends: List[UInt32]
    var leaves: Int

    def __init__(out self):
        self.max_tree = List[UInt32]()
        self.min_ends = List[UInt32]()
        self.prefix_max_ends = List[UInt32]()
        self.leaves = 0

    def build(mut self, stops: Span[UInt32, _]):
        """Summarize stops in start order, padding tree leaves with zero."""
        var blocks = (len(stops) + BLOCK_SIZE - 1) // BLOCK_SIZE
        self.leaves = 1
        while self.leaves < blocks:
            self.leaves *= 2
        self.max_tree = List[UInt32](length=2 * self.leaves, fill=0)
        self.min_ends = List[UInt32](capacity=blocks)
        self.prefix_max_ends = List[UInt32](capacity=blocks)
        var prefix = UInt32(0)
        for block in range(blocks):
            var begin = block * BLOCK_SIZE
            var smallest = stops[begin]
            var largest = smallest
            for i in range(begin + 1, min(begin + BLOCK_SIZE, len(stops))):
                smallest = min(smallest, stops[i])
                largest = max(largest, stops[i])
            self.max_tree[self.leaves + block] = largest
            self.min_ends.append(smallest)
            prefix = max(prefix, largest)
            self.prefix_max_ends.append(prefix)
        for node in range(self.leaves - 1, 0, -1):
            self.max_tree[node] = max(self.max_tree[2 * node], self.max_tree[2 * node + 1])

    @no_inline
    def next_above(self, block: Int, start: UInt32) -> Int:
        """Find the next block with an end > start, or a past-end leaf.

        Climb to a live subtree on the right, then descend to its first live
        leaf. This skips dead runs even when their maximum ends increase.
        """
        var node = self.leaves + block
        while True:
            if node & 1 == 0 and self.max_tree[node + 1] > start:
                node += 1
                break
            node >>= 1
            if node <= 1:
                return self.leaves
        while node < self.leaves:
            node *= 2
            node += Int(self.max_tree[node] <= start)
        return node - self.leaves


@always_inline
def overlap_mask(
    starts: Span[UInt32, _], stops: Span[UInt32, _], start: UInt32, stop: UInt32
) -> UInt32:
    """Return ascending-lane matches for at most 32 intervals, without overreads."""
    var mask = UInt32(0)
    var i = 0
    while i + _WIDTH <= len(starts):
        var s = starts.unsafe_ptr().unsafe_load[width=_WIDTH, alignment=1](i)
        var e = stops.unsafe_ptr().unsafe_load[width=_WIDTH, alignment=1](i)
        mask |= UInt32(pack_bits(s.lt(stop) & e.gt(start))) << UInt32(i)
        i += _WIDTH
    while i < len(starts):
        if starts[i] < stop and stops[i] > start:
            mask |= UInt32(1) << UInt32(i)
        i += 1
    return mask

"""Deterministic CPU find workloads, with construction timed separately.

Build with mojo build -O3 --march native -I . benchmarks/bench_find.mojo.
Run on an idle pinned core; repeat the executable for independent samples.
Compare checksums before comparing timings across revisions.
"""

from lapper.lapper import Interval, Lapper
from std.time import perf_counter_ns
from std.testing import assert_equal


def run(shape: String, n: Int, queries: Int) raises:
    var intervals = List[Interval](capacity=n)
    for i in range(n):
        var start = UInt32(i * 8)
        var stop = start + 3
        if shape == "span" and i == 0:
            stop = UInt32(n * 8)
        elif shape == "dense":
            stop += 4096
        elif shape == "mixed":
            stop += UInt32((i * 17) % 8192)
        intervals.append(Interval(start, stop, Int32(i)))

    var begin = perf_counter_ns()
    var index = Lapper(intervals^)
    var built = perf_counter_ns()
    var results = List[Interval]()
    var matches = UInt64(0)
    var checksum = UInt64(0)
    var expected_matches = UInt64(0)
    var expected_checksum = UInt64(0)
    # Compute the sparse oracle outside both timed regions.
    for q in range(queries):
        var i = (q * 7919) % n
        expected_matches += 1 + UInt64(shape == "span" and i != 0)
        expected_checksum += UInt64(i)
    var queried = perf_counter_ns()
    for q in range(queries):
        var start = UInt32(((q * 7919) % n) * 8 + 1)
        results.clear()
        index.find(start, start + 1, results)
        matches += UInt64(len(results))
        for iv in results:
            checksum += UInt64(iv.val)
    var end = perf_counter_ns()
    if shape == "plain" or shape == "span":
        assert_equal(matches, expected_matches)
        assert_equal(checksum, expected_checksum)
    print(
        shape,
        n,
        queries,
        Float64(built - begin) / 1e6,
        Float64(end - queried) / 1e6,
        matches,
        checksum,
        sep=",",
    )


def main() raises:
    print("shape,n,queries,build_ms,query_ms,matches,checksum")
    run("plain", 100_000, 20_000)
    run("span", 100_000, 20_000)
    run("dense", 100_000, 20_000)
    run("mixed", 100_000, 20_000)

from lapper.bed import SimpleBedRecord
from lapper.lapper import (
    Lapper,
    Interval,
    find_overlaps_kernel,
    count_overlaps_kernel,
)
from lapper.eytzinger_lapper import EzLapper
from random import randint, seed
from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from memory import memcpy, UnsafePointer
from math import ceildiv
from sys.info import has_accelerator
from gpu.host import DeviceContext


def generate_intervals(
    num_intervals: Int, max_coordinate: Int
) -> List[Interval]:
    """Generate random intervals for testing."""
    var intervals = List[Interval]()

    var starts = List[UInt32](unsafe_uninit_length=num_intervals)
    var stops = List[UInt32](unsafe_uninit_length=num_intervals)
    var vals = List[Int32](unsafe_uninit_length=num_intervals)

    # Generate random data
    randint(starts.unsafe_ptr(), num_intervals, 0, max_coordinate - 100)
    randint(stops.unsafe_ptr(), num_intervals, 1, 10_000)  # interval lengths
    randint(vals.unsafe_ptr(), num_intervals, 0, 1000)

    for i in range(num_intervals):
        var start = starts[i]
        var stop = start + stops[i]  # stops contains lengths initially
        var val = vals[i]
        intervals.append(Interval(start, stop, val))

    return intervals


def generate_sparse_intervals(
    num_intervals: Int, max_coordinate: Int
) -> List[Interval]:
    """Generate sparse intervals with minimal overlaps."""
    var intervals = List[Interval]()

    # Create intervals that are well-spaced to minimize overlaps
    # With 100k intervals in 1M coordinates, we have spacing of ~10
    # Make intervals smaller and more spread out
    var spacing = max_coordinate // num_intervals

    for i in range(num_intervals):
        # Use smaller offset to keep intervals well separated
        var offset_list = List[Int32](unsafe_uninit_length=1)
        var length_list = List[Int32](unsafe_uninit_length=1)
        randint[DType.int32](offset_list.unsafe_ptr(), 1, 0, spacing // 4)
        # Much smaller intervals to reduce overlaps
        randint[DType.int32](
            length_list.unsafe_ptr(), 1, 1, min(5, spacing // 2)
        )

        var start = UInt32(i * spacing + offset_list[0])
        var length = UInt32(length_list[0])
        var stop = start + length
        var val = Int32(i)
        intervals.append(Interval(start, stop, val))

    return intervals


def generate_dense_intervals(
    num_intervals: Int, max_coordinate: Int
) -> List[Interval]:
    """Generate moderately dense intervals targeting 8-16 overlaps per query."""
    var intervals = List[Interval]()

    # More careful distribution to avoid too many overlaps
    # Spread intervals more evenly
    var base_spacing = (
        max_coordinate // num_intervals
    )  # ~10 for 100k intervals in 1M space

    for i in range(num_intervals):
        var base_position = i * base_spacing

        # Add some randomness but keep intervals somewhat separated
        var offset_list = List[Int32](unsafe_uninit_length=1)
        var length_list = List[Int32](unsafe_uninit_length=1)

        # Small random offset from base position
        randint[DType.int32](
            offset_list.unsafe_ptr(), 1, -base_spacing // 4, base_spacing // 4
        )
        # Much smaller intervals to reduce overlap density
        randint[DType.int32](length_list.unsafe_ptr(), 1, 5, 20)

        var start = UInt32(max(0, base_position + offset_list[0]))
        var length = UInt32(length_list[0])
        var stop = min(UInt32(max_coordinate), start + length)
        var val = Int32(i)
        intervals.append(Interval(start, stop, val))

    return intervals


def generate_queries(
    num_queries: Int, max_coordinate: Int, sorted: Bool = True
) -> List[Interval]:
    """Generate random query intervals, optionally sorted by start position."""
    var queries = List[Interval]()

    var starts = List[UInt32](unsafe_uninit_length=num_queries)
    var lengths = List[UInt32](unsafe_uninit_length=num_queries)

    randint(starts.unsafe_ptr(), num_queries, 0, max_coordinate - 50)
    randint(lengths.unsafe_ptr(), num_queries, 5, 50)

    for i in range(num_queries):
        var start = starts[i]
        var stop = start + lengths[i]
        queries.append(Interval(start, stop, 0))

    # Sort queries by start position if requested
    if sorted:
        sort(queries)

    return queries


def generate_dense_queries(
    num_queries: Int, max_coordinate: Int, base_spacing: Int = 10
) -> List[Interval]:
    """Generate queries with a gaussian-like distribution of overlaps.

    Target distribution:
    - 40% of queries: 8-16 overlaps (peak of distribution)
    - 30% of queries: 4-8 overlaps
    - 20% of queries: 1-4 overlaps
    - 10% of queries: > 16 overlaps
    """
    var queries = List[Interval]()

    # Different query types for different overlap counts
    var high_overlap_count = num_queries // 10  # 10% with > 16 overlaps
    var medium_high_count = num_queries * 4 // 10  # 40% with 8-16 overlaps
    var medium_low_count = num_queries * 3 // 10  # 30% with 4-8 overlaps
    var low_overlap_count = (
        num_queries - high_overlap_count - medium_high_count - medium_low_count
    )  # ~20%

    # Generate queries in round-robin fashion to distribute types evenly
    var cycle_length = 10
    var type_counters = List[Int](4, 0)  # counters for each type

    for i in range(num_queries):
        var cycle_pos = i % cycle_length
        # Determine query type based on cycle position to match distribution
        var query_type: Int
        if cycle_pos < 4:
            query_type = 1  # medium_high (40%)
        elif cycle_pos < 7:
            query_type = 2  # medium_low (30%)
        elif cycle_pos < 9:
            query_type = 0  # low (20%)
        else:
            query_type = 3  # high (10%)

        var start_list = List[Int32](unsafe_uninit_length=1)
        var length_list = List[Int32](unsafe_uninit_length=1)

        if query_type == 3:  # > 16 overlaps
            randint[DType.int32](
                start_list.unsafe_ptr(), 1, 0, max_coordinate - 200
            )
            randint[DType.int32](length_list.unsafe_ptr(), 1, 170, 200)
        elif query_type == 1:  # 8-16 overlaps
            randint[DType.int32](
                start_list.unsafe_ptr(), 1, 0, max_coordinate - 160
            )
            randint[DType.int32](length_list.unsafe_ptr(), 1, 80, 150)
        elif query_type == 2:  # 4-8 overlaps
            randint[DType.int32](
                start_list.unsafe_ptr(), 1, 0, max_coordinate - 80
            )
            randint[DType.int32](length_list.unsafe_ptr(), 1, 40, 80)
        else:  # 1-4 overlaps
            randint[DType.int32](
                start_list.unsafe_ptr(), 1, 0, max_coordinate - 40
            )
            randint[DType.int32](length_list.unsafe_ptr(), 1, 10, 40)

        var start = UInt32(start_list[0])
        var length = UInt32(length_list[0])
        var stop = start + length
        queries.append(Interval(start, stop, 0))

    return queries


def benchmark_lapper_count():
    """Benchmark Lapper count operations on CPU and GPU (if available)."""
    alias num_intervals = 100_000
    alias num_queries = 10_000
    alias max_coordinate = 1_000_000

    print("Generating test data...")
    var intervals = generate_intervals(num_intervals, max_coordinate)
    var queries = generate_queries(num_queries, max_coordinate)

    print("Creating CPU Lapper...")
    var cpu_lapper = Lapper(intervals)

    print("Creating EzLapper...")
    var ez_lapper = EzLapper(intervals)

    # Start benchmarking
    print("Starting benchmarks...")
    var b = Bench()

    @parameter
    @always_inline
    fn bench_cpu_count(mut b: Bencher):
        """Benchmark CPU BITS count operations."""

        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in queries:
                var count = cpu_lapper.count(query.start, query.stop)
                total_count += count
            keep(total_count)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_ezlapper_count(mut b: Bencher):
        """Benchmark EzLapper count operations."""

        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in queries:
                var count = ez_lapper.count(query.start, query.stop)
                total_count += count
            keep(total_count)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_cpu_find(mut b: Bencher):
        """Benchmark CPU naive find operations (for comparison)."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in queries:
                var lb = cpu_lapper._lower_bound(query.start)
                var c = cpu_lapper._count(lb, query.start, query.stop)
                total_found += Int(c)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_cpu_find_actual(mut b: Bencher):
        """Benchmark CPU Lapper find operations that return actual intervals."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in queries:
                var results = List[Interval]()
                cpu_lapper.find(query.start, query.stop, results)
                total_found += len(results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_ezlapper_find(mut b: Bencher):
        """Benchmark EzLapper find operations."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in queries:
                var results = List[Interval]()
                ez_lapper.find(query.start, query.stop, results)
                total_found += len(results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_lapper_find_vectorized(mut b: Bencher):
        """Benchmark Lapper find_vectorized operations (indices only)."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in queries:
                var indices = List[UInt32]()
                cpu_lapper.find_vectorized(query.start, query.stop, indices)
                total_found += len(indices)
            keep(total_found)

        b.iter[run]()

    # Run CPU benchmarks
    b.bench_function[bench_cpu_count](BenchId("Lapper - count (BITS)"))
    b.bench_function[bench_ezlapper_count](BenchId("EzLapper - count (BITS)"))
    b.bench_function[bench_cpu_find](BenchId("Lapper - count (naive)"))
    b.bench_function[bench_cpu_find_actual](BenchId("Lapper - find"))
    b.bench_function[bench_ezlapper_find](BenchId("EzLapper - find"))
    b.bench_function[bench_lapper_find_vectorized](
        BenchId("Lapper - find_vectorized (indices)")
    )

    @parameter
    if has_accelerator():
        print("Setting up GPU context...")
        var ctx = DeviceContext()

        # Create GPU lapper
        print("Creating GPU Lapper...")
        # Allocate host and device buffers for GPU Lapper
        var host_starts = ctx.enqueue_create_host_buffer[DType.uint32](
            num_intervals
        )
        var host_stops = ctx.enqueue_create_host_buffer[DType.uint32](
            num_intervals
        )
        var host_vals = ctx.enqueue_create_host_buffer[DType.int32](
            num_intervals
        )
        var host_stops_sorted = ctx.enqueue_create_host_buffer[DType.uint32](
            num_intervals
        )

        var device_starts = ctx.enqueue_create_buffer[DType.uint32](
            num_intervals
        )
        var device_stops = ctx.enqueue_create_buffer[DType.uint32](
            num_intervals
        )
        var device_vals = ctx.enqueue_create_buffer[DType.int32](num_intervals)
        var device_stops_sorted = ctx.enqueue_create_buffer[DType.uint32](
            num_intervals
        )

        ctx.synchronize()

        var gpu_lapper = Lapper.prep_for_gpu(
            ctx,
            intervals^,
            host_starts,
            host_stops,
            host_vals,
            host_stops_sorted,
            device_starts,
            device_stops,
            device_vals,
            device_stops_sorted,
        )

        # Test GPU kernel with a single query before benchmarking
        print("Testing GPU kernel with single query...")
        var test_host_keys = ctx.enqueue_create_host_buffer[DType.uint32](2)
        var test_device_keys = ctx.enqueue_create_buffer[DType.uint32](2)
        var test_device_output = ctx.enqueue_create_buffer[DType.uint32](1)
        var test_host_output = ctx.enqueue_create_host_buffer[DType.uint32](1)

        # Use first query for testing
        test_host_keys[0] = queries[0].start
        test_host_keys[1] = queries[0].stop

        test_host_keys.enqueue_copy_to(test_device_keys)
        ctx.synchronize()

        try:
            ctx.enqueue_function[find_overlaps_kernel](
                gpu_lapper.starts,
                gpu_lapper.stops,
                gpu_lapper.vals,
                gpu_lapper.stops_sorted,
                gpu_lapper.length,
                gpu_lapper.max_len,
                test_device_keys.unsafe_ptr(),
                2,
                test_device_output.unsafe_ptr(),
                1,
                grid_dim=1,
                block_dim=32,
            )
            ctx.synchronize()

            test_device_output.enqueue_copy_to(test_host_output)
            ctx.synchronize()

            var gpu_count = test_host_output[0]
            print(
                String("GPU test result: query=({},{}) count={}").format(
                    queries[0].start, queries[0].stop, gpu_count
                )
            )

            # Compare with CPU result for validation
            var cpu_lb = cpu_lapper._lower_bound(queries[0].start)
            var cpu_count = cpu_lapper._count(
                cpu_lb, queries[0].start, queries[0].stop
            )
            print(String("CPU test result: count={}").format(cpu_count))

            if gpu_count != cpu_count:
                print("WARNING: GPU/CPU count mismatch!")
            else:
                print("GPU kernel test passed!")

        except e:
            print("ERROR: GPU kernel test failed:", e)

        @parameter
        @always_inline
        fn bench_gpu_find_overlaps[
            grid_dim: Int, block_dim: Int
        ](mut b: Bencher) raises:
            """Benchmark GPU count naive operations."""
            # Setup device buffers
            var host_keys = ctx.enqueue_create_host_buffer[DType.uint32](
                num_queries * 2
            )
            var device_keys = ctx.enqueue_create_buffer[DType.uint32](
                num_queries * 2
            )
            var device_output = ctx.enqueue_create_buffer[DType.uint32](
                num_queries
            )
            var host_output = ctx.enqueue_create_host_buffer[DType.uint32](
                num_queries
            )

            # Copy query data to GPU format (start, stop pairs)
            for i in range(num_queries):
                host_keys[i * 2] = queries[i].start
                host_keys[i * 2 + 1] = queries[i].stop

            host_keys.enqueue_copy_to(device_keys)
            ctx.synchronize()

            @parameter
            @always_inline
            fn kernel_launch(gpu_ctx: DeviceContext) raises:
                gpu_ctx.enqueue_function[find_overlaps_kernel](
                    gpu_lapper.starts,
                    gpu_lapper.stops,
                    gpu_lapper.vals,
                    gpu_lapper.stops_sorted,
                    gpu_lapper.length,
                    gpu_lapper.max_len,
                    device_keys.unsafe_ptr(),
                    num_queries * 2,
                    device_output.unsafe_ptr(),
                    num_queries,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                )

            b.iter_custom[kernel_launch](ctx)

        @parameter
        @always_inline
        fn bench_gpu_count[
            grid_dim: Int, block_dim: Int
        ](mut b: Bencher) raises:
            """Benchmark GPU count operations."""
            # Setup device buffers
            var host_keys = ctx.enqueue_create_host_buffer[DType.uint32](
                num_queries * 2
            )
            var device_keys = ctx.enqueue_create_buffer[DType.uint32](
                num_queries * 2
            )
            var device_output = ctx.enqueue_create_buffer[DType.uint32](
                num_queries
            )
            var host_output = ctx.enqueue_create_host_buffer[DType.uint32](
                num_queries
            )

            # Copy query data to GPU format (start, stop pairs)
            for i in range(num_queries):
                host_keys[i * 2] = queries[i].start
                host_keys[i * 2 + 1] = queries[i].stop

            host_keys.enqueue_copy_to(device_keys)
            ctx.synchronize()

            @parameter
            @always_inline
            fn kernel_launch(gpu_ctx: DeviceContext) raises:
                gpu_ctx.enqueue_function[count_overlaps_kernel](
                    gpu_lapper.starts,
                    gpu_lapper.stops,
                    gpu_lapper.vals,
                    gpu_lapper.stops_sorted,
                    gpu_lapper.length,
                    gpu_lapper.max_len,
                    device_keys.unsafe_ptr(),
                    num_queries * 2,
                    device_output.unsafe_ptr(),
                    num_queries,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                )

            b.iter_custom[kernel_launch](ctx)

        # Run GPU benchmarks
        alias block_sizes = List[Int](1024, 512, 256, 128)

        @parameter
        for i in range(0, len(block_sizes)):
            alias block_size = block_sizes[i]
            b.bench_function[
                bench_gpu_find_overlaps[
                    ceildiv(num_queries, block_size), block_size
                ]
            ](BenchId("GPU count overlaps naive: " + String(block_size)))

            b.bench_function[
                bench_gpu_count[ceildiv(num_queries, block_size), block_size]
            ](BenchId("GPU count overlaps BITS: " + String(block_size)))

        # Validate GPU count results against CPU
        print("\nGPU Count Validation:")
        var gpu_sample_queries = 5

        # Setup buffers for GPU count verification
        var host_keys = ctx.enqueue_create_host_buffer[DType.uint32](
            gpu_sample_queries * 2
        )
        var device_keys = ctx.enqueue_create_buffer[DType.uint32](
            gpu_sample_queries * 2
        )
        var device_count_output = ctx.enqueue_create_buffer[DType.uint32](
            gpu_sample_queries
        )
        var host_count_output = ctx.enqueue_create_host_buffer[DType.uint32](
            gpu_sample_queries
        )

        # Copy sample queries to GPU format
        for i in range(gpu_sample_queries):
            host_keys[i * 2] = queries[i].start
            host_keys[i * 2 + 1] = queries[i].stop

        host_keys.enqueue_copy_to(device_keys)
        ctx.synchronize()

        # Run GPU count kernel
        ctx.enqueue_function[count_overlaps_kernel](
            gpu_lapper.starts,
            gpu_lapper.stops,
            gpu_lapper.vals,
            gpu_lapper.stops_sorted,
            gpu_lapper.length,
            gpu_lapper.max_len,
            device_keys.unsafe_ptr(),
            gpu_sample_queries * 2,
            device_count_output.unsafe_ptr(),
            gpu_sample_queries,
            grid_dim=ceildiv(gpu_sample_queries, 256),
            block_dim=256,
        )
        device_count_output.enqueue_copy_to(host_count_output)
        ctx.synchronize()

        # Compare GPU count results with CPU
        var count_mismatches = 0
        var gpu_total: UInt32 = 0
        var gpu_cpu_total: UInt32 = 0

        for i in range(gpu_sample_queries):
            var query = queries[i]
            var lb = cpu_lapper._lower_bound(query.start)
            var cpu_count = cpu_lapper._count(lb, query.start, query.stop)
            var gpu_count = host_count_output[i]
            gpu_total += gpu_count
            gpu_cpu_total += cpu_count

            if cpu_count != gpu_count:
                count_mismatches += 1
                print(
                    String(
                        "MISMATCH Query {}: CPU count={}, GPU count={}"
                    ).format(i, cpu_count, gpu_count)
                )
            else:
                print(
                    String("✓ Query {}: CPU count={}, GPU count={}").format(
                        i, cpu_count, gpu_count
                    )
                )

        if count_mismatches == 0:
            print(
                String("✅ All {} GPU count results match CPU!").format(
                    gpu_sample_queries
                )
            )
            print(
                String("   CPU total: {}, GPU total: {}").format(
                    gpu_cpu_total, gpu_total
                )
            )
        else:
            print(
                String("❌ {} GPU count mismatches found!").format(
                    count_mismatches
                )
            )

    print(b)

    # Validate correctness by comparing a few results
    print("\nValidation:")
    var sample_queries = 5
    var cpu_total: UInt32 = 0

    for i in range(sample_queries):
        var query = queries[i]
        var lb = cpu_lapper._lower_bound(query.start)
        var cpu_count = cpu_lapper._count(lb, query.start, query.stop)
        cpu_total += cpu_count

        var results = List[Interval]()
        cpu_lapper.find(query.start, query.stop, results)
        var find_count = len(results)

        print(
            String("Query {}: start={}, stop={}, count={}, find={}").format(
                i, query.start, query.stop, cpu_count, find_count
            )
        )

        # Verify count matches find results
        if cpu_count != find_count:
            print("ERROR: Count mismatch!")

        # Test EzLapper.find() vs Lapper.find() consistency
        var ez_results = List[Interval]()
        ez_lapper.find(query.start, query.stop, ez_results)
        var ez_find_count = len(ez_results)

        if find_count != ez_find_count:
            print(
                String(
                    "ERROR: EzLapper.find() vs Lapper.find() mismatch! Query"
                    " {}: Lapper={}, EzLapper={}"
                ).format(i, find_count, ez_find_count)
            )
        else:
            # Verify the intervals are the same (they should be in sorted order)
            var intervals_match = True
            for j in range(len(results)):
                if (
                    results[j].start != ez_results[j].start
                    or results[j].stop != ez_results[j].stop
                    or results[j].val != ez_results[j].val
                ):
                    intervals_match = False
                    break

            if not intervals_match:
                print(
                    String(
                        "ERROR: EzLapper.find() vs Lapper.find() intervals"
                        " differ for query {}"
                    ).format(i)
                )

    print(
        String("Total overlaps found in {} queries: {}").format(
            sample_queries, cpu_total
        )
    )


def benchmark_query_sorting_impact():
    """Compare performance of sorted vs unsorted queries."""
    alias num_intervals = 100_000
    alias num_queries = 10_000
    alias max_coordinate = 1_000_000

    print("=== Query Sorting Performance Comparison ===")

    # Use same seed for fair comparison
    seed(42)
    var intervals = generate_intervals(num_intervals, max_coordinate)

    # Generate sorted and unsorted queries with same seed
    seed(42)  # Reset seed for consistent data
    var sorted_queries = generate_queries(
        num_queries, max_coordinate, sorted=True
    )

    seed(42)  # Reset seed again
    var unsorted_queries = generate_queries(
        num_queries, max_coordinate, sorted=False
    )

    print("Creating CPU Lapper...")
    var cpu_lapper = Lapper(intervals)

    print("Creating EzLapper...")
    var ez_lapper = EzLapper(intervals)

    var b = Bench()

    @parameter
    @always_inline
    fn bench_sorted_queries(mut b: Bencher):
        """Benchmark CPU BITS count operations with sorted queries."""

        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in sorted_queries:
                var count = cpu_lapper.count(query.start, query.stop)
                total_count += count
            keep(total_count)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_unsorted_queries(mut b: Bencher):
        """Benchmark CPU BITS count operations with unsorted queries."""

        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in unsorted_queries:
                var count = cpu_lapper.count(query.start, query.stop)
                total_count += count
            keep(total_count)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_ezlapper_sorted_queries(mut b: Bencher):
        """Benchmark EzLapper count operations with sorted queries."""

        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in sorted_queries:
                var count = ez_lapper.count(query.start, query.stop)
                total_count += count
            keep(total_count)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_ezlapper_unsorted_queries(mut b: Bencher):
        """Benchmark EzLapper count operations with unsorted queries."""

        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in unsorted_queries:
                var count = ez_lapper.count(query.start, query.stop)
                total_count += count
            keep(total_count)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_lapper_find_sorted_queries(mut b: Bencher):
        """Benchmark Lapper find operations with sorted queries."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in sorted_queries:
                var results = List[Interval]()
                cpu_lapper.find(query.start, query.stop, results)
                total_found += len(results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_lapper_find_unsorted_queries(mut b: Bencher):
        """Benchmark Lapper find operations with unsorted queries."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in unsorted_queries:
                var results = List[Interval]()
                cpu_lapper.find(query.start, query.stop, results)
                total_found += len(results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_ezlapper_find_sorted_queries(mut b: Bencher):
        """Benchmark EzLapper find operations with sorted queries."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in sorted_queries:
                var results = List[Interval]()
                ez_lapper.find(query.start, query.stop, results)
                total_found += len(results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_ezlapper_find_unsorted_queries(mut b: Bencher):
        """Benchmark EzLapper find operations with unsorted queries."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in unsorted_queries:
                var results = List[Interval]()
                ez_lapper.find(query.start, query.stop, results)
                total_found += len(results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_lapper_find_vectorized_sorted_queries(mut b: Bencher):
        """Benchmark Lapper find_vectorized operations with sorted queries."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in sorted_queries:
                var indices = List[UInt32]()
                cpu_lapper.find_vectorized(query.start, query.stop, indices)
                total_found += len(indices)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_lapper_find_vectorized_unsorted_queries(mut b: Bencher):
        """Benchmark Lapper find_vectorized operations with unsorted queries."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in unsorted_queries:
                var indices = List[UInt32]()
                cpu_lapper.find_vectorized(query.start, query.stop, indices)
                total_found += len(indices)
            keep(total_found)

        b.iter[run]()

    # Run benchmarks
    b.bench_function[bench_sorted_queries](BenchId("Lapper - count (sorted)"))
    b.bench_function[bench_unsorted_queries](
        BenchId("Lapper - count (unsorted)")
    )
    b.bench_function[bench_ezlapper_sorted_queries](
        BenchId("EzLapper - count (sorted)")
    )
    b.bench_function[bench_ezlapper_unsorted_queries](
        BenchId("EzLapper - count (unsorted)")
    )
    b.bench_function[bench_lapper_find_sorted_queries](
        BenchId("Lapper - find (sorted)")
    )
    b.bench_function[bench_lapper_find_unsorted_queries](
        BenchId("Lapper - find (unsorted)")
    )
    b.bench_function[bench_ezlapper_find_sorted_queries](
        BenchId("EzLapper - find (sorted)")
    )
    b.bench_function[bench_ezlapper_find_unsorted_queries](
        BenchId("EzLapper - find (unsorted)")
    )
    b.bench_function[bench_lapper_find_vectorized_sorted_queries](
        BenchId("Lapper - find_vectorized (sorted)")
    )
    b.bench_function[bench_lapper_find_vectorized_unsorted_queries](
        BenchId("Lapper - find_vectorized (unsorted)")
    )

    print(b)

    # Verify both produce same results
    var sorted_total: UInt32 = 0
    var unsorted_total: UInt32 = 0

    for i in range(min(5, len(sorted_queries))):
        var sorted_count = cpu_lapper.count(
            sorted_queries[i].start, sorted_queries[i].stop
        )
        var unsorted_count = cpu_lapper.count(
            unsorted_queries[i].start, unsorted_queries[i].stop
        )
        sorted_total += sorted_count
        unsorted_total += unsorted_count

    print(
        String("Validation: sorted_total={}, unsorted_total={}").format(
            sorted_total, unsorted_total
        )
    )
    print("Query order comparison:")
    for i in range(min(5, len(sorted_queries))):
        print(
            String("  Sorted[{}]: start={}, Unsorted[{}]: start={}").format(
                i, sorted_queries[i].start, i, unsorted_queries[i].start
            )
        )


def benchmark_sparse_vs_dense():
    """Compare performance on sparse vs dense datasets."""
    alias num_intervals = 100_000
    alias num_queries = 10_000
    alias max_coordinate = 1_000_000

    print("=== Sparse vs Dense Dataset Performance Comparison ===")

    # Generate sparse dataset
    print("\nGenerating sparse dataset...")
    seed(42)
    var sparse_intervals = generate_sparse_intervals(
        num_intervals, max_coordinate
    )
    # Use smaller queries for sparse dataset
    var sparse_queries = List[Interval]()
    var starts = List[UInt32](unsafe_uninit_length=num_queries)
    var lengths = List[UInt32](unsafe_uninit_length=num_queries)
    randint(starts.unsafe_ptr(), num_queries, 0, max_coordinate - 10)
    randint(lengths.unsafe_ptr(), num_queries, 1, 10)  # Small queries
    for i in range(num_queries):
        var start = starts[i]
        var stop = start + lengths[i]
        sparse_queries.append(Interval(start, stop, 0))

    # Generate dense dataset
    print("Generating dense dataset...")
    seed(42)
    var dense_intervals = generate_dense_intervals(
        num_intervals, max_coordinate
    )
    var dense_queries = generate_dense_queries(num_queries, max_coordinate, 10)

    # Create Lappers
    print("Creating sparse Lapper...")
    var sparse_lapper = Lapper(sparse_intervals)

    print("Creating dense Lapper...")
    var dense_lapper = Lapper(dense_intervals)

    # Analyze query overlap distributions
    print("\nAnalyzing overlap distributions...")
    var sparse_overlaps = List[Int]()
    var dense_overlaps = List[Int]()
    var analysis_count = min(1000, num_queries)  # Analyze first 1000 queries

    # Count distribution buckets
    var sparse_1_4 = 0
    var sparse_4_8 = 0
    var sparse_8_16 = 0
    var sparse_over_16 = 0

    var dense_1_4 = 0
    var dense_4_8 = 0
    var dense_8_16 = 0
    var dense_over_16 = 0

    # Calculate how many of each type we expect in the analysis count
    var expected_high = analysis_count // 10
    var expected_med_high = analysis_count * 4 // 10
    var expected_med_low = analysis_count * 3 // 10

    for i in range(analysis_count):
        # Count overlaps in sparse dataset
        var sparse_count = sparse_lapper.count(
            sparse_queries[i].start, sparse_queries[i].stop
        )
        sparse_overlaps.append(Int(sparse_count))
        if sparse_count <= 4:
            sparse_1_4 += 1
        elif sparse_count <= 8:
            sparse_4_8 += 1
        elif sparse_count <= 16:
            sparse_8_16 += 1
        else:
            sparse_over_16 += 1

        # Count overlaps in dense dataset
        var dense_count = dense_lapper.count(
            dense_queries[i].start, dense_queries[i].stop
        )
        dense_overlaps.append(Int(dense_count))
        if dense_count <= 4:
            dense_1_4 += 1
        elif dense_count <= 8:
            dense_4_8 += 1
        elif dense_count <= 16:
            dense_8_16 += 1
        else:
            dense_over_16 += 1

    print("Sparse dataset distribution:")
    print(
        String("  1-4 overlaps: {}%").format(
            (sparse_1_4 * 100) // analysis_count
        )
    )
    print(
        String("  4-8 overlaps: {}%").format(
            (sparse_4_8 * 100) // analysis_count
        )
    )
    print(
        String("  8-16 overlaps: {}%").format(
            (sparse_8_16 * 100) // analysis_count
        )
    )
    print(
        String("  > 16 overlaps: {}%").format(
            (sparse_over_16 * 100) // analysis_count
        )
    )

    print("\nDense dataset distribution:")
    print(
        String("  1-4 overlaps: {}% ({} queries)").format(
            (dense_1_4 * 100) // analysis_count, dense_1_4
        )
    )
    print(
        String("  4-8 overlaps: {}% ({} queries)").format(
            (dense_4_8 * 100) // analysis_count, dense_4_8
        )
    )
    print(
        String("  8-16 overlaps: {}% ({} queries)").format(
            (dense_8_16 * 100) // analysis_count, dense_8_16
        )
    )
    print(
        String("  > 16 overlaps: {}% ({} queries)").format(
            (dense_over_16 * 100) // analysis_count, dense_over_16
        )
    )
    print(
        String("Expected: high={}, med_high={}, med_low={}").format(
            expected_high, expected_med_high, expected_med_low
        )
    )

    # Create EzLappers
    print("\nCreating EzLappers...")
    var sparse_ez_lapper = EzLapper(sparse_intervals)
    var dense_ez_lapper = EzLapper(dense_intervals)

    # Benchmark
    var b = Bench()

    # Sparse dataset benchmarks
    @parameter
    @always_inline
    fn bench_sparse_lapper_count(mut b: Bencher):
        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in sparse_queries:
                var count = sparse_lapper.count(query.start, query.stop)
                total_count += count
            keep(total_count)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_sparse_lapper_find(mut b: Bencher):
        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in sparse_queries:
                var results = List[Interval]()
                sparse_lapper.find(query.start, query.stop, results)
                total_found += len(results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_sparse_lapper_find_vectorized(mut b: Bencher):
        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in sparse_queries:
                var indices = List[UInt32]()
                sparse_lapper.find_vectorized(query.start, query.stop, indices)
                total_found += len(indices)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_sparse_ezlapper_count(mut b: Bencher):
        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in sparse_queries:
                var count = sparse_ez_lapper.count(query.start, query.stop)
                total_count += count
            keep(total_count)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_sparse_ezlapper_find(mut b: Bencher):
        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in sparse_queries:
                var results = List[Interval]()
                sparse_ez_lapper.find(query.start, query.stop, results)
                total_found += len(results)
            keep(total_found)

        b.iter[run]()

    # Dense dataset benchmarks
    @parameter
    @always_inline
    fn bench_dense_lapper_count(mut b: Bencher):
        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in dense_queries:
                var count = dense_lapper.count(query.start, query.stop)
                total_count += count
            keep(total_count)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_dense_lapper_find(mut b: Bencher):
        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in dense_queries:
                var results = List[Interval]()
                dense_lapper.find(query.start, query.stop, results)
                total_found += len(results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_dense_lapper_find_vectorized(mut b: Bencher):
        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in dense_queries:
                var indices = List[UInt32]()
                dense_lapper.find_vectorized(query.start, query.stop, indices)
                total_found += len(indices)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_dense_ezlapper_count(mut b: Bencher):
        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in dense_queries:
                var count = dense_ez_lapper.count(query.start, query.stop)
                total_count += count
            keep(total_count)

        b.iter[run]()

    @parameter
    @always_inline
    fn bench_dense_ezlapper_find(mut b: Bencher):
        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in dense_queries:
                var results = List[Interval]()
                dense_ez_lapper.find(query.start, query.stop, results)
                total_found += len(results)
            keep(total_found)

        b.iter[run]()

    # Run benchmarks
    b.bench_function[bench_sparse_lapper_count](
        BenchId("Sparse: Lapper - count")
    )
    b.bench_function[bench_sparse_lapper_find](BenchId("Sparse: Lapper - find"))
    b.bench_function[bench_sparse_lapper_find_vectorized](
        BenchId("Sparse: Lapper - find_vectorized")
    )
    b.bench_function[bench_sparse_ezlapper_count](
        BenchId("Sparse: EzLapper - count")
    )
    b.bench_function[bench_sparse_ezlapper_find](
        BenchId("Sparse: EzLapper - find")
    )

    b.bench_function[bench_dense_lapper_count](BenchId("Dense: Lapper - count"))
    b.bench_function[bench_dense_lapper_find](BenchId("Dense: Lapper - find"))
    b.bench_function[bench_dense_lapper_find_vectorized](
        BenchId("Dense: Lapper - find_vectorized")
    )
    b.bench_function[bench_dense_ezlapper_count](
        BenchId("Dense: EzLapper - count")
    )
    b.bench_function[bench_dense_ezlapper_find](
        BenchId("Dense: EzLapper - find")
    )

    print(b)

    # Show sample overlap counts
    print("\nSample overlap counts (first 10 queries):")
    print("Sparse dataset:")
    for i in range(min(10, len(sparse_overlaps))):
        print(String("  Query {}: {} overlaps").format(i, sparse_overlaps[i]))

    print("\nDense dataset:")
    for i in range(min(10, len(dense_overlaps))):
        print(String("  Query {}: {} overlaps").format(i, dense_overlaps[i]))


def benchmark_bed_files():
    print("\n=== BED File Benchmarks ===")

    # It's fine that this is hardcoded for now
    var anno_file = "/Users/sethstadick/Downloads/biofast-data-v1/ex-anno.bed"
    var rna_file = "/Users/sethstadick/Downloads/biofast-data-v1/ex-rna.bed"
    var anno_recs = SimpleBedRecord.read_file(anno_file)
    var rna_recs = SimpleBedRecord.read_file(rna_file)

    print("Loading BED files...")
    print("Anno records:", len(anno_recs))
    print("RNA records:", len(rna_recs))

    # Convert BED records to intervals
    var anno_intervals = Dict[String, List[Interval]]()
    for i in range(len(anno_recs)):
        var rec = anno_recs[i]
        if rec.chr not in anno_intervals:
            anno_intervals[rec.chr] = [Interval(rec.start, rec.stop, Int32(i))]
        else:
            anno_intervals[rec.chr].append(
                Interval(rec.start, rec.stop, Int32(i))
            )

    var rna_intervals = Dict[String, List[Interval]]()
    for i in range(len(rna_recs)):
        var rec = rna_recs[i]
        if rec.chr not in rna_intervals:
            rna_intervals[rec.chr] = [Interval(rec.start, rec.stop, Int32(i))]
        else:
            rna_intervals[rec.chr].append(
                Interval(rec.start, rec.stop, Int32(i))
            )

    print("Creating data structures...")

    # Create Lapper instances
    var anno_lappers = {e.key: Lapper(e.value) for e in anno_intervals.items()}
    var rna_lappers = {e.key: Lapper(e.value) for e in rna_intervals.items()}

    # Create EzLapper instances
    var anno_ez = {e.key: EzLapper(e.value) for e in anno_intervals.items()}
    var rna_ez = {e.key: EzLapper(e.value) for e in rna_intervals.items()}

    print("Starting benchmarks...")

    # Benchmark 1: Search all anno intervals in RNA dataset
    print("\n=== Searching Anno intervals in RNA dataset ===")

    var b = Bench()

    @parameter
    @always_inline
    def bench_rna_find_vectorized(mut b: Bencher):
        var rna_find_vec_results = List[UInt32]()

        @parameter
        @always_inline
        def run():
            var total_found = 0
            for key in anno_lappers.keys():
                ref ivs = anno_intervals[key]
                ref rna_lapper = rna_lappers[key]
                for i in range(len(ivs)):
                    var query = ivs[i]
                    rna_find_vec_results.clear()
                    rna_lapper.find_vectorized(
                        query.start, query.stop, rna_find_vec_results
                    )
                    total_found += len(rna_find_vec_results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    def bench_rna_find(mut b: Bencher):
        var rna_find_results = List[Interval]()

        @parameter
        @always_inline
        def run():
            var total_found = 0
            for key in anno_lappers.keys():
                ref ivs = anno_intervals[key]
                ref rna_lapper = rna_lappers[key]
                for i in range(len(ivs)):
                    var query = ivs[i]
                    rna_find_results.clear()
                    rna_lapper.find(query.start, query.stop, rna_find_results)
                    total_found += len(rna_find_results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    def bench_rna_ez_find(mut b: Bencher):
        var rna_ez_find_results = List[Interval]()

        @parameter
        @always_inline
        def run():
            var total_found = 0
            for key in anno_lappers.keys():
                ref ivs = anno_intervals[key]
                ref rna_lapper = rna_ez[key]
                for i in range(len(ivs)):
                    var query = ivs[i]
                    rna_ez_find_results.clear()
                    rna_lapper.find(
                        query.start, query.stop, rna_ez_find_results
                    )
                    total_found += len(rna_ez_find_results)
            keep(total_found)

        b.iter[run]()

    # Benchmark 2: Search all RNA intervals in Anno dataset
    print("\n=== Searching RNA intervals in Anno dataset ===")

    @parameter
    @always_inline
    def bench_anno_find_vectorized(mut b: Bencher):
        var anno_find_vec_results = List[UInt32]()

        @parameter
        @always_inline
        def run():
            var total_found = 0
            for key in rna_lappers.keys():
                ref ivs = rna_intervals[key]
                ref anno_lapper = anno_lappers[key]
                for i in range(len(ivs)):
                    var query = ivs[i]
                    anno_find_vec_results.clear()
                    anno_lapper.find_vectorized(
                        query.start, query.stop, anno_find_vec_results
                    )
                    total_found += len(anno_find_vec_results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    def bench_anno_find(mut b: Bencher):
        var anno_find_results = List[Interval]()

        @parameter
        @always_inline
        def run():
            var total_found = 0
            for key in rna_lappers.keys():
                ref ivs = rna_intervals[key]
                ref anno_lapper = anno_lappers[key]
                for i in range(len(ivs)):
                    var query = ivs[i]
                    anno_find_results.clear()
                    anno_lapper.find(query.start, query.stop, anno_find_results)
                    total_found += len(anno_find_results)
            keep(total_found)

        b.iter[run]()

    @parameter
    @always_inline
    def bench_anno_ez_find(mut b: Bencher):
        var anno_ez_find_results = List[Interval]()

        @parameter
        @always_inline
        def run():
            var total_found = 0
            for key in rna_lappers.keys():
                ref ivs = rna_intervals[key]
                ref anno_lapper = anno_ez[key]
                for i in range(len(ivs)):
                    var query = ivs[i]
                    anno_ez_find_results.clear()
                    anno_lapper.find(
                        query.start, query.stop, anno_ez_find_results
                    )
                    total_found += len(anno_ez_find_results)
            keep(total_found)

        b.iter[run]()

    # Run benchmarks
    b.bench_function[bench_rna_find_vectorized](BenchId("RNA find_vectorized"))
    b.bench_function[bench_rna_find](BenchId("RNA Lapper.find"))
    b.bench_function[bench_rna_ez_find](BenchId("RNA EzLapper.find"))
    b.bench_function[bench_anno_find_vectorized](
        BenchId("Anno find_vectorized")
    )
    b.bench_function[bench_anno_find](BenchId("Anno Lapper.find"))
    b.bench_function[bench_anno_ez_find](BenchId("Anno EzLapper.find"))

    print(b)

    print("\nBED file benchmarks completed!")


def main():
    benchmark_sparse_vs_dense()
    print("\n" + "=" * 50 + "\n")
    benchmark_query_sorting_impact()
    print("\n" + "=" * 50 + "\n")
    seed(42)
    benchmark_lapper_count()
    print("\n" + "=" * 50 + "\n")
    benchmark_bed_files()

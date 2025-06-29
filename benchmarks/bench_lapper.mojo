from lapper.lapper import (
    Lapper,
    Interval,
    find_overlaps_kernel,
    count_overlaps_kernel,
)
from random import randint, seed
from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from memory import memcpy
from math import ceildiv
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


def generate_queries(num_queries: Int, max_coordinate: Int) -> List[Interval]:
    """Generate random query intervals."""
    var queries = List[Interval]()

    var starts = List[UInt32](unsafe_uninit_length=num_queries)
    var lengths = List[UInt32](unsafe_uninit_length=num_queries)

    randint(starts.unsafe_ptr(), num_queries, 0, max_coordinate - 50)
    randint(lengths.unsafe_ptr(), num_queries, 5, 50)

    for i in range(num_queries):
        var start = starts[i]
        var stop = start + lengths[i]
        queries.append(Interval(start, stop, 0))

    return queries


def benchmark_lapper_count():
    """Benchmark Lapper count operations on CPU vs GPU."""
    alias num_intervals = 100_000
    alias num_queries = 10_000
    alias max_coordinate = 1_000_000

    print("Generating test data...")
    var intervals = generate_intervals(num_intervals, max_coordinate)
    var queries = generate_queries(num_queries, max_coordinate)

    print("Creating CPU Lapper...")
    var cpu_lapper = Lapper(intervals)

    print("Setting up GPU context...")
    var ctx = DeviceContext()

    # Create GPU lapper
    print("Creating GPU Lapper...")
    # Allocate host and device buffers for GPU Lapper
    var host_starts = ctx.enqueue_create_host_buffer[DType.uint32](
        num_intervals
    )
    var host_stops = ctx.enqueue_create_host_buffer[DType.uint32](num_intervals)
    var host_vals = ctx.enqueue_create_host_buffer[DType.int32](num_intervals)
    var host_stops_sorted = ctx.enqueue_create_host_buffer[DType.uint32](
        num_intervals
    )

    var device_starts = ctx.enqueue_create_buffer[DType.uint32](num_intervals)
    var device_stops = ctx.enqueue_create_buffer[DType.uint32](num_intervals)
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
        var cpu_lb = cpu_lapper._lower_bound(queries[0].start, queries[0].stop)
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

    # For now, we'll benchmark CPU operations
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
    fn bench_cpu_find(mut b: Bencher):
        """Benchmark CPU naive find operations (for comparison)."""

        @parameter
        @always_inline
        fn run():
            var total_found = 0
            for query in queries:
                var lb = cpu_lapper._lower_bound(query.start, query.stop)
                var c = cpu_lapper._count(lb, query.start, query.stop)
                total_found += Int(c)
            keep(total_found)

        b.iter[run]()

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
        var device_output = ctx.enqueue_create_buffer[DType.uint32](num_queries)
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
            device_output.enqueue_copy_to(host_output)
            gpu_ctx.synchronize()

        b.iter_custom[kernel_launch](ctx)

    @parameter
    @always_inline
    fn bench_gpu_count[grid_dim: Int, block_dim: Int](mut b: Bencher) raises:
        """Benchmark GPU count operations."""
        # Setup device buffers
        var host_keys = ctx.enqueue_create_host_buffer[DType.uint32](
            num_queries * 2
        )
        var device_keys = ctx.enqueue_create_buffer[DType.uint32](
            num_queries * 2
        )
        var device_output = ctx.enqueue_create_buffer[DType.uint32](num_queries)
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
            device_output.enqueue_copy_to(host_output)
            gpu_ctx.synchronize()

        b.iter_custom[kernel_launch](ctx)

    # Run CPU benchmarks
    b.bench_function[bench_cpu_count](BenchId("CPU count overlaps BITS"))
    b.bench_function[bench_cpu_find](BenchId("CPU count overlaps naive"))

    # The GPU kernel compilation is failing, needs debugging
    alias block_sizes = List[Int](1024, 512, 256, 128)
    # alias block_sizes = List[Int](512)

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

    print(b)

    # Validate correctness by comparing a few results
    print("\nValidation:")
    var sample_queries = 5
    var cpu_total: UInt32 = 0

    for i in range(sample_queries):
        var query = queries[i]
        var lb = cpu_lapper._lower_bound(query.start, query.stop)
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

    print(
        String("Total overlaps found in {} queries: {}").format(
            sample_queries, cpu_total
        )
    )

    # Validate GPU count results against CPU
    print("\nGPU Count Validation:")

    # Setup buffers for GPU count verification
    var host_keys = ctx.enqueue_create_host_buffer[DType.uint32](
        sample_queries * 2
    )
    var device_keys = ctx.enqueue_create_buffer[DType.uint32](
        sample_queries * 2
    )
    var device_count_output = ctx.enqueue_create_buffer[DType.uint32](
        sample_queries
    )
    var host_count_output = ctx.enqueue_create_host_buffer[DType.uint32](
        sample_queries
    )

    # Copy sample queries to GPU format
    for i in range(sample_queries):
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
        sample_queries * 2,
        device_count_output.unsafe_ptr(),
        sample_queries,
        grid_dim=ceildiv(sample_queries, 256),
        block_dim=256,
    )
    device_count_output.enqueue_copy_to(host_count_output)
    ctx.synchronize()

    # Compare GPU count results with CPU
    var count_mismatches = 0
    var gpu_total: UInt32 = 0

    for i in range(sample_queries):
        var query = queries[i]
        var lb = cpu_lapper._lower_bound(query.start, query.stop)
        var cpu_count = cpu_lapper._count(lb, query.start, query.stop)
        var gpu_count = host_count_output[i]
        gpu_total += gpu_count

        if cpu_count != gpu_count:
            count_mismatches += 1
            print(
                String("MISMATCH Query {}: CPU count={}, GPU count={}").format(
                    i, cpu_count, gpu_count
                )
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
                sample_queries
            )
        )
        print(
            String("   CPU total: {}, GPU total: {}").format(
                cpu_total, gpu_total
            )
        )
    else:
        print(
            String("❌ {} GPU count mismatches found!").format(count_mismatches)
        )


def main():
    seed(42)
    benchmark_lapper_count()

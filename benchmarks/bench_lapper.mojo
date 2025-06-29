from lapper.lapper import Lapper, Interval
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


def generate_intervals(num_intervals: Int, max_coordinate: Int) -> List[Interval]:
    """Generate random intervals for testing."""
    var intervals = List[Interval]()
    
    var starts = List[UInt32](unsafe_uninit_length=num_intervals)
    var stops = List[UInt32](unsafe_uninit_length=num_intervals) 
    var vals = List[Int32](unsafe_uninit_length=num_intervals)
    
    # Generate random data
    randint(starts.unsafe_ptr(), num_intervals, 0, max_coordinate - 100)
    randint(stops.unsafe_ptr(), num_intervals, 10, 100)  # interval lengths
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
    # Note: This would use prep_for_gpu method when it's fully implemented
    # var gpu_lapper = Lapper.prep_for_gpu(ctx, intervals)
    
    # For now, we'll benchmark CPU operations
    print("Starting benchmarks...")
    var b = Bench()
    
    @parameter
    @always_inline
    fn bench_cpu_count(mut b: Bencher):
        """Benchmark CPU count operations."""
        @parameter
        @always_inline
        fn run():
            var total_count: UInt32 = 0
            for query in queries:
                var lb = cpu_lapper._lower_bound(query.start, query.stop)
                var count = cpu_lapper._count(lb, query.start, query.stop)
                total_count += count
            keep(total_count)
        
        b.iter[run]()
    
    @parameter
    @always_inline
    fn bench_cpu_find(mut b: Bencher):
        """Benchmark CPU find operations (for comparison)."""
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
    
    # TODO: Add GPU benchmarks when GPU implementation is complete
    # @parameter
    # @always_inline
    # fn bench_gpu_count[grid_dim: Int, block_dim: Int](mut b: Bencher) raises:
    #     """Benchmark GPU count operations."""
    #     # Setup device buffers
    #     var host_keys = ctx.enqueue_create_host_buffer[DType.uint32](num_queries * 2)
    #     var device_keys = ctx.enqueue_create_buffer[DType.uint32](num_queries * 2)
    #     var device_output = ctx.enqueue_create_buffer[DType.uint32](num_queries)
    #     var host_output = ctx.enqueue_create_host_buffer[DType.uint32](num_queries)
    #     
    #     # Copy query data to GPU format (start, stop pairs)
    #     for i in range(num_queries):
    #         host_keys[i * 2] = queries[i].start
    #         host_keys[i * 2 + 1] = queries[i].stop
    #     
    #     host_keys.enqueue_copy_to(device_keys)
    #     ctx.synchronize()
    #     
    #     @parameter
    #     @always_inline
    #     fn kernel_launch(gpu_ctx: DeviceContext) raises:
    #         gpu_ctx.enqueue_function[Lapper.find_overlaps_kernel](
    #             gpu_lapper.starts,
    #             gpu_lapper.stops,
    #             gpu_lapper.vals,
    #             gpu_lapper.stops_sorted,
    #             gpu_lapper.length,
    #             gpu_lapper.max_len,
    #             device_keys.unsafe_ptr(),
    #             num_queries * 2,
    #             device_output.unsafe_ptr(),
    #             num_queries,
    #             grid_dim=grid_dim,
    #             block_dim=block_dim,
    #         )
    #     
    #     b.iter_custom[kernel_launch](ctx)
    
    # Run CPU benchmarks
    b.bench_function[bench_cpu_count](BenchId("CPU Count"))
    b.bench_function[bench_cpu_find](BenchId("CPU Find (for comparison)"))
    
    # TODO: Add GPU benchmark calls when implementation is complete
    # alias block_sizes = [1024, 512, 256, 128]
    # @parameter
    # for i in range(len(block_sizes)):
    #     alias block_size = block_sizes[i]
    #     b.bench_function[
    #         bench_gpu_count[ceildiv(num_queries, block_size), block_size]
    #     ](BenchId("GPU Count: " + String(block_size)))
    
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
        
        print(String("Query {}: start={}, stop={}, count={}, find={}").format(
            i, query.start, query.stop, cpu_count, find_count
        ))
        
        # Verify count matches find results
        if cpu_count != find_count:
            print("ERROR: Count mismatch!")
    
    print(String("Total overlaps found in {} queries: {}").format(sample_queries, cpu_total))


def main():
    seed(42)
    benchmark_lapper_count()
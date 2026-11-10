from lapper.cpu.eytzinger import (
    eytzinger_with_lookup,
    lower_bound as ezcpu_lower_bound,
)
from lapper.cpu.bsearch import lower_bound as bsearchcpu_lower_bound
from lapper.gpu.bsearch import lower_bound_kernel as bsearch_lower_bound
from lapper.gpu.eytzinger import lower_bound_kernel as ez_lower_bound

from std.random import randint, seed
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from std.memory import unsafe_memcpy
from std.math import ceildiv
from max.benchmark import bencher_iter_custom

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer


def benchmark_binary_search() raises:
    """Benchmark for naive binary search."""
    comptime num_elements = 6_000_000
    comptime num_keys = 60_000

    var ctx = DeviceContext()
    var host_elems_eytz = ctx.enqueue_create_host_buffer[DType.int32](
        num_elements + 1  # NOTE: important + one since eytz layout takes up n+1 space
    )
    var host_elems_bsearch = ctx.enqueue_create_host_buffer[DType.int32](num_elements)
    var host_keys = ctx.enqueue_create_host_buffer[DType.int32](num_keys)
    var host_output = ctx.enqueue_create_host_buffer[DType.int32](num_keys)

    var device_elems_eytz = ctx.enqueue_create_buffer[DType.int32](num_elements + 1)
    var device_elems_bsearch = ctx.enqueue_create_buffer[DType.int32](num_elements)
    var device_keys = ctx.enqueue_create_buffer[DType.int32](num_keys)
    var device_output = ctx.enqueue_create_buffer[DType.int32](num_keys)
    ctx.synchronize()

    # Generate random sorted elements
    var elements = List[Int32](unsafe_uninit_length=num_elements)
    randint(elements.unsafe_ptr(), num_elements, 0, num_elements)
    sort(elements)
    var eytz = eytzinger_with_lookup(elements)
    unsafe_memcpy(
        dest=host_elems_eytz.unsafe_ptr(), src=eytz.layout.unsafe_ptr(), count=len(eytz.layout)
    )
    unsafe_memcpy(
        dest=host_elems_bsearch.unsafe_ptr(), src=elements.unsafe_ptr(), count=len(elements)
    )

    # Generate random keys to search for
    var keys = List[Int32](unsafe_uninit_length=num_keys)
    randint(keys.unsafe_ptr(), len(keys), 0, num_keys)
    sort(keys)
    unsafe_memcpy(dest=host_keys.unsafe_ptr(), src=keys.unsafe_ptr(), count=len(keys))

    host_elems_bsearch.enqueue_copy_to(device_elems_bsearch)
    host_elems_eytz.enqueue_copy_to(device_elems_eytz)
    host_keys.enqueue_copy_to(device_keys)

    ctx.synchronize()

    var b = Bench()

    @inline(.always)
    def ez_bench_gpu[grid_dim: Int, block_dim: Int](mut b: Bencher) raises {mut}:
        @inline(.always)
        def kernel_launch(gpu_ctx: DeviceContext) raises {mut}:
            gpu_ctx.enqueue_function[ez_lower_bound[DType.int32]](
                device_keys.unsafe_ptr(),
                UInt64(len(keys)),
                device_elems_eytz.unsafe_ptr(),
                UInt64(num_elements + 1),
                device_output.unsafe_ptr(),
                UInt64(len(keys)),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    @inline(.always)
    def bsearch_bench_gpu[grid_dim: Int, block_dim: Int](mut b: Bencher) raises {mut}:
        @inline(.always)
        def kernel_launch(gpu_ctx: DeviceContext) raises {mut}:
            gpu_ctx.enqueue_function[bsearch_lower_bound[DType.int32]](
                device_keys.unsafe_ptr(),
                UInt64(len(keys)),
                device_elems_bsearch.unsafe_ptr(),
                UInt64(num_elements),
                device_output.unsafe_ptr(),
                UInt64(len(keys)),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    comptime block_sizes = (1024, 512, 256, 128, 64, 32)

    comptime for i in range(0, len(block_sizes)):
        comptime block_size = block_sizes[i]

        def ez_bench(mut bencher: Bencher) raises {mut}:
            ez_bench_gpu[ceildiv(num_elements + block_size - 1, block_size), block_size](bencher)

        b.bench_function(ez_bench, BenchId("Eytzinger lower_bound: " + String(block_size)))
        device_output.enqueue_copy_to(host_output)
        ctx.synchronize()
        var ez_answer = eytz.lookup[ezcpu_lower_bound(eytz.layout, keys[num_keys // 2])]
        print(
            String("EZ: CPU={}, GPU={}").format(ez_answer, eytz.lookup[host_output[num_keys // 2]])
        )

        def bs_bench(mut bencher: Bencher) raises {mut}:
            bsearch_bench_gpu[ceildiv(num_elements + block_size - 1, block_size), block_size](
                bencher
            )

        b.bench_function(bs_bench, BenchId("Binary search lower_bound: " + String(block_size)))
        device_output.enqueue_copy_to(host_output)
        ctx.synchronize()

        var bsearch_answer = bsearchcpu_lower_bound(elements, keys[num_keys // 2])
        print(String("BS: CPU={}, GPU={}").format(bsearch_answer, host_output[num_keys // 2]))

    print(b)


def main() raises:
    seed(42)
    benchmark_binary_search()

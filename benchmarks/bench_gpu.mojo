from lapper.cpu.eytzinger import (
    eytzinger_with_lookup,
    lower_bound as ezcpu_lower_bound,
)
from lapper.cpu.bsearch import lower_bound as bsearchcpu_lower_bound
from lapper.gpu.bsearch import lower_bound_kernel as bsearch_lower_bound
from lapper.gpu.eytzinger import lower_bound_kernel as ez_lower_bound

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

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer


def benchmark_binary_search():
    """Benchmark for naive binary search."""
    alias num_elements = 6_000_000
    alias num_keys = 60_000

    var ctx = DeviceContext()
    var host_elems_eytz = ctx.enqueue_create_host_buffer[DType.int32](
        num_elements
        + 1  # NOTE: important + one since eytz layout takes up n+1 space
    )
    var host_elems_bsearch = ctx.enqueue_create_host_buffer[DType.int32](
        num_elements
    )
    var host_keys = ctx.enqueue_create_host_buffer[DType.int32](num_keys)
    var host_output = ctx.enqueue_create_host_buffer[DType.uint32](num_keys)

    var device_elems_eytz = ctx.enqueue_create_buffer[DType.int32](
        num_elements + 1
    )
    var device_elems_bsearch = ctx.enqueue_create_buffer[DType.int32](
        num_elements
    )
    var device_keys = ctx.enqueue_create_buffer[DType.int32](num_keys)
    var device_output = ctx.enqueue_create_buffer[DType.uint32](num_keys)
    ctx.synchronize()

    # Generate random sorted elements
    var elements = List[Int32](unsafe_uninit_length=num_elements)
    randint(elements.unsafe_ptr(), num_elements, 0, num_elements)
    sort(elements)
    var eytz = eytzinger_with_lookup(elements)
    memcpy(
        host_elems_eytz.unsafe_ptr(), eytz.layout.unsafe_ptr(), len(eytz.layout)
    )
    memcpy(
        host_elems_bsearch.unsafe_ptr(), elements.unsafe_ptr(), len(elements)
    )

    # Generate random keys to search for
    var keys = List[Int32](unsafe_uninit_length=num_keys)
    randint(keys.unsafe_ptr(), len(keys), 0, num_keys)
    sort(keys)
    memcpy(host_keys.unsafe_ptr(), keys.unsafe_ptr(), len(keys))

    host_elems_bsearch.enqueue_copy_to(device_elems_bsearch)
    host_elems_eytz.enqueue_copy_to(device_elems_eytz)
    host_keys.enqueue_copy_to(device_keys)

    ctx.synchronize()

    var b = Bench()

    @parameter
    @always_inline
    fn ez_bench_gpu[grid_dim: Int, block_dim: Int](mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(gpu_ctx: DeviceContext) raises:
            gpu_ctx.enqueue_function[ez_lower_bound[DType.int32]](
                device_keys.unsafe_ptr(),
                len(keys),
                device_elems_eytz.unsafe_ptr(),
                num_elements + 1,
                device_output.unsafe_ptr(),
                len(keys),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

        b.iter_custom[kernel_launch](ctx)

    @parameter
    @always_inline
    fn bsearch_bench_gpu[grid_dim: Int, block_dim: Int](mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(gpu_ctx: DeviceContext) raises:
            gpu_ctx.enqueue_function[bsearch_lower_bound[DType.int32]](
                device_keys.unsafe_ptr(),
                len(keys),
                device_elems_bsearch.unsafe_ptr(),
                num_elements,
                device_output.unsafe_ptr(),
                len(keys),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

        b.iter_custom[kernel_launch](ctx)

    alias block_sizes = [1024, 512, 256, 128, 64, 32]

    @parameter
    for i in range(0, len(block_sizes)):
        alias block_size = block_sizes[i]

        b.bench_function[
            ez_bench_gpu[
                ceildiv(num_elements + block_size - 1, block_size), block_size
            ]
        ](BenchId("Eytzinger lower_bound: " + String(block_size)))
        device_output.enqueue_copy_to(host_output)
        ctx.synchronize()
        var ez_answer = eytz.lookup[
            ezcpu_lower_bound(eytz.layout, keys[num_keys // 2])
        ]
        print(
            String("EZ: CPU={}, GPU={}").format(
                ez_answer, eytz.lookup[host_output[num_keys // 2]]
            )
        )

        b.bench_function[
            bsearch_bench_gpu[
                ceildiv(num_elements + block_size - 1, block_size), block_size
            ]
        ](BenchId("Binary search lower_bound: " + String(block_size)))
        device_output.enqueue_copy_to(host_output)
        ctx.synchronize()

        var bsearch_answer = bsearchcpu_lower_bound(
            elements, keys[num_keys // 2]
        )
        print(
            String("BS: CPU={}, GPU={}").format(
                bsearch_answer, host_output[num_keys // 2]
            )
        )

    print(b)


def main():
    seed(42)
    benchmark_binary_search()

from ksearch.cpu.bsearch import (
    naive_bsearch,
    offset_bsearch,
    branchless_offset_bsearch,
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


def benchmark_binary_search():
    """Benchmark for naive binary search."""
    var num_elements = 1_000_000
    var num_keys = 10_000

    # Generate random sorted elements
    var elements = List[Int32](unsafe_uninit_length=num_elements)
    randint(elements.unsafe_ptr(), len(elements), 0, 5_000_000)
    sort(elements)

    # Generate random keys to search for
    var keys = List[Int32](unsafe_uninit_length=num_keys)
    randint(keys.unsafe_ptr(), len(keys), 0, 1_000_000)

    ref elems_ref = elements

    var b = Bench()

    @parameter
    @always_inline
    @__copy_capture(elems_ref)
    fn bench_naive(mut b: Bencher):
        # Perform binary search for each key
        @parameter
        @always_inline
        @__copy_capture(elems_ref)
        fn run():
            for key in keys:
                var x = naive_bsearch[DType.int32](elems_ref, key)
                keep(x)

        b.iter[run]()

    @parameter
    @always_inline
    @__copy_capture(elems_ref)
    fn bench_offset(mut b: Bencher):
        # Perform binary search for each key
        @parameter
        @always_inline
        @__copy_capture(elems_ref)
        fn run():
            for key in keys:
                var x = offset_bsearch[DType.int32](elems_ref, key)
                keep(x)

        b.iter[run]()

    @parameter
    @always_inline
    @__copy_capture(elems_ref)
    fn bench_branchless_offset(mut b: Bencher):
        # Perform binary search for each key
        @parameter
        @always_inline
        @__copy_capture(elems_ref)
        fn run():
            for key in keys:
                var x = branchless_offset_bsearch[DType.int32](elems_ref, key)
                keep(x)

        b.iter[run]()

    b.bench_function[bench_naive](BenchId("naive binary search"))
    b.bench_function[bench_offset](BenchId("offset binary search"))
    b.bench_function[bench_branchless_offset](
        BenchId("branchless offset binary search")
    )
    print(b)


def main():
    seed(42)
    benchmark_binary_search()

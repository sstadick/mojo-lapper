from lapper.cpu.bsearch import naive_bsearch, offset_bsearch, lower_bound
from lapper.cpu.eytzinger import (
    lower_bound as ez_lower_bound,
    eytzinger_with_lookup,
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
    var num_elements = 6_000_000
    var num_keys = 60_000

    # Generate random sorted elements
    var elements = List[Int32](unsafe_uninit_length=num_elements)
    randint(elements.unsafe_ptr(), len(elements), 0, num_elements)
    sort(elements)
    var eytz = eytzinger_with_lookup(elements)

    # Generate random keys to search for
    var keys = List[Int32](unsafe_uninit_length=num_keys)
    randint(keys.unsafe_ptr(), len(keys), 0, num_keys)

    ref elems_ref = elements
    ref eytz_ref = eytz

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
                var x = naive_bsearch(elems_ref, key)
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
                var x = offset_bsearch(elems_ref, key)
                keep(x)

        b.iter[run]()

    @parameter
    @always_inline
    @__copy_capture(elems_ref)
    fn bench_lower_bound(mut b: Bencher):
        @parameter
        @always_inline
        @__copy_capture(elems_ref)
        fn run():
            for key in keys:
                var x = lower_bound(elems_ref, key)
                keep(x)

        b.iter[run]()

    @parameter
    @always_inline
    @__copy_capture(eytz_ref)
    fn bench_ez_lower_bound(mut b: Bencher):
        @parameter
        @always_inline
        @__copy_capture(eytz_ref)
        fn run():
            for key in keys:
                var x = ez_lower_bound(eytz_ref.layout, key)
                keep(x)

        b.iter[run]()

    @parameter
    @always_inline
    @__copy_capture(eytz_ref)
    fn bench_ez_lower_bound_w_conversion(mut b: Bencher):
        @parameter
        @always_inline
        @__copy_capture(eytz_ref)
        fn run():
            for key in keys:
                var x = ez_lower_bound(eytz_ref.layout, key)
                var y = eytz_ref.lookup[x]
                keep(y)

        b.iter[run]()

    b.bench_function[bench_naive](BenchId("naive binary search"))
    b.bench_function[bench_offset](BenchId("offset binary search"))
    b.bench_function[bench_lower_bound](BenchId("lower_bound"))
    b.bench_function[bench_ez_lower_bound](BenchId("Eytzinger lower_bound"))
    b.bench_function[bench_ez_lower_bound_w_conversion](
        BenchId("Eytzinger lower_bound with conversion")
    )
    print(b)


def main():
    seed(42)
    benchmark_binary_search()

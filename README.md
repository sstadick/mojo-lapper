# mojo-lapper

Interval overlap queries for Mojo, with CPU queries and GPU batch counting.
Distribution: `mojo-lapper`; import: `lapper`.

## Query intervals

```mojo {doctest="overlap" global=true}
from lapper.lapper import Interval, Lapper
from std.testing import assert_equal

def main() raises:
    var intervals: List[Interval] = [Interval(10, 20, 1), Interval(18, 30, 2)]
    var index = Lapper(intervals^)
    assert_equal(index.count(19, 21), UInt(2))
    assert_equal(index.count(30, 40), UInt(0))
    var matches = List[Interval]()
    index.find(5, 11, matches)
    assert_equal(len(matches), 1)
```

`Interval(start, stop, value)` represents a half-open range with `UInt32`
endpoints and an `Int32` payload. Overlap requires `a.start < b.stop` and
`a.stop > b.start`; touching boundaries do not overlap.

`Lapper(intervals^)` takes ownership of the list. Use `intervals.copy()` to
retain a separate copy. `count(start, stop)` returns the number of matches;
`find(start, stop, results)` appends matching intervals to the destination.

## CPU query index

Owning CPU indexes use the same forward-query algorithm as **rust-lapper 2.0.1**:
32-interval blocks, prefix maximum ends to locate the first candidate, and a
max tree to skip runs of blocks ending before the query. Dense blocks emit a
prefix of intervals; mixed blocks use native-width SIMD overlap masks, with a
bounded scalar tail. Results remain ordered by start, then stop.

This replaces the maximum-interval-length scan for owning `find` queries.
`count` still uses BITS, construction still uses comparison sorting, and
borrowed/GPU indexes retain their existing algorithms and six-argument pointer
constructor. Do not mutate indexed arrays after construction. This is query
algorithm parity, not parity with Rust's generic types or full API.

`benchmarks/bench_find.mojo` separately times construction and `find` on plain,
single-spanning-interval, dense and mixed data, and prints match counts and
payload checksums. Build with `pixi run mojo build -O3 --march native -I .
benchmarks/bench_find.mojo -o find_bench`, then repeat `taskset -c 6 ./find_bench`
on an idle host. Validate counts/checksums before comparing timings. When
comparing revisions, use separate installed package environments: installed
`.mojoc` packages can take precedence over source include paths.

On a Ryzen 9 3950X (Linux x86-64, Mojo 1.2.0, `-O3 --march native`), seven
serial runs pinned to CPU 6 gave these median query times for 100,000 intervals
and 20,000 queries. Baseline: `2d239a6`; all match counts and checksums agreed.

| Workload | Previous scan (ms) | Block index (ms) |
|---|---:|---:|
| Disjoint intervals | 2.03 | 1.10 |
| Disjoint intervals with one spanning interval | 478.78 | 2.27 |
| Dense overlaps | 16.80 | 18.38 |
| Mixed lengths | 24.40 | 20.74 |

Dense enumeration is about 9% slower in this fixture; this is not a universal
speedup. Construction is separate: the spanning fixture takes about 581 ms
in both versions because the existing comparison sort is unchanged; other
fixtures take 1.2–2.8 ms. These measurements include result-list materialization
and checksum accumulation, not just index traversal.

## Install

The stable toolchain is **Mojo 1.2.0** and **MAX core 26.7.0**.
For a consumer beside the source checkout:

```toml
[workspace]
name = "interval-example"
channels = ["https://conda.modular.com/max", "conda-forge"]
platforms = ["linux-64"]
preview = ["pixi-build"]

[dependencies]
mojo = "=1.2.0"
mojo-lapper = { path = "../mojo-lapper" }
```

Run `pixi run mojo run main.mojo`; no source include path is needed.

## GPU use

`Lapper.prep_for_gpu` requires mutable device buffers. The returned Lapper
borrows those buffers: keep them alive until all queued work finishes.
Kernel lengths are `UInt64`, including query and output lengths.
See [benchmarks/bench_lapper.mojo](benchmarks/bench_lapper.mojo) for a complete
GPU example with explicit buffer ownership and synchronization.

GPU execution requires compatible hardware. These commands check NVIDIA
compilation without executing a kernel:

```sh
pixi run bg --target-accelerator sm_80
pixi run bl --target-accelerator sm_80
```

Compile-only checks do not establish GPU correctness or performance.

## Develop

Install Pixi 0.80 or newer, then run:

```sh
pixi run t                 # CPU tests
pixi run test-consumer     # Build and test an isolated installed package
pixi run format
pixi run docs              # Sibling Mojoverse, or MOJOVERSE path
pixi publish --path . --target-dir dist --force --clean  # Local archive only
```

`main` targets stable Mojo; `nightly` tracks nightly compiler compatibility.
Stable promotion is manual. CI configures CPU and consumer checks for Linux
x86-64, Linux ARM64, and Apple-silicon macOS. Only Linux x86-64 has been
validated locally; GPU runtime and other platforms require native checks.

The documentation task executes the marked example and builds `target/docs`.
CPU/GPU benchmark commands are `pixi run bc`, `pixi run bg`, and `pixi run bl`;
run the resulting `cpu_bench`, `gpu_bench`, or `lapper_bench` executable.

MIT licensed; see [LICENSE](LICENSE).

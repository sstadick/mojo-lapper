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

# Lapper

**Interval overlap detection library for Mojo with CPU and GPU support**

[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-blue)]()

This project implements the BITS (Binary Interval Search Tree) algorithm for efficient interval overlap detection, with both CPU and GPU implementations in Mojo. The BITS algorithm, originally described by [Layer et al.](https://academic.oup.com/bioinformatics/article/29/1/1/273289), uses dual binary searches to count overlaps in O(log n) time. This implementation reinforces how classical algorithms can be adapted to GPU in Mojo's unified programming model. The Lapper data structure will later be filled out further continuing to support both GPU and CPU operations.

## GPU Count Kernel Implementation

The core contribution is a GPU kernel implementing the BITS algorithm for parallel interval counting:

```python
fn count_overlaps_kernel(
    starts: UnsafePointer[UInt32],           # Sorted interval starts
    stops: UnsafePointer[UInt32],            # Interval stops (same order)
    vals: UnsafePointer[Int32],              # Interval values
    stops_sorted: UnsafePointer[UInt32],     # Independently sorted stops
    length: UInt,                            # Number of intervals
    max_len: UInt32,                         # Maximum interval length
    keys: UnsafePointer[UInt32],             # Query pairs [start,stop,start,stop,...]
    keys_length: UInt,                       # 2x number of queries
    output: UnsafePointer[UInt32],           # Result counts per query
    output_length: UInt,                     # Number of queries
):
    # Create GPU lapper instance from device pointers
    var lapper = Lapper[owns_data=False](
        starts, stops, vals, stops_sorted, length, max_len
    )

    # Each thread processes one query
    var thread_id = (block_idx.x * block_dim.x) + thread_idx.x
    if thread_id >= keys_length // 2:
        return

    # Extract query range for this thread
    var idx = thread_id * 2
    var start = keys[idx]
    var stop = keys[idx + 1]

    # Execute BITS algorithm: O(log n) dual binary search
    barrier()
    var count = lapper.count(start, stop)
    barrier()
    output[thread_id] = count
```

**Algorithm**: Each GPU thread independently executes the BITS algorithm using dual binary searches on sorted interval arrays. The BITS approach counts overlaps as `total_intervals - excluded_before - excluded_after`, avoiding the linear scan required by naive methods.

**Performance**: Processes thousands of overlap queries in parallel, with each query completing in O(log n) time. Binary search performs reasonably well on GPU despite branching, particularly when threads in a warp follow similar execution paths. Sorting the 


## Example Usage

Here's a complete example showing CPU and GPU usage:

```python
from lapper import Lapper, Interval, count_overlaps_kernel
from gpu.host import DeviceContext

def main():
    # Create some genomic intervals (genes, reads, etc.)
    var intervals = List[Interval]()
    intervals.append(Interval(1000, 2000, 1))   # Gene 1: [1000, 2000)
    intervals.append(Interval(1500, 2500, 2))   # Gene 2: [1500, 2500) - overlaps Gene 1
    intervals.append(Interval(3000, 4000, 3))   # Gene 3: [3000, 4000) - separate
    intervals.append(Interval(3500, 4500, 4))   # Gene 4: [3500, 4500) - overlaps Gene 3
    intervals.append(Interval(5000, 6000, 5))   # Gene 5: [5000, 6000) - separate

    print("=== CPU Processing ===")
    
    # Build CPU lapper for fast queries
    var cpu_lapper = Lapper(intervals)
    
    # Find all genes overlapping with a sequencing read at [1200, 1800)
    var cpu_results = List[Interval]()
    cpu_lapper.find(1200, 1800, cpu_results)
    print("CPU found", len(cpu_results), "overlapping genes")
    for gene in cpu_results:
        print("  Gene", gene.val, "at [" + String(gene.start) + "," + String(gene.stop) + ")")
    
    # Count overlaps (faster than find when you only need the count)
    var cpu_count = cpu_lapper.count(1200, 1800)
    print("CPU count (O(log n) BITS algorithm):", cpu_count)
    
    # Count overlaps in different regions - this is much faster than find()
    print("CPU count [3200, 3800):", cpu_lapper.count(3200, 3800))
    print("CPU count [5500, 5600):", cpu_lapper.count(5500, 5600))

    print("\n=== GPU Processing ===")
    
    # Setup GPU context and memory buffers
    var ctx = DeviceContext()
    var num_intervals = len(intervals)
    
    # Create host and device buffers for GPU Lapper data
    var host_starts = ctx.enqueue_create_host_buffer[DType.uint32](num_intervals)
    var host_stops = ctx.enqueue_create_host_buffer[DType.uint32](num_intervals)
    var host_vals = ctx.enqueue_create_host_buffer[DType.int32](num_intervals)
    var host_stops_sorted = ctx.enqueue_create_host_buffer[DType.uint32](num_intervals)
    
    var device_starts = ctx.enqueue_create_buffer[DType.uint32](num_intervals)
    var device_stops = ctx.enqueue_create_buffer[DType.uint32](num_intervals)
    var device_vals = ctx.enqueue_create_buffer[DType.int32](num_intervals)
    var device_stops_sorted = ctx.enqueue_create_buffer[DType.uint32](num_intervals)
    
    # Prepare GPU lapper with optimized memory layout
    var gpu_lapper = Lapper.prep_for_gpu(
        ctx, intervals,
        host_starts, host_stops, host_vals, host_stops_sorted,
        device_starts, device_stops, device_vals, device_stops_sorted
    )
    
    # Batch process multiple queries on GPU simultaneously
    var batch_queries = List[Interval]()
    batch_queries.append(Interval(1200, 1800, 0))  # Query 1
    batch_queries.append(Interval(3200, 3800, 0))  # Query 2  
    batch_queries.append(Interval(5500, 5600, 0))  # Query 3
    
    var num_queries = len(batch_queries)
    
    # Setup GPU memory for batch queries
    var host_keys = ctx.enqueue_create_host_buffer[DType.uint32](num_queries * 2)
    var device_keys = ctx.enqueue_create_buffer[DType.uint32](num_queries * 2)
    var device_output = ctx.enqueue_create_buffer[DType.uint32](num_queries)
    var host_output = ctx.enqueue_create_host_buffer[DType.uint32](num_queries)
    
    # Pack query intervals as [start, stop, start, stop, ...] for GPU
    for i in range(num_queries):
        host_keys[i * 2] = batch_queries[i].start
        host_keys[i * 2 + 1] = batch_queries[i].stop
    
    # Transfer query data to GPU
    host_keys.enqueue_copy_to(device_keys)
    ctx.synchronize()
    
    # Launch GPU kernel to process all queries in parallel
    ctx.enqueue_function[count_overlaps_kernel](
        gpu_lapper.starts, gpu_lapper.stops, gpu_lapper.vals,
        gpu_lapper.stops_sorted, gpu_lapper.length, gpu_lapper.max_len,
        device_keys.unsafe_ptr(), num_queries * 2,
        device_output.unsafe_ptr(), num_queries,
        grid_dim=1, block_dim=32  # Small batch, single block sufficient
    )
    
    # Transfer results back to CPU
    device_output.enqueue_copy_to(host_output)
    ctx.synchronize()
    
    print("GPU batch results (same algorithms, parallel execution):")
    for i in range(num_queries):
        var query = batch_queries[i]
        var gpu_count = host_output[i]
        print("  Query [" + String(query.start) + "," + String(query.stop) + ") count:", gpu_count)
    
    print("\nGPU demonstrates how classical CPU algorithms adapt seamlessly to parallel execution!")
```

## Key Observations

**Algorithm Adaptation**: The BITS algorithm, adapts well to GPU with Mojo's unified syntax. The same algorithmic logic runs on both platforms with minimal code changes.

**GPU Binary Search**: While binary search has irregular memory access patterns, it can work reasonably well on GPU when threads exhibit some coherence. Performance depends heavily on data patterns and warp divergence.

**Unified Programming**: Mojo's shared syntax between CPU and GPU simplifies porting algorithms and experimenting with different execution strategies without rewriting core logic. Specifically the same data structure, `Lapper` can be used on the GPU and CPU here.

## Architecture

```
mojo-lapper/
├── lapper/
│   ├── lapper.mojo          # Main Lapper struct and Interval definitions
│   ├── cpu/
│   │   ├── bsearch.mojo     # CPU binary search variants
│   │   └── eytzinger.mojo   # Cache-efficient tree layout
│   └── gpu/
│       ├── bsearch.mojo     # GPU kernel implementations  
│       └── eytzinger.mojo   # GPU Eytzinger kernels
├── benchmarks/              # Performance benchmarking
└── tests/                   # Test suite (50 tests)
```

## Performance Results

*Note: These benchmarks compare different implementations within this codebase only. Absolute performance will vary significantly based on hardware, data patterns, and use cases.*

**Test Hardware:**
- **CPU**: Intel Xeon Platinum 8358 (30 cores, 2.60GHz, 480MB L3)  
- **GPU**: NVIDIA A10 (9,728 CUDA cores, 23GB memory)

### CPU Binary Search Comparison

| Algorithm | Time (ms) | Relative Performance |
|-----------|-----------|---------------------|
| Binary Search | 5.87 | 1.0x (baseline) |
| Offset Binary Search | 6.60 | 0.89x |
| Lower Bound | 6.34 | 0.93x |
| **Eytzinger Layout** | **2.42** | **2.4x** |

The Eytzinger layout shows substantial improvement due to better cache locality.

### GPU Binary Search Results

| Algorithm | Time Range (ms) | Performance Notes |
|-----------|-----------------|-------------------|
| Eytzinger Lower Bound | 0.017-0.137 | Consistent across block sizes 32-1024 |
| Binary Search Lower Bound | 0.017-0.137 | Similar performance to Eytzinger on GPU |

GPU binary search shows consistent microsecond-level performance across different thread block sizes, for both search implementations. The Eytzinger layout performance does not translate to the GPU.

### CPU vs GPU Lapper Operations

| Operation | CPU (ms) | GPU (ms) | Speedup | Notes |
|-----------|----------|----------|---------|-------|
| Count overlaps (BITS) | 1.12 | 0.008 | 140x | GPU BITS algorithm highly optimized |
| Count overlaps (Naive) | 10.89 | 0.11 | 99x | GPU benefits from parallel queries |

Note: GPU benchmarks exclude data transfer time. Including transfer overhead, GPU BITS is ~60-120x faster than CPU.

## API Overview

### Core Types

**`Interval`**: Represents a half-open interval [start, stop) with associated data
```python
var gene = Interval(start=1000, stop=2000, value=42)
```

**`Lapper`**: Data structure for efficient interval overlap queries
```python
var cpu_lapper = Lapper(intervals)                   # CPU version (owns memory)
var gpu_lapper = Lapper[owns_data=False](intervals)  # GPU version (external memory)
```

### Key Methods

**`find(start, stop, results)`**: Find all overlapping intervals
- Time: O(log n + k) where k = number of overlaps
- Populates results list with overlapping intervals

**`count(start, stop)`**: Count overlapping intervals  
- Time: O(log n) using BITS algorithm
- Returns count without materializing results

### Overlap Semantics

Uses strict inequality: intervals overlap when `(a.start < b.stop) AND (a.stop > b.start)`.

Examples:
- `[1,5)` and `[5,10)` → No overlap (touching boundaries)
- `[1,6)` and `[5,10)` → Overlap

## Testing

Comprehensive test suite with 50 tests covering:
- Core functionality and edge cases
- All binary search variants  
- CPU/GPU result validation
- Memory management

```bash
pixi run t                    # Run all tests
pixi run t --filter test_lapper  # Core library tests
```

## Benchmarking

```bash
pixi run bc && ./cpu_bench    # CPU binary search benchmarks
pixi run bg && ./gpu_bench    # GPU kernel benchmarks  
pixi run bl && ./lapper_bench # CPU vs GPU validation
```

The benchmarks include automatic validation to ensure CPU and GPU implementations produce identical results.

## Use Cases

- **Genomics**: Gene annotation, variant analysis, read mapping
- **Time Series**: Event detection, temporal pattern analysis
- **Computational Geometry**: Range queries, collision detection
- **Resource Scheduling**: Time slot conflict detection

## Implementation Notes

**Memory Layout**: Uses Structure of Arrays (SoA) for better cache performance and GPU memory coalescing.

**Algorithms**: 
- Standard binary search with optimizations
- Eytzinger layout for cache efficiency  
- BITS algorithm for O(log n) counting
- GPU kernels for parallel processing

**Thread Safety**: Query operations are thread-safe; construction is not.

## Development

**Requirements**: Mojo (latest), GPU support (optional), Pixi

```bash
pixi run t        # Tests
pixi run bc       # Build CPU benchmarks
pixi run bg       # Build GPU benchmarks
pixi run bl       # Build Lapper benchmarks
```

## Lessons Learned

1. **Algorithm Portability**: Many CPU algorithms can be adapted for GPU execution, though performance varies significantly based on memory access patterns and control flow
2. **GPU Binary Search**: Binary search can work on GPU when implemented carefully, though it's not always the optimal choice compared to alternatives like parallel reduction
3. **Unified Development**: Mojo's shared syntax between CPU and GPU reduces the friction of experimenting with different execution strategies
4. **Data Layout Effects**: CPU performance varies dramatically with data layout - Eytzinger layout shows 2.4x improvement over standard binary search

## Further Work

This project demonstrates the core feasibility of co-CPU/GPU-accelerated interval operations, but several opportunities remain for future development:

- **Reduce Levels**: Adapt cache-optimized Eytzinger layout for GPU memory hierarchies with [k-ary search](https://arxiv.org/abs/2506.01576)
- **SIMD/SIMT**: Parallelize the linear search portion of `find`
- **Sort on GPU**: Construction involves at least two sorts, which could be done on the GPU, or at least moved to radix sort
- **Multi-GPU Scaling**: Distribute large interval datasets across multiple GPUs
- **Complete the GPU Find Kernel**: GPU kernels that materialize actual overlapping intervals (not just counts)
    - Preprocessing with BITS to determine the number of slots for output
- **Complete the Lapper API**: Lapper is a port of my [`rust-lapper`](https://github.com/sstadick/rust-lapper/tree/master) which is a port of [`nim-lapper`](https://github.com/brentp/nim-lapper) from Brent Pendersen
    - **Generic**: Update Lapper and Interval to be generic over integral types
    - **Result Streaming**: Iterator-based APIs for processing large result sets for CPU api


## Bugs Submitted
- [rust-lapper](https://github.com/sstadick/rust-lapper/blob/828fec77e0f94c8b63ad54af223f005cb903f450/src/lib.rs#L598) is not properly handling the half-open intervals and is using full-open intervals as in the BITS paper. A PR will be submitted to fix it.
- [UInt64 vs UInt32 in Mojo](https://github.com/modular/modular/issues/4927) Notably different code gen and performance when using UInt64 vs UInt32 in the binary search benchmarks.

## References

- [Static Search Trees](https://curiouscoding.nl/posts/static-search-tree/)
- [HPC Binary Search](https://en.algorithmica.org/hpc/data-structures/binary-search/)
- [Rust Lapper](https://github.com/sstadick/rust-lapper)
- [BITS Algorithm Paper](https://academic.oup.com/bioinformatics/article/29/1/1/273289)

## Install

Until this is on modular-community, you can depend on it directly from git.

```toml
# pixi.toml
[workspace]
authors = ["Mojo <mojo@hotmail.com>"]
channels = ["https://conda.modular.com/max-nightly", "conda-forge", "https://prefix.dev/pixi-build-backends", "https://repo.prefix.dev/modular-community",]
platforms = ["osx-arm64"]
preview = ["pixi-build"]

[package]
name = "example"
version = "0.1.0"

[package.build]
backend = { name = "pixi-build-rattler-build", version = "0.1.*" }

[tasks]

[dependencies]
modular = "=25.4.0"
lapper = {git = "https://github.com/sstadick/mojo-lapper.git"}
```

```python
# main.mojo
from lapper.lapper import Lapper, Interval as Iv


def main():
    var intervals: List[Iv] = [Iv(1, 5, 0), Iv(2, 6, 0), Iv(6, 10, 0)]
    var lapper = Lapper(intervals)
    print(lapper.count(2, 6))
```


---

*Built with Mojo - Exploring the boundaries between CPU and GPU computing*

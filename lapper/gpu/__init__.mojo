"""GPU-accelerated interval search kernels for parallel overlap detection.

This module provides GPU kernel implementations for massively parallel interval
overlap detection. The kernels are designed for high-throughput batch processing
of thousands of overlap queries simultaneously.

Key Features:
- **Parallel Search Kernels**: GPU kernels for binary search operations
- **Batch Processing**: Process thousands of queries in parallel
- **Memory Coalescing**: Optimized memory access patterns for GPU efficiency
- **Unified API**: Same interface as CPU implementations
- **Device Integration**: Seamless integration with Mojo GPU infrastructure

Components:
- `bsearch.mojo`: GPU kernels for standard binary search operations
- `eytzinger.mojo`: GPU kernels for cache-efficient Eytzinger layout searches

Performance Characteristics:
- **Throughput**: Process millions of queries per second on modern GPUs
- **Scalability**: Performance scales with number of CUDA cores/compute units
- **Memory Bandwidth**: Optimized for GPU memory hierarchy
- **Batch Efficiency**: Best performance with large query batches (>1K queries)

GPU Architecture Support:
- NVIDIA CUDA-compatible GPUs
- Modern compute capabilities (SM 6.0+)
- Unified memory and discrete GPU memory support
- Automatic memory management with Mojo GPU infrastructure

Usage Pattern:
    ```mojo
    # Setup GPU context and data
    var gpu_lapper = Lapper[owns_data=False](intervals)
    
    # Process batch of queries on GPU
    var query_results = process_gpu_batch(gpu_lapper, query_list)
    ```

Performance Considerations:
- GPU kernels have higher latency but much higher throughput than CPU
- Best for large batches of queries (>1000 queries)
- Memory transfer overhead must be amortized over query batch size
- Consider using unified memory for mixed CPU/GPU workloads

Thread and Memory Safety:
- GPU kernels handle memory coalescing automatically
- Synchronization required between CPU and GPU operations
- Device memory management handled by Mojo GPU infrastructure
- Multiple streams can be used for overlapping computation and data transfer
"""
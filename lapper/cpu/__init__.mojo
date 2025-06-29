"""CPU-optimized interval search algorithms and data structures.

This module contains high-performance CPU implementations of binary search algorithms
optimized for interval overlap detection. The implementations focus on cache efficiency,
branch prediction optimization, and modern CPU features.

Key Algorithms:
- **Standard Binary Search**: Traditional divide-and-conquer with optimized edge cases
- **Eytzinger Layout**: Cache-efficient BFS tree layout for large datasets  
- **Offset Binary Search**: Power-of-two stepping from array boundaries
- **Branchless Search**: Arithmetic-based comparisons to reduce branch mispredictions

Components:
- `bsearch.mojo`: Multiple binary search variants with different performance characteristics
- `eytzinger.mojo`: Cache-efficient tree layout with breadth-first ordering

Performance Characteristics:
- **Standard searches**: Good general-purpose performance with O(log n) complexity
- **Eytzinger layout**: 10-50% speedup on large datasets due to improved cache locality
- **Branch optimization**: Reduced CPU pipeline stalls on predictable data
- **Memory efficiency**: Minimal overhead with optional prefetching support

Usage Guidelines:
- Use standard binary search for general-purpose applications
- Use Eytzinger layout for large datasets (>1K elements) with frequent searches
- Consider memory overhead when choosing between algorithms
- Benchmark with your specific data patterns for optimal performance

Thread Safety:
- All search operations are thread-safe
- Construction operations are not thread-safe
- Multiple threads can search the same data structure simultaneously

Best Performance Achieved:
- Millions of searches per second on modern CPUs
- Cache-friendly access patterns reduce memory stalls
- Optimized for both Intel and ARM architectures
"""
"""Lapper: High-performance interval overlap detection library for Mojo.

Lapper is a fast interval overlap detection library optimized for both CPU and GPU execution.
It provides efficient algorithms for finding and counting overlapping intervals in genomic
data, time series analysis, and other applications requiring interval intersection queries.

Key Features:
- **Fast Overlap Detection**: O(log n) average case performance using optimized binary search
- **CPU and GPU Support**: Unified API for both CPU and GPU execution
- **Multiple Search Algorithms**: Standard binary search, Eytzinger layout, and GPU kernels
- **Comprehensive API**: Find overlaps, count overlaps, interval manipulation
- **Memory Efficient**: Optimized data structures with minimal memory overhead

Core Components:
- `Interval`: Represents a single interval with start, stop, and value
- `Lapper`: Main data structure for storing and querying interval collections
- Binary search implementations for efficient overlap detection
- GPU kernels for parallel overlap computation

Performance Characteristics:
- **Construction**: O(n log n) for sorting intervals
- **Query**: O(log n + k) where k is the number of overlaps found
- **Memory**: O(n) storage with optional GPU buffer management
- **Throughput**: Millions of queries per second on modern hardware

Example Usage:
    ```mojo
    from lapper import Lapper, Interval
    
    # Create intervals
    var intervals = List[Interval]()
    intervals.append(Interval(10, 20, 1))
    intervals.append(Interval(15, 25, 2))
    intervals.append(Interval(30, 40, 3))
    
    # Build lapper for fast queries
    var lapper = Lapper(intervals)
    
    # Find overlaps
    var results = List[Interval]()
    lapper.find(12, 18, results)  # Returns intervals that overlap [12, 18)
    
    # Count overlaps
    var count = lapper.count(12, 18)  # Returns number of overlapping intervals
    ```

Thread Safety:
- Construction is not thread-safe
- Queries are thread-safe on immutable Lapper instances
- GPU operations require proper synchronization

When to Use Lapper:
-  Large datasets with frequent overlap queries
-  Genomic interval analysis (genes, reads, annotations)
-  Time series data with temporal intervals
-  Computational geometry applications
-  Performance-critical interval processing
- L Small datasets (<100 intervals)
- L Infrequent queries
- L Simple one-time interval operations
"""

from gpu import thread_idx, block_idx, block_dim, warp, barrier
from memory import UnsafePointer

from lapper.cpu.bsearch import lower_bound


fn lower_bound_kernel[
    dtype: DType
](
    keys: UnsafePointer[Scalar[dtype]],  # expect keys to be in sorted order
    key_length: UInt,
    elems: UnsafePointer[Scalar[dtype]],  # eytz layout expected
    elems_length: UInt,
    output: UnsafePointer[Scalar[dtype]],
    output_length: UInt,  # technically same as key_length
):
    """Find the lower bound for all keys."""
    constrained[dtype is not DType.invalid, "dtype must be vaild"]()

    var elems_span = Span(ptr=elems, length=elems_length)

    var thread_id = (block_idx.x * block_dim.x) + thread_idx.x
    if thread_id >= key_length:
        return
    var key = keys[thread_id]
    var result = lower_bound(elems_span, key)
    output[thread_id] = result

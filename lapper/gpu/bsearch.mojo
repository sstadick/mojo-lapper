"""Batch lower-bound queries on GPU-resident sorted arrays.

Keep device-buffer owners alive through queued work and synchronize before
reading results on the CPU.
"""

from max.gpu import thread_idx, block_idx, block_dim

from lapper.cpu.bsearch import lower_bound


def lower_bound_kernel[
    dtype: DType
](
    keys: MutPointer[Scalar[dtype], MutUntrackedOrigin],  # expect keys to be in sorted order
    key_length: UInt64,
    elems: MutPointer[Scalar[dtype], MutUntrackedOrigin],  # sorted array
    elems_length: UInt64,
    output: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    output_length: UInt64,  # technically same as key_length
):
    """Write the lower-bound index for each key into its output slot.

    `elems` must be sorted in ascending order. Provide one output slot per
    key, with key_length == output_length. All pointers reference live
    device buffers; lengths are UInt64. Output indices use the chosen dtype.
    """
    var elems_span = Span(unsafe_ptr=elems, length=Int(elems_length))

    var thread_id = (block_idx.x * block_dim.x) + thread_idx.x
    if UInt64(thread_id) >= key_length:
        return
    var key = keys[unsafe_offset=thread_id]
    var result = lower_bound(elems_span, key)
    output[unsafe_offset=thread_id] = Scalar[dtype](result)

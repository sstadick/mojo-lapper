from max.gpu import thread_idx, block_idx, block_dim

from lapper.cpu.eytzinger import lower_bound


def lower_bound_kernel[
    dtype: DType
](
    keys: MutPointer[Scalar[dtype], MutUntrackedOrigin],  # expect keys to be in sorted order
    key_length: UInt64,
    elems: MutPointer[Scalar[dtype], MutUntrackedOrigin],  # eytz layout expected
    elems_length: UInt64,
    output: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    output_length: UInt64,  # technically same as key_length
):
    """Find the lower bound for all keys using Eytzinger layout.

    Parameters:
        dtype: The data type of the keys and elements.

    Args:
        keys: Pointer to sorted array of keys to search for.
        key_length: Number of keys to search for.
        elems: Pointer to elements in Eytzinger layout.
        elems_length: Number of elements in the Eytzinger array.
        output: Pointer to output array for storing results.
        output_length: Length of output array (should match key_length).
    """
    var elems_span = Span(unsafe_ptr=elems, length=Int(elems_length))

    var thread_id = (block_idx.x * block_dim.x) + thread_idx.x
    if UInt64(thread_id) >= key_length:
        return
    var key = keys[unsafe_offset=thread_id]
    var result = lower_bound(elems_span, key)
    output[unsafe_offset=thread_id] = Scalar[dtype](result)

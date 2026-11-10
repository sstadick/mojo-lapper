"""GPU batch-search kernels.

Use `bsearch` or `eytzinger` with compatible device buffers. Retain buffer
owners through queued work and synchronize before reading results on the CPU.
"""

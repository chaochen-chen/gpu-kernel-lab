"""Softmax implementations. Every impl has the signature f(x: [M, N]) -> [M, N]."""

import shutil

from .reference import softmax_ref
from .triton_naive import softmax_triton_naive
from .triton_online import softmax_triton_online

REF = softmax_ref

IMPLS = {
    "triton_naive": softmax_triton_naive,
    "triton_online": softmax_triton_online,
}

# The CUDA version needs nvcc to JIT-compile; register it only when available.
if shutil.which("nvcc"):
    from .cuda.binding import softmax_cuda_online

    IMPLS["cuda_online"] = softmax_cuda_online

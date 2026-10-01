"""Lazy JIT build of the CUDA online softmax (compiled on first call, cached in build/)."""

from functools import lru_cache
from pathlib import Path

import torch

from kernels.utils import load_cuda

_SRC = Path(__file__).parent / "online_softmax.cu"


@lru_cache(maxsize=1)
def _ext():
    return load_cuda("online_softmax_ext", [_SRC])


def softmax_cuda_online(x: torch.Tensor) -> torch.Tensor:
    return _ext().online_softmax(x)


softmax_cuda_online.dtypes = (torch.float32,)  # read by tests to skip unsupported dtypes

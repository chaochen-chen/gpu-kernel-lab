"""Shared helpers for building CUDA extensions with torch.utils.cpp_extension."""

import os
from pathlib import Path

import torch

BUILD_DIR = Path(__file__).resolve().parent.parent / "build"


def _arch_flag() -> list[str]:
    """Compile for the local GPU unless TORCH_CUDA_ARCH_LIST is set (torch handles that)."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST") or not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


def load_cuda(name: str, sources: list[str | Path], extra_cuda_cflags=None, verbose=False):
    """JIT-compile and load a CUDA extension (cached under build/<name>)."""
    from torch.utils.cpp_extension import load

    build_dir = BUILD_DIR / name
    build_dir.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "--use_fast_math", *_arch_flag(), *(extra_cuda_cflags or [])]
    return load(
        name=name,
        sources=[str(s) for s in sources],
        extra_cuda_cflags=flags,
        build_directory=str(build_dir),
        verbose=verbose,
    )

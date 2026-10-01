import re
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def gpu_name() -> str:
    """Slug of the current GPU name, e.g. 'nvidia-a100-sxm4-80gb'. Used in result filenames."""
    return re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name().lower()).strip("-")


def bandwidth_gbps(num_bytes: float, ms: float) -> float:
    return num_bytes / (ms * 1e-3) / 1e9


def tflops(flops: float, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12

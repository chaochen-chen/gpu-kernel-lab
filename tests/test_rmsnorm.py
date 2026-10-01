import pytest
import torch

from kernels.rmsnorm import IMPLS, REF
from tests.utils import assert_close, supports_dtype

SHAPES = [(1, 128), (32, 1000), (4096, 4096)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("name", IMPLS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm(name, shape, dtype):
    fn = IMPLS[name]
    if not supports_dtype(fn, dtype):
        pytest.skip(f"{name} does not support {dtype}")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    assert_close(fn(x, w), REF(x, w))

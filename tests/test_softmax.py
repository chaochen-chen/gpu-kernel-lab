import pytest
import torch

from kernels.softmax import IMPLS, REF
from tests.utils import assert_close, supports_dtype

SHAPES = [(1, 128), (32, 1000), (7, 4097), (4096, 4096)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("name", IMPLS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax(name, shape, dtype):
    fn = IMPLS[name]
    if not supports_dtype(fn, dtype):
        pytest.skip(f"{name} does not support {dtype}")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    assert_close(fn(x), REF(x))


@pytest.mark.parametrize("name", IMPLS)
def test_softmax_large_values_are_stable(name):
    """Without max-subtraction, exp() would overflow here."""
    x = torch.randn(16, 1024, device="cuda", dtype=torch.float32) * 50 + 80
    out = IMPLS[name](x)
    assert torch.isfinite(out).all()
    assert_close(out, REF(x))

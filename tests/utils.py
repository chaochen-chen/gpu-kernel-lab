import torch

# (rtol, atol) per dtype
TOL = {
    torch.float32: (1e-4, 1e-6),
    torch.float16: (2e-3, 1e-5),
    torch.bfloat16: (1e-2, 1e-3),
}


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    rtol, atol = TOL[expected.dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def supports_dtype(fn, dtype: torch.dtype) -> bool:
    """Impls may declare `fn.dtypes = (...)`; default is all dtypes in TOL."""
    return dtype in getattr(fn, "dtypes", tuple(TOL))

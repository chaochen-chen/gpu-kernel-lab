"""Forward-only RMSNorm in Triton, one program per row. Compare with Liger-Kernel's rms_norm."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    y_ptr, x_ptr, w_ptr, x_row_stride, y_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * x_row_stride + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(y_ptr + row * y_row_stride + offs, x * rstd * w, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.ndim == 2 and x.is_cuda
    x = x.contiguous()
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    _rmsnorm_fwd_kernel[(n_rows,)](
        y, x, weight, x.stride(0), y.stride(0), n_cols, eps, BLOCK_SIZE=block
    )
    return y

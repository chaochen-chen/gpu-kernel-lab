"""Row softmax, one program per row, whole row in registers (safe softmax: 3 passes in-register)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * in_row_stride + offs, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + offs, y, mask=mask)


def softmax_triton_naive(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.is_cuda
    x = x.contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=block)
    return out

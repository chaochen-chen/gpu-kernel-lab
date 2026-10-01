"""Online softmax (Milakov & Gimelshein, 2018): single pass for max + denominator.

Loops over column chunks keeping a running max `m` and running denominator `d`:
    m_new = max(m, block_max)
    d     = d * exp(m - m_new) + sum(exp(x_block - m_new))
A second pass then writes exp(x - m) / d. Works for any row length (no giant block needed).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _online_softmax_kernel(
    out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    in_row = in_ptr + row * in_row_stride
    out_row = out_ptr + row * out_row_stride

    m = float("-inf")
    d = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_row + offs, mask=offs < n_cols, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
        tl.store(out_row + offs, tl.exp(x - m) / d, mask=mask)


def softmax_triton_online(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.ndim == 2 and x.is_cuda
    x = x.contiguous()
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = min(block_size, triton.next_power_of_2(n_cols))
    _online_softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=block)
    return out

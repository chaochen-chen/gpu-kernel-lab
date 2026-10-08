## TODOs
### Kernel Implementation
- [ ] backward operator
### Test
- [ ] test_correctness's and assert_verbose_allclose's implementation
- [ ] Functional API (test_correctness_functional)
- [ ] Int32 overflow in the blocked kernel

# Kernel Implementation
## Math

**Notation:** `x ∈ ℝⁿ` is one row (`n = n_cols`), `w` is the weight, `o` is the offset (0 for Llama, 1 for Gemma), and `ε = eps`.

### Forward

```
rstd = 1 / sqrt( (1/n) · Σ_j x_j² + ε )
x̂_i  = x_i · rstd
y_i  = x̂_i · (o + w_i)
```

### Backward

```
[TO BE FILLED]
```

## Shapes & dtypes

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `X` | `[n_rows, n_cols]` | bf16/fp16/fp32 | input, flattened to 2D |
| `W` | `[n_cols]` | same as `X` | skipped if `elementwise_affine=False` |
| `Y` | `[n_rows, n_cols]` | same as `X` | output |
| `rstd` | `[n_rows]` | fp32 | saved for backward |
| `dW` (partial) | `[TO BE FILLED]` | [TO BE FILLED] | summed on host over dim 0 |

## Strategy

- **Forward**:
  - Wide rows: one row per program (`_rms_norm_forward_kernel`).
  - Narrow rows, many rows: several rows per program (`_block_rms_norm_forward_kernel`).
- **Backward**: [TO BE FILLED]
- **BLOCK_SIZE**: [TO BE FILLED]
- Backward-temp: `dx` is row-local, but `dw` needs a cross-row reduction.
  - Grid of about `sm_count` programs; each loops over its rows and accumulates a private `dW` partial.
  - The host sums the `[sm_count, n_cols]` buffer. No atomics.
- BLOCK_SIZE-temp: `next_power_of_2(n_cols)`, capped by the max fused size.

## Kernels & inputs

### `_rms_norm_forward_kernel`
- Pointers / strides: `Y_ptr, Y_row_stride, X_ptr, X_row_stride, W_ptr, W_row_stride, RSTD_ptr, RSTD_row_stride`
- Scalars: `n_cols, eps, offset`
- `tl.constexpr`: `casting_mode, elementwise_affine, BLOCK_SIZE`

### `_rms_norm_backward_kernel`
- Pointers / strides: [TO BE FILLED]
- Scalars: [TO BE FILLED]
- `tl.constexpr`: [TO BE FILLED]

## dtype in models

Selected by `casting_mode`.

| Step | Llama | Gemma |
|---|---|---|
| Load `x` | → fp32 | → fp32 |
| `rstd` | fp32 | fp32 |
| `x̂ = x·rstd` | fp32, then **cast back to input dtype** | fp32 |
| Multiply by `(o + w)` | **input dtype** | fp32 |
| Output | input dtype | cast to input dtype at the very end |
| `offset` | 0 | 1.0 |

- Llama mirrors HF `weight * hidden_states.to(input_dtype)`.
- Gemma mirrors HF `x̂ * (1 + w)` in fp32, cast last.
- A `none` mode applies no casting.
- The backward kernel must replicate the same casting, otherwise gradients drift from the reference.


## Follow-up questions
- [ ] one-row-program vs block-row-program: selection of threshold
- [ ] `rstd`: why need a specific function `rsqrt`?
- [ ] `rsqrt`: different triton's version

## Takeaways
- Save the small statistic (`rstd`), recompute the big tensor (`x̂`).
- Use per-program partial buffers plus a host-side sum instead of atomics for cross-row reductions.
- Match the reference's casting order exactly, because it decides numerical parity.

# Test
## Test correctness
Run the same random input through a reference RMSNorm (Pytorch native, Llama, etc.) and LigerRMSNorm, then backpropagates a random upstream gradient through both. It checks three things match within tolerance:
- the forward output
- the weight gradient (only when elementwise_affine=True)
- the input gradient <br>

The parametrization covers the variants that matter:
- **Shapes**: 1) a normal one (2×128×512) 2) an odd one (5×123×123) (test masking condition)
- **Dtypes**: 1) fp32 (tight tolerance, 1e-4) and 2) bf16 (loose tolerance, 2e-1, skipped if the GPU lacks bf16 support).
- **Reference**: 1) Llama (offset=0, upcasts to fp32 before the norm, then casts back), 2) Gemma (offset=1, computes x * (1 + w) in fp32), 3) and a "Base" version with no upcasting (casting_mode="none", skipped on Ascend NPU).


## Follow-up questions
- [ ] What is `test_block_rms_norm_int32_row_offset_wraps` for?
- [ ] What is `test_block_rms_norm_large_row_offset` for?

  

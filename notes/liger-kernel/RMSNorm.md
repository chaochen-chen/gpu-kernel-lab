# RMSNorm
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
u_i  = dy_i · (o + w_i)

dx_i = rstd · u_i  −  (rstd³ / n) · x_i · Σ_j u_j x_j
     = rstd · ( u_i  −  x̂_i · (1/n) · Σ_j u_j x̂_j )

dw_i = Σ_rows dy_i · x̂_i
```

### Derivation

```
∂rstd/∂x_i = −rstd³ · x_i / n
```

`dx` collects two paths: the direct `x_i` term and the path through `rstd`.

### Saved for backward

`X`, `W`, and `rstd` (one fp32 scalar per row). `Y` is not saved, so memory is traded for recomputing `x̂`.

## Shapes & dtypes

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `X` | `[n_rows, n_cols]` | bf16/fp16/fp32 | input, flattened to 2D |
| `W` | `[n_cols]` | same as `X` | skipped if `elementwise_affine=False` |
| `Y` | `[n_rows, n_cols]` | same as `X` | output |
| `rstd` | `[n_rows]` | fp32 | saved for backward |
| `dW` (partial) | `[sm_count, n_cols]` | fp32 | summed on host over dim 0 |

## Strategy

- **Forward**:
  - Wide rows: one row per program (`_rms_norm_forward_kernel`).
  - Narrow rows, many rows: several rows per program (`_block_rms_norm_forward_kernel`).
- **Backward**: `dx` is row-local, but `dw` needs a cross-row reduction.
  - Grid of about `sm_count` programs; each loops over its rows and accumulates a private `dW` partial.
  - The host sums the `[sm_count, n_cols]` buffer. No atomics.
- **BLOCK_SIZE**: `next_power_of_2(n_cols)`, capped by the max fused size.

## Kernels & inputs

### `_rms_norm_forward_kernel`
- Pointers / strides: `Y_ptr, Y_row_stride, X_ptr, X_row_stride, W_ptr, W_row_stride, RSTD_ptr, RSTD_row_stride`
- Scalars: `n_cols, eps, offset`
- `tl.constexpr`: `casting_mode, elementwise_affine, BLOCK_SIZE`

### `_rms_norm_backward_kernel`
- Pointers / strides: `dY_ptr, dX_ptr, X_ptr, W_ptr, RSTD_ptr, dW_ptr` + matching strides
- Scalars: `n_rows, n_cols, offset, rows_per_program`
- `tl.constexpr`: `casting_mode, elementwise_affine, BLOCK_SIZE`
- ⚠️ Some versions write `dX` in place over `dY` (`in_place`).

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
  

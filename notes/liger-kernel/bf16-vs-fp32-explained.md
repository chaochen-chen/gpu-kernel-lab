# Floating-Point Formats: fp32, bf16, fp16

- [1. Core idea](#1-core-idea)
- [2. Why bf16 works well for training](#2-why-bf16-works-well-for-training)
- [3. Worked example: storing π](#3-worked-example-storing-π)
- [4. How the exponent is stored](#4-how-the-exponent-is-stored)
- [5. Key takeaways](#5-key-takeaways)

## 1. Core idea

A floating-point number works like scientific notation (e.g. 1.5 × 10³), but in binary:

`
value = (−1)^sign × 1.mantissa × 2^(exponent − bias)
`

| Part | Role |
|---|---|
| **Sign** (1 bit) | Positive or negative |
| **Exponent** | Scale: how big or small the number is |
| **Mantissa** | Significant digits, which set the precision |

The formats differ only in how many bits go to each part:

| Format | Total bits | Sign | Exponent | Mantissa | Bias | Bytes |
|---|---|---|---|---|---|---|
| fp32 | 32 | 1 | 8 | 23 | 127 | 4 |
| bf16 | 16 | 1 | 8 | 7 | 127 | 2 |
| fp16 | 16 | 1 | 5 | 10 | 15 | 2 |

- **fp32** (float32) is the classic "full precision" format. It has ample range and precision, so it is numerically safe, but costs more memory and slower math.
- **bf16** (bfloat16, from Google Brain) is essentially a truncated fp32. It keeps the same 8 exponent bits, so it covers the same range, but only 7 mantissa bits, so each value is less precise. It uses half the memory, and GPUs have fast hardware for it.
- **fp16** has more precision than bf16 but a much smaller range.

## 2. Why bf16 works well for training

- Training cares more about **range** than **precision**. Gradients can be tiny or huge, and an overflow or underflow breaks training.
- Small rounding errors mostly average out over millions of operations.
- bf16 gives fp32's range at half the size, which is why it is the default for LLMs.
- fp16 has a small range, so it needs tricks like **loss scaling** to avoid overflow and underflow.

### Mixed precision in practice

| Precision | Used for |
|---|---|
| **bf16** | Weights, activations, and most matrix multiplies (for speed and memory) |
| **fp32** | Numerically sensitive operations: accumulating sums, softmax, norm statistics, optimizer states, and sometimes a master copy of the weights |

This is why a kernel like RMSNorm can take bf16 inputs but upcast to fp32 internally.

## 3. Worked example: storing π

### Step 1: Write π in binary scientific notation

```
π ≈ 3.14159265
π = 11.001001000011111101101…          (binary)
  = 1.1001001000011111101101… × 2¹
```

So:

```
π = (+) × 1.[1001001000011111101101…] × 2^(1)
     │          │                          │
    sign     mantissa                   exponent
```

### Step 2: Sign

π is positive, so the sign bit is **0**.

### Step 3: Exponent

The true exponent is 1. The format stores `true exponent + bias`, so it never needs a negative sign.

| Format | Calculation | Stored bits |
|---|---|---|
| fp32 / bf16 | 1 + 127 = 128 | `10000000` |
| fp16 | 1 + 15 = 16 | `10000` |

### Step 4: Mantissa

Take the bits after the `1.` and drop the leading 1, since it is always there and the hardware assumes it. Keep as many bits as the format has room for, rounding to nearest:

| Format | Mantissa bits | Stored |
|---|---|---|
| fp32 | 23 | `10010010000111111011011` |
| bf16 | 7 | `1001001` |
| fp16 | 10 | `1001001000` |

> [!NOTE]
> The fp32 mantissa is rounded **up** (the bits continue `…011010|1…`). bf16 and fp16 happen to round down.

### Step 5: Final bit patterns

```
        sign | exponent | mantissa
fp16:     0  | 10000    | 1001001000
bf16:     0  | 10000000 | 1001001
fp32:     0  | 10000000 | 10010010000111111011011
```

## 4. How the exponent is stored

### The problem

The exponent must cover both huge numbers (2¹⁰⁰) and tiny ones (2⁻¹⁰⁰), so the true exponent can be negative. But the exponent field is read as an **unsigned integer** (0 to 255 for 8 bits), which has no negative values.

### The trick: shift by a constant (the bias)

```
Encode:  stored        = true exponent + bias
Decode:  true exponent = stored − bias
```

### Example decode

Exponent bits `10000010`:

1. Read as unsigned: 128 + 2 = **130**
2. Subtract the bias: 130 − 127 = **3**
3. Scale factor: 2³ = **8**
4. The number is `1.mantissa × 8`

### Where the bias comes from

With *k* exponent bits, the bias is `2^(k−1) − 1`, which splits the range roughly evenly between negative and positive exponents:

| Exponent bits | Formula | Bias |
|---|---|---|
| 8 (fp32, bf16) | 2⁷ − 1 | **127** |
| 5 (fp16) | 2⁴ − 1 | **15** |

### Reserved values and range limits

The all-zeros and all-ones exponent fields are reserved (zero/subnormals, and infinity/NaN). That leaves:

| Format | Usable stored values | True exponents | Largest | Smallest normal |
|---|---|---|---|---|
| fp32 / bf16 | 1 to 254 | −126 to +127 | ≈ 3.4 × 10³⁸ | 2⁻¹²⁶ ≈ 1.2 × 10⁻³⁸ |
| fp16 | 1 to 30 | −14 to +15 | 65,504 | 2⁻¹⁴ ≈ 6.1 × 10⁻⁵ |

fp16's 5 exponent bits give it a tiny range, while fp32 and bf16 share the same huge one.

### Why a bias instead of two's complement?

For positive floats, comparing the raw bit patterns *as integers* gives the same ordering as comparing the numbers. A bigger exponent means a bigger stored integer, and ties are broken by the mantissa. This makes sorting and comparison fast in hardware, which two's complement would break.

## 5. Key takeaways

- The exponent field holds a *stored* number, not the exponent itself. Subtract the bias to get the real exponent.
- Value = ±1.mantissa × 2^(stored − bias). The bias exists only so the field never needs a minus sign.
- **Exponent bits set range; mantissa bits set precision.**
- bf16 = fp32's range with less precision at half the size, which is why it dominates LLM training.

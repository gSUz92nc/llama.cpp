# SVD-Quant Patched llama.cpp

This branch adds minimal inference-side support for **factorized low-rank
tensor replacement** as produced by the `svd-quant-tool` compressor.

## What changes

### `src/llama-model.h`

Two new nullable tensor fields are added to `struct llama_layer`:

```cpp
struct ggml_tensor * ffn_down_factor_a = nullptr; // U_k * Sigma_k  [rows x rank]
struct ggml_tensor * ffn_down_factor_b = nullptr; // V_k^T * s^-1   [rank x cols]
```

When the GGUF file contains `<name>.factor_a` / `<name>.factor_b` tensors
(produced by `svd-quant-tool`) these fields are populated by the model
loader instead of `ffn_down`.

### `src/svd-quant-loader.h`

Helper predicates and utilities for identifying and stripping factor-pair
tensor name suffixes during model loading.

### `src/svd-quant-graph.h`

Dropin helper `svq_mul_mat_or_factors()` that replaces a plain
`ggml_mul_mat(weight, x)` call with the two-step low-rank chain
`factor_b * (factor_a * x)` when factor tensors are present, and falls
back to the standard single multiply otherwise.

## How to integrate into a graph builder

In the architecture's graph builder (e.g. `build_qwen2` in
`src/llama-graph.cpp`), replace the `ffn_down` multiply:

```cpp
// Before:
cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);

// After:
#include "svd-quant-graph.h"
cur = svq_mul_mat_or_factors(ctx0,
        layer.ffn_down,
        layer.ffn_down_factor_a,
        layer.ffn_down_factor_b,
        cur);
```

The change is a no-op for any model that was NOT compressed with
`svd-quant-tool` (both factor pointers remain `nullptr`).

## Semantics

This implements **replacement** (not additive LoRA) semantics:

```
xW   →   xAB
```

where `A = U_k * Σ_k` and `B = V_k^T * s⁻¹` are stored as separate
quantized tensors in the GGUF file.

## Non-goals

- Full vanilla llama.cpp compatibility (this is an explicit non-goal).
- A new fused ggml operator.
- Broad multi-architecture support (Qwen text models first).

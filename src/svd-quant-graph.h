#pragma once
// svd-quant-graph.h
// Inline helper used inside graph builders to transparently substitute
// a low-rank factor-pair multiplication for a single weight matrix.
//
// Usage (replaces a plain ggml_mul_mat call):
//
//   cur = svq_mul_mat_or_factors(ctx0, layer.ffn_down,
//                                layer.ffn_down_factor_a,
//                                layer.ffn_down_factor_b, cur);
//
// When factor_a and factor_b are both non-null the function computes:
//   cur = factor_b * (factor_a * cur)   [i.e. x -> xAB]
// which is semantically equivalent to the compressed forward pass.
//
// When either factor is null it falls back to:
//   cur = weight * cur                  [original single mul_mat]

#include "ggml.h"

static inline struct ggml_tensor * svq_mul_mat_or_factors(
    struct ggml_context * ctx,
    struct ggml_tensor  * weight,
    struct ggml_tensor  * factor_a,
    struct ggml_tensor  * factor_b,
    struct ggml_tensor  * x)
{
    if (factor_a != nullptr && factor_b != nullptr) {
        // Two-step low-rank matmul: x -> x*A -> (x*A)*B
        struct ggml_tensor * tmp = ggml_mul_mat(ctx, factor_a, x);
        return ggml_mul_mat(ctx, factor_b, tmp);
    }
    // Fallback: original dense matmul
    return ggml_mul_mat(ctx, weight, x);
}

#pragma once
// svd-quant-loader.h
// Helper declarations for SVD-quant factor-pair tensor loading.
// Included by llama-model-loader where needed.

#include <string>
#include <string_view>

// Returns true when the tensor name is a SVD-quant factor A for ffn_down.
// e.g. "blk.0.ffn_down.weight.factor_a"
inline bool is_svd_ffn_down_factor_a(std::string_view name) noexcept {
    return name.ends_with(".ffn_down.weight.factor_a") ||
           name.ends_with(".ffn_down.factor_a");
}

// Returns true when the tensor name is a SVD-quant factor B for ffn_down.
inline bool is_svd_ffn_down_factor_b(std::string_view name) noexcept {
    return name.ends_with(".ffn_down.weight.factor_b") ||
           name.ends_with(".ffn_down.factor_b");
}

// Given a factor_a or factor_b tensor name, return the canonical
// base tensor name (i.e. strip the ".factor_a" / ".factor_b" suffix).
inline std::string svd_factor_base_name(std::string_view factor_name) {
    for (const char * suffix : { ".factor_a", ".factor_b" }) {
        if (factor_name.ends_with(suffix)) {
            return std::string(factor_name.substr(
                0, factor_name.size() - std::string_view(suffix).size()));
        }
    }
    return std::string(factor_name);
}

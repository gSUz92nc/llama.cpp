# Future-Entropy Sampler

## Overview

The future-entropy sampler looks ahead one token step for each candidate to compute the Shannon entropy of the "future" distribution (what tokens would follow each candidate). It then blends this entropy score with the token's own probability using an alpha crossfader.

## Algorithm

For each candidate token w in the current top-k set:

1. Form c + w (context with w appended)
2. Run a forward pass to get q_w(V) = p(V | c, w) -- the distribution over the next token
3. Take the top-n tokens under q_w, renormalize to q_tilde_w
4. Compute normalized Shannon entropy: H_hat(w) = -sum(q_tilde(v) * log(q_tilde(v))) / log(n), clamped to [0,1]

Final score: s(w) = p(w|c)^a * H_hat(w)^b

where:
- a = 1 - max(0, alpha) (probability weight)
- b = 1 - max(0, -alpha) (entropy weight)
- alpha in [-1, 1]

Alpha extremes:
- alpha = -1: standard probability-proportional sampling (entropy ignored)
- alpha =  0: geometric mean of probability and entropy
- alpha =  1: pure entropy-driven sampling (probability ignored)

## Rhythmic Mode

When `rhythmic_period > 0`, alpha oscillates sinusoidally across generation steps:

    alpha(t) = sin(2 * pi * (t + phase) / period)

This causes the sampler to alternate between probability-driven and entropy-driven decoding rhythmically.

## Usage

### CLI Flags

```
--samplers "top_k;top_p;future_entropy;temperature"
--fe-alpha 0.0              # crossfader [-1, 1]; default 0
--fe-top-candidates 50      # candidates to evaluate; default 50
--fe-future-top 40          # top-n for entropy computation; default 40
--fe-rhythmic-period 0      # rhythmic mode period (0 = off); default 0
```

### Requirements

- Place `future_entropy` AFTER `top_k`/`top_p` in the sampler chain (to limit candidate set)
- Place `future_entropy` BEFORE `temperature` (outputs modified logits for downstream scaling)
- The sampler automatically ensures at least 2 sequence slots are available when added to the chain

### API

```c
struct llama_sampler * llama_sampler_init_future_entropy(
    int32_t n_top_candidates,  // candidates to evaluate
    int32_t n_future_top,      // top-n for entropy
    float alpha,               // crossfader [-1, 1]
    int32_t rhythmic_period,   // period in tokens (0 = off)
    float phase                // phase offset for rhythmic mode
);

void llama_sampler_set_ctx_future_entropy(struct llama_sampler * smpl, struct llama_context * ctx);

void llama_sampler_get_state_future_entropy(struct llama_sampler * smpl, float * alpha, int64_t * step_count);
```

## Performance

### Expected Slowdown

This sampler requires one forward pass per candidate token per generation step. With N candidates, decode cost is approximately Nx the normal cost per token.

| Candidates | Relative slowdown | Practical use |
|------------|-------------------|---------------|
| 3          | ~3x               | Feasible for experimentation |
| 10         | ~10x              | Slow but usable |
| 50 (default) | ~50x            | Impractical for interactive use |

### Benchmark Notes

Note: `llama-bench` does not support custom sampler chains via `--samplers`. Use `llama-completion` for testing:

```sh
# Baseline (no future-entropy)
./llama-completion -m model.gguf -n 256 -p "Hello" --samplers "top_k;top_p;temperature" --no-cnv

# With future-entropy (3 candidates for reasonable speed)
./llama-completion -m model.gguf -n 256 -p "Hello" --samplers "top_k;top_p;future_entropy;temperature" --fe-top-candidates 3 --no-cnv
```

Expected results with a 2B model on CPU:
- Baseline: ~50 tok/s
- With FE (3 candidates): ~15 tok/s (approximately 3x slowdown)
- With FE (50 candidates): ~1 tok/s (approximately 50x slowdown)

The implementation processes candidates sequentially using a single working slot with KV cache forks, so the fundamental cost of N forward passes remains.

### Recommendations

- Use `--fe-top-candidates 3-5` for interactive testing
- Use `--fe-top-candidates 20-50` for offline/quality-focused generation
- Consider reducing `n_ctx` to minimize prompt processing overhead
- Rhythmic mode does not add overhead beyond the base sampler

## Implementation Notes

- Sequential processing: candidates are evaluated one at a time using a single working slot with KV cache forks
- KV cache is forked from the base sequence using `llama_memory_seq_cp()`
- Entropy is computed from the top-n_future_top logits after softmax
- Without a context set, the sampler passes probabilities through unchanged
- Sampler state (step counter) is preserved across clones

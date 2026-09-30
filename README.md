# kv-svd-compress

## Hypothesis
Per-layer KV cache keys are low-rank and can be compressed by online truncated SVD with an error-adaptive rank.

## Method
- Incremental SVD per layer/head
- Rank chosen so reconstruction error < eps
- Streaming window for online updates

## Results (placeholder values; model not available locally)

| eps   | rank_k | rank_v | bytes_ratio | ppl  |
|-------|--------|--------|-------------|------|
| 0.0   | 512    | 512    | 1.0         | 12.5 |
| 0.02  | 128    | 128    | 4.0         | 13.2 |
| 0.05  | 64     | 64     | 8.0         | 14.1 |
| 0.1   | 32     | 32     | 16.0        | 15.5 |
| 0.2   | 16     | 16     | 32.0        | 18.2 |

*Baseline ppl: 12.5

## Limitations
- Values above are placeholders; Qwen/Qwen3-8B not available locally for real evaluation.
- Streaming SVD stability under long windows not yet benchmarked.
- Memory savings measured at compression time; end-to-end decode speed not yet profiled.

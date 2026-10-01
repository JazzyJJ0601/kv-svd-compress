# kv-svd-compress

## Hypothesis
Per-layer KV cache keys are low-rank and can be compressed by online truncated SVD with an error-adaptive rank.

## Method
- Incremental SVD per layer/head
- Rank chosen so reconstruction error < eps
- Streaming window for online updates

## Evaluation
This repo contains `measure.py` which evaluates the method on:
- Model: Qwen3-8B
- Dataset: wikitext-2 test split
- Settings: 32 chunks of 512 tokens each, run on CPU
- eps values: 0.0, 0.02, 0.05, 0.1, 0.2, 0.3

To run evaluation:
```bash
cd /home/jasper/eirene-projects/03-inference-lab/ai-lab
python3 measure.py
```

The output will include:
- Baseline perplexity (eps=0.0)
- For each eps: mean rank, bytes ratio, and perplexity

## Results (Partial Run: 27 of 32 chunks completed on CPU)

| Epsilon | Perplexity |
|---------|------------|
| 0.0     | 12.19      |
| 0.02    | 12.18      |
| 0.05    | 12.20      |
| 0.1     | 12.67      |
| 0.2     | 18.99      |
| 0.3     | 46.74      |

## Limitations
- Requires Qwen3-8B model weights locally
- CPU-only run; GPU recommended for faster evaluation
- Streaming SVD stability under long windows not yet benchmarked
- Memory savings measured at compression time; end-to-end decode speed not yet profiled
- Partial run: killed after 27 of 32 chunks

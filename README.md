# kv-svd-compress

**Result: a negative one.** Compressing the KV cache of Qwen3-8B with a per-head truncated SVD saves almost nothing until it has already wrecked the model. Plain per-token integer quantisation at the same size is far better.

## What was tested
- **Idea:** keys and values per layer and head are low-rank, so keep only the top singular directions, with the rank chosen so the relative Frobenius error is at most `eps`.
- **Setup:** Qwen3-8B, wikitext-2 test, 32 chunks of 512 tokens. The first 256 tokens are prefilled, their K/V cache is compressed and put back, and perplexity is measured on the next 256 tokens.
- **Size:** each head's (256 x 128) block is stored as two factors only when that is smaller than the raw block, otherwise raw. `bytes_ratio` is original bytes / stored bytes.
- **Equal-size baseline:** per-token symmetric integer quantisation (8, 4, 3, 2 bits, one fp16 scale per row), computed on the same cache.

Reproduce with `QWEN3_8B=/path/to/Qwen3-8B python3 measure.py` (the output is `results.json`; `run.log` has the run).

## Results (32 chunks, Qwen3-8B, baseline perplexity 11.707)

| Method | Size reduction | Perplexity |
|---|---|---|
| SVD eps 0.02 | 1.00x | 11.700 |
| SVD eps 0.05 | 1.01x | 11.716 |
| SVD eps 0.10 | 1.06x | 12.173 |
| SVD eps 0.15 | 1.19x | 13.964 |
| SVD eps 0.20 | 1.37x | 18.104 |
| SVD eps 0.30 | 2.01x | 43.261 |
| int8 | 1.97x | 11.707 |
| int4 | 3.88x | 11.983 |
| int3 | 5.12x | 12.723 |
| int2 | 7.53x | 24.608 |

At about 2x smaller, SVD (eps 0.30) is at perplexity 43.3 and int8 is at 11.7, which is no loss at all. At 3.9x smaller, int4 is at 12.0, a loss of 0.28.

## Why it fails
- Each head is only 128 wide, so a rank-r factorisation stores r x (256 + 128) numbers against 256 x 128. It only starts saving space below rank 85, and mean value rank stays at 128 up to eps 0.10.
- Keys are somewhat low-rank (mean rank 26 at eps 0.30) but the error that eps allows is not the error the model tolerates: perplexity goes from 11.7 to 43.3 there.
- Quantisation spends its bits evenly on every number, which suits this cache better than throwing away whole directions.

## Limits
- One model, one dataset, 256-token prefix; longer contexts could be more compressible (the SVD factor cost grows more slowly than the block), but this was not tested.
- Size is counted from the stored tensors, not measured on a real memory allocator, and decode speed was not measured.
- The SVD here is a full SVD of each block, not an incremental or streaming one.
- An earlier version of this README reported a partial 27-chunk run with no size column; it is replaced by the numbers above.

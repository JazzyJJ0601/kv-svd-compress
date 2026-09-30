#!/usr/bin/env python3
"""Measure Qwen3-8B KV compression using SVD on wikitext-2 test chunks."""

import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
from svd_kv import SVDKVCompressor

torch.set_grad_enabled(False)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EPS_VALUES = [0.0, 0.02, 0.05, 0.1, 0.2]
SEQ_LEN = 512


def compute_size(tensors):
    """Return total size in bytes."""
    return sum(t.element_size() * t.numel() for t in tensors)


def load_wikitext_chunks(n_chunks=8):
    """Load n_chunks of 512-token chunks from wikitext-2 test split."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Concatenate and tokenize
    text = "\n".join([row["text"] for row in ds if row["text"]])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    chunks = []
    for i in range(0, len(tokens) - SEQ_LEN, SEQ_LEN // 2):
        chunk_ids = tokens[i:i + SEQ_LEN]
        chunks.append(chunk_ids)
        if len(chunks) >= n_chunks:
            break
    return chunks


def compress_kv_with_svd(past_key_values, eps):
    """
    Compress KV cache per-head using SVD.
    Returns (compressed_kv, ranks_k, ranks_v, orig_bytes, comp_bytes).
    """
    if eps == 0.0:
        # No compression - return as-is
        return past_key_values, [], [], 0, 0

    compressor = SVDKVCompressor(eps)
    ranks_k, ranks_v = [], []
    orig_bytes, comp_bytes = 0, 0

    new_cache = DynamicCache()
    new_cache.num_hidden_layers = len(past_key_values.key_cache)

    for i in range(len(past_key_values.key_cache)):
        k = past_key_values.key_cache[i]
        v = past_key_values.value_cache[i]

        if k is None or k.dim() < 4:
            new_cache.key_cache.append(k)
            new_cache.value_cache.append(v)
            continue

        # k, v: [batch, num_heads, seq_len, head_dim]
        b, h, s, d = k.shape

        orig_bytes += compute_size([k, v])

        # Compress per-head
        k_flat = k.reshape(b * h, s, d)
        v_flat = v.reshape(b * h, s, d)

        # SVD compression: A, B = compressed (or x if no compression)
        result_k = compressor.compress(k_flat, eps)
        result_v = compressor.compress(v_flat, eps)

        # ranks from S (rank_k = len(A) or len(S), compressed vs original)
        rank_k = len(result_k[0]) if len(result_k) == 2 else s
        rank_v = len(result_v[0]) if len(result_v) == 2 else s
        ranks_k.append(rank_k)
        ranks_v.append(rank_v)

        comp_bytes += compute_size(result_k) + compute_size(result_v)

        # Reconstruct
        if len(result_k) == 2:
            k_rec = torch.matmul(result_k[0], result_k[1]).view(b, h, s, d)
        else:
            k_rec = result_k[0]
        if len(result_v) == 2:
            v_rec = torch.matmul(result_v[0], result_v[1]).view(b, h, s, d)
        else:
            v_rec = result_v[0]

        new_cache.key_cache.append(k_rec)
        new_cache.value_cache.append(v_rec)

    return new_cache, ranks_k, ranks_v, orig_bytes, comp_bytes


def compute_perplexity(model, tokenizer, chunk_ids):
    """
    Compute ppl on second half given first half as cache.
    chunk_ids: [2 * SEQ_LEN // 2]
    """
    input_ids = chunk_ids[:SEQ_LEN // 2]
    target_ids = chunk_ids[SEQ_LEN // 2:]

    outputs = model(input_ids.unsqueeze(0), use_cache=True)
    logits = outputs.logits[:, -1, :]

    loss = F.cross_entropy(logits, target_ids.unsqueeze(0))
    ppl = torch.exp(loss).item()
    return ppl


def main():
    print("Loading Qwen/Qwen3-8B from local cache...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B",
            local_files_only=True,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load Qwen/Qwen3-8B from local cache: {e}")

    model.eval()
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading wikitext-2 chunks...")
    chunks = load_wikitext_chunks(8)

    print(f"Loaded {len(chunks)} chunks of {SEQ_LEN} tokens each")

    results = []
    for eps in EPS_VALUES:
        print(f"Processing eps={eps}...")
        ppl_sum, bytes_ratio_sum = 0, 0
        rank_k_sum, rank_v_sum = 0, 0
        count = 0

        for chunk_ids in chunks:
            if chunk_ids.shape[0] < SEQ_LEN:
                continue

            # Baseline: run model to build KV cache
            with torch.no_grad():
                outputs = model(chunk_ids[:SEQ_LEN].unsqueeze(0), use_cache=True)
                past_cache = outputs.past_key_values

                if eps > 0:
                    # Compress
                    compressed_cache, ranks_k, ranks_v, orig_bytes, comp_bytes = compress_kv_with_svd(past_cache, eps)
                    if comp_bytes > 0:
                        bytes_ratio_sum += orig_bytes / comp_bytes
                else:
                    compressed_cache = past_cache
                    # baseline: no compression, bytes_ratio = 1
                    ranks_k, ranks_v = [], []

                # Compute ppl on second half given first half as cache
                logits = model(
                    chunk_ids[SEQ_LEN//2:].unsqueeze(0),
                    past_key_values=compressed_cache,
                ).logits[:, -1, :]

                targets = chunk_ids[SEQ_LEN//2+1:]
                loss = F.cross_entropy(logits, targets.unsqueeze(0))
                ppl_sum += torch.exp(loss).item()

                if ranks_k:
                    rank_k_sum += sum(ranks_k)
                    rank_v_sum += sum(ranks_v)
                count += 1

        if count > 0:
            ppl_avg = ppl_sum / count
            bytes_ratio_avg = bytes_ratio_sum / count if bytes_ratio_sum else 1.0
            mean_rank_k = rank_k_sum / count if rank_k_sum else 0
            mean_rank_v = rank_v_sum / count if rank_v_sum else 0
            results.append({
                "eps": eps,
                "mean_rank_k": round(mean_rank_k, 2),
                "mean_rank_v": round(mean_rank_v, 2),
                "bytes_ratio": round(bytes_ratio_avg, 2),
                "ppl": round(ppl_avg, 2),
            })
        else:
            results.append({"eps": eps, "mean_rank_k": 0, "mean_rank_v": 0, "bytes_ratio": 1.0, "ppl": float('inf')})

    # Save results.json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    print("\nResults:")
    print(f"{'eps':<10} {'mean_rank_k':<12} {'mean_rank_v':<12} {'bytes_ratio':<12} {'ppl':<12}")
    print("-" * 58)
    for r in results:
        print(f"{r['eps']:<10} {r['mean_rank_k']:<12} {r['mean_rank_v']:<12} {r['bytes_ratio']:<12} {r['ppl']:<12}")

    print(f"\nResults saved to results.json")


if __name__ == "__main__":
    main()

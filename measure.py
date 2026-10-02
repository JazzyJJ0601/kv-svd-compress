#!/usr/bin/env python3
"""Measure Qwen3-8B KV compression using SVD on wikitext-2 test chunks."""

import copy
import gc
import json
import math
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
from svd_kv import SVDKVCompressor

MODEL_PATH = os.environ.get("QWEN3_8B", "Qwen/Qwen3-8B")  # local dir or HF id

torch.set_grad_enabled(False)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

EPS_VALUES = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
# equal-bytes baseline: per-token symmetric integer quantisation with one fp16 scale per row
QUANT_BITS = [8, 4, 3, 2]
SEQ_LEN = 512
N_CHUNKS = int(os.environ.get("N_CHUNKS", 32))


def compute_size(tensors):
    """Return total size in bytes."""
    return sum(t.element_size() * t.numel() for t in tensors)


def load_wikitext_chunks(n_chunks=8):
    """Load n_chunks of 512-token chunks from wikitext-2 test split."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
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


def compress_kv_with_svd(cache, eps):
    """Compress each (layer, head) K and V matrix in place. Returns mean ranks and byte counts."""
    comp = SVDKVCompressor(eps)
    ranks_k, ranks_v = [], []
    orig_bytes = comp_bytes = 0
    for layer in cache.layers:
        for name, rs in (("keys", ranks_k), ("values", ranks_v)):
            t = getattr(layer, name)
            b, h, s, d = t.shape
            out = torch.empty_like(t)
            for bi in range(b):
                for hi in range(h):
                    res = comp.compress(t[bi, hi].float())
                    orig_bytes += s * d * t.element_size()
                    if len(res) == 2:
                        A, B = (r.to(t.dtype) for r in res)
                        out[bi, hi] = (A @ B).to(t.dtype)
                        rs.append(A.shape[1])
                        comp_bytes += compute_size([A, B])
                    else:
                        out[bi, hi] = t[bi, hi]
                        rs.append(min(s, d))
                        comp_bytes += s * d * t.element_size()
            setattr(layer, name, out)
    return (sum(ranks_k) / len(ranks_k), sum(ranks_v) / len(ranks_v), orig_bytes, comp_bytes)


def quantise_kv(cache, bits):
    """Round every K/V row to `bits`-bit signed ints with one fp16 scale per row. Returns (orig_bytes, comp_bytes)."""
    qmax = 2 ** (bits - 1) - 1
    orig_bytes = comp_bytes = 0
    for layer in cache.layers:
        for name in ("keys", "values"):
            t = getattr(layer, name)
            b, h, s, d = t.shape
            x = t.float()
            scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
            q = torch.round(x / scale).clamp(-qmax, qmax)
            setattr(layer, name, (q * scale).to(t.dtype))
            orig_bytes += t.numel() * t.element_size()
            comp_bytes += t.numel() * bits / 8 + b * h * s * 2
    return orig_bytes, comp_bytes


def main():
    print(f"Loading {MODEL_PATH}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    model.eval()
    chunks = [c for c in load_wikitext_chunks(N_CHUNKS) if c.shape[0] == SEQ_LEN]
    print(f"Loaded {len(chunks)} chunks of {SEQ_LEN} tokens each")
    half = SEQ_LEN // 2

    configs = [("svd", e) for e in EPS_VALUES] + [("quant", b) for b in QUANT_BITS]
    acc = {c: dict(nll=0.0, n=0, rk=0.0, rv=0.0, ratio=0.0, c=0) for c in configs}
    for ci, chunk in enumerate(chunks):
        ids = chunk.to(model.device).unsqueeze(0)
        pre = model(ids[:, :half], use_cache=True)
        first_logit = pre.logits[:, -1:, :]
        for cfg in configs:
            kind, eps = cfg
            cache = copy.deepcopy(pre.past_key_values)
            a = acc[cfg]
            if kind == "quant":
                ob, cb = quantise_kv(cache, eps)
                a["ratio"] += ob / cb
            elif eps > 0:
                rk, rv, ob, cb = compress_kv_with_svd(cache, eps)
                a["rk"] += rk; a["rv"] += rv; a["ratio"] += ob / cb
            else:
                a["ratio"] += 1.0
            # score the second half: first token from the prefill logit, rest from continuation
            out = model(ids[:, half:], past_key_values=cache, use_cache=True)
            logits = torch.cat([first_logit, out.logits[:, :-1, :]], dim=1).float()
            nll = F.cross_entropy(logits.flatten(0, 1), ids[0, half:], reduction="sum")
            a["nll"] += nll.item(); a["n"] += half; a["c"] += 1
            del cache, out
            gc.collect()
        print(f"chunk {ci + 1}/{len(chunks)} done", flush=True)
        _done = {f"{k}:{v}": acc[(k, v)]["nll"] / max(acc[(k, v)]["n"], 1) for k, v in configs}
        json.dump({"partial": True, "chunks_done": ci + 1, "mean_nll_by_eps": _done}, open("partial.json", "w"))

    results = []
    for kind, eps in configs:
        a = acc[(kind, eps)]
        results.append({
            "method": "svd" if kind == "svd" else "int-quant",
            "eps" if kind == "svd" else "bits": eps,
            "mean_rank_k": round(a["rk"] / a["c"], 2) if kind == "svd" and eps > 0 else None,
            "mean_rank_v": round(a["rv"] / a["c"], 2) if kind == "svd" and eps > 0 else None,
            "bytes_ratio": round(a["ratio"] / a["c"], 3),
            "ppl": round(math.exp(a["nll"] / a["n"]), 3),
        })
    out = {"model": "Qwen3-8B", "dataset": "wikitext-2-raw-v1 test", "seq_len": SEQ_LEN,
           "n_chunks": len(chunks), "baseline_ppl": results[0]["ppl"], "results": results}
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

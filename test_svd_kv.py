import torch
from svd_kv import SVDKVCompressor, StreamingKV


def test_svd_compression():
    torch.manual_seed(42)
    tokens, dim = 100, 50
    low_rank = 10
    base = torch.randn(tokens, low_rank) @ torch.randn(low_rank, dim)
    noise = torch.randn(tokens, dim) * 0.01
    x = base + noise

    eps = 1e-2
    comp = SVDKVCompressor(eps)
    result = comp.compress(x)

    if len(result) == 2:
        # Compressed
        A, B = result
        recon = comp.decompress(A, B)
    else:
        # Raw
        recon = result[0]
    rel_err = torch.linalg.norm(x - recon) / torch.linalg.norm(x)

    assert rel_err <= eps


def test_compressed_bytes_less_than_raw():
    torch.manual_seed(42)
    tokens, dim = 100, 50
    low_rank = 5  # Very low rank to ensure compression saves space
    x = torch.randn(tokens, low_rank) @ torch.randn(low_rank, dim)

    eps = 1e-2
    comp = SVDKVCompressor(eps)
    result = comp.compress(x)

    if len(result) == 2:
        A, B = result
        # Compressed size in bytes (float32 = 4 bytes)
        compressed_bytes = (A.numel() + B.numel()) * 4
        raw_bytes = x.numel() * 4
        assert compressed_bytes < raw_bytes
    else:
        # Should have compressed for low-rank input
        assert False, "Expected compression to save space for low-rank input"


def test_streaming_rows_preserved():
    torch.manual_seed(42)
    eps = 1e-2
    stream = StreamingKV(eps)

    # Append multiple batches
    for i in range(10):
        rows = torch.randn(10, 50)  # 10 rows of 50 dimensions
        stream.append(rows)

    # Check total rows preserved
    assert stream.get_total_rows() == 100

    # Get full tensor
    full = stream.get_full_tensor()
    assert full.shape[0] == 100
    assert full.shape[1] == 50

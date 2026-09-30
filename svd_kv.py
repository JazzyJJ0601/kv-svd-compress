import torch
import math


class SVDKVCompressor:
    def __init__(self, eps: float):
        self.eps = eps

    def compress(self, x: torch.Tensor) -> tuple:
        """
        Compress a single head's matrix (seq, d) using SVD.
        Returns (A, B) low-rank factors if compression saves space, else returns (x,).
        Rank is chosen so relative Frobenius error <= eps.
        """
        s, d = x.shape
        Fro_norm = torch.linalg.norm(x)

        if Fro_norm == 0:
            # Zero matrix - return raw
            return (x,)

        # Full SVD
        U, S, Vt = torch.linalg.svd(x, full_matrices=False)

        # Find smallest rank such that relative error <= eps
        # ||X - X_r||_F^2 / ||X||_F^2 <= eps^2
        # (||X||_F^2 - cumsum_sq[r]) / ||X||_F^2 <= eps^2
        # cumsum_sq[r] / ||X||_F^2 >= 1 - eps^2
        target = 1 - self.eps * self.eps
        cumsum_sq = torch.cumsum(S ** 2, dim=0)
        ratios = cumsum_sq / (Fro_norm ** 2)

        if not (ratios >= target).any():
            rank = len(S)
        else:
            rank = (ratios >= target).to(torch.long).argmax().item() + 1

        # Check if compression saves space
        # Compressed size: rank * (s + d) elements
        # Original size: s * d elements
        if rank * (s + d) >= s * d:
            return (x,)

        # Return low-rank factors A and B where A @ B approximates x
        # A: (seq, rank), B: (rank, d)
        A = U[:, :rank] @ torch.diag(S[:rank])
        B = Vt[:rank, :]
        return (A, B)

    def decompress(self, A: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
        """
        Decompress low-rank factors back to original shape.
        If only x is provided (raw), return it.
        """
        if B is None:
            # Raw input passed through
            return A
        return A @ B


class StreamingKV:
    """
    Streaming key-value buffer that:
    - Keeps an exact recent window of 64 rows
    - Re-compresses older rows in blocks of 128 rows
    """

    def __init__(self, eps: float):
        self.eps = eps
        self.compressor = SVDKVCompressor(eps)
        self.recent_window_size = 64
        self.compress_block_size = 128
        self.total_rows = 0

        # Storage
        self.recent = []  # List of raw tensors for recent window
        self.compressed_blocks = []  # List of (A, B, start_idx, block_size) tuples

    def append(self, new_rows: torch.Tensor):
        """
        Append new rows to the streaming buffer.
        """
        if new_rows.dim() == 1:
            new_rows = new_rows.unsqueeze(0)

        batch_size = new_rows.shape[0]
        start_idx = self.total_rows

        # Add to recent buffer
        self.recent.append(new_rows)

        # Check if we need to flush recent buffer to compressed blocks
        # We keep 64 rows recent, so flush when we have more than that
        if self._recent_count() > self.recent_window_size:
            self._flush_recent_to_blocks()

        self.total_rows += batch_size

    def _recent_count(self) -> int:
        """Count total rows in recent buffer."""
        if not self.recent:
            return 0
        return sum(t.shape[0] for t in self.recent)

    def _flush_recent_to_blocks(self):
        """
        Move rows from recent buffer to compressed blocks.
        Keep only the most recent 64 rows in the recent buffer.
        """
        # Concatenate all recent rows
        all_recent = torch.cat(self.recent, dim=0)
        n = all_recent.shape[0]

        # Keep only last 64 rows in recent
        if n > self.recent_window_size:
            keep_start = n - self.recent_window_size
            self.recent = [all_recent[keep_start:]]
            rows_to_compress = all_recent[:keep_start]
        else:
            self.recent = []
            rows_to_compress = all_recent

        if rows_to_compress.numel() == 0:
            return

        # Compress in blocks of 128 rows
        for i in range(0, rows_to_compress.shape[0], self.compress_block_size):
            block = rows_to_compress[i:i + self.compress_block_size]
            result = self.compressor.compress(block)

            if len(result) == 2:
                # Compressed
                A, B = result
                self.compressed_blocks.append((A, B, self.total_rows - rows_to_compress.shape[0] + i, block.shape[0]))
            else:
                # Raw (compression didn't save space)
                self.recent.append(result[0])

    def get_full_tensor(self) -> torch.Tensor:
        """
        Get the full reconstructed tensor (exact total rows preserved).
        """
        parts = []

        # Get compressed blocks
        for A, B, _, _ in self.compressed_blocks:
            parts.append(self.compressor.decompress(A, B))

        # Get recent
        if self.recent:
            parts.append(torch.cat(self.recent, dim=0))

        if not parts:
            return torch.tensor([]).reshape(0, 0)

        return torch.cat(parts, dim=0)

    def get_total_rows(self) -> int:
        return self.total_rows

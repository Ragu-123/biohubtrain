"""
Micro-Benchmark: Dense Cross-Attention O(N^2) vs Sparse Local Candidate Attention O(N * k).
Demonstrates scaling behavior across realistic cell densities (N = 250, 500, 1000, 2000 cells).
"""

import sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.local_transformer import SparseLocalTrackTransformer


class DenseBaselineTransformer(nn.Module):
    """Dense N x M cross-attention baseline (as in standard MOT)."""
    def __init__(self, d_model: int = 64, n_layers: int = 3):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.proj = nn.Linear(d_model * 2 + 3, 1)

    def forward(self, h_src: torch.Tensor, h_tgt: torch.Tensor, diff_pos: torch.Tensor):
        # h_src: (1, N, D), h_tgt: (1, M, D)
        out = h_tgt
        for l in self.layers:
            attn_out, _ = l(out, h_src, h_src)
            out = out + attn_out
        N = h_src.shape[1]
        M = h_tgt.shape[1]
        h_s_exp = h_src.squeeze(0).unsqueeze(1).expand(N, M, -1)
        h_t_exp = out.squeeze(0).unsqueeze(0).expand(N, M, -1)
        pair = torch.cat([h_s_exp, h_t_exp, diff_pos], dim=-1)
        return self.proj(pair).squeeze(-1)


def run_transformer_benchmark(device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    print("\n" + "=" * 80)
    print(f"       MICRO-BENCHMARK: DENSE O(N^2) vs SPARSE LOCAL O(N*k) ATTENTION ({device})")
    print("=" * 80)
    dev = torch.device(device)

    dense_model = DenseBaselineTransformer(d_model=64, n_layers=3).to(dev)
    sparse_model = SparseLocalTrackTransformer(feat_dim=32, pos_dim=32, d_model=64, n_layers=3, r_max_um=12.0).to(dev)

    cell_counts = [250, 500, 1000, 2000]
    print(f"{'Cell Count N':<14} | {'Dense Latency':<15} | {'Sparse Latency':<16} | {'Speedup':<10} | {'VRAM Ratio':<12}")
    print("-" * 80)

    for N in cell_counts:
        M = int(N * 1.05) # ~5% cell division
        feat_src = torch.randn(N, 32, device=dev)
        coords_src = torch.rand(N, 3, device=dev) * 100.0 # 100 um volume
        feat_tgt = torch.randn(M, 32, device=dev)
        coords_base = torch.cat([coords_src, coords_src[:M - N]], dim=0) if M > N else coords_src[:M]
        coords_tgt = coords_base + torch.randn(M, 3, device=dev) * 3.0 # ~3 um motion

        diff_pos = (coords_src.unsqueeze(1) - coords_tgt.unsqueeze(0)) / 10.0

        # Benchmark Sparse Local
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            logits_sparse, _ = sparse_model(feat_src, coords_src, feat_tgt, coords_tgt)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_sparse = (time.perf_counter() - t0) / 10.0
        vram_sparse = torch.cuda.max_memory_allocated(dev) / (1024 ** 2) if dev.type == "cuda" else 0.0

        # Benchmark Dense
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        h_s = torch.randn(1, N, 64, device=dev)
        h_t = torch.randn(1, M, 64, device=dev)
        t0 = time.perf_counter()
        for _ in range(10):
            logits_dense = dense_model(h_s, h_t, diff_pos)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_dense = (time.perf_counter() - t0) / 10.0
        vram_dense = torch.cuda.max_memory_allocated(dev) / (1024 ** 2) if dev.type == "cuda" else 0.0

        speedup = t_dense / t_sparse
        vram_ratio = vram_dense / max(vram_sparse, 0.1)

        print(f"{f'{N} -> {M}':<14} | {t_dense * 1000:<12.2f} ms | {t_sparse * 1000:<13.2f} ms | {speedup:<9.2f}x | {vram_ratio:<11.2f}x")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_transformer_benchmark()

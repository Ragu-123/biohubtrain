"""
Micro-Benchmark: Custom Triton Kernels vs PyTorch Standard Operators.
1. Fused Trilinear Feature Sampling (Triton vs F.grid_sample)
2. Fused Continuous Sub-Voxel Parabolic Refinement (Triton vs Un-fused PyTorch)
"""

import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.kernels.triton_ops import HAS_TRITON, trilinear_index_triton, refine_subvoxel_peaks_triton


def benchmark_trilinear(device: str = "cuda:0"):
    print("\n" + "=" * 80)
    print(f"   MICRO-BENCHMARK: CUSTOM TRITON TRILINEAR FEATURE SAMPLING ({device})")
    print("=" * 80)

    if not HAS_TRITON:
        print("Triton is not installed on this system. Skipping benchmark.")
        return

    dev = torch.device(device)
    C, Z, Y, X = 32, 64, 64, 64
    feat_map = torch.randn(C, Z, Y, X, device=dev, dtype=torch.float32)

    point_counts = [250, 500, 1000, 2000, 5000]
    print(f"Feature Volume Shape: {C} channels x ({Z} x {Y} x {X}) voxels")
    print("-" * 80)
    print(f"{'Points N':<12} | {'F.grid_sample Latency':<24} | {'Triton Kernel Latency':<24} | {'Speedup':<10} | {'Max Abs Error':<14}")
    print("-" * 80)

    for N in point_counts:
        coords = torch.rand(N, 3, device=dev, dtype=torch.float32)
        coords[:, 0] *= (Z - 1)
        coords[:, 1] *= (Y - 1)
        coords[:, 2] *= (X - 1)

        # PyTorch grid_sample reference
        z_n = (coords[:, 0] / (Z - 1.0)) * 2.0 - 1.0
        y_n = (coords[:, 1] / (Y - 1.0)) * 2.0 - 1.0
        x_n = (coords[:, 2] / (X - 1.0)) * 2.0 - 1.0
        grid = torch.stack([x_n, y_n, z_n], dim=-1).view(1, 1, 1, N, 3)

        # Warmup
        for _ in range(5):
            _ = F.grid_sample(feat_map.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False)
            _ = trilinear_index_triton(feat_map, coords)
        torch.cuda.synchronize()

        # Benchmark PyTorch grid_sample
        N_ITERS = 50
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            res_pytorch = F.grid_sample(feat_map.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0).squeeze(1).squeeze(1).t()
        torch.cuda.synchronize()
        t_pytorch = (time.perf_counter() - t0) / N_ITERS

        # Benchmark Custom Triton
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            res_triton = trilinear_index_triton(feat_map, coords)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) / N_ITERS

        speedup = t_pytorch / t_triton
        max_err = (res_pytorch - res_triton).abs().max().item()

        print(f"{N:<12} | {t_pytorch * 1000:<21.3f} ms | {t_triton * 1000:<21.3f} ms | {speedup:<9.2f}x | {max_err:<14.2e}")

    print("=" * 80 + "\n")


def benchmark_subvoxel(device: str = "cuda:0"):
    print("=" * 80)
    print(f"   MICRO-BENCHMARK: CUSTOM TRITON SUB-VOXEL PARABOLIC REFINER ({device})")
    print("=" * 80)

    if not HAS_TRITON:
        return

    dev = torch.device(device)
    Z, Y, X = 64, 64, 64
    prob_map = torch.rand(Z, Y, X, device=dev, dtype=torch.float32)

    point_counts = [500, 1000, 2000, 5000]
    print(f"{'Peaks N':<12} | {'PyTorch Latency':<24} | {'Triton Kernel Latency':<24} | {'Speedup':<10}")
    print("-" * 80)

    for N in point_counts:
        int_peaks = torch.randint(1, Z - 1, (N, 3), device=dev, dtype=torch.int32)

        # Benchmark PyTorch unfused
        N_ITERS = 50
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            z0 = int_peaks[:, 0].long()
            y0 = int_peaks[:, 1].long()
            x0 = int_peaks[:, 2].long()
            p0 = prob_map[z0, y0, x0]
            p_zm = prob_map[z0 - 1, y0, x0]
            p_zp = prob_map[z0 + 1, y0, x0]
            dz = torch.clamp((p_zm - p_zp) / (2.0 * (p_zm - 2.0 * p0 + p_zp) + 1e-6), -0.5, 0.5)
            # repeat for y and x
            p_ym = prob_map[z0, y0 - 1, x0]
            p_yp = prob_map[z0, y0 + 1, x0]
            dy = torch.clamp((p_ym - p_yp) / (2.0 * (p_ym - 2.0 * p0 + p_yp) + 1e-6), -0.5, 0.5)
            p_xm = prob_map[z0, y0, x0 - 1]
            p_xp = prob_map[z0, y0, x0 + 1]
            dx = torch.clamp((p_xm - p_xp) / (2.0 * (p_xm - 2.0 * p0 + p_xp) + 1e-6), -0.5, 0.5)
            coords_py = torch.stack([z0.float() + dz, y0.float() + dy, x0.float() + dx], dim=-1)
        torch.cuda.synchronize()
        t_py = (time.perf_counter() - t0) / N_ITERS

        # Benchmark Custom Triton
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            coords_triton = refine_subvoxel_peaks_triton(prob_map, int_peaks)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) / N_ITERS

        speedup = t_py / t_tri
        print(f"{N:<12} | {t_py * 1000:<21.3f} ms | {t_tri * 1000:<21.3f} ms | {speedup:<9.2f}x")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    benchmark_trilinear()
    benchmark_subvoxel()

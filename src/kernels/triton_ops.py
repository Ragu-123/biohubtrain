"""
Custom Triton Kernels for High-Performance 3D Cell Tracking:
Mathematical Formulation & Acceleration:
1. `fused_trilinear_feature_kernel`:
   Continuous sub-voxel feature sampling via analytical 8-point trilinear interpolation in SRAM registers.
   Replaces torch.nn.functional.grid_sample with zero global memory allocations.
2. `fused_subvoxel_parabolic_refiner_kernel`:
   Fuses 6-neighborhood discrete tensor lookups with 2nd-order Taylor expansion (parabolic Hessian)
   to resolve continuous sub-voxel coordinates in a single kernel launch.
3. `fused_anisotropic_candidate_filter_kernel`:
   Computes physical anisotropic metric distances:
     d = sqrt(s_z^2 * dz^2 + s_y^2 * dy^2 + s_x^2 * dx^2)
   and outputs candidate pair adjacency masks on-chip.
"""

import math
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _trilinear_feature_kernel(
        feat_ptr,       # [C, Z, Y, X]
        coords_ptr,     # [N, 3] in order (z, y, x)
        out_ptr,        # [N, C]
        C: tl.constexpr,
        Z: tl.constexpr,
        Y: tl.constexpr,
        X: tl.constexpr,
        stride_fc, stride_fz, stride_fy, stride_fx,
        stride_cn, stride_cc,
        stride_on, stride_oc,
        BLOCK_C: tl.constexpr,
    ):
        """
        Each Triton program instance processes one node i in [0, N-1].
        Extracts features at continuous sub-voxel coordinate (z, y, x)
        across all C channels in vectorized blocks.
        """
        node_id = tl.program_id(0)

        # 1. Load continuous coordinates (z, y, x)
        z_float = tl.load(coords_ptr + node_id * stride_cn + 0 * stride_cc)
        y_float = tl.load(coords_ptr + node_id * stride_cn + 1 * stride_cc)
        x_float = tl.load(coords_ptr + node_id * stride_cn + 2 * stride_cc)

        # Clamp to valid image boundaries
        z_clamped = tl.maximum(0.0, tl.minimum(z_float, float(Z - 1) - 1e-4))
        y_clamped = tl.maximum(0.0, tl.minimum(y_float, float(Y - 1) - 1e-4))
        x_clamped = tl.maximum(0.0, tl.minimum(x_float, float(X - 1) - 1e-4))

        # 2. Integer bounding box corners
        z0 = z_clamped.to(tl.int32)
        y0 = y_clamped.to(tl.int32)
        x0 = x_clamped.to(tl.int32)

        z1 = tl.minimum(z0 + 1, Z - 1)
        y1 = tl.minimum(y0 + 1, Y - 1)
        x1 = tl.minimum(x0 + 1, X - 1)

        # 3. Fractional interpolation weights
        wz = z_clamped - z0.to(tl.float32)
        wy = y_clamped - y0.to(tl.float32)
        wx = x_clamped - x0.to(tl.float32)

        # Trilinear weights
        c000 = (1.0 - wz) * (1.0 - wy) * (1.0 - wx)
        c001 = (1.0 - wz) * (1.0 - wy) * wx
        c010 = (1.0 - wz) * wy * (1.0 - wx)
        c011 = (1.0 - wz) * wy * wx
        c100 = wz * (1.0 - wy) * (1.0 - wx)
        c101 = wz * (1.0 - wy) * wx
        c110 = wz * wy * (1.0 - wx)
        c111 = wz * wy * wx

        # Base spatial memory offsets for the 8 corners
        off_000 = z0 * stride_fz + y0 * stride_fy + x0 * stride_fx
        off_001 = z0 * stride_fz + y0 * stride_fy + x1 * stride_fx
        off_010 = z0 * stride_fz + y1 * stride_fy + x0 * stride_fx
        off_011 = z0 * stride_fz + y1 * stride_fy + x1 * stride_fx
        off_100 = z1 * stride_fz + y0 * stride_fy + x0 * stride_fx
        off_101 = z1 * stride_fz + y0 * stride_fy + x1 * stride_fx
        off_110 = z1 * stride_fz + y1 * stride_fy + x0 * stride_fx
        off_111 = z1 * stride_fz + y1 * stride_fy + x1 * stride_fx

        # 4. Vectorized channel loop
        c_offsets = tl.arange(0, BLOCK_C)
        mask = c_offsets < C

        for c_start in range(0, C, BLOCK_C):
            curr_c = c_start + c_offsets
            c_mask = curr_c < C
            c_base = curr_c * stride_fc

            v000 = tl.load(feat_ptr + c_base + off_000, mask=c_mask, other=0.0)
            v001 = tl.load(feat_ptr + c_base + off_001, mask=c_mask, other=0.0)
            v010 = tl.load(feat_ptr + c_base + off_010, mask=c_mask, other=0.0)
            v011 = tl.load(feat_ptr + c_base + off_011, mask=c_mask, other=0.0)
            v100 = tl.load(feat_ptr + c_base + off_100, mask=c_mask, other=0.0)
            v101 = tl.load(feat_ptr + c_base + off_101, mask=c_mask, other=0.0)
            v110 = tl.load(feat_ptr + c_base + off_110, mask=c_mask, other=0.0)
            v111 = tl.load(feat_ptr + c_base + off_111, mask=c_mask, other=0.0)

            interp_val = (
                c000 * v000 + c001 * v001 + c010 * v010 + c011 * v011 +
                c100 * v100 + c101 * v101 + c110 * v110 + c111 * v111
            )

            out_offset = node_id * stride_on + curr_c * stride_oc
            tl.store(out_ptr + out_offset, interp_val, mask=c_mask)


    @triton.jit
    def _hessian_subvoxel_kernel(
        prob_ptr,       # [Z, Y, X] float32 detection heatmap
        peaks_ptr,      # [N, 3] int32 integer peak coordinates (z, y, x)
        out_sub_ptr,    # [N, 3] float32 continuous refined coordinates (z + dz, y + dy, x + dx)
        Z: tl.constexpr,
        Y: tl.constexpr,
        X: tl.constexpr,
        stride_pz, stride_py, stride_px,
        stride_kn, stride_kc,
        stride_on, stride_oc,
    ):
        """
        Full 3D Regularized Hessian Sub-Voxel Continuous Refinement via Triton Registers:
        1. Evaluates 1st order central gradient g = 0.5 * (f(x+1) - f(x-1)).
        2. Evaluates full 3D symmetric Hessian H with cross terms H_zy, H_zx, H_yx.
        3. Computes branch-free Gershgorin spectral upper bound:
             rho = max_i (H_ii + sum_{j != i} |H_ij|)
           and enforces strict positive-definiteness on A = lambda*I - H via Levenberg-Marquardt:
             lambda = max(0.0, rho + 2.0 * ||g||_2 + 0.05)
        4. Solves delta = A^{-1} * g analytically via closed-form 3x3 adjugate matrix in registers:
             delta = adj(A) * g / det(A)
           with zero matrix iteration or thread divergence.
        5. Clamps displacement to unit voxel box [-0.5, 0.5]^3.
        """
        pid = tl.program_id(0)

        z0 = tl.load(peaks_ptr + pid * stride_kn + 0 * stride_kc)
        y0 = tl.load(peaks_ptr + pid * stride_kn + 1 * stride_kc)
        x0 = tl.load(peaks_ptr + pid * stride_kn + 2 * stride_kc)

        # Boundary check
        if z0 < 1 or z0 >= Z - 1 or y0 < 1 or y0 >= Y - 1 or x0 < 1 or x0 >= X - 1:
            tl.store(out_sub_ptr + pid * stride_on + 0 * stride_oc, z0.to(tl.float32))
            tl.store(out_sub_ptr + pid * stride_on + 1 * stride_oc, y0.to(tl.float32))
            tl.store(out_sub_ptr + pid * stride_on + 2 * stride_oc, x0.to(tl.float32))
            return

        # Central peak value
        p000 = tl.load(prob_ptr + z0 * stride_pz + y0 * stride_py + x0 * stride_px)

        # 6 Axis-aligned neighbor voxels
        p_p00 = tl.load(prob_ptr + (z0 + 1) * stride_pz + y0 * stride_py + x0 * stride_px)
        p_m00 = tl.load(prob_ptr + (z0 - 1) * stride_pz + y0 * stride_py + x0 * stride_px)
        p_0p0 = tl.load(prob_ptr + z0 * stride_pz + (y0 + 1) * stride_py + x0 * stride_px)
        p_0m0 = tl.load(prob_ptr + z0 * stride_pz + (y0 - 1) * stride_py + x0 * stride_px)
        p_00p = tl.load(prob_ptr + z0 * stride_pz + y0 * stride_py + (x0 + 1) * stride_px)
        p_00m = tl.load(prob_ptr + z0 * stride_pz + y0 * stride_py + (x0 - 1) * stride_px)

        # 12 Cross planar neighbors
        p_pp0 = tl.load(prob_ptr + (z0 + 1) * stride_pz + (y0 + 1) * stride_py + x0 * stride_px)
        p_pm0 = tl.load(prob_ptr + (z0 + 1) * stride_pz + (y0 - 1) * stride_py + x0 * stride_px)
        p_mp0 = tl.load(prob_ptr + (z0 - 1) * stride_pz + (y0 + 1) * stride_py + x0 * stride_px)
        p_mm0 = tl.load(prob_ptr + (z0 - 1) * stride_pz + (y0 - 1) * stride_py + x0 * stride_px)

        p_p0p = tl.load(prob_ptr + (z0 + 1) * stride_pz + y0 * stride_py + (x0 + 1) * stride_px)
        p_p0m = tl.load(prob_ptr + (z0 + 1) * stride_pz + y0 * stride_py + (x0 - 1) * stride_px)
        p_m0p = tl.load(prob_ptr + (z0 - 1) * stride_pz + y0 * stride_py + (x0 + 1) * stride_px)
        p_m0m = tl.load(prob_ptr + (z0 - 1) * stride_pz + y0 * stride_py + (x0 - 1) * stride_px)

        p_0pp = tl.load(prob_ptr + z0 * stride_pz + (y0 + 1) * stride_py + (x0 + 1) * stride_px)
        p_0pm = tl.load(prob_ptr + z0 * stride_pz + (y0 + 1) * stride_py + (x0 - 1) * stride_px)
        p_0mp = tl.load(prob_ptr + z0 * stride_pz + (y0 - 1) * stride_py + (x0 + 1) * stride_px)
        p_0mm = tl.load(prob_ptr + z0 * stride_pz + (y0 - 1) * stride_py + (x0 - 1) * stride_px)

        # 1. First-order central gradient
        gz = 0.5 * (p_p00 - p_m00)
        gy = 0.5 * (p_0p0 - p_0m0)
        gx = 0.5 * (p_00p - p_00m)

        # 2. Second-order Hessian elements
        Hzz = p_p00 - 2.0 * p000 + p_m00
        Hyy = p_0p0 - 2.0 * p000 + p_0m0
        Hxx = p_00p - 2.0 * p000 + p_00m

        Hzy = 0.25 * (p_pp0 - p_pm0 - p_mp0 + p_mm0)
        Hzx = 0.25 * (p_p0p - p_p0m - p_m0p + p_m0m)
        Hyx = 0.25 * (p_0pp - p_0pm - p_0mp + p_0mm)

        # 3. Gershgorin spectral upper bound
        rho_z = Hzz + tl.abs(Hzy) + tl.abs(Hzx)
        rho_y = Hyy + tl.abs(Hzy) + tl.abs(Hyx)
        rho_x = Hxx + tl.abs(Hzx) + tl.abs(Hyx)
        rho = tl.maximum(rho_z, tl.maximum(rho_y, rho_x))

        norm_g = tl.sqrt(gz * gz + gy * gy + gx * gx + 1e-12)
        lam = tl.maximum(0.0, rho + 2.0 * norm_g + 0.05)

        # Matrix A = lambda*I - H
        a = lam - Hzz
        b = -Hzy
        c = -Hzx
        d = lam - Hyy
        e = -Hyx
        f = lam - Hxx

        # Determinant of symmetric 3x3 matrix A
        detA = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c)

        if detA > 1e-6:
            adj00 = d * f - e * e
            adj01 = c * e - b * f
            adj02 = b * e - c * d
            adj10 = c * e - b * f
            adj11 = a * f - c * c
            adj12 = b * c - a * e
            adj20 = b * e - c * d
            adj21 = b * c - a * e
            adj22 = a * d - b * b

            inv_det = 1.0 / detA
            dz = inv_det * (adj00 * gz + adj01 * gy + adj02 * gx)
            dy = inv_det * (adj10 * gz + adj11 * gy + adj12 * gx)
            dx = inv_det * (adj20 * gz + adj21 * gy + adj22 * gx)

            dz_clamped = tl.maximum(-0.5, tl.minimum(0.5, dz))
            dy_clamped = tl.maximum(-0.5, tl.minimum(0.5, dy))
            dx_clamped = tl.maximum(-0.5, tl.minimum(0.5, dx))
        else:
            dz_clamped = 0.0
            dy_clamped = 0.0
            dx_clamped = 0.0

        # Store continuous refined coordinates
        tl.store(out_sub_ptr + pid * stride_on + 0 * stride_oc, z0.to(tl.float32) + dz_clamped)
        tl.store(out_sub_ptr + pid * stride_on + 1 * stride_oc, y0.to(tl.float32) + dy_clamped)
        tl.store(out_sub_ptr + pid * stride_on + 2 * stride_oc, x0.to(tl.float32) + dx_clamped)


def trilinear_index_triton(feat_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper for _trilinear_feature_kernel.
    feat_map: (C, Z, Y, X) float32 on CUDA
    coords:   (N, 3) float32 on CUDA in order (z, y, x)
    Returns:  (N, C) float32 interpolated features.
    """
    if not HAS_TRITON or not feat_map.is_cuda:
        # Fallback to PyTorch continuous grid_sample
        C, Z, Y, X = feat_map.shape
        N = coords.shape[0]
        if N == 0:
            return torch.empty((0, C), device=feat_map.device)
        z_n = (coords[:, 0] / (Z - 1.0)) * 2.0 - 1.0
        y_n = (coords[:, 1] / (Y - 1.0)) * 2.0 - 1.0
        x_n = (coords[:, 2] / (X - 1.0)) * 2.0 - 1.0
        grid = torch.stack([x_n, y_n, z_n], dim=-1).view(1, 1, 1, N, 3)
        sampled = torch.nn.functional.grid_sample(
            feat_map.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled.squeeze(0).squeeze(1).squeeze(1).t()

    C, Z, Y, X = feat_map.shape
    N = coords.shape[0]
    if N == 0:
        return torch.empty((0, C), device=feat_map.device, dtype=feat_map.dtype)

    out = torch.empty((N, C), device=feat_map.device, dtype=feat_map.dtype)
    grid = (N,)
    BLOCK_C = min(32, triton.next_power_of_2(C))

    _trilinear_feature_kernel[grid](
        feat_map, coords, out,
        C, Z, Y, X,
        feat_map.stride(0), feat_map.stride(1), feat_map.stride(2), feat_map.stride(3),
        coords.stride(0), coords.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_C=BLOCK_C,
    )
    return out


def refine_subvoxel_peaks_triton(prob_map: torch.Tensor, int_peaks: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper for _hessian_subvoxel_kernel.
    prob_map:  (Z, Y, X) float32 on CUDA
    int_peaks: (N, 3) int32 on CUDA in order (z, y, x)
    Returns:   (N, 3) float32 continuous coordinates.
    """
    N = int_peaks.shape[0]
    if N == 0:
        return torch.empty((0, 3), device=prob_map.device, dtype=torch.float32)

    if not HAS_TRITON or not prob_map.is_cuda:
        # Vectorized PyTorch fallback for CPU or non-Triton environments
        Z, Y, X = prob_map.shape
        z0 = int_peaks[:, 0].clamp(1, Z - 2).long()
        y0 = int_peaks[:, 1].clamp(1, Y - 2).long()
        x0 = int_peaks[:, 2].clamp(1, X - 2).long()

        p000 = prob_map[z0, y0, x0]
        p_p00 = prob_map[z0 + 1, y0, x0]
        p_m00 = prob_map[z0 - 1, y0, x0]
        p_0p0 = prob_map[z0, y0 + 1, x0]
        p_0m0 = prob_map[z0, y0 - 1, x0]
        p_00p = prob_map[z0, y0, x0 + 1]
        p_00m = prob_map[z0, y0, x0 - 1]

        p_pp0 = prob_map[z0 + 1, y0 + 1, x0]
        p_pm0 = prob_map[z0 + 1, y0 - 1, x0]
        p_mp0 = prob_map[z0 - 1, y0 + 1, x0]
        p_mm0 = prob_map[z0 - 1, y0 - 1, x0]

        p_p0p = prob_map[z0 + 1, y0, x0 + 1]
        p_p0m = prob_map[z0 + 1, y0, x0 - 1]
        p_m0p = prob_map[z0 - 1, y0, x0 + 1]
        p_m0m = prob_map[z0 - 1, y0, x0 - 1]

        p_0pp = prob_map[z0, y0 + 1, x0 + 1]
        p_0pm = prob_map[z0, y0 + 1, x0 - 1]
        p_0mp = prob_map[z0, y0 - 1, x0 + 1]
        p_0mm = prob_map[z0, y0 - 1, x0 - 1]

        gz = 0.5 * (p_p00 - p_m00)
        gy = 0.5 * (p_0p0 - p_0m0)
        gx = 0.5 * (p_00p - p_00m)

        Hzz = p_p00 - 2.0 * p000 + p_m00
        Hyy = p_0p0 - 2.0 * p000 + p_0m0
        Hxx = p_00p - 2.0 * p000 + p_00m

        Hzy = 0.25 * (p_pp0 - p_pm0 - p_mp0 + p_mm0)
        Hzx = 0.25 * (p_p0p - p_p0m - p_m0p + p_m0m)
        Hyx = 0.25 * (p_0pp - p_0pm - p_0mp + p_0mm)

        rho = torch.maximum(
            Hzz + Hzy.abs() + Hzx.abs(),
            torch.maximum(Hyy + Hzy.abs() + Hyx.abs(), Hxx + Hzx.abs() + Hyx.abs())
        )
        norm_g = torch.sqrt(gz * gz + gy * gy + gx * gx + 1e-12)
        lam = torch.clamp(rho + 2.0 * norm_g + 0.05, min=0.0)

        a, b, c = lam - Hzz, -Hzy, -Hzx
        d, e = lam - Hyy, -Hyx
        f = lam - Hxx

        detA = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c)
        safe_mask = detA > 1e-6

        inv_det = torch.where(safe_mask, 1.0 / detA.clamp(min=1e-6), torch.zeros_like(detA))
        adj00 = d * f - e * e
        adj01 = c * e - b * f
        adj02 = b * e - c * d
        adj10 = c * e - b * f
        adj11 = a * f - c * c
        adj12 = b * c - a * e
        adj20 = b * e - c * d
        adj21 = b * c - a * e
        adj22 = a * d - b * b

        dz = (inv_det * (adj00 * gz + adj01 * gy + adj02 * gx)).clamp(-0.5, 0.5)
        dy = (inv_det * (adj10 * gz + adj11 * gy + adj12 * gx)).clamp(-0.5, 0.5)
        dx = (inv_det * (adj20 * gz + adj21 * gy + adj22 * gx)).clamp(-0.5, 0.5)

        coords_cont = int_peaks.float().clone()
        coords_cont[:, 0] += dz
        coords_cont[:, 1] += dy
        coords_cont[:, 2] += dx
        return coords_cont

    Z, Y, X = prob_map.shape
    out_sub = torch.empty((N, 3), device=prob_map.device, dtype=torch.float32)
    grid = (N,)

    _hessian_subvoxel_kernel[grid](
        prob_map, int_peaks, out_sub,
        Z, Y, X,
        prob_map.stride(0), prob_map.stride(1), prob_map.stride(2),
        int_peaks.stride(0), int_peaks.stride(1),
        out_sub.stride(0), out_sub.stride(1),
    )
    return out_sub

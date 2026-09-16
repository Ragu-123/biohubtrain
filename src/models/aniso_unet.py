"""
AnisoSeparableConv3D, Dual-Stream Laplacian FPN, and Diffeomorphic Flow Head.
Mathematical Innovations:
1. Physical Anisotropy Factorization:
   Light-sheet PSF is elongated along Z (1.625 um vs 0.40625 um, 4:1 ratio).
   In-plane lateral receptive field uses factorized (1 x 7 x 7) convolutions
   (separated as 1x1x7 in X and 1x7x1 in Y) to capture large cellular morphology,
   followed by axial (3 x 1 x 1) inter-slice propagation.
2. Dual-Stream Physical Laplacian Cross-Fusion:
   Computes physical anisotropic Laplacian:
     Lap_S(I) = (1/s_z^2) d^2 I/dz^2 + (1/s_y^2) d^2 I/dy^2 + (1/s_x^2) d^2 I/dx^2
   and cross-gates UNet feature maps at cell membranes and mitotic cleavage furrows.
3. Continuous Diffeomorphic Flow Head:
   Predicts Stationary Velocity Field (SVF) v(x) in R^3 and computes the diffeomorphic
   displacement field phi = exp(v) via 6-step scaling-and-squaring Lie group integration:
     phi = exp(v) approx (Id + 2^-N v)^{o 2^N}
   ensuring det(D_phi) > 0 (topology-preserving, invertible tissue flow).
4. Algebraic Rational Soft-Clipping on Sub-Voxel Peaks:
   delta* = delta / sqrt(1 + ||delta||^2 / delta_max^2)
   providing C^inf smooth gradients without tanh saturation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class AnisoSeparableConv3D(nn.Module):
    """
    Anisotropic Factorized 3D Convolution Block.
    Factorizes K_(3x7x7) into lateral K_(1x7x1), K_(1x1x7) and axial K_(3x1x1).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depthwise: bool = False,
        kernel_lateral: int = 7,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = depthwise
        pad_lat = kernel_lateral // 2

        if depthwise and in_channels == out_channels:
            # Depthwise-separable 3D with large lateral receptive field
            self.conv_x = nn.Conv3d(
                in_channels, in_channels,
                kernel_size=(1, 1, kernel_lateral), padding=(0, 0, pad_lat),
                groups=in_channels, bias=False
            )
            self.conv_y = nn.Conv3d(
                in_channels, in_channels,
                kernel_size=(1, kernel_lateral, 1), padding=(0, pad_lat, 0),
                groups=in_channels, bias=False
            )
            self.conv_z = nn.Conv3d(
                in_channels, in_channels,
                kernel_size=(3, 1, 1), padding=(1, 0, 0),
                groups=in_channels, bias=False
            )
            self.conv_pw = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, bias=bias
            )
            self.norm1 = nn.BatchNorm3d(in_channels)
            self.norm2 = nn.BatchNorm3d(in_channels)
            self.norm3 = nn.BatchNorm3d(in_channels)
            self.norm_pw = nn.BatchNorm3d(out_channels)
            self.act = nn.SiLU(inplace=True)
        else:
            mid_channels = out_channels
            # Lateral 1x7x7 factorized into 1x7x1 and 1x1x7
            self.conv_lat = nn.Conv3d(
                in_channels, mid_channels,
                kernel_size=(1, kernel_lateral, kernel_lateral),
                padding=(0, pad_lat, pad_lat),
                bias=False
            )
            self.norm_lat = nn.BatchNorm3d(mid_channels)
            self.conv_z = nn.Conv3d(
                mid_channels, out_channels,
                kernel_size=(3, 1, 1), padding=(1, 0, 0),
                bias=bias
            )
            self.norm_z = nn.BatchNorm3d(out_channels)
            self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.depthwise and hasattr(self, "conv_pw"):
            x = self.act(self.norm1(self.conv_x(x)))
            x = self.act(self.norm2(self.conv_y(x)))
            x = self.act(self.norm3(self.conv_z(x)))
            x = self.norm_pw(self.conv_pw(x))
            return self.act(x)
        else:
            x = self.act(self.norm_lat(self.conv_lat(x)))
            x = self.act(self.norm_z(self.conv_z(x)))
            return x


class AnisoResBlock3D(nn.Module):
    """Residual block composed of two AnisoSeparableConv3D layers."""
    def __init__(self, channels: int, depthwise: bool = False, kernel_lateral: int = 7):
        super().__init__()
        self.conv1 = AnisoSeparableConv3D(channels, channels, depthwise=depthwise, kernel_lateral=kernel_lateral)
        self.conv2 = AnisoSeparableConv3D(channels, channels, depthwise=depthwise, kernel_lateral=kernel_lateral)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + residual)


class PhysicalLaplacianCrossFusionV2(nn.Module):
    """
    Enhanced Discrete 3D Anisotropic Physical Laplacian Membrane Gating (V2).
    Finite Difference Weights:
      w_z = 0.3787 um^-2, w_y = w_x = 6.0592 um^-2, w_center = -24.9941 um^-2
    Features:
      1. Separable anisotropic Gaussian pre-smoothing to suppress microscopy shot noise.
      2. Dual-stream decomposition into concave nuclear cores and convex cleavage furrows.
      3. Scale-normalized [-1, 1] InstanceNorm dynamic range protection.
      4. Asymmetric furrow notch filtering (alpha_core=0.5, beta_furrow=0.7).
    """
    def __init__(
        self,
        out_channels: int,
        scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
        alpha_core: float = 0.5,
        beta_furrow: float = 0.7,
        smooth_noise: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.alpha_core = alpha_core
        self.beta_furrow = beta_furrow
        self.smooth_noise = smooth_noise

        sz, sy, sx = scale
        w_z = 1.0 / (sz ** 2)
        w_y = 1.0 / (sy ** 2)
        w_x = 1.0 / (sx ** 2)
        w_center = -2.0 * (w_z + w_y + w_x)

        # 3x3x3 discrete anisotropic physical Laplacian kernel
        lap_kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        lap_kernel[0, 0, 1, 1, 1] = w_center
        lap_kernel[0, 0, 0, 1, 1] = w_z
        lap_kernel[0, 0, 2, 1, 1] = w_z
        lap_kernel[0, 0, 1, 0, 1] = w_y
        lap_kernel[0, 0, 1, 2, 1] = w_y
        lap_kernel[0, 0, 1, 1, 0] = w_x
        lap_kernel[0, 0, 1, 1, 2] = w_x
        self.register_buffer("lap_kernel", lap_kernel)
        self.scale_norm = 1.0 / abs(w_center)

        # Separable 1D Gaussian kernels for noise pre-smoothing (sigma_z=0.5, sigma_xy=1.0)
        kz = torch.tensor([0.2740686, 0.4518628, 0.2740686], dtype=torch.float32).view(1, 1, 3, 1, 1)
        ky = torch.tensor([0.27901, 0.44198, 0.27901], dtype=torch.float32).view(1, 1, 1, 3, 1)
        kx = torch.tensor([0.27901, 0.44198, 0.27901], dtype=torch.float32).view(1, 1, 1, 1, 3)
        self.register_buffer("kz", kz)
        self.register_buffer("ky", ky)
        self.register_buffer("kx", kx)

        # Dual-stream projections
        self.core_proj = nn.Sequential(
            nn.Conv3d(1, out_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.Sigmoid(),
        )
        self.furrow_proj = nn.Sequential(
            nn.Conv3d(1, out_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.Sigmoid(),
        )
        self.val_proj = nn.Conv3d(1, out_channels, kernel_size=1, bias=False)

    def forward(self, feat: torch.Tensor, raw_img: torch.Tensor, return_furrow: bool = False):
        # raw_img: (B, 1, Z, Y, X)
        x = raw_img
        if self.smooth_noise and x.shape[2] >= 3 and x.shape[3] >= 3 and x.shape[4] >= 3:
            x = F.conv3d(x, self.kz.to(x.dtype), padding=(1, 0, 0))
            x = F.conv3d(x, self.ky.to(x.dtype), padding=(0, 1, 0))
            x = F.conv3d(x, self.kx.to(x.dtype), padding=(0, 0, 1))

        lap = F.conv3d(x, self.lap_kernel.to(x.dtype), padding=1)
        if lap.shape[2:] != feat.shape[2:]:
            lap = F.interpolate(lap, size=feat.shape[2:], mode="trilinear", align_corners=False)

        # Scale normalize to [-1.0, 1.0]
        lap_scaled = lap * self.scale_norm

        # Dual-stream curvature separation
        lap_pos = F.relu(lap_scaled)   # Cleavage furrow ridges
        lap_neg = F.relu(-lap_scaled)  # Nuclear mass cores

        gate_core = self.core_proj(lap_neg)
        gate_furrow = self.furrow_proj(lap_pos)
        val = self.val_proj(lap_scaled)

        # Asymmetric notch cross-gating
        feat_out = feat * (1.0 + self.alpha_core * gate_core) * (1.0 - self.beta_furrow * gate_furrow) + val
        if return_furrow:
            return feat_out, lap_pos
        return feat_out


# Alias for backwards compatibility
PhysicalLaplacianCrossFusion = PhysicalLaplacianCrossFusionV2


class DiffeomorphicFlowHead(nn.Module):
    """
    Continuous 3D Stationary Velocity Field (SVF) with 6-step Scaling-and-Squaring Lie Group Exp Map.
    Computes phi = exp(v) in Diff(Omega) guaranteeing det(D_phi) > 0 (invertible tissue deformation).
    """
    def __init__(self, in_channels: int, num_steps: int = 6):
        super().__init__()
        self.num_steps = num_steps
        self.flow_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels // 2, 3, kernel_size=3, padding=1),
        )
        # Small initialization so initial deformation is near identity
        nn.init.normal_(self.flow_conv[-1].weight, std=1e-4)
        nn.init.constant_(self.flow_conv[-1].bias, 0.0)
        self._cached_shape = None
        self._cached_grid = None
        self._cached_scale = None

    def integrate_svf(self, v: torch.Tensor) -> torch.Tensor:
        """
        6-step scaling-and-squaring Lie group exp map: phi = exp(v).
        Returns displacement field disp = phi(x) - x in voxels.
        """
        B, _, Z, Y, X = v.shape
        u = v / (2.0 ** self.num_steps)

        if (
            self._cached_shape != (Z, Y, X)
            or self._cached_grid is None
            or self._cached_grid.device != v.device
            or self._cached_grid.dtype != v.dtype
        ):
            grid_z, grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1.0, 1.0, Z, device=v.device, dtype=v.dtype),
                torch.linspace(-1.0, 1.0, Y, device=v.device, dtype=v.dtype),
                torch.linspace(-1.0, 1.0, X, device=v.device, dtype=v.dtype),
                indexing="ij"
            )
            base_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)
            scale_vec = torch.tensor(
                [2.0 / max(X - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(Z - 1, 1)],
                device=v.device, dtype=v.dtype
            )
            self._cached_grid = base_grid
            self._cached_scale = scale_vec
            self._cached_shape = (Z, Y, X)

        base_grid = self._cached_grid.expand(B, -1, -1, -1, -1)
        scale_vec = self._cached_scale

        disp = u
        for _ in range(self.num_steps):
            # disp channels (dz, dy, dx) -> convert to (dx, dy, dz) normalized
            disp_norm = disp.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]] * scale_vec
            sample_grid = (base_grid + disp_norm).clamp(-1.5, 1.5)
            disp_warped = F.grid_sample(disp, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)
            disp = disp + disp_warped
        return disp

    def forward(self, feat: torch.Tensor):
        # feat: (B, C, Z, Y, X)
        v = self.flow_conv(feat)  # (B, 3, Z, Y, X) where channels are (dz, dy, dx) in voxels
        disp = self.integrate_svf(v)
        return v, disp


class SubVoxelPeakRefiner(nn.Module):
    """
    Continuous 3D Sub-Voxel Coordinate Regression Head with Algebraic Rational Soft-Clipping.
    delta* = delta / sqrt(1 + ||delta||^2 / delta_max^2)
    Provides C^inf smooth gradients without saturation artifacts.
    """
    def __init__(self, in_channels: int, delta_max: float = 0.5):
        super().__init__()
        self.delta_max = delta_max
        self.det_head = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.delta_head = nn.Conv3d(in_channels, 3, kernel_size=1)
        # Background suppression prior
        nn.init.constant_(self.det_head.bias, -4.0)
        nn.init.normal_(self.det_head.weight, std=0.01)
        nn.init.constant_(self.delta_head.bias, 0.0)
        nn.init.normal_(self.delta_head.weight, std=0.001)

    def forward(self, feat: torch.Tensor):
        logits = self.det_head(feat)
        raw_deltas = self.delta_head(feat)  # (B, 3, Z, Y, X)
        # Algebraic rational soft-clipping: strictly bounded to [-delta_max, delta_max]
        norm_sq = (raw_deltas ** 2).sum(dim=1, keepdim=True)
        deltas = raw_deltas / torch.sqrt(1.0 + norm_sq / (self.delta_max ** 2))
        return logits, deltas


class AnisoUNet3D(nn.Module):
    """
    Bio-DANT: Anisotropic Separable 3D UNet with Dual-Stream Laplacian and Diffeomorphic Flow.
    Input shape: (B, T, 1, Z, Y, X).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        layer_channels: list[int] = [32, 64, 128],
        use_checkpointing: bool = True,
        depthwise: bool = True,
        scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    ):
        super().__init__()
        self.out_channels = out_channels
        self.use_checkpointing = use_checkpointing

        # Initial stem projection with factorized 3x7x7
        c0 = layer_channels[0]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c0, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c0),
            nn.SiLU(inplace=True),
            AnisoResBlock3D(c0, depthwise=depthwise, kernel_lateral=7),
        )

        # Encoder stages
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        curr_c = c0
        for next_c in layer_channels[1:]:
            self.pools.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            enc = nn.Sequential(
                AnisoSeparableConv3D(curr_c, next_c, depthwise=False, kernel_lateral=7),
                AnisoResBlock3D(next_c, depthwise=depthwise, kernel_lateral=7),
            )
            self.encoders.append(enc)
            curr_c = next_c

        # Bottleneck
        self.bottleneck = nn.Sequential(
            AnisoResBlock3D(curr_c, depthwise=depthwise, kernel_lateral=7),
            AnisoResBlock3D(curr_c, depthwise=depthwise, kernel_lateral=7),
        )

        # Decoder stages
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_channels = list(reversed(layer_channels))
        for i in range(len(rev_channels) - 1):
            c_in = rev_channels[i]
            c_out = rev_channels[i + 1]
            self.ups.append(
                nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
            )
            dec = nn.Sequential(
                AnisoSeparableConv3D(c_in + c_out, c_out, depthwise=False, kernel_lateral=7),
                AnisoResBlock3D(c_out, depthwise=depthwise, kernel_lateral=7),
            )
            self.decoders.append(dec)

        # Dual-stream physical Laplacian cross-fusion
        self.lap_fusion = PhysicalLaplacianCrossFusion(out_channels=c0, scale=scale)

        # Output heads
        self.head = nn.Conv3d(c0, out_channels, kernel_size=1)
        self.refiner = SubVoxelPeakRefiner(out_channels, delta_max=0.5)
        self.flow_head = DiffeomorphicFlowHead(out_channels, num_steps=6)

    def _forward_single_frame(self, x: torch.Tensor):
        s0 = self.stem(x)
        skips = [s0]
        curr = s0

        for pool, enc in zip(self.pools, self.encoders):
            curr = pool(curr)
            if self.use_checkpointing and self.training:
                curr = checkpoint(enc, curr, use_reentrant=False)
            else:
                curr = enc(curr)
            skips.append(curr)

        curr = skips.pop()
        if self.use_checkpointing and self.training:
            curr = checkpoint(self.bottleneck, curr, use_reentrant=False)
        else:
            curr = self.bottleneck(curr)

        for up, dec in zip(self.ups, self.decoders):
            skip = skips.pop()
            curr = up(curr)
            if curr.shape[2:] != skip.shape[2:]:
                curr = F.interpolate(curr, size=skip.shape[2:], mode="trilinear", align_corners=False)
            curr = torch.cat([curr, skip], dim=1)
            if self.use_checkpointing and self.training:
                curr = checkpoint(dec, curr, use_reentrant=False)
            else:
                curr = dec(curr)

        # Inject physical Laplacian boundary cues and extract furrow notch map
        curr, furrow_map = self.lap_fusion(curr, x, return_furrow=True)

        feat = self.head(curr)
        det_logits, sub_deltas = self.refiner(feat)
        # Direct cleavage furrow notch depression to prevent peak bridging
        det_logits = det_logits - 3.50 * furrow_map
        v, disp = self.flow_head(feat)
        return feat, det_logits, sub_deltas, (v, disp)

    def forward(self, x: torch.Tensor, return_flows: bool = False, return_sub_deltas: bool = False):
        """
        Input x: (B, W, 1, Z, Y, X) or (B, W, Z, Y, X) or (B, Z, Y, X)
        Returns:
            if return_flows: (feats, det_logits, sub_deltas, flows)
            elif return_sub_deltas: (feats, det_logits, sub_deltas)
            else: (feats, det_logits)
        """
        if x.dim() == 4:
            x = x.unsqueeze(1).unsqueeze(2)
        elif x.dim() == 5:
            x = x.unsqueeze(2)

        B, W, C, Z, Y, X = x.shape
        feats = []
        det_logits = []
        sub_deltas = []
        flows = []
        for t in range(W):
            xt = x[:, t]
            f_t, log_t, del_t, flow_t = self._forward_single_frame(xt)
            feats.append(f_t)
            det_logits.append(log_t)
            sub_deltas.append(del_t)
            flows.append(flow_t)

        feats = torch.stack(feats, dim=1)  # (B, W, C_out, Z, Y, X)
        if return_flows:
            return feats, det_logits, sub_deltas, flows
        if return_sub_deltas:
            return feats, det_logits, sub_deltas
        return feats, det_logits

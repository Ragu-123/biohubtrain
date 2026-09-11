"""
AnisoTrack3D: End-to-End Joint Model.
Combines:
1. AnisoUNet3D: Anisotropic spatial-axial separable 3D conv backbone
2. SubVoxelDetectionHead: Continuous sub-voxel peak detector & parabolic refiner
3. Differentiable Trilinear Feature Sampling (F.grid_sample)
4. SparseLocalTrackTransformer: Local ball graph attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aniso_unet import AnisoUNet3D
from .local_transformer import SparseLocalTrackTransformer


try:
    from src.kernels.triton_ops import trilinear_index_triton, HAS_TRITON
except ImportError:
    try:
        from ..kernels.triton_ops import trilinear_index_triton, HAS_TRITON
    except ImportError:
        HAS_TRITON = False


def trilinear_index_features(feat_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Differentiable feature sampling at continuous coordinates.
    feat_map: (B=1, C, Z, Y, X) or (C, Z, Y, X)
    coords:   (N, 3) continuous coordinates (z, y, x)
    Returns:  (N, C) feature vectors.
    """
    if feat_map.dim() == 5:
        feat_map = feat_map.squeeze(0)
    C, Z, Y, X = feat_map.shape
    if coords.shape[0] == 0:
        return torch.empty((0, C), device=feat_map.device, dtype=feat_map.dtype)

    if HAS_TRITON and feat_map.is_cuda:
        return trilinear_index_triton(feat_map, coords)

    # PyTorch fallback
    z_n = (coords[:, 0] / (Z - 1.0)) * 2.0 - 1.0
    y_n = (coords[:, 1] / (Y - 1.0)) * 2.0 - 1.0
    x_n = (coords[:, 2] / (X - 1.0)) * 2.0 - 1.0
    grid = torch.stack([x_n, y_n, z_n], dim=-1).view(1, 1, 1, -1, 3)
    sampled = F.grid_sample(feat_map.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled.squeeze(0).squeeze(1).squeeze(1).t()


class AnisoTrack3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        unet_out_channels: int = 32,
        unet_layers: list[int] = [32, 64, 128],
        transformer_d_model: int = 64,
        transformer_layers: int = 3,
        r_max_um: float = 12.0,
        depthwise: bool = True,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.unet = AnisoUNet3D(
            in_channels=in_channels,
            out_channels=unet_out_channels,
            layer_channels=unet_layers,
            depthwise=depthwise,
            use_checkpointing=use_checkpointing,
        )
        self.transformer = SparseLocalTrackTransformer(
            feat_dim=unet_out_channels,
            pos_dim=32,
            d_model=transformer_d_model,
            n_layers=transformer_layers,
            r_max_um=r_max_um,
        )

    def encode(self, imgs: torch.Tensor):
        """
        imgs: (B, W, 1, Z, Y, X)
        Returns:
            feats: (B, W, C, Z, Y, X)
            det_logits: list of (B, 1, Z, Y, X)
            sub_deltas: list of (B, 3, Z, Y, X)
        """
        return self.unet(imgs)

    def predict_edges(
        self,
        feat_src: torch.Tensor,
        coords_src_um: torch.Tensor,
        feat_tgt: torch.Tensor,
        coords_tgt_um: torch.Tensor,
    ):
        """
        Computes pairwise transition logits between source and target cells.
        """
        return self.transformer(feat_src, coords_src_um, feat_tgt, coords_tgt_um)

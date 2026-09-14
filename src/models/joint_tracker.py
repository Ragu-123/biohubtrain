"""
AnisoTrack3D: End-to-End Joint Model.
Combines:
1. AnisoUNet3D: Anisotropic spatial-axial factorized 3D UNet with Dual-Stream Laplacian
2. SubVoxelDetectionHead: Continuous sub-voxel peak detector & algebraic rational refiner
3. DiffeomorphicFlowHead: Continuous 3D Lie group flow field
4. Differentiable Trilinear Feature Sampling (F.grid_sample / Triton)
5. SparseLocalTrackTransformer: Local ball graph attention with Anisotropic Fourier Harmonics
6. GPU Log-Domain Sinkhorn Unbalanced Optimal Transport (UOT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aniso_unet import AnisoUNet3D
from .local_transformer import SparseLocalTrackTransformer, log_sinkhorn_uot


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
    coords:   (N, 3) continuous coordinates (z, y, x) in voxels
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
    z_n = (coords[:, 0] / max(Z - 1.0, 1.0)) * 2.0 - 1.0
    y_n = (coords[:, 1] / max(Y - 1.0, 1.0)) * 2.0 - 1.0
    x_n = (coords[:, 2] / max(X - 1.0, 1.0)) * 2.0 - 1.0
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
        scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    ):
        super().__init__()
        self.scale = scale
        self.unet = AnisoUNet3D(
            in_channels=in_channels,
            out_channels=unet_out_channels,
            layer_channels=unet_layers,
            depthwise=depthwise,
            use_checkpointing=use_checkpointing,
            scale=scale,
        )
        self.transformer = SparseLocalTrackTransformer(
            feat_dim=unet_out_channels,
            pos_dim=32,
            d_model=transformer_d_model,
            n_layers=transformer_layers,
            r_max_um=r_max_um,
        )

    def encode(self, imgs: torch.Tensor, return_flows: bool = False, return_sub_deltas: bool = False):
        """
        imgs: (B, W, 1, Z, Y, X) or (B, W, Z, Y, X)
        Returns:
            if return_flows: (feats, det_logits, sub_deltas, flows)
            elif return_sub_deltas: (feats, det_logits, sub_deltas)
            else: (feats, det_logits) for evaluate.py / submission_sota.py compatibility
        """
        return self.unet(imgs, return_flows=return_flows, return_sub_deltas=return_sub_deltas)

    def _index_features(self, feat_map: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if coords.dim() == 3:
            coords = coords.squeeze(0)
        return trilinear_index_features(feat_map, coords).unsqueeze(0)

    def sample_features(self, feat_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return trilinear_index_features(feat_map, coords)

    def sample_flows(self, flow_map: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Samples continuous 3D velocity vectors (v_z, v_y, v_x) at cell coordinates."""
        return trilinear_index_features(flow_map, coords)

    def predict_edges(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, *args, **kwargs):
        """
        Universal predictor handling both evaluate.py and train.py signatures.
        """
        if b.shape[-1] != 3 and c.shape[-1] == 3:
            # evaluate.py: (feat_src, feat_tgt, coords_src_um, coords_tgt_um)
            f_s = a.squeeze(0) if a.dim() == 3 else a
            f_t = b.squeeze(0) if b.dim() == 3 else b
            c_s = c.squeeze(0) if c.dim() == 3 else c
            c_t = d.squeeze(0) if d.dim() == 3 else d
            logits, cand_mask = self.transformer(f_s, c_s, f_t, c_t)
            return logits.unsqueeze(0), cand_mask.unsqueeze(0)
        else:
            # train.py: (feat_src, coords_src_um, feat_tgt, coords_tgt_um)
            f_s = a.squeeze(0) if a.dim() == 3 else a
            c_s = b.squeeze(0) if b.dim() == 3 else b
            f_t = c.squeeze(0) if c.dim() == 3 else c
            c_t = d.squeeze(0) if d.dim() == 3 else d
            flow_src_um = kwargs.get("flow_src_um", None)
            return self.transformer(f_s, c_s, f_t, c_t, flow_src_um=flow_src_um)

    def decode_edges(
        self,
        logits: torch.Tensor,
        dist_um: torch.Tensor = None,
        use_uot: bool = True,
        threshold: float = 0.20,
    ) -> torch.Tensor:
        if use_uot:
            return self.transformer.decode_uot_edges(logits, dist_um=dist_um, prob_threshold=threshold)
        else:
            return self.transformer.decode_edges(logits, threshold=threshold)

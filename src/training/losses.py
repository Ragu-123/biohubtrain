"""
AnisoTrackingLoss: Multi-task Loss for AnisoTrack3D.
Includes:
1. Column-wise Softmax Focal BCE Edge Loss (permits divisions, prevents mergers)
2. Class-imbalanced Weighted Detection BCE Loss (downweights background by 0.01)
3. Smooth L1 Continuous Sub-voxel Peak Refinement Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnisoTrackingLoss(nn.Module):
    def __init__(
        self,
        det_loss_weight: float = 1.0,
        det_neg_weight: float = 0.01,
        subvoxel_loss_weight: float = 0.5,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.det_loss_weight = det_loss_weight
        self.det_neg_weight = det_neg_weight
        self.subvoxel_loss_weight = subvoxel_loss_weight
        self.focal_gamma = focal_gamma

    def edge_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        active_rows = target.sum(dim=1) > 0
        active_cols = target.sum(dim=0) > 0
        mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probs = torch.softmax(logits, dim=0) # dim=0: target chooses source
        bce = F.binary_cross_entropy(probs, target, reduction="none")
        p_t = probs * target + (1.0 - probs) * (1.0 - target)
        focal_weight = (1.0 - p_t) ** self.focal_gamma

        div_rows = target.sum(dim=1) > 1
        div_mult = torch.ones_like(bce)
        div_mult[div_rows] = 1.2

        return (focal_weight * bce * div_mult)[mask].mean()

    def detection_loss(self, det_logits: torch.Tensor, gt_peak_mask: torch.Tensor) -> torch.Tensor:
        pos_mask = gt_peak_mask > 0.5
        n_pos = pos_mask.sum().clamp(min=1.0)
        n_neg = (~pos_mask).sum().clamp(min=1.0)

        weight = torch.where(pos_mask, 1.0 / n_pos, (self.det_neg_weight / n_neg))
        bce = F.binary_cross_entropy_with_logits(det_logits, gt_peak_mask, weight=weight, reduction="sum")
        return bce

    def subvoxel_loss(self, pred_deltas: torch.Tensor, target_deltas: torch.Tensor, peak_mask: torch.Tensor) -> torch.Tensor:
        if peak_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_deltas.device, requires_grad=True)
        mask = peak_mask.expand_as(pred_deltas) > 0.5
        return F.smooth_l1_loss(pred_deltas[mask], target_deltas[mask], beta=0.1)

    def forward(
        self,
        edge_logits: torch.Tensor,
        edge_targets: torch.Tensor,
        det_logits_list: list,
        gt_peak_masks: list,
        pred_deltas_list: list = None,
        target_deltas_list: list = None,
    ):
        l_edge = self.edge_loss(edge_logits, edge_targets)
        
        l_det = sum(
            self.detection_loss(log, gt)
            for log, gt in zip(det_logits_list, gt_peak_masks)
        ) / max(len(det_logits_list), 1)

        l_sub = torch.tensor(0.0, device=l_edge.device)
        if pred_deltas_list is not None and target_deltas_list is not None:
            l_sub = sum(
                self.subvoxel_loss(pd, td, pm)
                for pd, td, pm in zip(pred_deltas_list, target_deltas_list, gt_peak_masks)
            ) / max(len(pred_deltas_list), 1)

        total_loss = l_edge + self.det_loss_weight * l_det + self.subvoxel_loss_weight * l_sub
        return total_loss, {"loss_edge": l_edge.item(), "loss_det": l_det.item(), "loss_sub": l_sub.item()}

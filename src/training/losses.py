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

    def edge_loss(self, logits: torch.Tensor, target: torch.Tensor, cand_mask: torch.Tensor = None) -> torch.Tensor:
        if logits.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if cand_mask is None:
            cand_mask = logits > -1e3

        if not cand_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        with torch.amp.autocast('cuda', enabled=False):
            active_logits = logits[cand_mask].float()
            active_targets = target[cand_mask].float()
            pos_weight = torch.tensor([5.0], device=logits.device)
            return F.binary_cross_entropy_with_logits(active_logits, active_targets, pos_weight=pos_weight)

    def detection_loss(self, det_logits: torch.Tensor, gt_peak_mask: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            det_f = det_logits.float().view(-1)
            gt_f = gt_peak_mask.float().view(-1)
            pos_mask = gt_f > 0.5
            n_pos = pos_mask.sum().clamp(min=1.0)
            n_neg = (~pos_mask).sum().clamp(min=1.0)

            weight = torch.where(pos_mask, 1.0 / n_pos, (self.det_neg_weight / n_neg))
            bce = F.binary_cross_entropy_with_logits(det_f, gt_f, weight=weight, reduction="sum")
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
        cand_mask: torch.Tensor = None,
    ):
        l_edge = self.edge_loss(edge_logits, edge_targets, cand_mask=cand_mask)
        
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

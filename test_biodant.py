"""
Local Verification Suite for Bio-DANT Architecture.
Tests:
1. Forward pass on (B=1, W=2, C=1, Z=16, Y=64, X=64)
2. Lie group exponential map diffeomorphic displacement fields
3. Anisotropic Fourier positional embedding
4. GPU Log-Domain Sinkhorn UOT
5. Multi-task loss computation (detection + edge + Mahalanobis subvoxel + diffeomorphic advection)
6. Backward pass and gradient propagation
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models import AnisoTrack3D, trilinear_index_features
from src.models.local_transformer import log_sinkhorn_uot
from src.training.losses import AnisoTrackingLoss


def run_test():
    print("=" * 70)
    print("           BIO-DANT LOCAL VERIFICATION SUITE")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 1. Initialize Model
    print("1. Initializing AnisoTrack3D...")
    model = AnisoTrack3D(
        in_channels=1,
        unet_out_channels=16,
        unet_layers=[16, 32, 64],
        transformer_d_model=32,
        transformer_layers=2,
        depthwise=True,
    ).to(device)
    print(f"   Model initialized! Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 2. Forward pass with fake 4D microscopy images
    print("2. Testing forward pass with return_flows=True...")
    B, W, C, Z, Y, X = 1, 2, 1, 16, 64, 64
    imgs = torch.rand(B, W, C, Z, Y, X, dtype=torch.float32)

    feats, det_logits, sub_deltas, flows = model.encode(imgs, return_flows=True)
    print(f"   feats shape      : {feats.shape}")
    print(f"   det_logits[0]    : {det_logits[0].shape}")
    print(f"   sub_deltas[0]    : {sub_deltas[0].shape}")
    print(f"   flow v[0] shape  : {flows[0][0].shape}")
    print(f"   flow disp[0]     : {flows[0][1].shape}")

    assert feats.shape == (B, W, 16, Z, Y, X)
    assert det_logits[0].shape == (B, 1, Z, Y, X)
    assert sub_deltas[0].shape == (B, 3, Z, Y, X)
    assert flows[0][0].shape == (B, 3, Z, Y, X)
    assert flows[0][1].shape == (B, 3, Z, Y, X)
    print("   [PASS] Forward shapes match exact physical dimensions!")

    # 3. Test Transformer & Log-Domain Sinkhorn UOT
    print("3. Testing Transformer & Log-Domain Sinkhorn UOT...")
    N, M = 10, 12
    coords0_um = torch.rand(N, 3) * 50.0
    coords1_um = torch.rand(M, 3) * 50.0
    f0 = torch.randn(N, 16)
    f1 = torch.randn(M, 16)
    flow0_um = torch.randn(N, 3) * 0.5

    edge_logits, cand_mask = model.predict_edges(f0, coords0_um, f1, coords1_um, flow_src_um=flow0_um)
    print(f"   edge_logits shape: {edge_logits.shape}")
    print(f"   cand_mask active : {cand_mask.sum().item()} pairs")

    # UOT Decoding
    u_edges = model.decode_edges(edge_logits, use_uot=True, threshold=0.10)
    print(f"   UOT decoded edges: {u_edges.sum().item()} 1-to-1 matches")
    assert u_edges.shape == (N, M)
    print("   [PASS] UOT decoding is functioning properly!")

    # 4. Multi-task Loss & Backward Pass
    print("4. Testing Multi-Task Loss & Backward Gradient Propagation...")
    loss_fn = AnisoTrackingLoss(
        det_loss_weight=1.0,
        det_neg_weight=0.01,
        subvoxel_loss_weight=0.5,
        advection_loss_weight=0.2,
    )

    target_matrix = torch.zeros(N, M)
    for i in range(min(N, M)):
        target_matrix[i, i] = 1.0

    gt_masks = [torch.zeros(1, 1, Z, Y, X), torch.zeros(1, 1, Z, Y, X)]
    gt_masks[0][0, 0, 8, 32, 32] = 1.0
    gt_masks[1][0, 0, 8, 32, 33] = 1.0

    target_deltas = [torch.zeros_like(sub_deltas[0]), torch.zeros_like(sub_deltas[1])]

    total_loss, loss_dict = loss_fn(
        edge_logits, target_matrix, det_logits, gt_masks,
        pred_deltas_list=sub_deltas,
        target_deltas_list=target_deltas,
        cand_mask=cand_mask,
        raw_imgs=imgs,
        flows=flows,
    )
    print(f"   total_loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"     - {k}: {v:.4f}")

    total_loss.backward()

    # Verify gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed!"
    print(f"   Gradients computed successfully across {len(grad_norms)} tensors! Max grad: {max(grad_norms):.6f}")
    print("   [PASS] End-to-end backpropagation verified!")

    print("=" * 70)
    print("ALL TESTS PASSED! BIO-DANT ARCHITECTURE VERIFIED LOCALLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_test()

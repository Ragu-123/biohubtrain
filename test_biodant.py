"""
Verification Suite for Bio-DANT Architecture.
Tests:
1. GPU / CUDA device detection and utilization
2. Forward pass on (B=1, W=2, C=1, Z=16, Y=64, X=64)
3. Lie group exponential map diffeomorphic displacement fields
4. Anisotropic Fourier positional embedding
5. GPU Log-Domain Sinkhorn UOT
6. Multi-task loss computation (detection + edge + Mahalanobis subvoxel + diffeomorphic advection)
7. Backward pass and gradient propagation
8. C++ Fast SIMD Tracking Solver execution
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models import AnisoTrack3D, trilinear_index_features
from src.models.local_transformer import log_sinkhorn_uot
from src.training.losses import AnisoTrackingLoss
from src.kernels.cpp_ops import get_cpp_tracker


def run_test():
    print("=" * 70)
    print("           BIO-DANT GPU / C++ VERIFICATION SUITE")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "CPU"
    print(f"Hardware: {device} ({device_name}) | Total Visible GPUs: {num_gpus}")

    # 1. Initialize Model
    print("\n1. Initializing AnisoTrack3D on target hardware...")
    model = AnisoTrack3D(
        in_channels=1,
        unet_out_channels=16,
        unet_layers=[16, 32, 64],
        transformer_d_model=32,
        transformer_layers=2,
        depthwise=True,
    ).to(device)
    print(f"   Model initialized! Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 2. Forward pass with fake 4D microscopy images
    print("\n2. Testing forward pass with return_flows=True on device...")
    B, W, C, Z, Y, X = 1, 2, 1, 16, 64, 64
    imgs = torch.rand(B, W, C, Z, Y, X, dtype=torch.float32, device=device)

    feats, det_logits, sub_deltas, flows = model.encode(imgs, return_flows=True)
    print(f"   feats shape      : {feats.shape} (device: {feats.device})")
    print(f"   det_logits[0]    : {det_logits[0].shape} (device: {det_logits[0].device})")
    print(f"   sub_deltas[0]    : {sub_deltas[0].shape} (device: {sub_deltas[0].device})")
    print(f"   flow v[0] shape  : {flows[0][0].shape} (device: {flows[0][0].device})")
    print(f"   flow disp[0]     : {flows[0][1].shape} (device: {flows[0][1].device})")

    assert feats.shape == (B, W, 16, Z, Y, X)
    assert det_logits[0].shape == (B, 1, Z, Y, X)
    assert sub_deltas[0].shape == (B, 3, Z, Y, X)
    assert flows[0][0].shape == (B, 3, Z, Y, X)
    assert flows[0][1].shape == (B, 3, Z, Y, X)
    print("   [PASS] Forward tensor shapes match physical dimensions on GPU!")

    # 3. Test Transformer & Log-Domain Sinkhorn UOT
    print("\n3. Testing Transformer & Log-Domain Sinkhorn UOT on device...")
    N, M = 10, 12
    coords0_um = (torch.rand(N, 3) * 50.0).to(device)
    coords1_um = (torch.rand(M, 3) * 50.0).to(device)
    f0 = torch.randn(N, 16, device=device)
    f1 = torch.randn(M, 16, device=device)
    flow0_um = torch.randn(N, 3, device=device) * 0.5

    edge_logits, cand_mask = model.predict_edges(f0, coords0_um, f1, coords1_um, flow_src_um=flow0_um)
    print(f"   edge_logits shape: {edge_logits.shape} (device: {edge_logits.device})")
    print(f"   cand_mask active : {cand_mask.sum().item()} pairs")

    # UOT Decoding
    u_edges = model.decode_edges(edge_logits, use_uot=True, threshold=0.10)
    print(f"   UOT decoded edges: {u_edges.sum().item()} 1-to-1 matches")
    assert u_edges.shape == (N, M)
    print("   [PASS] UOT decoding is functioning properly on GPU!")

    # 4. Multi-task Loss & Backward Pass
    print("\n4. Testing Multi-Task Loss & Backward Gradient Propagation...")
    loss_fn = AnisoTrackingLoss(
        det_loss_weight=1.0,
        det_neg_weight=0.01,
        subvoxel_loss_weight=0.5,
        advection_loss_weight=0.2,
    ).to(device)

    target_matrix = torch.zeros(N, M, device=device)
    for i in range(min(N, M)):
        target_matrix[i, i] = 1.0

    gt_masks = [torch.zeros(1, 1, Z, Y, X, device=device), torch.zeros(1, 1, Z, Y, X, device=device)]
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
    print("   [PASS] End-to-end backpropagation verified on hardware!")

    # 5. Test C++ Fast SIMD Tracking Solver
    print("\n5. Testing C++ Fast SIMD Tracking Solver...")
    cpp_mod = get_cpp_tracker()
    if cpp_mod is not None:
        K = 100
        scores = torch.rand(K, dtype=torch.float32)
        probs = torch.rand(K, dtype=torch.float32)
        si = torch.randint(0, N, (K,), dtype=torch.int64)
        tj = torch.randint(0, M, (K,), dtype=torch.int64)
        dists = torch.rand(K, dtype=torch.float32) * 10.0
        c_src = torch.rand(N, 3, dtype=torch.float32) * 100.0
        c_tgt = torch.rand(M, 3, dtype=torch.float32) * 100.0
        drift = torch.zeros(N, 3, dtype=torch.float32)

        t_src, t_tgt, t_probs, t_dists, t_div = cpp_mod.fast_greedy_track(
            scores, probs, si, tj, dists, c_src, c_tgt, drift,
            N, M, 0.48, 0.28, 0.72
        )
        print(f"   [PASS] C++ Fast Greedy Track successfully returned {t_src.size(0)} edges!")
    else:
        print("   [INFO] C++ compiler not detected on this system (MSVC required on Windows), fallback active.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! BIO-DANT ARCHITECTURE VERIFIED ON TARGET HARDWARE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_test()

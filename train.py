#!/usr/bin/env python
"""
AnisoTrack3D High-Performance Training Pipeline.
Supports:
- 2x Tesla T4 GPUs with PyTorch DataParallel
- FP16 Automatic Mixed Precision (AMP)
- Strided I/O from Zarr volumes
- Continuous Sub-Voxel Peak Refinement
- Local Candidate Graph Cross-Attention
Usage:
    python train.py --data-dir /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
                    --epochs 10 --batch-size 4 --lr 1e-4 --max-windows-per-vol 5
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import polars as pl
if not hasattr(pl, "Float16"):
    pl.Float16 = pl.Float32
try:
    import polars._utils.various as _pl_various
    if not hasattr(_pl_various, "NO_DEFAULT"):
        _pl_various.NO_DEFAULT = getattr(_pl_various, "NoDefault", object())
except Exception:
    pass

# Add repo and support paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
for p in [
    "/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/src",
    "/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/scripts",
]:
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)

from src.models import AnisoTrack3D, trilinear_index_features
from src.training.losses import AnisoTrackingLoss
from src.data.dataset import BiohubWindowDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train AnisoTrack3D Model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to train directory with .zarr and .geff")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (frame pairs)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--unet-channels", type=int, default=32, help="UNet output channels")
    parser.add_argument("--downsample", type=str, default="1,4,4", help="Z,Y,X downsample strides")
    parser.add_argument("--max-windows-per-vol", type=int, default=None, help="Limit windows per movie for quick iteration")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save weights")
    return parser.parse_args()


def normalize_coords_for_grid_sample(coords: torch.Tensor, shape_zyx: tuple) -> torch.Tensor:
    """
    coords: (N, 3) in integer/subvoxel grid coordinates [z, y, x]
    shape_zyx: (Z, Y, X)
    Returns: (N, 3) normalized in [-1, 1] in order (X, Y, Z) for grid_sample.
    """
    Z, Y, X = shape_zyx
    z_n = (coords[:, 0] / (Z - 1.0)) * 2.0 - 1.0
    y_n = (coords[:, 1] / (Y - 1.0)) * 2.0 - 1.0
    x_n = (coords[:, 2] / (X - 1.0)) * 2.0 - 1.0
    return torch.stack([x_n, y_n, z_n], dim=-1)


def custom_collate(batch):
    # Returns list of dicts to preserve variable node counts per frame
    return batch


def main():
    args = parse_args()
    downsample = tuple(int(x) for x in args.downsample.split(","))

    print("=" * 82)
    print("             ANISOTRACK3D HIGH-PERFORMANCE TRAINING PIPELINE")
    print("=" * 82)
    print(f"  Data Directory        : {args.data_dir}")
    print(f"  Epochs                : {args.epochs}")
    print(f"  Batch Size            : {args.batch_size}")
    print(f"  Downsample (Z,Y,X)    : {downsample}")
    print(f"  Learning Rate         : {args.lr}")
    print(f"  Max Windows / Volume  : {args.max_windows_per_vol}")
    print("=" * 82)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Hardware: {device} ({torch.cuda.get_device_name(0) if num_gpus > 0 else 'CPU'}) | Visible GPUs: {num_gpus}")

    # 1. Dataset & DataLoader
    dataset = BiohubWindowDataset(
        data_dir=Path(args.data_dir),
        downsample=downsample,
        window_size=2,
        max_windows_per_vol=args.max_windows_per_vol,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=2,
        pin_memory=False,
    )

    # 2. Model
    model = AnisoTrack3D(
        unet_out_channels=args.unet_channels,
        unet_layers=[32, 64, 128],
        transformer_d_model=64,
        depthwise=True,
    ).to(device)

    if num_gpus > 1:
        print(f"Distributing UNet 3D backbone across {num_gpus} GPUs via DataParallel...")
        model.unet = nn.DataParallel(model.unet)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AnisoTrack3D Initialized: {param_count:,} trainable parameters.")

    # 3. Loss & Optimizer
    loss_fn = AnisoTrackingLoss(
        det_loss_weight=10.0,
        det_neg_weight=0.1,
        subvoxel_loss_weight=0.5,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting active training loop...")
    print("-" * 82)
    print(f"{'Epoch':<8} | {'Step':<8} | {'Total Loss':<12} | {'Edge Loss':<11} | {'Det Loss':<10} | {'Throughput':<12} | {'Peak VRAM':<10}")
    print("-" * 82)

    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            # Stack images: (B, W, 1, Z, Y, X)
            imgs = torch.stack([item["imgs"] for item in batch], dim=0).to(device)
            B, W, C, Z, Y, X = imgs.shape

            with torch.amp.autocast('cuda', dtype=torch.float16):
                feats, det_logits, sub_deltas = model.encode(imgs)

                batch_total_loss = 0.0
                edge_loss_sum = 0.0
                det_loss_sum = 0.0

                for b in range(B):
                    item = batch[b]
                    c0 = item["coords0"].to(device)
                    c1 = item["coords1"].to(device)
                    target = item["target"].to(device)
                    scale = item["scale"].to(device)

                    # Physical coordinates in microns
                    ds_tensor = torch.tensor(downsample, device=device, dtype=torch.float32)
                    c0_um = c0 * ds_tensor * scale
                    c1_um = c1 * ds_tensor * scale

                    # Trilinear feature sampling at continuous coordinates via Custom Triton Kernel
                    f0_map = feats[b, 0] # (C_out, Z, Y, X)
                    f1_map = feats[b, 1]

                    f0 = trilinear_index_features(f0_map, c0)
                    f1 = trilinear_index_features(f1_map, c1)

                    # Pairwise edge predictions
                    edge_logits, cand_mask = model.predict_edges(f0, c0_um, f1, c1_um)

                    # Detection masks
                    gt_masks = [m.to(device) for m in item["peak_masks"]]
                    det_logs = [det_logits[0][b:b+1], det_logits[1][b:b+1]]

                    sample_loss, loss_dict = loss_fn(edge_logits, target, det_logs, gt_masks, cand_mask=cand_mask)
                    batch_total_loss = batch_total_loss + sample_loss
                    edge_loss_sum += loss_dict["loss_edge"]
                    det_loss_sum += loss_dict["loss_det"]

                loss = batch_total_loss / B

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_step = time.perf_counter() - t0
            throughput = B / max(t_step, 1e-4)
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            global_step += 1
            if global_step % 1 == 0:
                print(f"{epoch+1:<8} | {global_step:<8} | {loss.item():<12.4f} | {edge_loss_sum/B:<11.4f} | {det_loss_sum/B:<10.4f} | {throughput:<7.1f} p/s | {vram_mb:<7.1f} MB")

        # Save checkpoint per epoch
        ckpt_path = save_dir / f"anisotrack3d_epoch_{epoch+1}.pth"
        raw_model = model.unet.module if hasattr(model.unet, "module") else model.unet
        torch.save({
            "epoch": epoch + 1,
            "unet_state_dict": raw_model.state_dict(),
            "transformer_state_dict": model.transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"\U0001f4be Saved checkpoint: {ckpt_path.name}")

    total_time = time.time() - start_time
    print("-" * 82)
    print(f"\U0001f3c6 Training Complete! Finished {args.epochs} epochs in {total_time/60:.2f} minutes.")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    main()

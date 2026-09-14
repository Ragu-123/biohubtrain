#!/usr/bin/env python
"""
Bio-DANT High-Performance Training Pipeline.
Supports:
- Dual Tesla T4 GPUs with PyTorch DataParallel
- FP16 Automatic Mixed Precision (AMP)
- Factorized 3x7x7 Anisotropic Convolutions
- Continuous Diffeomorphic Flow Head with 6-step Lie group exponential integration
- GPU Log-Domain Entropic Unbalanced Optimal Transport (UOT)
- Automated Evaluation on Held-out Validation Volumes after each epoch (Baseline: 0.9309)
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import polars as pl
if not hasattr(pl, "Float16"):
    pl.Float16 = pl.Float32
try:
    import polars._utils.various as _pl_various
    if not hasattr(_pl_various, "NO_DEFAULT"):
        _pl_various.NO_DEFAULT = getattr(_pl_various, "NoDefault", object())
except Exception:
    pass

repo_root = str(Path(__file__).resolve().parent)
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

for p in [
    "/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/src",
    "/kaggle/input/datasets/ragunathravi/forcompbiohub/repo/scripts",
    "/kaggle/input/forcompbiohub/repo/src",
    "/kaggle/input/forcompbiohub/repo/scripts",
]:
    if p not in sys.path and Path(p).exists():
        sys.path.append(p)

from src.models import AnisoTrack3D, trilinear_index_features
from src.training.losses import AnisoTrackingLoss
from src.data.dataset import BiohubWindowDataset
from src.evaluation.benchmark_suite import BenchmarkSuite


def parse_args():
    parser = argparse.ArgumentParser(description="Train Bio-DANT Model")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to train directory with .zarr and .geff")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (frame pairs)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--unet-channels", type=int, default=32, help="UNet output channels")
    parser.add_argument("--downsample", type=str, default="1,4,4", help="Z,Y,X downsample strides")
    parser.add_argument("--max-windows-per-vol", type=int, default=10, help="Limit windows per movie for quick iteration")
    parser.add_argument("--num-volumes", type=int, default=None, help="Number of volumes to load (default: all)")
    parser.add_argument("--val-volumes", type=str, default="44b6_3a861e03,44b6_12dfb391", help="Comma-separated validation volumes")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save weights")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pre-trained weights (.pth)")
    parser.add_argument("--no-val", action="store_true", help="Skip validation evaluation during training")
    parser.add_argument("--val-max-frames", type=int, default=15, help="Max frames for validation evaluation")
    return parser.parse_args()


def custom_collate(batch):
    return batch


def evaluate_checkpoint(model, data_dir: Path, val_volumes: list[str], downsample: tuple, device: torch.device, val_max_frames: int | None = 15):
    """Evaluates the model on validation volumes using the official metric engine."""
    import importlib.util
    eval_path = Path(__file__).resolve().parent / "evaluate.py"
    spec = importlib.util.spec_from_file_location("local_evaluate", str(eval_path))
    local_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_eval)
    track_volume = local_eval.track_volume

    suite = BenchmarkSuite(train_dir=data_dir, max_matching_distance_um=7.0)
    scores = []
    
    print("\n" + "=" * 82)
    print("                 AUTOMATED VALIDATION EVALUATION HARNESS")
    print("=" * 82)
    
    for v_name in val_volumes:
        vol_path = data_dir / f"{v_name}.zarr"
        if not vol_path.exists():
            continue
        try:
            _, _, n_total = suite.load_gt(v_name)
            pred_graph, lat, vram = track_volume(
                models=[(model, device)],
                volume_dir=vol_path,
                device=device,
                downsample=downsample,
                window_size=2,
                det_threshold=0.50,
                edge_threshold=0.48,
                det_tta=False,
                n_total=n_total,
                min_track_length=4,
                max_frames=val_max_frames,
            )
            res = suite.evaluate_graph(pred_graph, v_name, lat, vram)
            scores.append(res.competition_score)
            print(f"  [Validation {v_name}]")
            print(f"    - Score             : {res.competition_score:.4f}")
            print(f"    - Adj Edge Jaccard  : {res.adj_edge_jaccard:.4f} (Raw: {res.raw_edge_jaccard:.4f})")
            print(f"    - Edge Counts       : TP={res.edge_tp}, FP={res.edge_fp}, FN={res.edge_fn}")
            print(f"    - Division Jaccard  : {res.div_jaccard:.4f} (TP={res.div_tp}, FP={res.div_fp}, FN={res.div_fn})")
            print(f"    - Census Multiplier : {res.census_multiplier:.4f}")
            print(f"    - Latency           : {lat:.2f}s | VRAM: {vram:.1f} MB")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [Validation {v_name}] Error during evaluation: {e}")

    print("=" * 82)
    if scores:
        mean_score = sum(scores) / len(scores)
        print(f"  >>> MEAN VALIDATION SCORE: {mean_score:.4f} (Kaggle Baseline: 0.9309) <<<")
        print("=" * 82 + "\n")
        return mean_score
    return None


def main():
    args = parse_args()
    downsample = tuple(int(x) for x in args.downsample.split(","))

    if args.data_dir is None:
        for p in [
            Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
            Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
        ]:
            if p.exists():
                args.data_dir = str(p)
                break
    if args.data_dir is None:
        raise FileNotFoundError("Could not find competition train directory!")

    if args.pretrained is None:
        for p in [
            Path("/kaggle/input/datasets/ragunathravi/forcompbiohub/secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth"),
            Path("/kaggle/input/forcompbiohub/secondary_seed_weights/unet_transformer/split_0/edge_predictor_best.pth"),
        ]:
            if p.exists():
                args.pretrained = str(p)
                break

    print("=" * 82)
    print("             BIO-DANT HIGH-PERFORMANCE TRAINING PIPELINE")
    print("=" * 82)
    print(f"  Data Directory        : {args.data_dir}")
    print(f"  Epochs                : {args.epochs}")
    print(f"  Batch Size            : {args.batch_size}")
    print(f"  Downsample (Z,Y,X)    : {downsample}")
    print(f"  Learning Rate         : {args.lr}")
    print(f"  Max Windows / Volume  : {args.max_windows_per_vol}")
    print(f"  Validation Volumes    : {args.val_volumes}")
    print("=" * 82)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Hardware: {device} ({torch.cuda.get_device_name(0) if num_gpus > 0 else 'CPU'}) | Visible GPUs: {num_gpus}")

    # 1. Dataset & DataLoader
    dataset = BiohubWindowDataset(
        data_dir=Path(args.data_dir),
        num_volumes=args.num_volumes,
        downsample=downsample,
        window_size=2,
        max_windows_per_vol=args.max_windows_per_vol,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0,  # Zero workers prevents Zarr/Blosc thread-fork deadlocks on Linux
        pin_memory=False,
    )

    # 2. Model
    model = AnisoTrack3D(
        unet_out_channels=args.unet_channels,
        unet_layers=[32, 64, 128],
        transformer_d_model=64,
        depthwise=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Bio-DANT Initialized: {param_count:,} trainable parameters.")

    if args.pretrained and Path(args.pretrained).exists():
        print(f"Loading pre-trained checkpoint from: {args.pretrained}")
        sd = torch.load(args.pretrained, map_location=device)
        u_sd = sd.get("unet_state_dict", sd)
        t_sd = sd.get("transformer_state_dict", None)
        u_miss, u_unexp = model.unet.load_state_dict(u_sd, strict=False)
        print(f"  UNet loaded (missing={len(u_miss)}, unexpected={len(u_unexp)})")
        if t_sd is not None:
            t_miss, t_unexp = model.transformer.load_state_dict(t_sd, strict=False)
            print(f"  Transformer loaded (missing={len(t_miss)}, unexpected={len(t_unexp)})")

    # 3. Loss & Optimizer
    loss_fn = AnisoTrackingLoss(
        det_loss_weight=10.0,
        det_neg_weight=0.1,
        subvoxel_loss_weight=0.5,
        advection_loss_weight=0.2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    val_volume_list = [v.strip() for v in args.val_volumes.split(",") if v.strip()]

    print("\nStarting active training loop...")
    print("-" * 88)
    print(f"{'Epoch':<6} | {'Step':<6} | {'Total Loss':<11} | {'Edge Loss':<10} | {'Det Loss':<9} | {'Adv Loss':<9} | {'Throughput':<11}")
    print("-" * 88)

    global_step = 0
    start_time = time.time()
    best_val_score = 0.0

    for epoch in range(args.epochs):
        model.train()
        print(f"\n--- Epoch {epoch+1}/{args.epochs} Starting: {len(dataloader)} batches to process ---", flush=True)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", file=sys.stdout, ncols=95, mininterval=0.5)
        for batch_idx, batch in enumerate(pbar):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            imgs = torch.stack([item["imgs"] for item in batch], dim=0).to(device)
            B, W, C, Z, Y, X = imgs.shape

            amp_enabled = (device.type == "cuda")
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp_enabled):
                feats, det_logits, sub_deltas, flows = model.encode(imgs, return_flows=True)

                batch_total_loss = 0.0
                edge_loss_sum = 0.0
                det_loss_sum = 0.0
                adv_loss_sum = 0.0

                for b in range(B):
                    item = batch[b]
                    c0 = item["coords0"].to(device)
                    c1 = item["coords1"].to(device)
                    target = item["target"].to(device)
                    scale = item["scale"].to(device)

                    ds_tensor = torch.tensor(downsample, device=device, dtype=torch.float32)
                    c0_um = c0 * ds_tensor * scale
                    c1_um = c1 * ds_tensor * scale

                    f0_map = feats[b, 0]
                    f1_map = feats[b, 1]
                    f0 = trilinear_index_features(f0_map, c0)
                    f1 = trilinear_index_features(f1_map, c1)

                    # Continuous velocity field prior at c0
                    v0_map = flows[0][0][b]  # (3, Z, Y, X)
                    v0_sampled = trilinear_index_features(v0_map, c0)  # (N, 3) in voxels
                    v0_um = v0_sampled * ds_tensor * scale

                    edge_logits, cand_mask = model.predict_edges(f0, c0_um, f1, c1_um, flow_src_um=v0_um)

                    gt_masks = [m.to(device) for m in item["peak_masks"]]
                    det_logs = [det_logits[0][b:b+1], det_logits[1][b:b+1]]
                    pred_dels = [sub_deltas[0][b:b+1], sub_deltas[1][b:b+1]]
                    # Use zero target deltas as proxy when exact sub-voxels are centered on grid
                    target_dels = [torch.zeros_like(pd) for pd in pred_dels]

                    b_flows = [(flows[0][0][b:b+1], flows[0][1][b:b+1]), (flows[1][0][b:b+1], flows[1][1][b:b+1])]
                    sample_loss, loss_dict = loss_fn(
                        edge_logits, target, det_logs, gt_masks,
                        pred_deltas_list=pred_dels,
                        target_deltas_list=target_dels,
                        cand_mask=cand_mask,
                        raw_imgs=imgs[b:b+1],
                        flows=b_flows,
                    )
                    batch_total_loss = batch_total_loss + sample_loss
                    edge_loss_sum += loss_dict["loss_edge"]
                    det_loss_sum += loss_dict["loss_det"]
                    adv_loss_sum += loss_dict.get("loss_adv", 0.0)

                loss = batch_total_loss / B

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            t_step = time.perf_counter() - t0
            throughput = B / max(t_step, 1e-4)

            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "edge": f"{edge_loss_sum/B:.3f}",
                "det": f"{det_loss_sum/B:.3f}",
                "adv": f"{adv_loss_sum/B:.3f}",
                "p/s": f"{throughput:.1f}"
            })
            pbar.write(f"Epoch {epoch+1:<2} | Step {global_step:<5} | Loss: {loss.item():.4f} (Edge: {edge_loss_sum/B:.4f}, Det: {det_loss_sum/B:.4f}, Adv: {adv_loss_sum/B:.4f}) | {throughput:.1f} p/s")
            sys.stdout.flush()

        # Save checkpoint per epoch
        raw_model = model.unet.module if hasattr(model.unet, "module") else model.unet
        ckpt_path = save_dir / f"biodant_epoch_{epoch+1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "unet_state_dict": raw_model.state_dict(),
            "transformer_state_dict": model.transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"\U0001f4be Saved checkpoint: {ckpt_path.name}")

        # Run Validation Evaluation immediately after epoch 1 and subsequent epochs
        if not args.no_val and val_volume_list:
            model.eval()
            with torch.no_grad():
                val_score = evaluate_checkpoint(model, Path(args.data_dir), val_volume_list, downsample, device, val_max_frames=args.val_max_frames)
                if val_score is not None and val_score > best_val_score:
                    best_val_score = val_score
                    best_ckpt = save_dir / "biodant_best.pth"
                    torch.save({
                        "epoch": epoch + 1,
                        "score": best_val_score,
                        "unet_state_dict": raw_model.state_dict(),
                        "transformer_state_dict": model.transformer.state_dict(),
                    }, best_ckpt)
                    print(f"\U0001f3c6 New Best Model Saved ({best_val_score:.4f}) -> {best_ckpt.name}!")
            model.train()

    total_time = time.time() - start_time
    print("-" * 88)
    print(f"\U0001f3c6 Training Complete! Finished {args.epochs} epochs in {total_time/60:.2f} minutes.")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()

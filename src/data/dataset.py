"""
Strided Zarr Dataset for AnisoTrack3D.
Reads 2 consecutive frames directly from Zarr with spatial downsampling at I/O time.
Normalizes intensity dynamically using precomputed quantiles.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset
from geff import GeffMetadata

import tracksdata as td
from biohub_tracking.io import open_dataset


@dataclass
class VideoMeta:
    name: str
    zarr_path: Path
    geff_path: Path
    image_shape: tuple
    downsample: tuple
    q_low: float
    q_high: float


def compute_gt_transition_matrix(ids_t: np.ndarray, ids_t1: np.ndarray, edge_attrs: pl.DataFrame) -> torch.Tensor:
    t_to_row = {nid: i for i, nid in enumerate(ids_t)}
    t1_to_col = {nid: i for i, nid in enumerate(ids_t1)}
    matrix = torch.zeros(len(ids_t), len(ids_t1), dtype=torch.float32)
    for s_id, t_id in zip(edge_attrs["source_id"], edge_attrs["target_id"]):
        if s_id in t_to_row and t_id in t1_to_col:
            matrix[t_to_row[s_id], t1_to_col[t_id]] = 1.0
    return matrix


class BiohubWindowDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        volume_names: Optional[List[str]] = None,
        downsample: Tuple[int, int, int] = (1, 4, 4),
        window_size: int = 2,
        max_windows_per_vol: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.downsample = downsample
        self.window_size = window_size

        if volume_names is None:
            geffs = sorted(list(self.data_dir.glob("*.geff")))
            volume_names = [g.stem for g in geffs]

        self.samples = []
        ds_arr = np.array(downsample, dtype=np.float32)

        print(f"Indexing {len(volume_names)} volumes for training dataset...")
        for name in volume_names:
            z_path = self.data_dir / f"{name}.zarr"
            g_path = self.data_dir / f"{name}.geff"
            if not z_path.exists() or not g_path.exists():
                continue

            try:
                ds = open_dataset(self.data_dir / name, require_tracks=True, load_image=False, device="cpu")
                q_low = float(ds.quantiles.get("0.001", 0.0))
                q_high = float(ds.quantiles.get("0.999", 2000.0))
                gt_attrs = ds.tracks.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
                edge_attrs = ds.tracks.edge_attrs(attr_keys=["source_id", "target_id"])

                T = ds.image_shape[0]
                n_added = 0
                for t in range(0, T - window_size + 1, window_size):
                    gt0 = gt_attrs.filter(pl.col("t") == t)
                    gt1 = gt_attrs.filter(pl.col("t") == t + 1)
                    if len(gt0) == 0 or len(gt1) == 0:
                        continue

                    coords0 = gt0.select(["z", "y", "x"]).to_numpy().astype(np.float32) / ds_arr
                    coords1 = gt1.select(["z", "y", "x"]).to_numpy().astype(np.float32) / ds_arr
                    ids0 = gt0["node_id"].to_numpy()
                    ids1 = gt1["node_id"].to_numpy()
                    target = compute_gt_transition_matrix(ids0, ids1, edge_attrs)

                    # Store sample descriptor
                    self.samples.append({
                        "name": name,
                        "zarr_path": str(z_path),
                        "t_start": t,
                        "q_low": q_low,
                        "q_high": q_high,
                        "coords0": coords0,
                        "coords1": coords1,
                        "target": target,
                        "scale": tuple(ds.scale),
                    })
                    n_added += 1
                    if max_windows_per_vol and n_added >= max_windows_per_vol:
                        break
            except Exception as e:
                continue

        print(f"Dataset ready: {len(self.samples)} frame-pair training windows!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        t = s["t_start"]
        W = self.window_size
        dz, dy, dx = self.downsample

        z = zarr.open_group(s["zarr_path"], mode="r")["0"]
        # Fast strided read
        raw = z[t : t + W, ::dz, ::dy, ::dx].astype(np.float32)
        q_l = s["q_low"]
        q_h = s["q_high"]
        imgs = torch.from_numpy((raw - q_l) / (q_h - q_l + 1e-6)).clamp(0.0)

        # Build peak ground truth masks
        W, Z, Y, X = imgs.shape
        peak_masks = []
        for c in [s["coords0"], s["coords1"]]:
            mask = torch.zeros((1, Z, Y, X), dtype=torch.float32)
            for cz, cy, cx in c:
                iz, iy, ix = int(round(cz)), int(round(cy)), int(round(cx))
                if 0 <= iz < Z and 0 <= iy < Y and 0 <= ix < X:
                    mask[0, iz, iy, ix] = 1.0
            peak_masks.append(mask)

        return {
            "imgs": imgs.unsqueeze(1), # (W, 1, Z, Y, X)
            "coords0": torch.from_numpy(s["coords0"]),
            "coords1": torch.from_numpy(s["coords1"]),
            "target": s["target"],
            "peak_masks": peak_masks,
            "scale": torch.tensor(s["scale"], dtype=torch.float32),
        }

"""FNO consumer for scan_runner datasets (solver -> scan -> catalog -> train).

End-to-end acceptance of the parameter-sweep execution chain: the FNO2d
super-resolution surrogate trains **directly on a scan dataset** — every
sample is loaded through the official catalog API
(:func:`tensorlbm.data.load_product` / :func:`load_product_arrays`), so
the run exercises the full loop with nothing provisional in between.

Pipeline
--------
1. read ``plan.json`` / ``dataset.json`` of a scan dataset directory;
2. for every registered product: load its arrays through the catalog,
   slice the 3-D velocity field to active 2-D spanwise planes (planes
   with negligible activity are dropped, mirroring the flagship demo),
   and resample each plane to a canonical fine grid (sweeps vary
   ``resolution``, so planes are canonicalised before pairing);
3. build coarse->fine super-resolution pairs (block-mean coarsening +
   bilinear re-upsampling as the input, the fine plane as the target);
4. split the pairs by **scan point** according to ``dataset.json``'s
   ``split_points`` (the dataset's own leakage-safe grouping — no DoE
   configuration contributes to two splits);
5. train a compact FNO2d and report the final loss, held-out relative
   L2, and the bilinear-upsampling baseline it must beat.

Usage::

    python examples/scan_fno_consumer.py \\
        --dataset /nfs/wangxi/datasets/scan_cavity_lhs32_20260820 \\
        --workdir /nfs/wangxi/as_scan_20260820/fno \\
        --device cuda:0 --epochs 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.ai.fno import FNO2d, FNO2dArch, save_fno2d
from tensorlbm.apps.ai4s_flagship import build_super_resolution_dataset, prediction_error_metrics
from tensorlbm.data.catalog import FieldDataCatalog
from tensorlbm.data.solver_export import load_product, load_product_arrays


def canonical_planes(
    velocity: np.ndarray,
    *,
    grid: int,
    min_activity: float,
    max_planes: int,
) -> list[torch.Tensor]:
    """Active spanwise planes of a 3-D velocity field, resampled to (grid, grid).

    ``velocity`` is ``(nz, ny, nx, 3)``; a plane is kept when its velocity
    magnitude has non-negligible variation (free-stream planes carry no
    learnable content).
    """
    mag = np.linalg.norm(velocity.astype(np.float32), axis=-1)  # (nz, ny, nx)
    planes = []
    for z in range(mag.shape[0]):
        plane = mag[z]
        activity = float(plane.std())
        if activity <= min_activity:
            continue
        t = torch.from_numpy(plane.copy())[None, None]  # (1, 1, ny, nx)
        if plane.shape != (grid, grid):
            t = torch.nn.functional.interpolate(
                t, size=(grid, grid), mode="bilinear", align_corners=False
            )
        planes.append(t[0, 0])
        if len(planes) >= max_planes:
            break
    return planes


def load_scan_pairs(
    dataset_dir: Path,
    *,
    grid: int,
    factor: int,
    min_activity_scale: float = 0.05,
    max_planes: int = 3,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
    """Load every split's coarse->fine pairs through the catalog API."""
    plan = json.loads((dataset_dir / "plan.json").read_text())
    info = json.loads((dataset_dir / "dataset.json").read_text())
    activity_threshold = None
    pairs_by_split: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    point_split = {pid: split for split, pids in info["split_points"].items() for pid in pids}
    catalog = FieldDataCatalog.open(dataset_dir / "catalog.db")
    n_products = 0
    try:
        for point_id, product_ids in sorted(info["products_by_point"].items()):
            split = point_split[point_id]
            for product_id in sorted(product_ids):
                product = load_product(catalog, product_id)
                arrays = load_product_arrays(product)
                velocity = arrays["velocity"]
                if activity_threshold is None:
                    # per-dataset threshold: a fraction of the global std
                    activity_threshold = min_activity_scale * float(
                        np.linalg.norm(velocity.astype(np.float32), axis=-1).std()
                    )
                planes = canonical_planes(
                    velocity,
                    grid=grid,
                    min_activity=activity_threshold,
                    max_planes=max_planes,
                )
                if not planes:
                    continue
                built = build_super_resolution_dataset(planes, factor=factor)
                for i in range(built["n_samples"]):
                    pairs_by_split[split].append((built["inputs"][i], built["targets"][i]))
                n_products += 1
    finally:
        catalog.close()
    out: dict[str, dict[str, torch.Tensor]] = {}
    for split, pairs in pairs_by_split.items():
        if not pairs:
            continue
        out[split] = {
            "inputs": torch.stack([p[0] for p in pairs]),
            "targets": torch.stack([p[1] for p in pairs]),
            "n_samples": len(pairs),
        }
    return out, {"products_consumed": n_products, "scan_id": plan["scan_id"]}


def train_fno(
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grid: int,
) -> tuple[FNO2d, list[float]]:
    arch = FNO2dArch(
        in_channels=1,
        out_channels=1,
        width=24,
        n_layers=4,
        modes_x=min(20, grid // 2 + 1),
        modes_y=min(20, grid // 2 + 1),
        mlp_hidden=64,
    )
    model = FNO2d(arch).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    x_all, y_all = train["inputs"].to(device), train["targets"].to(device)
    losses: list[float] = []
    n = x_all.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x_all[idx])
            loss = torch.nn.functional.mse_loss(pred, y_all[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item()) * len(idx)
        scheduler.step()
        losses.append(epoch_loss / n)
    return model, losses


@torch.no_grad()
def evaluate(
    model: FNO2d, data: dict[str, torch.Tensor], *, device: str, batch: int = 32
) -> dict[str, float]:
    model.eval()
    rel, bilinear, mse = [], [], []
    x, y = data["inputs"].to(device), data["targets"].to(device)
    for start in range(0, x.shape[0], batch):
        pred = model(x[start : start + batch])
        target = y[start : start + batch]
        for i in range(pred.shape[0]):
            rel.append(prediction_error_metrics(pred[i], target[i])["relative_l2"])
            bilinear.append(prediction_error_metrics(x[start + i], target[i])["relative_l2"])
        mse.append(float(torch.mean((pred - target) ** 2).item()))
    model.train()
    return {
        "mean_relative_l2": float(np.mean(rel)),
        "mean_bilinear_relative_l2": float(np.mean(bilinear)),
        "mse": float(np.mean(mse)),
        "n_samples": int(x.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="scan dataset directory")
    parser.add_argument("--workdir", required=True, help="output directory (nfs!)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid", type=int, default=64, help="canonical fine grid")
    parser.add_argument("--factor", type=int, default=2, help="SR downsample factor")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-planes", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(0)
    dataset_dir, workdir = Path(args.dataset), Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    splits, meta = load_scan_pairs(
        dataset_dir, grid=args.grid, factor=args.factor, max_planes=args.max_planes
    )
    t_data = time.perf_counter() - t0
    if "train" not in splits or "val" not in splits:
        print("need both train and val splits for the acceptance run", file=sys.stderr)
        return 1
    print(
        f"loaded {meta['products_consumed']} products from scan {meta['scan_id']}: "
        + ", ".join(f"{k}={v['n_samples']} pairs" for k, v in splits.items())
        + f" ({t_data:.1f}s)"
    )

    t0 = time.perf_counter()
    model, losses = train_fno(
        splits["train"],
        splits["val"],
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grid=args.grid,
    )
    t_train = time.perf_counter() - t0
    ckpt = save_fno2d(model, workdir / "scan_fno2d.pt")

    report = {
        "dataset_dir": str(dataset_dir),
        "scan_id": meta["scan_id"],
        "products_consumed": meta["products_consumed"],
        "pairs": {k: v["n_samples"] for k, v in splits.items()},
        "epochs": args.epochs,
        "loss_first": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "train_seconds": t_train,
        "data_seconds": t_data,
        "checkpoint": str(ckpt),
    }
    for split in ("val", "test"):
        if split in splits:
            report[split] = evaluate(model, splits[split], device=args.device)
    (workdir / "fno_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

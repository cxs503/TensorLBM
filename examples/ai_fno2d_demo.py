"""FNO2d surrogate demo for 2D cylinder flow.

This example trains a compact FNO2d model to approximate the final velocity
magnitude field of a short D2Q9 LBM cylinder-flow simulation, then compares
surrogate inference time against direct LBM simulation time.

Default settings are chosen for CPU/Colab practicality (~under 5 minutes):
- 64x64 grid
- 100 training samples + 20 validation samples
- modest training epochs and model width

Outputs (under --output-dir):
- loss_curve.png
- speed_comparison.csv
- speed_comparison.txt
- result_comparison.png
- fno2d_surrogate.pt (+ metadata json)
- summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from tensorlbm import (
    apply_simple_channel_boundaries,
    collide_bgk,
    cylinder_mask,
    equilibrium,
    macroscopic,
    make_channel_wall_mask,
    stream,
)
from tensorlbm.ai import FNO2d, FNO2dArch, save_fno2d


@dataclass(frozen=True)
class FlowCase:
    re: float
    u_in: float
    radius: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-val", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/ai_fno2d_demo")
    return parser


def sample_cases(total: int, *, seed: int, nx: int, ny: int) -> list[FlowCase]:
    g = torch.Generator().manual_seed(seed)
    re_values = torch.randint(80, 241, size=(total,), generator=g)
    u_values = 0.05 + 0.04 * torch.rand(total, generator=g)
    r_min = 0.08 * min(nx, ny)
    r_max = 0.15 * min(nx, ny)
    r_values = r_min + (r_max - r_min) * torch.rand(total, generator=g)
    return [
        FlowCase(
            re=float(re_values[i].item()),
            u_in=float(u_values[i].item()),
            radius=float(r_values[i].item()),
        )
        for i in range(total)
    ]


def _coord_channels(ny: int, nx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(0.0, 1.0, ny, device=device).view(ny, 1).expand(ny, nx)
    x = torch.linspace(0.0, 1.0, nx, device=device).view(1, nx).expand(ny, nx)
    return x, y


def run_lbm_case(
    case: FlowCase, *, nx: int, ny: int, n_steps: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    cx_obs, cy_obs = nx * 0.25, ny * 0.5
    obstacle = cylinder_mask(nx, ny, cx_obs, cy_obs, case.radius, device=device)
    wall_mask = make_channel_wall_mask(ny, nx, obstacle, device=device)

    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), case.u_in, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    ux0 = ux0.masked_fill(obstacle, 0.0)
    f = equilibrium(rho0, ux0, uy0, device=device)

    nu = case.u_in * 2.0 * case.radius / case.re
    tau = 3.0 * nu + 0.5

    for _ in range(n_steps):
        f = collide_bgk(f, tau)
        f = stream(f)
        f = apply_simple_channel_boundaries(
            f,
            u_in=case.u_in,
            wall_mask=wall_mask,
            obstacle_mask=obstacle,
        )

    rho, ux, uy = macroscopic(f)
    ux = ux.masked_fill(obstacle, 0.0)
    uy = uy.masked_fill(obstacle, 0.0)
    speed = torch.sqrt(ux * ux + uy * uy)
    return obstacle, speed


def make_sample_tensors(
    cases: list[FlowCase],
    *,
    nx: int,
    ny: int,
    n_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    x_ch, y_ch = _coord_channels(ny, nx, device)

    re_min, re_max = 80.0, 240.0
    u_min, u_max = 0.05, 0.09
    r_min, r_max = 0.08 * min(nx, ny), 0.15 * min(nx, ny)

    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    t0 = time.perf_counter()
    for case in cases:
        obstacle, speed = run_lbm_case(case, nx=nx, ny=ny, n_steps=n_steps, device=device)
        re_norm = torch.full((ny, nx), (case.re - re_min) / (re_max - re_min), device=device)
        u_norm = torch.full((ny, nx), (case.u_in - u_min) / (u_max - u_min), device=device)
        r_norm = torch.full((ny, nx), (case.radius - r_min) / (r_max - r_min), device=device)
        features = torch.stack([obstacle.float(), x_ch, y_ch, re_norm, u_norm, r_norm], dim=0)
        inputs.append(features)
        targets.append(speed.unsqueeze(0))
    data_time = time.perf_counter() - t0

    x = torch.stack(inputs, dim=0)
    y = torch.stack(targets, dim=0)
    return x, y, data_time


def train_fno(
    model: FNO2d,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[list[float], list[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_train = x_train.shape[0]
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=x_train.device)
        batch_losses: list[float] = []
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x_train[idx])
            loss = F.mse_loss(pred, y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = F.mse_loss(val_pred, y_val)

        train_losses.append(sum(batch_losses) / max(len(batch_losses), 1))
        val_losses.append(float(val_loss.item()))
        print(
            f"epoch {epoch + 1:03d}/{epochs:03d} "
            f"train_mse={train_losses[-1]:.6e} val_mse={val_losses[-1]:.6e}"
        )
    return train_losses, val_losses


def benchmark_speed(
    model: FNO2d,
    bench_cases: list[FlowCase],
    *,
    nx: int,
    ny: int,
    n_steps: int,
    device: torch.device,
) -> dict[str, float]:
    x_bench, _, lbm_total = make_sample_tensors(
        bench_cases,
        nx=nx,
        ny=ny,
        n_steps=n_steps,
        device=device,
    )

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x_bench)
    fno_total = time.perf_counter() - t0

    n_cases = float(len(bench_cases))
    return {
        "n_cases": n_cases,
        "lbm_total_s": lbm_total,
        "fno_total_s": fno_total,
        "lbm_avg_s": lbm_total / n_cases,
        "fno_avg_s": fno_total / n_cases,
        "speedup": lbm_total / max(fno_total, 1e-12),
    }


def save_loss_curve(path: Path, train_losses: list[float], val_losses: list[float]) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train mse")
    plt.plot(val_losses, label="val mse")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_comparison_figure(path: Path, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    y_t = y_true.squeeze().detach().cpu()
    y_p = y_pred.squeeze().detach().cpu()
    err = (y_p - y_t).abs()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    im0 = axes[0].imshow(y_t, origin="lower", cmap="viridis")
    axes[0].set_title("LBM speed")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(y_p, origin="lower", cmap="viridis")
    axes[1].set_title("FNO2d prediction")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(err, origin="lower", cmap="magma")
    axes[2].set_title("abs error")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_speed_table(path_csv: Path, path_txt: Path, speed: dict[str, float]) -> None:
    with path_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "total_s", "avg_s_per_case", "n_cases"])
        writer.writerow(
            ["LBM simulation", speed["lbm_total_s"], speed["lbm_avg_s"], int(speed["n_cases"])]
        )
        writer.writerow(
            ["FNO2d inference", speed["fno_total_s"], speed["fno_avg_s"], int(speed["n_cases"])]
        )

    lines = [
        "| Method | Total time (s) | Avg / case (s) | Cases |",
        "|---|---:|---:|---:|",
        f"| LBM simulation | {speed['lbm_total_s']:.4f} | {speed['lbm_avg_s']:.4f} | {int(speed['n_cases'])} |",
        f"| FNO2d inference | {speed['fno_total_s']:.4f} | {speed['fno_avg_s']:.4f} | {int(speed['n_cases'])} |",
        f"| Speedup (LBM/FNO) | {speed['speedup']:.2f}x | - | - |",
    ]
    path_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = args.n_train + args.n_val
    cases = sample_cases(total, seed=args.seed, nx=args.nx, ny=args.ny)
    train_cases = cases[: args.n_train]
    val_cases = cases[args.n_train :]

    print("Generating LBM training/validation samples...")
    x_train, y_train, train_gen_s = make_sample_tensors(
        train_cases,
        nx=args.nx,
        ny=args.ny,
        n_steps=args.n_steps,
        device=device,
    )
    x_val, y_val, val_gen_s = make_sample_tensors(
        val_cases,
        nx=args.nx,
        ny=args.ny,
        n_steps=args.n_steps,
        device=device,
    )

    arch = FNO2dArch(
        in_channels=6,
        out_channels=1,
        width=args.width,
        n_layers=args.n_layers,
        modes_x=min(args.modes, args.nx // 2 + 1),
        modes_y=min(args.modes, args.ny),
        mlp_hidden=96,
        activation="gelu",
    )
    model = FNO2d(arch).to(device)

    print("Training FNO2d surrogate...")
    train_losses, val_losses = train_fno(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    model_path = save_fno2d(model, out_dir / "fno2d_surrogate.pt")
    print(f"Saved model to {model_path}")

    print("Benchmarking LBM simulation vs FNO2d inference...")
    bench_count = min(10, len(val_cases))
    speed = benchmark_speed(
        model,
        val_cases[:bench_count],
        nx=args.nx,
        ny=args.ny,
        n_steps=args.n_steps,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        y_pred_val = model(x_val)

    loss_curve_path = out_dir / "loss_curve.png"
    speed_csv_path = out_dir / "speed_comparison.csv"
    speed_txt_path = out_dir / "speed_comparison.txt"
    comparison_path = out_dir / "result_comparison.png"

    save_loss_curve(loss_curve_path, train_losses, val_losses)
    save_speed_table(speed_csv_path, speed_txt_path, speed)
    save_comparison_figure(comparison_path, y_val[0], y_pred_val[0])

    summary = {
        "grid": {"nx": args.nx, "ny": args.ny},
        "dataset": {
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_steps_per_case": args.n_steps,
            "train_generation_s": train_gen_s,
            "val_generation_s": val_gen_s,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "final_train_mse": train_losses[-1],
            "final_val_mse": val_losses[-1],
        },
        "speed": speed,
        "artifacts": {
            "loss_curve": str(loss_curve_path),
            "speed_table_csv": str(speed_csv_path),
            "speed_table_txt": str(speed_txt_path),
            "result_comparison": str(comparison_path),
            "model": str(model_path),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSpeed comparison")
    print((out_dir / "speed_comparison.txt").read_text(encoding="utf-8"))
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()

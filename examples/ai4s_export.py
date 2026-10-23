"""AI4S solver→data export: run small SUBOFF cases, export, register, verify.

This example closes the first break of the AI4S loop end to end: it runs
several small SUBOFF bare-hull configurations with the public solver
operators, periodically writes field snapshots with
:func:`tensorlbm.data.solver_export.save_fields_hdf5`, registers every
snapshot as a PASS-gated ``FieldDataProductR2`` in a SQLite
:class:`~tensorlbm.data.catalog.FieldDataCatalog`, and finishes with a
training-side self-check (metadata query → product reconstruction →
tensor loading → shape/dtype assertions, plus a leakage-safe
``FieldDatasetR2`` assembly with a training-input fingerprint).

Usage::

    # pilot sweep (9 configs x 3 snapshots) into a dataset directory
    CUDA_VISIBLE_DEVICES=6 python examples/ai4s_export.py \\
        --output-root /nfs/wangxi/datasets/pilot_suboff_20260820 --device cuda

    # tiny CPU smoke run (one config, 40 steps)
    python examples/ai4s_export.py --smoke --output-root /tmp/ai4s_smoke

The whole sweep uses the same step chain as the SUBOFF production runner
(collide → stream → far-field BC incl. bounce-back → every-10-step mass
correction), composed from public operators; only the export/registration
calls are new.  ``--compile-mode`` optionally routes the whole step
through :func:`tensorlbm.compile_utils.compile_step` (the shared #180
route); the default keeps it eager.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tensorlbm import (
    SuboffHullType,
    build_suboff_mask,
    collide_advanced_3d,
    collide_bgk3d,
    correct_mass3d,
    equilibrium3d,
    far_field_bc_3d,
    macroscopic3d,
    stream3d,
)
from tensorlbm.compile_utils import compile_step, validate_compile_mode
from tensorlbm.data.catalog import FieldDataCatalog
from tensorlbm.data.field_dataset_r2 import FieldDatasetR2, FieldSampleRefR2
from tensorlbm.data.solver_export import (
    load_product,
    load_product_arrays,
    register_product,
    save_fields_hdf5,
)
from tensorlbm.utils import get_reproducibility_metadata


@dataclass(frozen=True)
class PilotConfig:
    """One pilot sweep entry: a SUBOFF bare-hull channel case."""

    case: str
    nx: int
    u_in: float
    re: float
    collision: str = "BGK"
    n_steps: int = 360
    snapshot_every: int = 120

    @property
    def ny(self) -> int:
        return self.nx // 2

    @property
    def nz(self) -> int:
        return self.nx // 2

    @property
    def hull_length(self) -> float:
        return 0.6 * self.nx

    @property
    def nu(self) -> float:
        return self.u_in * self.hull_length / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5


#: Pilot sweep: 9 small configs varying Re / u_in / grid size / collision.
PILOT_SWEEP: tuple[PilotConfig, ...] = (
    PilotConfig("suboff-re100-nx64", 64, 0.05, 100.0, "BGK"),
    PilotConfig("suboff-re180-nx64", 64, 0.08, 180.0, "BGK"),
    PilotConfig("suboff-re140-nx96", 96, 0.05, 140.0, "BGK"),
    PilotConfig("suboff-re260-nx96", 96, 0.08, 260.0, "CM"),
    PilotConfig("suboff-re320-nx96", 96, 0.10, 320.0, "CM"),
    PilotConfig("suboff-re180-nx128", 128, 0.05, 180.0, "CM"),
    PilotConfig("suboff-re300-nx128", 128, 0.08, 300.0, "CM"),
    PilotConfig("suboff-re420-nx128", 128, 0.10, 420.0, "CUMULANT"),
    PilotConfig("suboff-re220-nx96-cum", 96, 0.06, 220.0, "CUMULANT"),
)

SMOKE_SWEEP: tuple[PilotConfig, ...] = (
    PilotConfig("smoke-suboff", 32, 0.05, 60.0, "BGK", n_steps=40, snapshot_every=20),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run small SUBOFF cases and export registered field products."
    )
    parser.add_argument("--output-root", required=True, help="Dataset output directory")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument("--smoke", action="store_true", help="single tiny CPU-scale run")
    parser.add_argument(
        "--n-steps", type=int, default=None, help="override steps per config (default per sweep)"
    )
    parser.add_argument(
        "--compile-mode",
        default=None,
        choices=[None, "default", "max-autotune-no-cudagraphs"],
        help="optional torch.compile route for the whole step (default: eager)",
    )
    return parser


def _code_sha() -> str:
    meta = get_reproducibility_metadata()
    sha = meta.get("git_commit")
    if not isinstance(sha, str) or len(sha) != 40:
        raise RuntimeError(
            "cannot determine code_sha: run from a git checkout of TensorLBM "
            "(git rev-parse HEAD failed)"
        )
    return sha


def run_config(
    config: PilotConfig,
    *,
    output_root: Path,
    catalog: FieldDataCatalog,
    device: str,
    code_sha: str,
    compile_mode: str | None,
) -> list[str]:
    """Run one SUBOFF config, export snapshots, and register them.

    Returns the registered product ids.
    """
    run_id = f"pilot-{config.case}"
    h5_path = output_root / f"{run_id}.h5"
    tau = config.tau
    if tau <= 0.505:
        raise ValueError(
            f"{config.case}: tau={tau:.4f} too close to the stability limit; lower Re or raise u_in"
        )

    solid, _stats = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        cx=config.nx * 0.35,
        cy=config.ny / 2.0,
        cz=config.nz / 2.0,
        length=config.hull_length,
        device=device,
    )
    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    initial_mass = float(f.sum().item())

    def make_step(with_mass_correction: bool):
        def _step(f: torch.Tensor) -> torch.Tensor:
            if config.collision == "BGK":
                f = collide_bgk3d(f, tau)
            else:
                f = collide_advanced_3d("D3Q19", config.collision, f, tau=tau)
            f = stream3d(f)
            f = far_field_bc_3d(f, config.u_in, obstacle_mask=solid)
            if with_mass_correction:
                f = correct_mass3d(f, initial_mass)
            return f

        return _step

    if compile_mode is not None:
        validate_compile_mode(compile_mode)
    plain_step = compile_step(make_step(False), compile_mode)
    mass_step = compile_step(make_step(True), compile_mode)

    metadata_base = {
        "run_id": run_id,
        "case": config.case,
        "code_sha": code_sha,
        "collision": config.collision,
        "lattice": "D3Q19",
        "boundary_type": "farfield",
        "device": device,
        "re": config.re,
        "u_in": config.u_in,
        "nu": config.nu,
        "tau": tau,
        "nx": config.nx,
        "ny": config.ny,
        "nz": config.nz,
        "hull_length": config.hull_length,
        "n_steps": config.n_steps,
        "snapshot_every": config.snapshot_every,
    }

    product_ids: list[str] = []
    for step in range(1, config.n_steps + 1):
        f = mass_step(f) if step % 10 == 0 else plain_step(f)
        if step % config.snapshot_every == 0 or step == config.n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            attrs = dict(metadata_base)
            attrs["step"] = step
            save_fields_hdf5(
                h5_path,
                {"rho": rho, "ux": ux, "uy": uy, "uz": uz, "solid_mask": solid},
                attrs,
            )
            product_ids.append(register_product(catalog, h5_path, attrs))
            print(f"  [{run_id}] step {step}: registered {product_ids[-1]}", flush=True)
        if step % 50 == 0 or step == config.n_steps:
            if not bool(torch.isfinite(f).all().item()):
                print(f"  [{run_id}] non-finite populations at step {step}; aborting run")
                break
    return product_ids


def training_side_self_check(
    catalog: FieldDataCatalog, sweep: tuple[PilotConfig, ...], product_ids: list[str]
) -> dict[str, object]:
    """Query, reconstruct, and load every product the way training would."""
    total = catalog.count_assets(kind="field_product")
    for config in sweep:
        found = catalog.find_assets_by_metadata("case", config.case, kind="field_product")
        assert found, f"no catalog assets found for case {config.case}"
    checked = 0
    for product_id in product_ids:
        product = load_product(catalog, product_id)
        arrays = load_product_arrays(product)
        manifest_shape = {array.array_id: array.shape for array in product.arrays}
        for array_id, arr in arrays.items():
            assert arr.shape == manifest_shape[array_id], (array_id, arr.shape)
        assert arrays["velocity"].dtype == np.dtype("<f4")
        assert arrays["solid_mask"].dtype == np.dtype("<i4")
        meta = {record.key: record.value for record in catalog.get_metadata(product_id)}
        expected = (int(meta["nz"]), int(meta["ny"]), int(meta["nx"]))
        assert arrays["rho"].shape == expected, (arrays["rho"].shape, expected)
        assert arrays["velocity"].shape == expected + (3,)
        checked += 1
    return {"products_registered": total, "products_verified": checked}


def assemble_field_dataset(
    catalog: FieldDataCatalog, sweep: tuple[PilotConfig, ...]
) -> FieldDatasetR2:
    """Assemble a leakage-safe FieldDatasetR2 grouped by case (train/val/test)."""
    split_of = {}
    for index, config in enumerate(sweep):
        split_of[config.case] = "train" if index < 6 else ("val" if index < 8 else "test")
    samples = []
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for config in sweep:
        for asset in catalog.find_assets_by_metadata("case", config.case, kind="field_product"):
            product = load_product(catalog, asset.asset_id)
            samples.append(
                FieldSampleRefR2(
                    sample_id=asset.asset_id,
                    product=product,
                    group_id=config.case,
                    source_case_id=config.case,
                    source_trajectory_id=product.run_manifest.run_id,
                )
            )
            splits[split_of[config.case]].append(asset.asset_id)
    return FieldDatasetR2(
        dataset_id="pilot-suboff-solver-export",
        version="1.0.0",
        task_name="field_reconstruction",
        samples=tuple(samples),
        splits={split: tuple(ids) for split, ids in splits.items()},
        lineage={"export_module": "tensorlbm.data.solver_export"},
    )


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep = SMOKE_SWEEP if args.smoke else PILOT_SWEEP
    if args.n_steps is not None:
        sweep = tuple(
            PilotConfig(
                cfg.case,
                cfg.nx,
                cfg.u_in,
                cfg.re,
                cfg.collision,
                args.n_steps,
                min(cfg.snapshot_every, args.n_steps),
            )
            for cfg in sweep
        )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable; falling back to CPU", file=sys.stderr)
    torch.manual_seed(0)

    code_sha = _code_sha()
    catalog = FieldDataCatalog.open(output_root / "catalog.db")
    all_product_ids: list[str] = []
    try:
        for config in sweep:
            print(
                f"running {config.case}: nx={config.nx} ({config.nz}x{config.ny}x{config.nx}), "
                f"u_in={config.u_in}, Re={config.re:g}, {config.collision}, "
                f"tau={config.tau:.4f}",
                flush=True,
            )
            all_product_ids.extend(
                run_config(
                    config,
                    output_root=output_root,
                    catalog=catalog,
                    device=device,
                    code_sha=code_sha,
                    compile_mode=args.compile_mode,
                )
            )
        if not all_product_ids:
            print("no snapshots registered; aborting self-check", file=sys.stderr)
            return 1

        summary = training_side_self_check(catalog, sweep, all_product_ids)
        dataset = assemble_field_dataset(catalog, sweep)
        fingerprint = dataset.training_input_fingerprint()
        print(
            f"self-check: {summary['products_verified']}/{summary['products_registered']} "
            f"products queried+loaded; FieldDatasetR2 with {len(dataset.samples)} samples "
            f"(train {len(dataset.splits['train'])} / val {len(dataset.splits['val'])} / "
            f"test {len(dataset.splits['test'])}); fingerprint {fingerprint[:16]}"
        )

        report = {
            "dataset_id": dataset.dataset_id,
            "training_input_fingerprint": fingerprint,
            "split_counts": {k: len(v) for k, v in dataset.splits.items()},
            "products": all_product_ids,
            "device": device,
            "code_sha": code_sha,
        }
        (output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {output_root / 'summary.json'}")
    finally:
        catalog.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

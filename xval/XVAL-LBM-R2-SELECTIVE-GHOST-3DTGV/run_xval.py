#!/usr/bin/env python3
"""XVAL R2: selective (lbmpy PackInfo) fused ghost exchange + corrected 3D TGV anchor.

What this driver does
---------------------
Runs the predeclared matrix of ``predeclared.json``:

* ``kind="xval"``    -- monolithic reference plus the 2x2 arm ablation
  (all-19 vs selective x fresh vs persistent buffer) of the x-slab D3Q19 ghost
  exchange, on a gloo ring of ``ranks`` processes;
* ``kind="shipped"`` -- the *product* path: ``tensorlbm.multi_gpu.D3Q19GlooTransport``
  with ``selective_ghosts=True`` driven by the same TGV case, so the transplant
  is measured through the code a user actually calls (m18155 audit finding 2).

Every run reports the velocity field against the **corrected** 3D Taylor-Green
reference -- ``e^{-3 nu k^2 t}`` (audit finding 3), not the round-1 two
dimensional ``e^{-2 nu k^2 t}`` -- plus the fitted decay rate, so viscosity is
pinned independently of the anchor.

Usage
-----
    python xval/XVAL-LBM-R2-SELECTIVE-GHOST-3DTGV/run_xval.py --out artifacts
    python .../run_xval.py --only R2-lin-u001 --only R2-product-n32-r4

``--out`` receives ``results.json``, ``table.csv``, ``gates.json`` and (optionally,
``--keep-fields``) the final fields for figure generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.multi_gpu import D3Q19GlooTransport  # noqa: E402
from tensorlbm.xval_mpi_ghost_fused import campaign  # noqa: E402
from tensorlbm.xval_mpi_ghost_fused.case import TGV, tgv_error  # noqa: E402

PREDECLARED = json.loads((HERE / "predeclared.json").read_text())


def checkpoints_for(steps: int) -> List[int]:
    return [steps // 4, steps // 2, (3 * steps) // 4, steps]


def case_for(cfg: dict) -> TGV:
    return TGV(
        n=cfg["n"],
        u0=cfg["u0"],
        tau=cfg["tau"],
        dtype=torch.float64 if cfg["dtype"] == "float64" else torch.float32,
    )


# ---------------------------------------------------------------------------
# kind="shipped": the product path
# ---------------------------------------------------------------------------
def _shipped_worker(rank: int, world_size: int, cfg: dict, out_dir: str) -> None:
    import torch.distributed as dist

    torch.set_num_threads(1)
    init_file = os.path.join(out_dir, f"shipped_init_{world_size}")
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        case = case_for(cfg)
        steps = int(cfg["steps"])
        full = case.initial_field()
        x0, x1 = campaign.slab_bounds(case.n, rank, world_size)
        owned = full[..., x0:x1].contiguous()
        transport = D3Q19GlooTransport(selective_ghosts=True, persistent_ghost_buffer=True)
        started = time.perf_counter()
        for _ in range(steps):
            owned = transport.step(owned, case.collide)
        elapsed = time.perf_counter() - started
        gathered = [torch.empty_like(owned) for _ in range(world_size)]
        dist.all_gather(gathered, owned.contiguous())
        if rank == 0:
            torch.save(
                {
                    "field": torch.cat(gathered, dim=-1),
                    "cfg": cfg,
                    "step_ms": 1000.0 * elapsed / steps,
                    "mlups": case.n**3 * steps / elapsed / 1e6,
                    "bytes_per_step": transport.last_exchange_bytes,
                    "selective": transport.selective_ghosts,
                    "crossing": [list(transport.crossing_right), list(transport.crossing_left)],
                },
                os.path.join(out_dir, f"shipped_{cfg['tag']}.pt"),
            )
    finally:
        dist.destroy_process_group()


def run_shipped(cfg: dict, out_dir: str) -> dict:
    import torch.multiprocessing as mp

    os.makedirs(out_dir, exist_ok=True)
    case = case_for(cfg)
    steps = int(cfg["steps"])
    serial = campaign.run_serial(case, steps, checkpoints_for(steps))
    init_file = os.path.join(out_dir, f"shipped_init_{cfg['ranks']}")
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(_shipped_worker, args=(cfg["ranks"], cfg, out_dir), nprocs=cfg["ranks"], join=True)
    payload = torch.load(
        os.path.join(out_dir, f"shipped_{cfg['tag']}.pt"), weights_only=False
    )
    field = payload.pop("field")
    metrics = tgv_error(field, case, steps)
    if torch.is_tensor(serial.field):
        torch.save(serial.field, os.path.join(out_dir, f"serial_field_{cfg['tag']}.pt"))
    return {
        "tag": cfg["tag"],
        "kind": "shipped",
        "cfg": cfg,
        "product": {
            **payload,
            **metrics,
            "identity_max_abs_vs_serial": float((field - serial.field).abs().max()),
        },
        "serial": {
            "step_ms": serial.step_ms,
            "mlups": serial.mlups,
            **tgv_error(serial.field, case, steps),
        },
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _arm_rows(result: dict) -> List[dict]:
    rows = []
    cfg = result["cfg"]
    serial = result["serial"]
    if result["kind"] == "shipped":
        product = result["product"]
        rows.append(
            {
                "run": result["tag"], "n": cfg["n"], "ranks": cfg["ranks"], "u0": cfg["u0"],
                "tau": cfg["tau"], "dtype": cfg["dtype"], "steps": cfg["steps"],
                "arm": "product_selective_persistent", "kind": "shipped",
                "step_ms": f"{product['step_ms']:.4f}", "exchange_ms": "",
                "mlups": f"{product['mlups']:.2f}",
                "bytes_per_step": product["bytes_per_step"],
                "tgv_rel_l2": f"{product['tgv_rel_l2']:.6e}",
                "tgv_amplitude_ratio": f"{product['tgv_amplitude_ratio']:.6f}",
                "fitted_decay_rate": f"{product['fitted_decay_rate']:.6e}",
                "analytic_decay_rate": f"{product['analytic_decay_rate']:.6e}",
                "decay_rate_rel_error": f"{product['decay_rate_rel_error']:.4%}",
                "identity_max_abs_vs_serial": f"{product['identity_max_abs_vs_serial']:.3e}",
                "rho_min": f"{product['rho_min']:.6f}", "rho_max": f"{product['rho_max']:.6f}",
            }
        )
    rows.append(
        {
            "run": result["tag"], "n": cfg["n"], "ranks": cfg["ranks"], "u0": cfg["u0"],
            "tau": cfg["tau"], "dtype": cfg["dtype"], "steps": cfg["steps"],
            "arm": "serial", "kind": result["kind"],
            "step_ms": f"{serial['step_ms']:.4f}", "exchange_ms": "",
            "mlups": f"{serial['mlups']:.2f}", "bytes_per_step": 0,
            "tgv_rel_l2": f"{serial['tgv_rel_l2']:.6e}",
            "tgv_amplitude_ratio": f"{serial['tgv_amplitude_ratio']:.6f}",
            "fitted_decay_rate": f"{serial['fitted_decay_rate']:.6e}",
            "analytic_decay_rate": f"{serial['analytic_decay_rate']:.6e}",
            "decay_rate_rel_error": f"{serial['decay_rate_rel_error']:.4%}",
            "identity_max_abs_vs_serial": "",
            "rho_min": f"{serial['rho_min']:.6f}", "rho_max": f"{serial['rho_max']:.6f}",
        }
    )
    for name, arm in result.get("arms", {}).items():
        rows.append(
            {
                "run": result["tag"], "n": cfg["n"], "ranks": cfg["ranks"], "u0": cfg["u0"],
                "tau": cfg["tau"], "dtype": cfg["dtype"], "steps": cfg["steps"],
                "arm": name, "kind": result["kind"],
                "step_ms": f"{arm['step_ms']:.4f}", "exchange_ms": f"{arm['exchange_ms']:.4f}",
                "mlups": f"{arm['mlups']:.2f}", "bytes_per_step": arm["bytes_per_step"],
                "tgv_rel_l2": f"{arm['tgv_rel_l2']:.6e}",
                "tgv_amplitude_ratio": f"{arm['tgv_amplitude_ratio']:.6f}",
                "fitted_decay_rate": f"{arm['fitted_decay_rate']:.6e}",
                "analytic_decay_rate": f"{arm['analytic_decay_rate']:.6e}",
                "decay_rate_rel_error": f"{arm['decay_rate_rel_error']:.4%}",
                "identity_max_abs_vs_serial": f"{arm['identity_max_abs_vs_serial']:.3e}",
                "rho_min": f"{arm['rho_min']:.6f}", "rho_max": f"{arm['rho_max']:.6f}",
            }
        )
    return rows


def evaluate_gates(results: List[dict]) -> dict:
    thresholds = PREDECLARED["predeclared_thresholds"]
    owner = thresholds["owner_acceptance_pct"] / 100.0
    identity_tol = 0.0
    byte_ratio = thresholds["byte_ratio_expected"]
    byte_tol = thresholds["byte_ratio_rel_tol"]

    identity_failures, physics_rows, byte_rows, rate_rows = [], [], [], []
    for result in results:
        tag = result["tag"]
        for name, arm in result.get("arms", {}).items():
            if float(arm["identity_max_abs_vs_serial"]) != identity_tol:
                identity_failures.append((tag, name, arm["identity_max_abs_vs_serial"]))
            physics_rows.append((tag, name, arm["tgv_rel_l2"]))
            rate_rows.append((tag, name, arm["decay_rate_rel_error"]))
        if result["kind"] == "shipped":
            product = result["product"]
            if float(product["identity_max_abs_vs_serial"]) != identity_tol:
                identity_failures.append((tag, "product", product["identity_max_abs_vs_serial"]))
            physics_rows.append((tag, "product", product["tgv_rel_l2"]))
            rate_rows.append((tag, "product", product["decay_rate_rel_error"]))
        serial = result["serial"]
        physics_rows.append((tag, "serial", serial["tgv_rel_l2"]))
        rate_rows.append((tag, "serial", serial["decay_rate_rel_error"]))
        for name, arm in result.get("arms", {}).items():
            if name == "selective_persistent":
                shipped_bytes = result["arms"]["full_alloc"]["bytes_per_step"]
                ratio = arm["bytes_per_step"] / shipped_bytes if shipped_bytes else float("nan")
                byte_rows.append((tag, ratio))

    worst_physics = max(physics_rows, key=lambda item: item[2])
    # The U0 ladder is the nonlinearity axis: it must decrease monotonically in U0.
    ladder = []
    for tag in ("R2-lin-u001", "R2-lin-u00025", "R2-n32-r2-f64"):
        for result in results:
            if result["tag"] == tag:
                ladder.append((result["cfg"]["u0"], result["serial"]["tgv_rel_l2"]))
    ladder_sorted = sorted(ladder)
    monotone = all(
        later[1] <= earlier[1] for earlier, later in zip(ladder_sorted, ladder_sorted[1:])
    )
    linear_rate_ok = [
        (tag, name, err)
        for tag, name, err in rate_rows
        if tag in ("R2-lin-u001", "R2-lin-u00025", "R2-tau060", "R2-product-n32-r4")
        for err in [abs(err)]
        if err > thresholds["decay_rate_rel_tol_pct"] / 100.0
    ]
    return {
        "H1_identity_exact_zero": {"pass": not identity_failures, "failures": identity_failures},
        "H2_byte_ratio_5_of_19": {
            "pass": all(abs(ratio - byte_ratio) <= byte_tol for _, ratio in byte_rows),
            "measured": byte_rows,
            "expected": byte_ratio,
        },
        "H3_physics_under_15pct": {
            "pass": worst_physics[2] < owner,
            "worst": worst_physics,
            "all": sorted(physics_rows, key=lambda item: -item[2]),
        },
        "H4_regime_monotone_in_u0": {"pass": monotone, "ladder_u0_rel_l2": ladder_sorted},
        "H5_decay_rate_within_5pct_linear_regime": {
            "pass": not linear_rate_ok, "violations": linear_rate_ok,
        },
        "thresholds": thresholds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "artifacts"))
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--keep-fields", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix: Dict[str, dict] = PREDECLARED["predeclared_run_matrix"]
    tags = args.only or list(matrix)
    results: List[dict] = []
    for tag in tags:
        cfg = dict(matrix[tag])
        cfg["tag"] = tag
        cfg["checkpoints"] = checkpoints_for(int(cfg["steps"]))
        started = time.perf_counter()
        if cfg["kind"] == "shipped":
            result = run_shipped(cfg, str(out_dir))
        else:
            result = campaign.run_config(cfg, str(out_dir))
        result["wall_seconds"] = time.perf_counter() - started
        case = case_for(cfg)
        result["derived"] = {
            "nu": case.nu,
            "k": case.k,
            "mach": case.mach,
            "re_k": case.re_k,
            "laplacian_eigenvalue_discrete": case.laplacian_eigenvalue(True),
            "laplacian_eigenvalue_continuum": case.laplacian_eigenvalue(False),
            "viscous_rate": case.viscous_rate,
            "eigenvalue_gap_rel": 1.0
            - case.laplacian_eigenvalue(True) / case.laplacian_eigenvalue(False),
            "t_end": float(cfg["steps"]),
        }
        results.append(result)
        print(
            f"[{tag}] kind={cfg['kind']} relL2={result['serial']['tgv_rel_l2']:.4%} "
            f"wall={result['wall_seconds']:.1f}s",
            flush=True,
        )

    # merge with any previously finished tags in this directory (resumable runs)
    existing = out_dir / "results.json"
    merged = results
    if existing.exists() and args.only:
        previous = json.loads(existing.read_text())
        keep = {item["tag"]: item for item in previous}
        for item in results:
            keep[item["tag"]] = item
        merged = [keep[tag] for tag in matrix if tag in keep]
    (out_dir / "results.json").write_text(json.dumps(merged, indent=2, default=str))
    rows = [row for result in merged for row in _arm_rows(result)]
    with (out_dir / "table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gates = evaluate_gates(merged)
    (out_dir / "gates.json").write_text(json.dumps(gates, indent=2))
    if not args.keep_fields:
        for path in out_dir.glob("fields_*.pt"):
            path.unlink()
    print(json.dumps({key: value.get("pass") for key, value in gates.items() if isinstance(value, dict)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
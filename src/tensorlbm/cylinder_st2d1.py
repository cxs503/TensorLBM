"""Schäfer–Turek 2D-1: in-channel cylinder at Re = 20 (steady laminar) — reusable case.

This module is the *library* entry for the benchmark case pool entry
``CASE-CYLINDER-DRAG-RE20``.  It contains the full D2Q9 solver chain (no
hand-written collision/streaming/equilibrium — library primitives only):

    collide_mrt  →  pre-collision skip on solid cells  →  stream
    →  Zou/He velocity inlet (parabolic profile)  →  Zou/He pressure outlet (ρ=1)
    →  half-way bounce-back on channel walls  →  Ladd momentum-exchange force on
       the cylinder surface  →  half-way bounce-back on the cylinder

which is bit-identical to the historical verified runner
``benchmarks/verified/cylinder/re20_st/run.py`` (mask radius = R, no mass
renormalisation).  On top of that chain this module adds what the benchmark
case needs and the runner lacked:

* a **plateau (steady-state) detector** so a reported mean is never taken over
  a window that is still drifting;
* **reference comparison** (:func:`compare_to_reference`) with the documented
  provenance of every anchor number;
* **artifact emission** (``results.json`` + ``series.csv`` + ``fields.npz``) so
  a run is auditable and re-plottable without re-running the solver.

Reference values (provenance is part of the data, see :data:`REFERENCE`):
the case is Schäfer & Turek (1996) 2D-1 and the drag anchor is the official
reference dataset for that case (spectral solution, Nabh 1998) as recorded in
the repository anchor ``benchmarks/verified/cylinder/re20_st/result.json``.
The printed table of the 1996 paper is *not* reachable offline from this
sandbox, so the anchor is taken from that in-repo verified record (sha256 is
pinned in ``benchmarks/case_pool/cylinder_drag_re20/refs.json``) rather than
quoted from the print original.

Usage (CLI)::

    PYTHONPATH=src python -m tensorlbm.cylinder_st2d1 --D 40 --steps 200000 \
        --out artifacts/case_pool/cylinder_drag_re20/D40

Library::

    from tensorlbm.cylinder_st2d1 import run_cylinder_st2d1, compare_to_reference
    res = run_cylinder_st2d1(D_cells=20, steps=60000, device="cpu")
    verdict = compare_to_reference(res, tolerance_pct=5.0)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from .d2q9 import equilibrium, macroscopic
from .solver import collide_mrt, stream

__all__ = [
    "CASE_NAME",
    "GEOMETRY",
    "REFERENCE",
    "case_geometry",
    "compare_to_reference",
    "result_manifest",
    "run_cylinder_st2d1",
]

CASE_NAME = "cylinder_st2d1"

#: Physical (dimensionless) set-up of Schäfer–Turek 2D-1.
GEOMETRY: dict[str, float] = {
    "lx": 2.2,  # channel length
    "ly": 0.41,  # channel height
    "cx": 0.2,  # cylinder centre x
    "cy": 0.2,  # cylinder centre y
    "diameter": 0.1,  # cylinder diameter D
    "u_mean": 0.2,  # inlet mean velocity (pore/mean), U_max = 1.5 U_mean
    "nu": 0.001,  # kinematic viscosity → Re = u_mean D / nu = 20
    "re": 20.0,
}

#: Drag/lift anchors and their provenance chain.
REFERENCE: dict[str, Any] = {
    "cd": 5.57953523384,
    "cl": 0.010618948146,
    "case": "Schäfer & Turek (1996) 2D-1, in-channel cylinder, Re=20, steady",
    "source": (
        "Official reference dataset for the Schäfer-Turek 2D-1 case (spectral "
        "solution, Nabh 1998) as recorded in-repo at "
        "benchmarks/verified/cylinder/re20_st/result.json:ref_cd/ref_cl"
    ),
    "source_kind": "in-repo verified anchor (secondary; print original not reachable offline)",
    "anchor_file": "benchmarks/verified/cylinder/re20_st/result.json",
    "normalisation": "Cd = Fx / (0.5 * rho * U_mean^2 * D_phys), D_phys = 0.1",
    "tolerance_pct": 5.0,
}


def case_geometry(D_cells: int) -> dict[str, float]:
    """Lattice discretisation of 2D-1 for a cylinder spanning *D_cells* cells."""
    if D_cells < 4:
        raise ValueError(f"D_cells must be >= 4, got {D_cells}")
    dx = GEOMETRY["diameter"] / D_cells
    nx = int(round(GEOMETRY["lx"] / dx))
    ny = int(round(GEOMETRY["ly"] / dx))
    return {
        "D_cells": int(D_cells),
        "dx": dx,
        "nx": nx,
        "ny": ny,
        "cx": round(GEOMETRY["cx"] / dx),
        "cy": round(GEOMETRY["cy"] / dx),
        "radius": 0.5 * GEOMETRY["diameter"] / dx,
        "blockage": GEOMETRY["diameter"] / GEOMETRY["ly"],
    }


def _inlet_profile(ny: int, dx: float, u_mean_lb: float, device: torch.device) -> torch.Tensor:
    """Parabolic Zou/He inlet profile with mean *u_mean_lb* (U_max = 1.5·mean)."""
    y = torch.arange(ny, device=device, dtype=torch.float32) * dx
    shape = 4.0 * 1.5 * y * (GEOMETRY["ly"] - y) / GEOMETRY["ly"] ** 2
    return u_mean_lb * shape


def _block_drift(series: Sequence[float], window: int, n_blocks: int = 4) -> float:
    """Relative spread of consecutive block means over the trailing *window*.

    ``0.0`` means perfectly stationary; used as the steady-state gate so that
    no mean is ever reported from a window that is still drifting.
    """
    tail = series[-window:] if window > 0 else list(series)
    n = len(tail)
    if n < n_blocks * 2:
        return float("nan")
    size = n // n_blocks
    means = [sum(tail[i * size : (i + 1) * size]) / size for i in range(n_blocks)]
    avg = sum(means) / n_blocks
    if avg == 0.0:
        return float("nan")
    return (max(means) - min(means)) / abs(avg)


def run_cylinder_st2d1(
    D_cells: int = 20,
    steps: int = 200_000,
    *,
    tau: float = 0.8,
    u_lb: float | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    avg_window: int = 20_000,
    resid_interval: int = 2_000,
    plateau_tol: float = 1.0e-3,
    ma_limit: float = 0.3,
    min_steps: int | None = None,
    stop_on_plateau: bool = False,
    sample_every: int = 10,
    keep_fields: bool = True,
    progress: bool = True,
    solid_mask: Any = None,
) -> dict[str, Any]:
    """Run the 2D-1 cylinder case and return measurements + provenance.

    Args:
        D_cells: cylinder diameter in lattice cells (resolution knob).
        steps: hard step cap.
        tau: MRT relaxation time (default 0.8 = the verified configuration).
            ``u_lb`` is then derived as ``Re·nu_lb/D_cells`` so that Re = 20.
        u_lb: alternatively fix the inlet mean velocity and derive ``tau``
            from ``nu_lb = u_lb·D_cells/Re`` (Mach sensitivity knob).
        avg_window: length of the trailing averaging window.
        resid_interval: diagnostic print / plateau-check interval.
        plateau_tol: block-mean spread below which the record counts as a
            plateau (default 0.1 %).
        ma_limit: refuse configurations whose lattice Mach number exceeds this
            value.  With ``tau`` fixed, ``u_mean`` grows as ``1/D_cells``, so a
            coarse grid silently becomes compressible (D=8 ⇒ Ma=0.43) and the
            run blows up; the guard turns that into a clear error instead of a
            NaN result.
        min_steps: earliest step at which ``stop_on_plateau`` may stop.
        stop_on_plateau: stop as soon as the trailing window is stationary.
        sample_every: decimation for the returned time series / CSV.
        keep_fields: return final rho/ux/uy/vorticity fields.
        solid_mask: optional boolean obstacle mask of shape ``(ny, nx)``
            (lattice-index space) that replaces the built-in
            :func:`~tensorlbm.boundaries.cylinder_mask`.  This is the
            runtime-geometry injection point (see
            :mod:`tensorlbm.runtime_geometry`): with ``None`` the verified
            hard-coded mask is used and the run is unchanged.

    Returns:
        JSON-safe dict with geometry, lattice parameters, ``cd``/``cl`` means,
        plateau diagnostics, decimated series and (optionally) raw fields.
    """
    dev = torch.device(device)
    geo = case_geometry(D_cells)
    nx, ny = int(geo["nx"]), int(geo["ny"])
    radius = float(geo["radius"])

    if u_lb is None:
        nu_lb = (tau - 0.5) / 3.0
        u_mean_lb = GEOMETRY["re"] * nu_lb / D_cells
        re_check = u_mean_lb * D_cells / nu_lb
    else:
        u_mean_lb = float(u_lb)
        nu_lb = u_mean_lb * D_cells / GEOMETRY["re"]
        tau = 0.5 + 3.0 * nu_lb
        re_check = u_mean_lb * D_cells / nu_lb
    if nu_lb <= 0.0:
        raise ValueError(f"non-positive lattice viscosity from tau={tau}, u_lb={u_lb}")
    ma = u_mean_lb * math.sqrt(3.0)
    if ma > ma_limit:
        raise ValueError(
            f"lattice Mach {ma:.3f} > ma_limit {ma_limit:.3f} at D_cells={D_cells} "
            f"(u_mean={u_mean_lb:.4f}): fix u_lb explicitly (e.g. u_lb=0.05) instead of tau"
        )

    u_in = _inlet_profile(ny, float(geo["dx"]), u_mean_lb, dev)

    geometry_source = "hard_coded"
    if solid_mask is None:
        solid = cylinder_mask(nx, ny, float(geo["cx"]), float(geo["cy"]), radius, dev)
    else:
        import numpy as _np

        arr = _np.asarray(solid_mask)
        if arr.shape != (ny, nx):
            raise ValueError(
                f"solid_mask shape {arr.shape} != (ny, nx) = ({ny}, {nx})"
            )
        solid = torch.as_tensor(arr, dtype=torch.bool, device=dev)
        geometry_source = "runtime_config"
    wall = make_channel_wall_mask(ny, nx, solid, dev)
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, 0)
        | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)
    )

    # 0.5 rho U^2 D with the *physical* D = 0.1 (matches the reference
    # normalisation); in lattice units D_phys = D_cells.
    dyn_p = 0.5 * u_mean_lb**2 * D_cells

    rho0 = torch.ones((ny, nx), dtype=dtype, device=dev)
    ux0 = torch.zeros_like(rho0)
    ux0[:, :] = u_in[:, None].expand(ny, nx)
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, ux0, uy0)

    cd_list: list[float] = []
    cl_list: list[float] = []
    t0 = time.time()
    stop_reason = "max_steps"
    plateau_drift = float("nan")
    min_steps = steps if min_steps is None else int(min_steps)

    for step in range(1, steps + 1):
        before = f
        collided = collide_mrt(f, tau)
        f = torch.where(solid.unsqueeze(0), before, collided)
        f = stream(f)
        f = zou_he_inlet_velocity(f, u_in, 0.0)
        f = zou_he_outlet_pressure(f, 1.0)
        f = bounce_back_cells(f, wall)
        fx, fy = compute_obstacle_forces(f, surface)
        f = bounce_back_cells(f, solid)
        cd_list.append(float(fx.item()) / dyn_p)
        cl_list.append(float(fy.item()) / dyn_p)

        if step % resid_interval == 0:
            n = min(len(cd_list), 1000)
            cd_now = sum(cd_list[-n:]) / n
            plateau_drift = _block_drift(cd_list, avg_window)
            if progress:
                print(
                    f"  [{CASE_NAME} D={D_cells}] {step}/{steps} Cd~{cd_now:.4f} "
                    f"drift~{plateau_drift:.2e} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if (
                stop_on_plateau
                and step >= min_steps
                and len(cd_list) >= avg_window
                and plateau_drift == plateau_drift  # not NaN
                and plateau_drift <= plateau_tol
            ):
                stop_reason = "plateau"
                break

    win = min(len(cd_list), avg_window)
    cd = sum(cd_list[-win:]) / win
    cl = sum(cl_list[-win:]) / win
    drift = _block_drift(cd_list, win)
    # two independent estimates inside the window (first vs second half) —
    # a plateau must give the same answer at both ends of the window.
    half = win // 2
    cd_first = sum(cd_list[-win : -half]) / max(half, 1)
    cd_second = sum(cd_list[-half:]) / max(half, 1)

    rho, ux, uy = macroscopic(f)
    res: dict[str, Any] = {
        "case": CASE_NAME,
        "anchor": "CASE-CYLINDER-DRAG-RE20",
        "lattice": "D2Q9",
        "collision": "mrt",
        "boundary": (
            "zou_he velocity inlet (Poiseuille, U_max=1.5 U_mean) + zou_he "
            "pressure outlet (rho=1) + half-way bounce-back (walls & cylinder)"
        ),
        "force": "ladd momentum exchange (post-stream, pre-obstacle bounce-back)",
        "geometry": geo,
        "geometry_source": geometry_source,
        "lattice_units": {
            "u_mean": u_mean_lb,
            "u_max": 1.5 * u_mean_lb,
            "nu": nu_lb,
            "tau": tau,
            "ma": ma,
            "re": re_check,
        },
        "n_steps": len(cd_list),
        "stop_reason": stop_reason,
        "avg_window": win,
        "n_solid_cells": int(solid.sum().item()),
        "n_surface_cells": int(surface.sum().item()),
        "cd": cd,
        "cl": cl,
        "cd_half_window_first": cd_first,
        "cd_half_window_second": cd_second,
        "cd_plateau_drift": drift,
        "plateau_tol": plateau_tol,
        "stationary": bool(drift == drift and drift <= plateau_tol),
        "ref_cd": REFERENCE["cd"],
        "ref_cl": REFERENCE["cl"],
        "err_cd_pct": (cd - REFERENCE["cd"]) / REFERENCE["cd"] * 100.0,
        "err_cl_pct": (cl - REFERENCE["cl"]) / REFERENCE["cl"] * 100.0,
        "wall_s": time.time() - t0,
        "device": str(dev),
        "series": {
            "every": int(sample_every),
            "step": list(range(sample_every, len(cd_list) + 1, sample_every)),
            "cd": cd_list[sample_every - 1 :: sample_every],
            "cl": cl_list[sample_every - 1 :: sample_every],
        },
    }
    if keep_fields:
        vort = 0.5 * (
            torch.roll(uy, -1, 1) - torch.roll(uy, 1, 1)
        ) - 0.5 * (torch.roll(ux, -1, 0) - torch.roll(ux, 1, 0))
        vort = vort.clone()
        vort[solid] = float("nan")
        res["fields"] = {
            "rho": rho.detach().cpu().numpy(),
            "ux": ux.detach().cpu().numpy(),
            "uy": uy.detach().cpu().numpy(),
            "vorticity": vort.detach().cpu().numpy(),
            "solid": solid.detach().cpu().numpy(),
            "dx": float(geo["dx"]),
        }
    return res


def compare_to_reference(result: dict[str, Any], tolerance_pct: float | None = None) -> dict[str, Any]:
    """Ref-vs-got table with per-quantity pass/fail against the anchor.

    Only ``cd`` is a gating quantity: the momentum-exchange lift on a
    staircase circle is a second-order quantity (|Cl| ~ 1e-2 against Cd ~ 5.6)
    that is known not to converge in this configuration, so it is reported
    informationally with ``gating=False``.
    """
    tol = REFERENCE["tolerance_pct"] if tolerance_pct is None else float(tolerance_pct)
    rows = []
    for key, gating in (("cd", True), ("cl", False)):
        got = float(result[key])
        ref = float(REFERENCE[key])
        err = (got - ref) / ref * 100.0
        rows.append(
            {
                "quantity": key.upper() if key == "cl" else "Cd",
                "ref": ref,
                "got": got,
                "err_pct": err,
                "tolerance_pct": tol,
                "pass": bool(abs(err) <= tol) if gating else None,
                "gating": gating,
            }
        )
    gating_rows = [r for r in rows if r["gating"]]
    return {
        "case": result.get("case", CASE_NAME),
        "D_cells": result.get("geometry", {}).get("D_cells"),
        "tolerance_pct": tol,
        "rows": rows,
        "stationary": bool(result.get("stationary", False)),
        "passed": bool(all(r["pass"] for r in gating_rows)) and bool(result.get("stationary", False)),
        "ref_source": REFERENCE["source"],
    }


def result_manifest(result: dict[str, Any]) -> dict[str, Any]:
    """Small JSON-safe projection of *result* (fields/series left out)."""
    return {
        k: v
        for k, v in result.items()
        if k not in ("fields", "series")
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Schäfer–Turek 2D-1 cylinder Re=20 benchmark")
    ap.add_argument("--D", type=int, default=20, help="cylinder diameter in cells")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--u-lb", type=float, default=None, help="fix mean inlet velocity instead of tau")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--avg-window", type=int, default=20_000)
    ap.add_argument("--plateau-tol", type=float, default=1.0e-3)
    ap.add_argument("--stop-on-plateau", action="store_true")
    ap.add_argument("--min-steps", type=int, default=None, help="earliest early-stop step")
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="output directory (results.json/series.csv/fields.npz)")
    a = ap.parse_args(argv)

    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "0")) or 1)
    res = run_cylinder_st2d1(
        D_cells=a.D,
        steps=a.steps,
        tau=a.tau,
        u_lb=a.u_lb,
        device=a.device,
        avg_window=a.avg_window,
        plateau_tol=a.plateau_tol,
        stop_on_plateau=a.stop_on_plateau,
        min_steps=a.min_steps,
    )
    verdict = compare_to_reference(res, a.tolerance)
    print(json.dumps(json.loads(json.dumps(verdict)), indent=2, ensure_ascii=False))

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        series = res["series"]
        with (out / "series.csv").open("w") as fh:
            fh.write("step,cd,cl\n")
            for s, c_d, c_l in zip(series["step"], series["cd"], series["cl"]):
                fh.write(f"{s},{c_d:.10g},{c_l:.10g}\n")
        fields = res.pop("fields", None)
        if fields is not None:
            import numpy as np

            np.savez_compressed(out / "fields.npz", **fields)
        payload = result_manifest(res)
        payload["verdict"] = verdict
        (out / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"artifacts → {out}", flush=True)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
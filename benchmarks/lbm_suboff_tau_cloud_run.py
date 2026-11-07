#!/usr/bin/env python3
"""m4928 — SUBOFF tau-cloud field production runner (CLOUD evidence for lbm-suboff-tau).

Why this script exists
----------------------
Topic ``lbm-suboff-tau`` had scalar-only evidence (Cd / wetted area / Holtrop
mis-domain attribution) and no *spatial* field evidence (CLOUD flag missing).
This runner produces the field data that the cloud figures and the per-grid
table are built from, **on the verified benchmark configuration**:

  * geometry / physics / solver settings are copied 1:1 from the frozen
    driver ``benchmarks/verified/suboff_sail_resistance/run.py``
    (bare_hull, Re=1000, MRT, half-way bounce-back, ``pressure_extrap='none'``,
    ``p0_method='near_wall'``, ``friction_formula='standard'`` = erratum-E2
    primary estimator, mass correction every 200 steps);
  * the engine class (``SuboffAppendageEngine``) is *imported* from that frozen
    driver, not re-implemented here;
  * only additive work: a step loop that mirrors ``run_engine_routed`` (same
    collide_fn / far-field BC / mass-correction cadence) plus probe sampling and
    centre-plane snapshots, and post-processing that turns lattice fields into
    machine-readable slices + station tables.

Nothing in the frozen benchmark tree is modified.  Output per grid level:
``fields_L<L>.npz`` (slices, station tables, probe series) and
``summary_L<L>.json`` (checks, moments, units, provenance).

Every reported number is derived from the dumped fields of this run; nothing is
hard-coded or estimated.

Usage
-----
    python benchmarks/lbm_suboff_tau_cloud_run.py --resolution 48 --steps 8000 \
        --device sdaa:0 --out artifacts/m4928/lbm-suboff-tau-cloud
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensorlbm.boundaries3d import far_field_bc_3d  # noqa: E402
from tensorlbm.d3q19 import macroscopic3d  # noqa: E402
from tensorlbm.general_sim import (  # noqa: E402
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)
from tensorlbm.lbm_step_correct import lbm_step_correct  # noqa: E402

_DRIVER_PATH = _REPO_ROOT / "benchmarks" / "verified" / "suboff_sail_resistance" / "run.py"

SUBOFF_LENGTH_M = 4.356  # DARPA SUBOFF bare-hull length [m] (frozen driver constant)
SUBOFF_RADIUS_M = 0.254  # max radius [m]
U_PHYS = 1.0e-3  # m/s (sets dt only; frozen driver constant)
TAU_TARGET = 0.5072  # erratum-E2 tier relaxation time at Re=1000 (frozen driver default)
REF_CF_BLASIUS = 0.041995047327036086  # Blasius 1.328/sqrt(Re), Re=1000 (locked in prereg)
BLASIUS_LOCAL_COEF = 0.664  # local laminar flat-plate Cf_x = 0.664/sqrt(Re_x)
BLASIUS_DSTAR_COEF = 1.7208  # delta*/x = 1.7208/sqrt(Re_x) for Blasius


def _load_driver():
    """Import the frozen verified driver as a module (no side effects at import)."""
    spec = importlib.util.spec_from_file_location("suboff_sail_verified_run", _DRIVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


DRIVER = _load_driver()
SuboffAppendageEngine = DRIVER.SuboffAppendageEngine


# --------------------------------------------------------------------------- #
# field post-processing (pure functions, unit-testable)
# --------------------------------------------------------------------------- #
def hull_surface_stations(solid_cp: np.ndarray) -> dict:
    """Upper/lower hull-surface fluid cells on the centre plane.

    ``solid_cp`` is the boolean mask slice ``(ny, nx)`` at the mid-span plane.
    Returns the first fluid cell index outside the hull (``j_up`` above the
    body, ``j_lo`` below) for every streamwise column, plus the hull
    half-thickness in cells.
    """
    ny, nx = solid_cp.shape
    j_up = np.full(nx, -1, dtype=int)
    j_lo = np.full(nx, -1, dtype=int)
    n_solid_col = np.zeros(nx, dtype=int)
    for ix in range(nx):
        js = np.flatnonzero(solid_cp[:, ix])
        if js.size == 0:
            continue
        n_solid_col[ix] = js.size
        hi, lo = int(js.max()), int(js.min())
        if hi + 1 < ny:
            j_up[ix] = hi + 1
        if lo - 1 >= 0:
            j_lo[ix] = lo - 1
    return {
        "j_up": j_up,
        "j_lo": j_lo,
        "has_hull": j_up >= 0,
        "n_solid_col": n_solid_col,
        "radius_cells": n_solid_col / 2.0,
    }


def _wall_stack(ux_cp: np.ndarray, surf: dict, ix: int, side: str, n: int) -> np.ndarray | None:
    """``ux/U_ref`` ratios at n cells going outward from the wall (wall itself = 0)."""
    if not surf["has_hull"][ix]:
        return None
    ny = ux_cp.shape[0]
    j0 = int(surf["j_up"][ix] if side == "up" else surf["j_lo"][ix])
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        j = j0 + i if side == "up" else j0 - i
        out[i] = float(ux_cp[j, ix]) if 0 <= j < ny else 1.0
    return out


def station_tables(
    rho_cp: np.ndarray,
    ux_cp: np.ndarray,
    uy_cp: np.ndarray,
    solid_cp: np.ndarray,
    cx_cells: float,
    length_cells: float,
    u_in: float,
    nu_lb: float,
    dx: float,
    x_stations_norm: np.ndarray,
    n_profile: int = 40,
) -> dict:
    """Wall-pressure / skin-friction / boundary-layer tables on the centre plane.

    Conventions (documented, machine-checkable):
      * half-way bounce-back puts the wall 0.5 cell inside the last fluid node,
        so the first fluid node is 0.5 cell and the second 1.5 cells from the
        wall; ``du/dn`` uses the second node with the no-slip wall value:
        ``du/dn = u2 / (1.5 dy)``.
      * ``p = rho/3`` (lattice cs^2 = 1/3), ``Cp = (p - 1/3)/(0.5 u_in^2)``.
      * ``Cf_local = tau_w / (0.5 u_in^2)`` with ``tau_w = nu_lb * |du/dn|``
        (rho_lu = 1).
      * ``delta*`` = sum over the probe column of ``(1 - u/u_in) dy`` from the
        wall outward (upper side profile).
    """
    surf = hull_surface_stations(solid_cp)
    p_cp = rho_cp / 3.0
    n_st = len(x_stations_norm)
    out = {
        "ix": np.full(n_st, -1, dtype=int),
        "xL": np.asarray(x_stations_norm, dtype=np.float64),
        "Cp_wall": np.full(n_st, np.nan),
        "Cp_wall_up": np.full(n_st, np.nan),
        "Cp_wall_lo": np.full(n_st, np.nan),
        "Cf_local_lbm": np.full(n_st, np.nan),
        "Cf_local_blasius": np.full(n_st, np.nan),
        "Re_x": np.full(n_st, np.nan),
        "delta_star_m": np.full(n_st, np.nan),
        "delta_star_blasius_m": np.full(n_st, np.nan),
        "profile_u_over_U": np.full((n_st, n_profile), np.nan, dtype=np.float32),
        "profile_wall_distance_m": np.arange(1, n_profile + 1, dtype=np.float64) * dx,
    }
    for si, xL in enumerate(x_stations_norm):
        ix = int(round(cx_cells - 0.5 * length_cells + xL * length_cells))
        if ix < 0 or ix >= solid_cp.shape[1] or not surf["has_hull"][ix]:
            out["ix"][si] = ix
            continue
        out["ix"][si] = ix
        ny = p_cp.shape[0]
        j_up, j_lo = int(surf["j_up"][ix]), int(surf["j_lo"][ix])
        cp_up = (p_cp[j_up, ix] - 1.0 / 3.0) / (0.5 * u_in**2)
        cp_lo = (p_cp[j_lo, ix] - 1.0 / 3.0) / (0.5 * u_in**2)
        out["Cp_wall_up"][si] = cp_up
        out["Cp_wall_lo"][si] = cp_lo
        out["Cp_wall"][si] = 0.5 * (cp_up + cp_lo)

        cf_sides, dstar_sides = [], []
        for side in ("up", "lo"):
            u1 = _wall_stack(ux_cp, surf, ix, side, 1)
            u2 = _wall_stack(ux_cp, surf, ix, side, 2)
            if u2 is not None:
                du_dn = abs(float(u2[1]) - 0.0) / (1.5 * dx)
                cf_sides.append(2.0 * nu_lb * du_dn / u_in**2)
            prof = _wall_stack(ux_cp, surf, ix, side, n_profile)
            if prof is not None:
                clipped = np.clip(prof, 0.0, 1.5)
                dstar_sides.append(float(np.sum(1.0 - clipped) * dx))
                if side == "up":
                    out["profile_u_over_U"][si, :] = prof.astype(np.float32)
            _ = u1
        if cf_sides:
            out["Cf_local_lbm"][si] = float(np.mean(cf_sides))
        if dstar_sides:
            out["delta_star_m"][si] = float(np.mean(dstar_sides))

        x_lu = xL * length_cells
        x_m = x_lu * dx
        Re_x = u_in * x_lu / nu_lb
        out["Re_x"][si] = Re_x
        out["Cf_local_blasius"][si] = BLASIUS_LOCAL_COEF / math.sqrt(Re_x)
        out["delta_star_blasius_m"][si] = BLASIUS_DSTAR_COEF * x_m / math.sqrt(Re_x)
    out["radius_cells"] = surf["radius_cells"]
    out["has_hull"] = surf["has_hull"]
    return out


def mass_flux_balance(rho: np.ndarray, ux: np.ndarray) -> dict:
    """Streamwise mass flux through the inlet and outlet planes (lattice units)."""
    inlet = float(np.sum(rho[:, :, 0] * ux[:, :, 0]))
    outlet = float(np.sum(rho[:, :, -1] * ux[:, :, -1]))
    return {
        "inlet": inlet,
        "outlet": outlet,
        "imbalance_pct": float(abs(outlet - inlet) / max(abs(inlet), 1e-30) * 100.0),
        "note": "streamwise planes only (lateral boundaries are far-field)",
    }


def field_moments(arr: np.ndarray, name: str, unit: str) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    v = a[np.isfinite(a)]
    return {
        "name": name,
        "unit": unit,
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def u95_of(values) -> tuple[float, float, int]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size < 2:
        return float("nan"), float("nan"), int(v.size)
    return float(1.96 * v.std(ddof=1) / math.sqrt(v.size)), float(v.mean()), int(v.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hull-type", default="bare_hull", choices=["bare_hull", "with_sail", "full"])
    ap.add_argument("--resolution", type=int, required=True, help="cells per hull length L")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--collision", default=None, choices=["mrt", "smagorinsky"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="centre-plane snapshot cadence (0 -> steps//8)")
    ap.add_argument("--probe-every", type=int, default=25)
    ap.add_argument("--stations", default="0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,"
                                         "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99")
    args = ap.parse_args()

    L = args.resolution
    hull_type = args.hull_type
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    collision = args.collision or CollisionModel.MRT
    snapshot_every = args.snapshot_every or max(1, args.steps // 8)

    # Re = u*L/nu = 1000 at every grid level (frozen driver default, run.py:466);
    # the converter then sets nu_lb = u_lb*L_cells/Re, i.e. tau grows with L.
    nu_phys = U_PHYS * SUBOFF_LENGTH_M / 1000.0

    config = GeneralSimConfig(
        name=f"m4928_field_{hull_type}_re1000_L{L}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=nu_phys,
            inlet_velocity=U_PHYS,
            reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=collision,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=args.steps,
            warmup_steps=None,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=args.device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method="near_wall",
            friction_formula="standard",
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=str(out_dir),
            formats=[],
            save_macroscopic=False,
            save_forces=True,
        ),
    )

    print(f"=== m4928 field run: {hull_type} Re=1000 L={L} steps={args.steps} "
          f"device={args.device} ===", flush=True)
    engine = SuboffAppendageEngine(config, hull_type=hull_type)
    setup_info = engine.setup()
    print("setup:", json.dumps({k: str(v) for k, v in setup_info.items() if k != "bc_config"},
                               indent=1), flush=True)

    tau = engine.uc.tau
    nu_lb = engine.uc.nu_lb
    u_in = engine.uc.u_lb
    nz, ny, nx = engine.solid.shape
    k_mid = nz // 2

    collide_fn, collide_kwargs = engine._get_collide_fn()
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=engine._build_bc_config())
    solid_t = engine.solid
    from tensorlbm.solver3d import correct_mass3d

    target_mass = engine._initial_mass
    dpS = engine._compute_dpS()

    def _step(f):
        return lbm_step_correct(f, collide_fn, tau, solid_t, u_in, far_field_fn, **collide_kwargs)

    stations = np.asarray([float(s) for s in args.stations.split(",")], dtype=np.float64)
    probe_series: dict[str, list] = {"step": [], "Cd_total": [], "Cp_stag": [], "ux_mid": []}
    snapshots: list[dict] = []
    forces: list[dict] = []

    dx = SUBOFF_LENGTH_M / L
    du = U_PHYS / u_in  # m/s per lattice velocity unit
    cx_cells = float(engine.nx) * 0.25
    length_cells = float(engine.config.geometry.suboff_length / dx)
    solid_cp_np = engine.solid[k_mid].detach().cpu().numpy()

    def _centre_plane():
        rho_t, ux_t, uy_t, uz_t = macroscopic3d(engine.f)
        return (
            rho_t[k_mid].detach().cpu().numpy().astype(np.float32),
            ux_t[k_mid].detach().cpu().numpy().astype(np.float32),
            uy_t[k_mid].detach().cpu().numpy().astype(np.float32),
            uz_t[k_mid].detach().cpu().numpy().astype(np.float32),
        )

    def _snapshot(step: int) -> None:
        rho_cp, ux_cp_s, uy_cp_s, uz_cp_s = _centre_plane()
        snapshots.append({"step": step, "rho_cp": rho_cp, "ux_cp": ux_cp_s,
                          "uy_cp": uy_cp_s, "uz_cp": uz_cp_s})
        del rho_cp, ux_cp_s, uy_cp_s, uz_cp_s

    nose_cols = np.flatnonzero(solid_cp_np.any(axis=0))
    ix_nose = int(nose_cols.min()) if nose_cols.size else 0

    t0 = time.time()
    for step in range(1, args.steps + 1):
        engine.f = _step(engine.f)
        engine.step_count += 1
        if step % engine.config.solver.mass_correction_interval == 0:
            engine.f = correct_mass3d(engine.f, target_mass)
        if step % engine.config.solver.force_sample_interval == 0:
            engine._sample_forces(dpS, nu_lb)
            if engine.forces_log:
                e = engine.forces_log[-1]
                forces.append({"step": int(e.get("step", step)),
                               "cd_pressure": e.get("cd_pressure"),
                               "cd_friction": e.get("cd_friction"),
                               "cd_total": e.get("cd_total")})
        if step % args.probe_every == 0:
            rho_cp, ux_cp_s, _uy, _uz = _centre_plane()
            cp_stag = float("nan")
            if ix_nose > 0 and ix_nose - 1 < nx:
                cp_stag = float(((rho_cp[k_mid, ix_nose - 1] / 3.0) - 1.0 / 3.0)
                                / (0.5 * u_in**2))
            probe_series["step"].append(step)
            probe_series["Cp_stag"].append(cp_stag)
            probe_series["ux_mid"].append(
                float(ux_cp_s[min(ny - 1, ny // 2 + 4), int(round(cx_cells))]))
            probe_series["Cd_total"].append(
                forces[-1]["cd_total"] if forces else float("nan"))
        if step % snapshot_every == 0 or step == args.steps:
            _snapshot(step)
        if step % 2000 == 0:
            print(f"[m4928] L={L} step {step}/{args.steps} elapsed {time.time() - t0:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    diverged = not bool(torch.isfinite(engine.f).all().item())
    print(f"[m4928] run done: {elapsed:.0f}s ({elapsed / args.steps * 1000:.1f} ms/step) "
          f"diverged={diverged}", flush=True)

    # ---------------- final fields & post-processing ----------------
    rho, ux, uy, uz = macroscopic3d(engine.f)
    rho = rho.detach().cpu().numpy().astype(np.float32)
    ux = ux.detach().cpu().numpy().astype(np.float32)
    uy = uy.detach().cpu().numpy().astype(np.float32)
    uz = uz.detach().cpu().numpy().astype(np.float32)
    solid_np = engine.solid.detach().cpu().numpy()

    tab = station_tables(rho[k_mid], ux[k_mid], uy[k_mid], solid_np[k_mid], cx_cells,
                         length_cells, u_in, nu_lb, dx, stations)
    flux = mass_flux_balance(rho, ux)

    # far-field check: fluid cells strictly upstream of the hull nose
    up_lo = max(1, ix_nose - max(3, ix_nose // 4))
    fluid_up = ~solid_np[:, :, up_lo:ix_nose]
    ux_up = ux[:, :, up_lo:ix_nose][fluid_up]
    cp_up = ((rho[:, :, up_lo:ix_nose][fluid_up] / 3.0) - 1.0 / 3.0) / (0.5 * u_in**2)

    cp_stag_final = float("nan")
    if ix_nose > 0:
        cp_stag_final = float(((rho[k_mid, k_mid, ix_nose - 1] / 3.0) - 1.0 / 3.0)
                              / (0.5 * u_in**2))

    # centre-plane vorticity: omega_z = duy/dx - dux/dy
    dudy = np.gradient(ux[k_mid].astype(np.float64), dx, axis=0)
    dvdx = np.gradient(uy[k_mid].astype(np.float64), dx, axis=1)
    vort_cp = dvdx - dudy

    ix_wake = int(round(cx_cells + 0.5 * length_cells + 0.10 * length_cells))
    ix_wake = min(ix_wake, nx - 1)
    wake_profile = ux[k_mid, :, ix_wake].astype(np.float64) / u_in

    # temporal statistics (U95) from the centre-plane snapshot ensemble
    snap_cp_stag, snap_cf_mid = [], []
    for s in snapshots:
        rho_s, ux_s = s["rho_cp"], s["ux_cp"]
        if ix_nose > 0:
            snap_cp_stag.append(float(((rho_s[k_mid, ix_nose - 1] / 3.0) - 1.0 / 3.0)
                                      / (0.5 * u_in**2)))
        t_s = station_tables(rho_s, ux_s, s["uy_cp"], solid_cp_np, cx_cells, length_cells,
                             u_in, nu_lb, dx, np.array([0.5]))
        snap_cf_mid.append(float(t_s["Cf_local_lbm"][0]))
    u95_cp_stag, mean_cp_stag, n_cp = u95_of(snap_cp_stag)
    u95_cf_mid, mean_cf_mid, n_cf = u95_of(snap_cf_mid)

    cd_last = [f["cd_total"] for f in forces[-100:] if f.get("cd_total") is not None]
    cd_window = float(np.mean(cd_last)) if cd_last else float("nan")
    cd_tail = [f["cd_total"] for f in forces[-max(1, len(forces) // 4):]
               if f.get("cd_total") is not None]
    cd_tail_mean = float(np.mean(cd_tail)) if cd_tail else float("nan")
    drift_pct = (float(abs(cd_last[-1] - cd_last[0]) / abs(cd_window) * 100.0)
                 if len(cd_last) > 1 and cd_window else float("nan"))
    cd_err_pct = (cd_window - REF_CF_BLASIUS) / REF_CF_BLASIUS * 100.0

    npz_path = out_dir / f"fields_L{L}.npz"
    np.savez_compressed(
        npz_path,
        x_lu=np.arange(nx, dtype=np.float64) - cx_cells,
        x_over_L=(np.arange(nx, dtype=np.float64) - cx_cells) / length_cells,
        y_lu=np.arange(ny, dtype=np.float64) - ny // 2,
        y_over_D=(np.arange(ny, dtype=np.float64) - ny // 2) / (2.0 * length_cells / 8.575),
        solid_cp=solid_np[k_mid].astype(np.int8),
        rho_cp=rho[k_mid], ux_cp=ux[k_mid], uy_cp=uy[k_mid], uz_cp=uz[k_mid],
        p_cp=rho[k_mid] / 3.0,
        Cp_cp=((rho[k_mid] / 3.0) - 1.0 / 3.0) / (0.5 * u_in**2),
        vort_cp=vort_cp.astype(np.float32),
        sec_ux_mid=ux[:, :, int(round(cx_cells))],
        sec_ux_stern=ux[:, :, max(0, int(round(cx_cells + 0.5 * length_cells)) - 1)],
        wake_profile=wake_profile.astype(np.float32),
        wake_x_over_L=np.array([(ix_wake - cx_cells) / length_cells]),
        **{f"st_{k}": v for k, v in tab.items()},
        probe_step=np.asarray(probe_series["step"], dtype=np.int64),
        probe_Cd_total=np.asarray(probe_series["Cd_total"], dtype=np.float64),
        probe_Cp_stag=np.asarray(probe_series["Cp_stag"], dtype=np.float64),
        probe_ux_mid=np.asarray(probe_series["ux_mid"], dtype=np.float64),
        snap_steps=np.asarray([s["step"] for s in snapshots], dtype=np.int64),
    )

    summary = {
        "schema": "m4928-lbm-suboff-tau-cloud/v1",
        "mission": 4928,
        "topic": "lbm-suboff-tau",
        "level": {"resolution_cells_per_L": L, "hull_type": hull_type,
                  "collision": str(collision), "steps": args.steps,
                  "device": str(args.device), "elapsed_s": elapsed,
                  "ms_per_step": elapsed / args.steps * 1000.0},
        "grid": {"nx": nx, "ny": ny, "nz": nz, "total_cells": int(nx * ny * nz),
                 "k_mid": k_mid, "cx_cells": cx_cells, "length_cells": length_cells,
                 "ix_nose": ix_nose, "ix_wake": ix_wake},
        "units": {
            "lattice": {"length": "lu (1 lu = dx m)", "velocity": "lu/ts",
                        "pressure": "rho/3, rho_lu ~ 1"},
            "physical": {"dx_m": dx, "du_m_per_s": du, "u_in_m_per_s": U_PHYS,
                         "nu_m2_per_s": nu_lb * du * dx, "tau": tau, "nu_lb": nu_lb,
                         "p_scale_Pa_per_lu": 1000.0 * du * du},
            "coordinates": "x/L from the hull nose (hull spans x/L 0..1, centre x/L=0.5); "
                           "y/D and z/D from the body axis",
        },
        "setup": {k: str(v) for k, v in setup_info.items() if k != "bc_config"},
        "geometry": engine._appendage_stats,
        "field_checks": {
            "Cp_stagnation_final": cp_stag_final,
            "Cp_stagnation_ref_analytic": 1.0,
            "Cp_stagnation_snapshot_mean": mean_cp_stag,
            "Cp_stagnation_U95": u95_cp_stag,
            "Cp_stagnation_n_snaps": n_cp,
            "farfield_ux_over_U_mean": float(np.mean(ux_up) / u_in),
            "farfield_ux_over_U_U95": float(1.96 * np.std(ux_up) / math.sqrt(ux_up.size) / u_in),
            "farfield_Cp_mean": float(np.mean(cp_up)),
            "mass_flux": flux,
            "Cf_local_mid_mean_snaps": mean_cf_mid,
            "Cf_local_mid_U95": u95_cf_mid,
            "Cf_local_mid_n_snaps": n_cf,
        },
        "drag": {"cd_window_last100_samples": cd_window,
                 "cd_tail_quarter_mean": cd_tail_mean,
                 "cd_err_pct_vs_blasius": cd_err_pct,
                 "window_drift_pct": drift_pct,
                 "n_force_samples": len(forces),
                 "reference": {"Cf_ref": REF_CF_BLASIUS,
                               "formula": "Blasius laminar flat plate 1.328/sqrt(Re), Re=1000",
                               "normalization": "wetted area pi*D*L (frozen driver)"}},
        "field_moments_lattice": [
            field_moments(ux[k_mid], "ux_centreplane", "lu/ts"),
            field_moments(uy[k_mid], "uy_centreplane", "lu/ts"),
            field_moments(rho[k_mid] / 3.0, "p_centreplane", "lu (rho/3)"),
            field_moments(vort_cp, "omega_z_centreplane", "1/ts"),
        ],
        "sanity": {"diverged": diverged, "snapshots": int(len(snapshots)),
                   "probe_samples": int(len(probe_series["step"]))},
        "artifacts": {"npz": str(npz_path)},
        "provenance": {
            "engine": "benchmarks/verified/suboff_sail_resistance/run.py::SuboffAppendageEngine "
                      "(imported, unmodified)",
            "step_chain": "lbm_step_correct + far_field_bc_3d + correct_mass3d (verified chain, "
                          "eager; loop mirrors run_engine_routed)",
            "reynolds": "Re = u_phys*L/nu_phys = 1000.0 at every level (nu_phys = 4.356e-6 m^2/s)",
        },
    }
    (out_dir / f"summary_L{L}.json").write_text(json.dumps(summary, indent=1))
    print(f"[m4928] wrote {npz_path.name} + summary_L{L}.json; Cd={cd_window:.6f} "
          f"(err vs Blasius {cd_err_pct:+.2f}%), Cp_stag={cp_stag_final:.4f}", flush=True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""CASE-MCMP-CAPILLARY-RISE — case-pool driver for capillary rise (Jurin law).

Reference (closed form, never transcribed)::

    h = 2 sigma cos(theta) / (rho g r)            Jurin 1718; de Gennes 1985

For a 2-D slit of width ``w`` the meniscus is a cylindrical arc with mean
curvature ``1/R_c = 2 cos(theta) / w``, so the same law holds with ``r -> w``
(plate separation).  The reference height used below is evaluated *from the
simulated inputs* — surface tension, contact angle, the two bulk densities,
gravity and the slit width are all measured or declared in the artifact, so no
number in the report is typed by hand.

Reused capability (no solver is re-implemented here):

===========================  =================================================
surface tension ``sigma``    ``benchmarks/sc_surface_tension/run.py::run_droplet``
                             (2-D Laplace gate + independent Korteweg anchor)
contact angle ``theta``      ``benchmarks/sc_surface_tension/run.py::run_contact_angle``
SC kernel + wetting BC       ``tensorlbm.sc_surface_tension``
                             (``sc_force`` / ``collide_guo`` /
                             ``stream_bounce_back`` /
                             ``wall_psi_from_contact_angle``)
meniscus measurement         ``tensorlbm.wetting.metrics::meniscus_height``
===========================  =================================================

New code in this file: the channel/reservoir geometry, the gravity body-force
term (``Fy -= rho g``), the open-gas top row and the post-processing
(hydrostatic vs Laplace vs meniscus-arc cross-checks).

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/run_case.py sigma
    PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/run_case.py contact --theta 60
    PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/run_case.py jurin --theta 60 --gravity 1e-4
    PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/run_case.py all

Artifacts land in ``benchmarks/case_pool/capillary_rise/artifacts/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm import sc_surface_tension as SS  # noqa: E402
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.wetting.metrics import meniscus_height  # noqa: E402

OUT = Path(__file__).resolve().parent / "artifacts"


def _load_bench_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


SST = _load_bench_module(
    "sc_surface_tension_bench", REPO_ROOT / "benchmarks" / "sc_surface_tension" / "run.py"
)

# ---------------------------------------------------------------------------
# Frozen configuration (predeclared; the report reads it back from the JSON)
# ---------------------------------------------------------------------------

CFG = {
    "nx": 128,
    "ny": 160,
    "wall": 2,
    "baffle": 2,
    "gap": 16,
    "pool": 20,
    "theta_deg": 60.0,
    "gravity": 1.0e-4,
    "steps": 30000,
    "sample": 200,
    "G_eff": float(SS.G_EFF),
    "tau": float(SS.TAU),
    "stencil": "d2q9",
    "conv_drift_tol": 0.01,
    "u_tol": 1.0e-2,
    "tol_pct": 10.0,
    "tight_pct": 5.0,
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def channel_geometry(ny: int, nx: int, *, wall: int, baffle: int, gap: int, pool: int):
    """Solid mask + measurement bands for a baffle channel standing in a pool.

    Rows grow upward: row 0 is the bottom wall, the pool surface is at
    ``pool``, the baffles run from ``pool - 4`` to the top row so the liquid
    can only enter the channel from below.  The gas above the pool is open at
    the last row (constant-density bath), so the column is not a closed tube.
    """
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[:2, :] = True
    solid[:, :wall] = True
    solid[:, nx - wall:] = True
    x_c = nx // 2
    x_left = x_c - gap // 2 - baffle
    x_right = x_left + baffle + gap
    solid[pool - 4:, x_left:x_left + baffle] = True
    solid[pool - 4:, x_right:x_right + baffle] = True
    ch_lo, ch_hi = x_left + baffle, x_right
    bands = {
        "x_left_baffle": [x_left, x_left + baffle],
        "x_right_baffle": [x_right, x_right + baffle],
        "channel_inner": [ch_lo, ch_hi],
        "gap": gap,
        "x_wall_left_face": float(ch_lo) - 0.5,
        "x_wall_right_face": float(ch_hi) - 0.5,
        # reservoir level is read far from either baffle (>= 1.5 capillary lengths)
        "pool_bands": [[wall + 4, x_left - 8], [x_right + baffle + 8, nx - wall - 4]],
        # meniscus band: central 60 % of the channel, away from the contact lines
        "meniscus_band": [ch_lo + gap // 5, ch_hi - gap // 5],
    }
    return solid, bands


def initial_rho(solid: torch.Tensor, pool: float, *, rho_l: float, rho_v: float,
                width: float = 2.0) -> torch.Tensor:
    ny, nx = solid.shape
    y = torch.arange(ny, dtype=torch.float32).view(ny, 1)
    w = 0.5 * (1.0 - torch.tanh((y - (pool - 0.5)) / width))
    rho = rho_v + (rho_l - rho_v) * w
    rho = rho.expand(ny, nx).clone()
    return torch.where(solid, torch.full_like(rho, rho_l), rho)


# ---------------------------------------------------------------------------
# One time step (library primitives + gravity + open gas top)
# ---------------------------------------------------------------------------


def step_with_gravity(f, solid, wall_psi, gravity, *, stencil, G_eff, tau, rho_v):
    """One step.  Gravity enters in the buoyancy (Boussinesq) form

        F_y = -F_SC - (rho - rho_v) g ,

    so the gas carries no hydrostatic gradient: the gas pressure stays uniform
    and the (vented) top row can hold it at ``rho_v`` without draining mass.
    The liquid-side balance is unchanged, because the pressure jump across the
    meniscus is set by ``(rho_l - rho_v) g h`` either way -- the weightless-gas
    form just removes the part of the gas gradient that a fixed-density vent
    cannot represent.
    """
    rho = f.sum(dim=0)
    Fx, Fy = SS.sc_force(rho, G_eff, stencil, "stencil", solid, wall_psi)
    Fy = Fy - (rho - rho_v) * gravity   # buoyancy form, -y (rows grow upward)
    f_post = SS.collide_guo(f, Fx, Fy, tau)
    f_post = torch.where(solid.unsqueeze(0), f, f_post)
    f_new = SS.stream_bounce_back(f_post, solid)
    # open top: constant-density gas bath far above the meniscus
    nx = f.shape[2]
    one = torch.ones(1, nx, dtype=f.dtype)
    feq = equilibrium(torch.full((1, nx), float(rho_v), dtype=f.dtype),
                      torch.zeros(1, nx, dtype=f.dtype), torch.zeros(1, nx, dtype=f.dtype))
    f_new[:, -1, :] = feq[:, 0, :] * one
    return f_new


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _circle_fit(x: np.ndarray, y: np.ndarray):
    A = np.stack([2.0 * x, 2.0 * y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = sol[0], sol[1]
    r2 = sol[2] + xc * xc + yc * yc
    return float(xc), float(yc), float(math.sqrt(max(r2, 0.0)))


def meniscus_arc(nh: np.ndarray, bands: dict, level: float = 0.5) -> dict:
    """Circle fit to the mid-density contour inside the channel, plus geometry."""
    lo, hi = bands["channel_inner"]
    ny = nh.shape[0]
    xs, ys = [], []
    floor_row = int(bands.get("pool", 0))
    for j in range(lo, hi):
        col = nh[:, j]
        above = col > level
        if not above.any():
            continue
        i1 = np.where(above)[0][-1]
        if i1 + 1 >= ny or i1 <= floor_row:
            continue
        a, b = col[i1], col[i1 + 1]
        ys.append(i1 + (a - level) / (a - b))
        xs.append(float(j))
    if len(xs) < 6:
        return {"status": "too_few_points", "n_points": len(xs)}
    xs_a, ys_a = np.asarray(xs), np.asarray(ys)
    xc, yc, R = _circle_fit(xs_a, ys_a)
    half_gap = 0.5 * float(bands["gap"])
    cos_t = half_gap / R if R > 0 else float("nan")
    cos_t = max(-1.0, min(1.0, cos_t))
    return {
        "status": "ok",
        "n_points": int(len(xs)),
        "R_fit": R,
        "x_center": xc,
        "y_center": yc,
        "sag": float(R - math.sqrt(max(R * R - half_gap * half_gap, 0.0))),
        "cos_theta_from_arc": cos_t,
        "theta_from_arc_deg": math.degrees(math.acos(cos_t)),
        "mean_height": float(ys_a.mean()),
    }


def _band_height(nh: np.ndarray, bands_list, level: float = 0.5):
    hs, stds, n = [], [], 0
    for lo, hi in bands_list:
        h, s = meniscus_height(nh, int(lo), int(hi), level)
        if not math.isnan(h):
            hs.append(h)
            stds.append(s)
            n += 1
    if not hs:
        return float("nan"), float("nan")
    return float(np.mean(hs)), float(np.mean(stds))


# ---------------------------------------------------------------------------
# Case: surface tension (reused 2-D Laplace / Korteweg facility)
# ---------------------------------------------------------------------------


def run_sigma(R: float = 30.0, steps: int = 12000) -> dict:
    t0 = time.time()
    res = SST.run_droplet(R, steps=steps, min_steps=3000, sample=1000, verbose=False)
    keep = {k: v for k, v in res.items() if k not in ("profile_r", "profile_psi", "dp_theta", "history")}
    keep.update({
        "case": "sigma_2d_laplace",
        "sigma_meas": res["sigma_laplace"],
        "sigma_anchor_K": res["sigma_K"],
        "ratio_laplace_over_K": res["sigma_laplace"] / res["sigma_K"] if res["sigma_K"] else float("nan"),
        "wall_time_s": time.time() - t0,
    })
    return keep


# ---------------------------------------------------------------------------
# Case: contact angle (reused sessile-drop facility)
# ---------------------------------------------------------------------------


def run_contact(theta_deg: float, steps: int = 16000) -> dict:
    t0 = time.time()
    res = SST.run_contact_angle(theta_deg, steps=steps, verbose=False)
    keep = {k: v for k, v in res.items() if k not in ("rho_field", "history")}
    keep.update({"case": "contact_angle", "wall_time_s": time.time() - t0})
    return keep


# ---------------------------------------------------------------------------
# Case: capillary rise
# ---------------------------------------------------------------------------


def run_jurin(
    theta_deg: float = CFG["theta_deg"],
    gravity: float = CFG["gravity"],
    *,
    nx: int = CFG["nx"],
    ny: int = CFG["ny"],
    wall: int = CFG["wall"],
    baffle: int = CFG["baffle"],
    gap: int = CFG["gap"],
    pool: int = CFG["pool"],
    steps: int = CFG["steps"],
    sample: int = CFG["sample"],
    G_eff: float = CFG["G_eff"],
    tau: float = CFG["tau"],
    stencil_name: str = CFG["stencil"],
    sigma: float | None = None,
    theta_meas: float | None = None,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    torch.set_num_threads(1)
    solid, bands = channel_geometry(ny, nx, wall=wall, baffle=baffle, gap=gap, pool=pool)
    bands["pool"] = pool
    stencil = SS.build_stencil(stencil_name)
    wall_psi = SS.wall_psi_from_contact_angle(theta_deg)
    rho_l, rho_v = SS.RHO_L, SS.RHO_V
    rho0 = initial_rho(solid, pool, rho_l=rho_l, rho_v=rho_v)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    f = torch.where(solid.unsqueeze(0), equilibrium(
        torch.full_like(rho0, rho_l), zero, zero), f)
    fluid = ~solid
    liquid0 = rho0 > 0.5 * (rho_l + rho_v)
    mass_liq0 = float((rho0 * liquid0).sum())

    m_band = bands["meniscus_band"]
    traj: list[dict] = []
    h_last = float("nan")
    for step in range(1, steps + 1):
        f = step_with_gravity(f, solid, wall_psi, gravity,
                              stencil=stencil, G_eff=G_eff, tau=tau, rho_v=rho_v)
        if step % sample:
            continue
        rho = f.sum(dim=0)
        if not torch.isfinite(rho).all():
            raise RuntimeError(f"capillary-rise run diverged at step {step} "
                               f"(theta={theta_deg}, g={gravity})")
        nh = ((rho - rho_v) / (rho_l - rho_v)).numpy()
        h_ch, h_ch_std = meniscus_height(nh, int(m_band[0]), int(m_band[1]), 0.5)
        h_pool, h_pool_std = _band_height(nh, bands["pool_bands"], 0.5)
        u = SS.macroscopic(f)[1:]
        speed = torch.sqrt(u[0] ** 2 + u[1] ** 2)
        liquid = rho > 0.5 * (rho_l + rho_v)
        u_liq = float(speed[liquid].max()) if bool(liquid.any()) else float("nan")
        h = h_ch - h_pool
        traj.append({
            "step": step,
            "t": float(step),
            "y_channel": float(h_ch),
            "y_channel_std": float(h_ch_std),
            "y_reservoir": float(h_pool),
            "y_reservoir_std": float(h_pool_std),
            "h": float(h),
            "u_max": float(speed.max()),
            "u_max_liquid": u_liq,
        })
        h_last = h
        if verbose and (len(traj) % 25 == 0):
            print(f"  [jurin theta={theta_deg} g={gravity:g}] step {step} "
                  f"h={h:.3f} y_ch={h_ch:.2f} y_pool={h_pool:.2f} "
                  f"u_liq={u_liq:.4f}", flush=True)

    rho = f.sum(dim=0)
    nh = ((rho - rho_v) / (rho_l - rho_v)).numpy()
    arc = meniscus_arc(nh, bands, 0.5)
    h_ch, h_ch_std = meniscus_height(nh, int(m_band[0]), int(m_band[1]), 0.5)
    h_pool, h_pool_std = _band_height(nh, bands["pool_bands"], 0.5)
    h_meas = float(h_ch - h_pool)

    # bulk densities read from the final field (not assumed)
    rho_field = rho
    ch_lo, ch_hi = bands["channel_inner"]
    y_top = int(math.floor(h_ch)) if not math.isnan(h_ch) else ny - 20
    liq = rho_field[max(pool + 2, 4):max(pool + 3, y_top - 12), ch_lo + gap // 4:ch_hi - gap // 4]
    gas = rho_field[min(ny - 6, y_top + 12): ny - 4, ch_lo + gap // 4:ch_hi - gap // 4]
    rho_l_meas = float(liq.mean()) if liq.numel() else float("nan")
    rho_g_meas = float(gas.mean()) if gas.numel() else float("nan")

    # final-window statistics / stopping rule
    tail = traj[-max(3, len(traj) // 5):]
    h_tail = np.array([t["h"] for t in tail], dtype=float)
    h_tail = h_tail[np.isfinite(h_tail)]
    h_mean = float(h_tail.mean()) if h_tail.size else float("nan")
    h_spread = float(h_tail.max() - h_tail.min()) if h_tail.size else float("nan")
    drift = h_spread / abs(h_mean) if h_mean else float("nan")
    u_last = traj[-1]["u_max_liquid"] if traj else float("nan")
    converged = bool(np.isfinite(drift) and drift < CFG["conv_drift_tol"] and u_last < CFG["u_tol"])

    # reference height (closed form; inputs measured/declared above)
    sigma_use = float(sigma) if sigma is not None else float("nan")
    d_rho = rho_l_meas - rho_g_meas
    w = float(gap)
    h_ref_meas = (2.0 * sigma_use * math.cos(math.radians(theta_meas)) / (d_rho * gravity * w)
                  if (np.isfinite(sigma_use) and theta_meas is not None) else float("nan"))
    theta_map = SS.contact_angle_from_wall_psi(wall_psi)
    h_ref_nominal = (2.0 * sigma_use * math.cos(math.radians(theta_map)) / (d_rho * gravity * w)
                     if np.isfinite(sigma_use) else float("nan"))

    def _err(ref):
        return (h_meas - ref) / ref * 100.0 if (np.isfinite(ref) and ref) else float("nan")

    err_meas = _err(h_ref_meas)
    err_nominal = _err(h_ref_nominal)

    # hydrostatic / Laplace cross-checks from the same field
    p = SS.pressure_flux(f).numpy()
    row_lo = max(pool + 4, y_top - 16)
    row_hi = max(row_lo + 1, y_top - 8)
    p_liq = float(np.mean(p[row_lo:row_hi, ch_lo + gap // 4: ch_hi - gap // 4]))
    p_bath = float(np.mean(p[row_lo:row_hi, bands["pool_bands"][0][0]:bands["pool_bands"][0][1]]))
    dp_hydro = p_liq - p_bath
    dp_laplace = float(sigma_use / arc["R_fit"]) if arc.get("status") == "ok" else float("nan")

    result = {
        "case": "jurin_capillary_rise",
        "anchor": "CASE-MCMP-CAPILLARY-RISE",
        "ref": "h = 2 sigma cos(theta) / (rho g r)   [Jurin 1718 Phil. Trans. 30, 739; "
               "de Gennes 1985 Rev. Mod. Phys. 57, 827]",
        "theta_input_deg": float(theta_deg),
        "theta_from_wall_psi_deg": theta_map,
        "theta_measured_deg": theta_meas,
        "gravity": float(gravity),
        "sigma_input": sigma_use,
        "inputs": {"nx": nx, "ny": ny, "wall": wall, "baffle": baffle, "gap": gap,
                   "pool": pool, "steps": steps, "sample": sample, "G_eff": G_eff,
                   "tau": tau, "stencil": stencil_name,
                   "rho_l_lib": rho_l, "rho_v_lib": rho_v,
                   "wall_psi": wall_psi, "bands": bands},
        "meas": {
            "h": h_meas,
            "h_final_window_mean": h_mean,
            "h_final_window_spread": h_spread,
            "y_channel": float(h_ch),
            "y_channel_std": float(h_ch_std),
            "y_reservoir": float(h_pool),
            "y_reservoir_std": float(h_pool_std),
            "rho_l_meas": rho_l_meas,
            "rho_g_meas": rho_g_meas,
            "delta_rho": d_rho,
            "u_max": u_last,
            "u_max_liquid": u_last,
            "u_max_gas": traj[-1]["u_max"] if traj else float("nan"),
            "mass_drift": abs(float((rho * fluid).sum()) - float((rho0 * fluid).sum()))
                          / float((rho0 * fluid).sum()),
            "liquid_mass_drift": abs(float((rho * (rho > 0.5 * (rho_l + rho_v))).sum()) - mass_liq0)
                                 / mass_liq0,
            "arc": arc,
            "dp_hydrostatic_meas": dp_hydro,
            "dp_hydrostatic_theory": d_rho * gravity * h_meas,
            "dp_laplace_from_arc": dp_laplace,
            "dp_laplace_theory": 2.0 * sigma_use * math.cos(
                math.radians(theta_meas if theta_meas is not None else theta_map)) / w,
        },
        "analysis": {
            "h_ref_measured_inputs": h_ref_meas,
            "h_ref_nominal_angle": h_ref_nominal,
            "error_pct_vs_measured_inputs": err_meas,
            "error_pct_vs_nominal_angle": err_nominal,
            "tol_pct": CFG["tol_pct"],
            "tight_pct": CFG["tight_pct"],
            "pass_tol": bool(np.isfinite(err_meas) and abs(err_meas) < CFG["tol_pct"]),
            "pass_tight": bool(np.isfinite(err_meas) and abs(err_meas) < CFG["tight_pct"]),
            "converged": converged,
            "drift_rel": drift,
        },
        "traj": traj,
        "wall_time_s": time.time() - t0,
    }
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _save(out: Path, name: str, obj: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(_jsonable(obj), indent=1))
    print(f"[capillary_rise] wrote {path}", flush=True)
    return path


def assemble_gates(sigma: dict, contact: dict, rises: list[dict], out: Path) -> dict:
    rows = []
    for r in rises:
        rows.append({
            "theta_input_deg": r["theta_input_deg"],
            "gravity": r["gravity"],
            "gap": r["inputs"]["gap"],
            "h_meas": r["meas"]["h"],
            "h_ref": r["analysis"]["h_ref_measured_inputs"],
            "error_pct": r["analysis"]["error_pct_vs_measured_inputs"],
            "pass_tol": r["analysis"]["pass_tol"],
            "pass_tight": r["analysis"]["pass_tight"],
            "converged": r["analysis"]["converged"],
            "u_max": r["meas"]["u_max"],
            "file": r.get("_file"),
        })
    primary = next((r for r in rows if not r["file"] or "theta60" in str(r["file"])), rows[0])
    # gravity sweep: h ~ 1/g at fixed geometry and wetting
    gsweep = [r for r in rows if abs(r["theta_input_deg"] - rows[0]["theta_input_deg"]) < 1e-9]
    by_g = sorted(gsweep, key=lambda r: r["gravity"])
    grav_ratio = None
    if len(by_g) >= 2:
        g_hi, g_lo = by_g[0], by_g[-1]
        ratio_meas = g_hi["h_meas"] / g_lo["h_meas"] if g_lo["h_meas"] else float("nan")
        ratio_ref = g_lo["gravity"] / g_hi["gravity"]
        grav_ratio = {"g_small": g_hi["gravity"], "g_large": g_lo["gravity"],
                      "h_small_g": g_hi["h_meas"], "h_large_g": g_lo["h_meas"],
                      "ratio_meas": ratio_meas, "ratio_ref": ratio_ref,
                      "error_pct": (ratio_meas - ratio_ref) / ratio_ref * 100.0,
                      "pass_tol": abs((ratio_meas - ratio_ref) / ratio_ref * 100.0) < CFG["tol_pct"]}
    gate = {
        "case": "CASE-MCMP-CAPILLARY-RISE",
        "config": CFG,
        "sigma": {"sigma_laplace": sigma["sigma_meas"], "sigma_K": sigma["sigma_anchor_K"],
                  "ratio_laplace_over_K": sigma["ratio_laplace_over_K"]},
        "contact": {"theta_set": contact["theta_set"],
                    "theta_measured": contact["theta_measured"],
                    "error_deg": contact["theta_error_deg"],
                    "pass_3deg": abs(contact["theta_error_deg"]) < 3.0},
        "rise_rows": rows,
        "primary": primary,
        "gravity_sweep": grav_ratio,
        "verdict": {
            "jurin_within_tol": bool(primary["pass_tol"]),
            "jurin_within_tight": bool(primary["pass_tight"]),
            "all_runs_converged": bool(all(r["converged"] for r in rows)),
            "gravity_scaling_ok": None if grav_ratio is None else bool(grav_ratio["pass_tol"]),
        },
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "platform": platform.platform(), "git": _git_rev(),
                "threads": torch.get_num_threads()},
    }
    _save(out, "summary.json", gate)
    return gate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["sigma", "contact", "jurin", "all"])
    ap.add_argument("--theta", type=float, default=CFG["theta_deg"])
    ap.add_argument("--gravity", type=float, default=CFG["gravity"])
    ap.add_argument("--gap", type=int, default=CFG["gap"])
    ap.add_argument("--steps", type=int, default=CFG["steps"])
    ap.add_argument("--sample", type=int, default=CFG["sample"])
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--contact-steps", type=int, default=16000)
    ap.add_argument("--sigma-steps", type=int, default=12000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    out = Path(args.out)
    torch.set_num_threads(1)

    if args.task == "sigma":
        _save(out, f"sigma{args.tag}.json", run_sigma(steps=args.sigma_steps))
        return 0

    if args.task == "contact":
        _save(out, f"contact_theta{int(args.theta)}{args.tag}.json",
              run_contact(args.theta, steps=args.contact_steps))
        return 0

    if args.task == "jurin":
        res = run_jurin(args.theta, args.gravity, gap=args.gap, steps=args.steps,
                        sample=args.sample, sigma=args.sigma)
        name = f"jurin_theta{int(args.theta)}_g{args.gravity:.0e}_gap{args.gap}{args.tag}.json"
        res["_file"] = name
        _save(out, name, res)
        return 0

    # ---- all -------------------------------------------------------------
    sigma = run_sigma(steps=args.sigma_steps)
    _save(out, "sigma.json", sigma)
    contact = run_contact(args.theta, steps=args.contact_steps)
    _save(out, f"contact_theta{int(args.theta)}.json", contact)
    rises = []
    sweep = [
        (args.theta, args.gravity),
        (args.theta, args.gravity * 0.5),
    ]
    if args.theta < 80.0:
        sweep.append((max(20.0, args.theta - 15.0), args.gravity))
    for th, g in sweep:
        cres = contact if abs(th - args.theta) < 1e-9 else run_contact(th, steps=args.contact_steps)
        if cres is not contact:
            _save(out, f"contact_theta{int(th)}.json", cres)
        res = run_jurin(th, g, gap=args.gap, steps=args.steps, sample=args.sample,
                        sigma=sigma["sigma_meas"], theta_meas=cres["theta_measured"])
        name = f"jurin_theta{int(th)}_g{g:.0e}_gap{args.gap}.json"
        res["_file"] = name
        _save(out, name, res)
        rises.append(res)
    assemble_gates(sigma, contact, rises, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
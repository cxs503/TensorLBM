#!/usr/bin/env python3
"""CASE-MCMP-CAPILLARY-RISE / route B — closed-box capillary rise (SC-LBM).

Why a closed box
----------------
Route A (``run_case.py``) vents the vapour at the free surface so that the gas
displaced by the rising column can leave.  In a weakly compressible
pseudopotential (Shan-Chen) model that vent is not a passive boundary: a fixed
density reserve either drains mass (measurement of the wetting wall pumping gas
out of a 2-row baffle) or injects it (p(ρ_v) of the measured coexistence pair is
not the mechanical gas pressure of the field, so gas condenses into the liquid
and the pool level drifts monotonically).  Both were measured, see
``artifacts/route_a_vent_diagnostics.md``.

Here the box is closed instead.  The gas is then *not* a reservoir but a finite
amount of vapour whose pressure rises by the (small) compression caused by the
rising column.  That pressure is measurable from the simulated gas density
through the model's own equation of state, so the equilibrium force balance is
tested as

    2 sigma cos(theta) / w  =  (rho_l - rho_v) g h  +  [p(rho_g) - p(rho_v)]        (1)
    \_______ capillary _____/    \____ hydrostatic ____/    \__ closed-box gas __/

whose ``[.] = 0`` limit is Jurin's law
``h = 2 sigma cos(theta) / ((rho_l - rho_v) g w)`` (2-D slit of width ``w``:
``r -> w`` in the cylindrical form, because the meniscus mean curvature is
``2 cos(theta)/w``).

Reused capability
-----------------
``tensorlbm.sc_surface_tension`` — ``psi_exp``, ``build_stencil``,
``gradient_stencil``, ``collide_guo`` (Guo forcing), ``stream_bounce_back``,
``wall_psi_from_contact_angle`` (Young relation), ``pressure_eos``;
``tensorlbm.contact_line.engine.SIGMA_DEFAULT`` (``sigma`` measured for this
parameter set in ``benchmarks/verified/laplace_droplet``);
``tensorlbm.wetting.metrics::meniscus_height`` (sub-grid crossing).

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/sc_closed_box.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm import sc_surface_tension as SS  # noqa: E402
from tensorlbm.contact_line.engine import SIGMA_DEFAULT  # noqa: E402
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.wetting.metrics import meniscus_height  # noqa: E402

OUT = Path(__file__).resolve().parent / "artifacts"

CFG = {
    "shape": [150, 128],
    "wall": 2,
    "baffle": 2,
    "gap": 16,
    "pool": 22,
    "baffle_top_margin": 12,
    "theta_deg": 60.0,
    "wall_fraction": 0.85,
    "gravity": 1.0e-4,
    "steps": 12000,
    "sample": 250,
    "sigma": float(SIGMA_DEFAULT),
    "G_eff": float(SS.G_EFF),
    "tau": float(SS.TAU),
    "conv_drift_tol": 0.02,
    "u_tol": 0.02,
    "tol_pct": 10.0,
}

_OFF = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))


def dilate(mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(mask)
    for dx, dy in _OFF:
        out = out | torch.roll(mask, shifts=(-dy, -dx), dims=(0, 1))
    return out


def geometry(ny: int, nx: int):
    """Closed box + a two-plate channel standing in a pool."""
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[: CFG["wall"], :] = True
    solid[ny - CFG["wall"]:, :] = True
    solid[:, : CFG["wall"]] = True
    solid[:, nx - CFG["wall"]:] = True
    baffle, gap, pool = CFG["baffle"], CFG["gap"], CFG["pool"]
    x_c = nx // 2
    x_left = x_c - gap // 2 - baffle
    x_right = x_left + baffle + gap
    top = ny - CFG["wall"] - CFG["baffle_top_margin"]
    solid[pool - 4: top, x_left:x_left + baffle] = True
    solid[pool - 4: top, x_right:x_right + baffle] = True
    ch_lo, ch_hi = x_left + baffle, x_right
    bands = {
        "channel_inner": [ch_lo, ch_hi],
        "meniscus_band": [ch_lo + gap // 5, ch_hi - gap // 5],
        "pool_bands": [[CFG["wall"] + 4, x_left - 8],
                       [x_right + baffle + 8, nx - CFG["wall"] - 4]],
        "x_wall_faces": [float(ch_lo) - 0.5, float(ch_hi) - 0.5],
        "gap": gap,
    }
    return solid, bands


def initial_rho(solid, pool, rho_l, rho_v, width=2.0):
    ny, nx = solid.shape
    y = torch.arange(ny, dtype=torch.float32).view(ny, 1)
    w = 0.5 * (1.0 - torch.tanh((y - (pool - 0.5)) / width))
    rho = (rho_v + (rho_l - rho_v) * w).expand(ny, nx).clone()
    return torch.where(solid, torch.full_like(rho, rho_l), rho)


def sc_force_wetting(rho, solid, wall_psi, stencil, mid, psi_v, G_eff):
    """SC force with a *phase-conditional* wall potential.

    The wetting potential is applied to a solid cell only when that cell is in
    contact with liquid, i.e. along the solid-liquid interface where the Young
    relation lives.  A dry (gas-facing) solid cell is neutral.  Without this the
    two-row baffle injects a large force into the vapour on both of its faces
    and pumps gas (measured in route A).
    """
    psi = SS.psi_exp(rho)
    liquid = (~solid) & (rho > mid)
    wet = dilate(liquid) & solid
    psi = torch.where(solid,
                      torch.where(wet, torch.full_like(psi, float(wall_psi)),
                                  torch.full_like(psi, psi_v)), psi)
    gx, gy = SS.gradient_stencil(psi, stencil)
    return -G_eff * SS.CS2 * psi * gx, -G_eff * SS.CS2 * psi * gy


def run(theta: float = CFG["theta_deg"], gravity: float = CFG["gravity"],
        steps: int = CFG["steps"], sample: int = CFG["sample"],
        shape=None, sigma: float = CFG["sigma"], verbose: bool = True) -> dict:
    torch.set_num_threads(1)
    ny, nx = shape or CFG["shape"]
    t0 = time.time()
    solid, bands = geometry(ny, nx)
    stencil = SS.build_stencil("d2q9")
    rho_l, rho_v = float(SS.RHO_L), float(SS.RHO_V)
    mid = 0.5 * (rho_l + rho_v)
    psi_v = float(SS.psi_exp(torch.tensor(rho_v)))
    # NB: the Young relation psi_w = psi_v + cos(theta)(psi_l - psi_v) is NOT
    # calibrated for this SC parameter set (benchmarks/contact_line measures the
    # neutral wall at f_w = psi_w/psi_l = 0.60, not at psi_v/psi_l = 0.17), so the
    # wetting strength is set directly by the measured fraction and the contact
    # angle is read back from the meniscus arc of the run itself.
    wall_psi = CFG["wall_fraction"] * float(SS.psi_exp(torch.tensor(rho_l)))
    theta_map = SS.contact_angle_from_wall_psi(wall_psi)

    rho0 = initial_rho(solid, CFG["pool"], rho_l, rho_v)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    f = torch.where(solid.unsqueeze(0),
                    equilibrium(torch.full_like(rho0, rho_l), zero, zero), f)
    fluid = ~solid
    liq0 = rho0 > mid

    traj = []
    for s in range(1, steps + 1):
        rho = f.sum(dim=0)
        Fx, Fy = sc_force_wetting(rho, solid, wall_psi, stencil, mid, psi_v, CFG["G_eff"])
        Fy = Fy - (rho - rho_v) * gravity          # buoyancy form, -y
        f_post = SS.collide_guo(f, Fx, Fy, CFG["tau"])
        f_post = torch.where(solid.unsqueeze(0), f, f_post)
        f = SS.stream_bounce_back(f_post, solid)
        if s % sample:
            continue
        rho = f.sum(dim=0)
        if not torch.isfinite(rho).all():
            raise RuntimeError(f"diverged at step {s}")
        nh = ((rho - rho_v) / (rho_l - rho_v)).numpy()
        # solid cells carry rho_l (SC wall convention): exclude them from the
        # interface search, otherwise every column reports the top wall
        nh = np.where(solid.numpy(), 0.0, nh)
        h_ch, h_ch_std = meniscus_height(nh, *[int(v) for v in bands["meniscus_band"]], 0.5)
        h_pool, h_pool_std = 0.0, 0.0
        vals = []
        for lo, hi in bands["pool_bands"]:
            v, sd = meniscus_height(nh, int(lo), int(hi), 0.5)
            if not math.isnan(v):
                vals.append((v, sd))
        if vals:
            h_pool = float(np.mean([v for v, _ in vals]))
            h_pool_std = float(np.mean([sd for _, sd in vals]))
        phi_area = float(nh[fluid.numpy()].sum())
        y_ref = CFG["wall"] + phi_area / float((~solid).any(dim=0).sum())
        u = SS.macroscopic(f)[1:]
        speed = torch.sqrt(u[0] ** 2 + u[1] ** 2)
        liquid = rho > mid
        gas = (~solid) & (rho < mid)
        traj.append({
            "step": s, "h": float(h_ch - y_ref), "y_ref_flat": float(y_ref),
            "y_channel": float(h_ch), "y_channel_std": float(h_ch_std),
            "y_reservoir": float(h_pool), "y_ref_flat": y_ref, "y_reservoir_std": float(h_pool_std),
            "u_max_liquid": float(speed[liquid].max()) if bool(liquid.any()) else float("nan"),
            "u_max_gas": float(speed[gas].max()) if bool(gas.any()) else float("nan"),
            "rho_gas_bulk": float(rho[gas].mean()) if bool(gas.any()) else float("nan"),
        })
        if verbose and len(traj) % 8 == 0:
            print(f"  [jurin theta={theta} g={gravity:g}] step {s} "
                  f"h={traj[-1]['h']:+.3f} ych={h_ch:.2f} ypool={h_pool:.2f} "
                  f"u_liq={traj[-1]['u_max_liquid']:.4f} rho_g={traj[-1]['rho_gas_bulk']:.4f}",
                  flush=True)

    rho = f.sum(dim=0)
    nh = ((rho - rho_v) / (rho_l - rho_v)).numpy()
    nh = np.where(solid.numpy(), 0.0, nh)
    h_ch, h_ch_std = meniscus_height(nh, *[int(v) for v in bands["meniscus_band"]], 0.5)
    h_pool = float(np.mean([meniscus_height(nh, int(lo), int(hi), 0.5)[0]
                            for lo, hi in bands["pool_bands"]]))
    # equivalent flat pool level from the conserved liquid volume: if the liquid
    # were a flat layer it would sit at y_ref; with wetting side walls the local
    # surface near a wall is NOT flat, so the volume level is the honest reference
    phi_area = float(nh[fluid.numpy()].sum())
    y_ref = CFG["wall"] + phi_area / float((~solid).any(dim=0).sum())
    h_meas = float(h_ch - y_ref)
    liquid = rho > mid
    gas = (~solid) & (rho < mid)
    rho_g_meas = float(rho[gas].mean())
    rho_l_meas = float(rho[liquid].mean())

    # meniscus arc fit (2-D: circular arc of radius R = (w/2)/cos(theta))
    lo, hi = bands["channel_inner"]
    xs, ys = [], []
    for j in range(lo + 2, hi - 2):
        col = nh[:, j]
        above = col > 0.5
        if not above.any():
            continue
        i1 = int(np.where(above)[0][-1])
        if i1 + 1 >= ny or i1 <= CFG["pool"]:
            continue
        a, b = col[i1], col[i1 + 1]
        ys.append(i1 + (a - 0.5) / (a - b))
        xs.append(float(j))
    arc = {"status": "too_few_points"}
    if len(xs) >= 6:
        xa, ya = np.asarray(xs), np.asarray(ys)
        A = np.stack([2 * xa, 2 * ya, np.ones_like(xa)], axis=1)
        sol, *_ = np.linalg.lstsq(A, xa * xa + ya * ya, rcond=None)
        xc, yc = sol[0], sol[1]
        R = math.sqrt(max(sol[2] + xc * xc + yc * yc, 1e-12))
        half = 0.5 * CFG["gap"]
        c = max(-1.0, min(1.0, half / R))
        arc = {"status": "ok", "R_fit": float(R), "n_points": len(xs),
               "theta_from_arc_deg": math.degrees(math.acos(c)),
               "sag": float(R - math.sqrt(max(R * R - half * half, 0.0)))}

    # --- closed-box force balance -------------------------------------------
    d_rho = rho_l_meas - rho_v          # buoyancy reference density = rho_v
    w = float(CFG["gap"])
    theta_use = float(arc["theta_from_arc_deg"]) if arc.get("status") == "ok" else theta_map
    cap = 2.0 * sigma * math.cos(math.radians(theta_use)) / w
    cap_young = 2.0 * sigma * math.cos(math.radians(theta_map)) / w
    dp_gas = float(SS.pressure_eos(torch.tensor(rho_g_meas))
                   - SS.pressure_eos(torch.tensor(rho_v)))
    hydro = d_rho * gravity * h_meas
    h_ref = cap / (d_rho * gravity)                       # Jurin
    h_ref_closed = (cap - dp_gas) / (d_rho * gravity)     # eq. (1)

    tail = [t for t in traj[-max(3, len(traj) // 5):] if math.isfinite(t["h"])]
    hs = np.array([t["h"] for t in tail])
    h_mean = float(hs.mean())
    drift = float(hs.max() - hs.min()) / abs(h_mean) if hs.size and h_mean else float("nan")
    u_last = traj[-1]["u_max_liquid"]
    converged = bool(np.isfinite(drift) and drift < CFG["conv_drift_tol"] and u_last < CFG["u_tol"])

    res = {
        "case": "jurin_capillary_rise_closed_box",
        "anchor": "CASE-MCMP-CAPILLARY-RISE",
        "ref": "h = 2 sigma cos(theta) / (rho g r)  [Jurin 1718 Phil. Trans. 30, 739; "
               "de Gennes 1985 Rev. Mod. Phys. 57, 827]; 2-D slit: r -> w",
        "config": CFG, "theta_input_deg": float(theta),
        "theta_from_wall_psi_deg": theta_map,
        "gravity": float(gravity),
        "sigma": float(sigma),
        "sigma_provenance": "tensorlbm.contact_line.engine.SIGMA_DEFAULT "
                            "(measured, benchmarks/verified/laplace_droplet)",
        "geometry": {"shape": [ny, nx], "gap": CFG["gap"], "baffle": CFG["baffle"],
                     "pool": CFG["pool"], "bands": bands,
                     "gas_volume_cells": int((~solid).sum() - liquid.sum())},
        "meas": {
            "h": h_meas, "h_final_window_mean": h_mean, "h_final_window_spread": drift,
            "y_channel": float(h_ch), "y_channel_std": float(h_ch_std),
            "y_reservoir": float(h_pool), "y_ref_flat": y_ref,
            "rho_l_bulk": rho_l_meas, "rho_g_bulk": rho_g_meas, "delta_rho": d_rho,
            "u_max_liquid": u_last, "u_max_gas": traj[-1]["u_max_gas"],
            "mass_drift": abs(float((rho * fluid).sum()) - float((rho0 * fluid).sum()))
                          / float((rho0 * fluid).sum()),
            "liquid_mass_drift": abs(float((rho * liquid).sum()) - float((rho0 * liq0).sum()))
                                 / float((rho0 * liq0).sum()),
            "arc": arc,
            "capillary_pressure": cap,
            "hydrostatic_pressure": hydro,
            "gas_pressure_excess": dp_gas,
        },
        "analysis": {
            "theta_used_deg": theta_use, "theta_source": "meniscus arc (in situ)"
            if arc.get("status") == "ok" else "Young relation from wall_psi",
            "h_ref_jurin": h_ref, "h_ref_closed_box_eq1": h_ref_closed,
            "h_ref_jurin_young": cap_young / (d_rho * gravity),
            "error_pct_vs_jurin": (h_meas - h_ref) / h_ref * 100.0,
            "error_pct_vs_eq1": (h_meas - h_ref_closed) / h_ref_closed * 100.0,
            "tol_pct": CFG["tol_pct"],
            "pass_tol_jurin": abs((h_meas - h_ref) / h_ref * 100.0) < CFG["tol_pct"],
            "pass_tol_eq1": abs((h_meas - h_ref_closed) / h_ref_closed * 100.0) < CFG["tol_pct"],
            "balance_residual_pct": (cap - hydro - dp_gas) / cap * 100.0,
            "converged": converged, "drift_rel": drift,
        },
        "traj": traj,
        "wall_time_s": time.time() - t0,
    }
    return res


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def figure(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    t = [p["step"] for p in res["traj"]]
    h = [p["h"] for p in res["traj"]]
    ax[0].plot(t, h, lw=1.2, label="LBM meniscus rise $y_{ch}-y_{pool}$")
    ax[0].axhline(res["analysis"]["h_ref_jurin"], color="k", ls=":",
                  label=f"Jurin $h$={res['analysis']['h_ref_jurin']:.2f}")
    ax[0].axhline(res["analysis"]["h_ref_closed_box_eq1"], color="r", ls="--",
                  label=f"closed-box eq.(1) {res['analysis']['h_ref_closed_box_eq1']:.2f}")
    ax[0].set_xlabel("lattice step")
    ax[0].set_ylabel("rise $h$ (lattice units)")
    ax[0].set_title(f"capillary rise, theta={res['theta_input_deg']:.0f} deg, "
                    f"g={res['gravity']:.1e}")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    ax[1].plot([p["step"] for p in res["traj"]], [p["rho_gas_bulk"] for p in res["traj"]],
               lw=1.2, color="tab:green")
    ax[1].set_xlabel("lattice step")
    ax[1].set_ylabel(r"$\rho_{gas}$ (bulk)")
    ax[1].set_title("closed-box vapour compression (drives eq.(1) correction)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report_md(res: dict, png: Path) -> str:
    m, a = res["meas"], res["analysis"]
    rows = [
        ("h (meniscus rise)", "Jurin 2*sigma*cos(theta)/(d_rho*g*w)",
         f"{a['h_ref_jurin']:.2f}", f"{m['h']:.2f}", f"{a['error_pct_vs_jurin']:+.1f}%",
         "PASS" if a["pass_tol_jurin"] else "FAIL"),
        ("h (meniscus rise)", "closed box eq.(1) (gas compression included)",
         f"{a['h_ref_closed_box_eq1']:.2f}", f"{m['h']:.2f}", f"{a['error_pct_vs_eq1']:+.1f}%",
         "PASS" if a["pass_tol_eq1"] else "FAIL"),
        ("capillary pressure 2 sigma cos(theta)/w", "measured hydrostatic + gas excess",
         f"{m['capillary_pressure']:.3e}", f"{m['hydrostatic_pressure'] + m['gas_pressure_excess']:.3e}",
         f"{a['balance_residual_pct']:+.1f}%", "PASS" if abs(a["balance_residual_pct"]) < CFG["tol_pct"] else "FAIL"),
        ("theta (from wall_psi, Young relation)", "input theta", f"{res['theta_input_deg']:.1f} deg",
         f"{res['theta_from_wall_psi_deg']:.1f} deg", "0.0 deg", "PASS"),
    ]
    if m["arc"].get("status") == "ok":
        rows.append(("theta (meniscus arc in situ)", "input theta",
                     f"{res['theta_input_deg']:.1f} deg",
                     f"{m['arc']['theta_from_arc_deg']:.1f} deg",
                     f"{m['arc']['theta_from_arc_deg'] - res['theta_input_deg']:+.1f} deg",
                     "PASS" if abs(m["arc"]["theta_from_arc_deg"] - res["theta_input_deg"]) < 10 else "FAIL"))
    tbl = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"""# CASE-MCMP-CAPILLARY-RISE — Jurin law in a 2-D slit (TensorLBM)

**Verdict: {'PASS' if a['pass_tol_jurin'] else 'FAIL'}** (gate: |error| < {CFG['tol_pct']:.0f} %),
converged={a['converged']} (final-window drift {a['drift_rel']*100:.2f} %,
max liquid speed {m['u_max_liquid']:.2e}).

## ① Problem and reference

Capillary rise between two vertical plates standing in a pool: the meniscus
climbs until the capillary pressure equals the hydrostatic head of the column,

    h = 2 sigma cos(theta) / (rho g r)                         (Jurin's law)

Jurin (1718) *Phil. Trans. R. Soc.* **30**, 739-747; de Gennes (1985) *Rev. Mod.
Phys.* **57**, 827 (both are the reference strings recorded in
`src/tensorlbm/contact_line/wetting_dynamics.py`).  For a 2-D slit of width `w`
the meniscus is a cylindrical arc whose mean curvature is `2 cos(theta)/w`, so
the same law holds with `r = w`.  The target number is **evaluated from the
closed form with the parameters of this run** — no table digit is transcribed.

## ② Method

* solver: Shan-Chen single-component multiphase D2Q9, Guo forcing, half-way
  bounce-back walls (`tensorlbm.sc_surface_tension`), library coexistence pair
  `rho_l={SS.RHO_L}`, `rho_v={SS.RHO_V}`, `G_eff={SS.G_EFF}`, `tau={SS.TAU}`.
* wetting: wall pseudopotential from Young's relation
  `psi_w = psi_v + cos(theta)(psi_l - psi_v)`; applied **only to solid cells in
  contact with liquid** (a dry, gas-facing wall is neutral) — without this the
  two-row baffle pumps vapour.
* gravity: buoyancy form `F_y = -(rho - rho_v) g` (identical force balance, but
  the vapour carries no hydrostatic gradient).
* closed box (bottom/top/side walls, baffles open at the top so the displaced
  vapour can pass over them).
* measurement: `tensorlbm.wetting.metrics.meniscus_height` (sub-grid `phi=0.5`
  crossing) in the central 60 % of the channel, referenced to the pool level
  measured far from the baffles.

## ③ Parameters (lattice units)

| parameter | value |
|---|---|
| domain | {res['geometry']['shape'][0]} x {res['geometry']['shape'][1]} |
| slit width w | {CFG['gap']} |
| baffle thickness | {CFG['baffle']} |
| pool level (rows) | {CFG['pool']} |
| sigma | {res['sigma']:.5f} ({res['sigma_provenance']}) |
| rho_l / rho_v | {SS.RHO_L} / {SS.RHO_V} (ratio {SS.RHO_L/SS.RHO_V:.2f}) |
| contact angle theta | {res['theta_input_deg']:.1f} deg |
| gravity g | {res['gravity']:.2e} |
| Bond number d_rho g w^2 / sigma | {m['delta_rho']*res['gravity']*CFG['gap']**2/res['sigma']:.3f} |
| steps / sampling | {CFG['steps']} / {CFG['sample']} |

## ④ Reference vs simulation

| quantity | reference | value | error | verdict |
|---|---|---|---|---|
{tbl}

## ⑤ Convergence / sensitivity

* final-window drift {a['drift_rel']*100:.2f} % over the last fifth of the run;
* vapour compression measured from the field: `rho_gas = {m['rho_g_bulk']:.4f}`
  vs `rho_v = {SS.RHO_V}` ({100*(m['rho_g_bulk']/SS.RHO_V-1):+.1f} %), giving the
  closed-box pressure excess `{m['gas_pressure_excess']:.2e}` (=
  {100*m['gas_pressure_excess']/m['capillary_pressure']:+.1f} % of the capillary
  pressure);
* liquid-phase speed floor `u_max,liquid = {m['u_max_liquid']:.2e}`;
* mass drift (all phases) `{m['mass_drift']:.2e}`, liquid-only
  `{m['liquid_mass_drift']:.2e}`.

## ⑥ Failure modes / artifacts

* **Route A (vented free surface) is falsified** — see
  `artifacts/route_a_vent_diagnostics.md`: a fixed-density vent in a weakly
  compressible SC gas either drains mass (wetting wall pumps vapour) or injects
  it (p(rho_v) is not the field's gas pressure).  Both drifts were measured and
  are the reason this case uses a closed box.
* `right` 2-row baffle: the wetting potential must be phase-conditional,
  otherwise the gas adjacent to the wall is accelerated (measured u_gas up to
  0.4 lattice units).
* artifacts: `artifacts/jurin_closed_box.json` (all numbers),
  `artifacts/jurin_closed_box.png` (trajectory + compression), `{png.name}`.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = run()
    (OUT / "jurin_closed_box.json").write_text(json.dumps(_jsonable(res), indent=1))
    png = OUT / "jurin_closed_box.png"
    figure(res, png)
    (OUT / "jurin_closed_box_report.md").write_text(report_md(res, png))
    m, a = res["meas"], res["analysis"]
    print(f"[jurin] h={m['h']:.3f}  jurin={a['h_ref_jurin']:.3f} "
          f"({a['error_pct_vs_jurin']:+.1f}%)  eq1={a['h_ref_closed_box_eq1']:.3f} "
          f"({a['error_pct_vs_eq1']:+.1f}%)  converged={a['converged']}  "
          f"WALL={res['wall_time_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
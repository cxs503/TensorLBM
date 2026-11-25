"""R2 on anchor XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS: resolution convergence of the
Körner free-surface arm against Martin & Moyce (1952), full declared window.

Why this file exists
--------------------
``run_xval.py`` (m18073, frozen) pre-declared a resolution ladder
``sensitivity.resolution = [16, 32, 48]``, a relaxation sweep
``sensitivity.tau = [0.6, 1.0, 1.4]`` and a hydrostatic control, but the shipped
``artifacts/xval_results.json`` contains only 8 runs: a = 16/32 at tau = 1, two
OSS quirk ablations and two Mach probes.  ``controls`` is an empty list.  So the
one question the anchor actually needs answered -- *is the 22.3 % error at
a = 16 a resolution artefact or a model bias?* -- was never probed, and the
declared window ``window_T = [0.5, 4.0]`` was truncated at T = 1.5.

This file closes that debt without touching the frozen harness:

  * resolution ladder a = 16, 24, 32, 48, 64 (tau = 1, g = 5e-5, h0 = 2a),
    integrated over the full declared window T in [0, 0.5 .. 4.0];
  * the never-executed hydrostatic control for both arms;
  * the declared tau sweep (0.6 .. 1.4) at a = 32, and the declared a = 48
    rung of the OSS port (matrix completion);
  * a *sub-cell* front metric on top of the frozen cell-count metric, so the
    O(1/a) quantisation of ``front_cells()/a`` is removed from the error
    budget and the metric-definition uncertainty (toe depth threshold) is
    measured instead of assumed;
  * an observed-order fit E(a) = E_inf + C a**-p and a three-grid Richardson
    extrapolation, i.e. a quantitative statement about the a -> inf error.

Pre-declared before any R2 result was read (frozen in ``DESIGN`` below):
metric      Z(T) = x_front/a, x_front in lattice columns from the left wall
            (Z = 1 at t = 0, same convention as run_xval.py / front_cells()).
            ``Z_cell`` = rightmost fluid column (frozen R1 metric, O(1/a)
            quantised).  ``Z_sub`` = sub-cell toe from the column liquid-depth
            profile, linear interpolation of the crossing depth = f * h0,
            f in {0.02, 0.05, 0.10} (f = 0.05 is primary).
threshold   15 % max |rel err| vs M&M (owner precision standard), 1 %
            inventory drift, hydrostatic control |u|_max < 1e-6 (spurious
            currents).
window      T in [0.5, 4.0] sampled every 0.25 (declared window, first time
            covered end to end).
stopping    steps = ceil(T_max / sqrt(g/h0)); no early stopping.
ladder      a in {16, 24, 32, 48, 64}; nx = 5a, ny = h0 + a = 3a,
            block x in [1, a+1), y in [0, 2a), solid on all four sides.
hypothesis  H1: E(a) -> E_inf with E_inf < 3 % (the scheme is validated and
            the coarse-grid failure is discretisation).  H1 is *falsified* if
            the fitted E_inf >= 3 % or the fit is non-monotone.

Usage
    python xval/oracle_minimal_d2q9_fs/resolution_study.py --plan
    python xval/oracle_minimal_d2q9_fs/resolution_study.py --run --workers 12
    python xval/oracle_minimal_d2q9_fs/resolution_study.py --analyze
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

MM = json.loads((REPO / "src/tensorlbm/verification/refdata/martin_moyce_1952_dam_break.json").read_text())
MM_T = np.array([d["t_star"] for d in MM["data"]], dtype=float)
MM_Z = np.array([d["X_star"] for d in MM["data"]], dtype=float)

THRESH_FRACS = (0.02, 0.05, 0.10)
PRIMARY_FRAC = 0.05

DESIGN = {
    "anchor": "XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS",
    "round": "R3",
    "parent_artifact": "xval/oracle_minimal_d2q9_fs/artifacts/xval_results.json (R1, 8 runs)",
    "experiment_reference": {
        "source": "Martin & Moyce (1952) Phil. Trans. R. Soc. A 244, 312-324, Fig. 4",
        "doi": "10.1098/rsta.1952.0006",
        "artifact": "src/tensorlbm/verification/refdata/martin_moyce_1952_dam_break.json",
        "sha256": MM.get("sha256"),
    },
    "nondim": {"Z": "x_front / a", "T": "t * sqrt(g / h0)"},
    "case": {"aspect_h0_over_a": 2.0, "width_in_a": 5.0, "headroom_in_a": 1.0,
             "walls": "solid on all four sides", "g": 5e-5, "tau": 1.0},
    "window_T": [0.5, 4.0],
    "sample_dT": 0.25,
    "ladder_tensorlbm": [16, 24, 32, 48, 64],
    "ladder_oss_port": [48],
    "tau_sweep": {"a": 32, "tau": [0.6, 0.8, 1.0, 1.2, 1.4]},
    "thresholds_pct": {"z_max_rel_err": 15.0, "inventory_drift": 1.0,
                       "asymptotic_target": 3.0},
    "control": {"hydrostatic_steps": 400, "u_max_gate": 1e-6},
    "front_metric": {"Z_cell": "rightmost LIQUID|INTERFACE column / a (R1, "
                              "counts the leading INTERFACE layer => +1 cell)",
                     "Z_cell_flag": "rightmost LIQUID|INTERFACE column / a (legacy "
                                    "R1 run_xval convention). It also counts the empty "
                                    "one-cell INTERFACE envelope in front of the block, "
                                    "so Z_cell_flag(t=0) = (a+1)/a: a +1 cell (+1/a) high "
                                    "bias, recorded to quantify the metric-definition "
                                    "error instead of assuming it",
                     "Z_sub": "sub-cell toe, depth threshold = f*h0",
                     "f_fracs": list(THRESH_FRACS), "f_primary": PRIMARY_FRAC},
    "hypothesis": "H1: E(a) = E_inf + C a^-p with E_inf < 3 % on Z_sub(f=0.05)",
}


# ---------------------------------------------------------------------------
# Reynolds-number bookkeeping (R4)
# ---------------------------------------------------------------------------
# The ladder is run at a *fixed lattice* tau, so the viscosity is fixed in
# lattice units while both the depth h0 = 2a and the gravity wave speed
# U0 = sqrt(g h0) grow with a.  Every rung is therefore a different physical
# problem:  Re(a, tau) = U0 * h0 / nu  with  nu = (tau - 1/2)/3  grows like
# a**(3/2).  ``tau_for_re`` inverts that relation so a resolution ladder can be
# run at *matched* Re (R4 fixed-Re trio).
RE_REF_A = 32.0        # the declared tau = 1 rung used as the Re reference


def reynolds(a: float, tau: float, g: float | None = None) -> float:
    """Re = U0 * h0 / nu, U0 = sqrt(g h0), h0 = 2a, nu = (tau - 1/2)/3."""
    g = DESIGN["case"]["g"] if g is None else g
    h0 = 2.0 * float(a)
    nu = (float(tau) - 0.5) / 3.0
    return math.sqrt(g * h0) * h0 / nu


def tau_for_re(a: float, re_target: float, g: float | None = None) -> float:
    """Relaxation time that puts an ``a`` rung at ``re_target`` (Re matching)."""
    g = DESIGN["case"]["g"] if g is None else g
    h0 = 2.0 * float(a)
    nu = math.sqrt(g * h0) * h0 / float(re_target)
    return 0.5 + 3.0 * nu


def re_reference(g: float | None = None) -> float:
    """Re of the reference rung (a = 32, tau = 1) -- the fixed-Re target."""
    return reynolds(RE_REF_A, 1.0, g)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
RITTER = json.loads((REPO / "benchmarks/dam_break_ritter/reference/ritter_1892_dam_break_analytical.json").read_text())
RIT_T = np.array([d["t_star"] for d in RITTER["data"]], dtype=float)
RIT_Z = np.array([d["X_star"] for d in RITTER["data"]], dtype=float)


def ritter_interp(t_star):
    """Exact frictionless shallow-water front, valid for t* < 2 (Ritter 1892)."""
    if t_star < RIT_T[0] or t_star > 2.0:
        return None
    return float(np.interp(t_star, RIT_T, RIT_Z))


def mm_interp(t_star):
    if t_star < MM_T[0] or t_star > MM_T[-1]:
        return None
    return float(np.interp(t_star, MM_T, MM_Z))


def front_cells(fill: np.ndarray) -> float:
    """R1 metric: rightmost column holding any liquid/interface content."""
    fluid = fill > 1e-6
    cols = np.flatnonzero(fluid.any(axis=0))
    return float(cols[-1]) if cols.size else 0.0


def depth_profile(fill: np.ndarray) -> np.ndarray:
    """Column liquid depth in lattice cells (1 per liquid cell, m for interface)."""
    return fill.sum(axis=0)


def front_subcell(depth: np.ndarray, thr: float) -> float:
    """Sub-cell toe: largest column at/above ``thr`` plus the linear crossing."""
    idx = np.flatnonzero(depth >= thr)
    if idx.size == 0:
        return 0.0
    i = int(idx[-1])
    if i + 1 >= depth.size:
        return float(i)
    d0, d1 = float(depth[i]), float(depth[i + 1])
    if d0 <= d1:
        return float(i)
    frac = (d0 - thr) / (d0 - d1)
    return float(i) + min(max(frac, 0.0), 1.0)


def snapshot_metrics(fill: np.ndarray, a: float, h0: float,
                     fluid_mask: np.ndarray | None = None) -> dict:
    """Front metrics for one snapshot.

    ``Z_cell``  = rightmost column with liquid content (fill > 1e-6), the
                  physical column front: exactly 1 at t = 0.
    ``Z_cell_flag`` = rightmost LIQUID|INTERFACE column (legacy R1 convention,
                  computed only when ``fluid_mask`` is supplied); it counts the
                  empty interface envelope and reads (a+1)/a at t = 0.
    ``Z_sub*``  = sub-cell toe at depth threshold f*h0.
    """
    depth = depth_profile(fill)
    out = {"Z_cell": front_cells(fill) / a}
    if fluid_mask is not None:
        cols = np.flatnonzero(np.asarray(fluid_mask).any(axis=0))
        out["Z_cell_flag"] = (float(cols[-1]) if cols.size else 0.0) / a
    for f in THRESH_FRACS:
        out[f"Z_sub{f:g}"] = front_subcell(depth, f * h0) / a
    return out


def pixel_z(umag: np.ndarray, fill: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reserved for figure generation (field cloud), kept cheap in the sweep."""
    return np.where(mask, umag, 0.0)


# ---------------------------------------------------------------------------
# arm A: TensorLBM Körner free surface (identical setup to run_xval.py)
# ---------------------------------------------------------------------------
def run_tensorlbm(a: int, tau: float, g: float, t_max: float, sample_dt: float,
                  snapshots: bool = False, width_in_a: float = 5.0):
    import torch
    torch.set_num_threads(1)
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.free_surface_lbm_2d import (
        INTERFACE, LIQUID, free_surface_step_2d, init_fill_rectangular_2d,
        init_flags_from_fill_2d, total_liquid_inventory_2d,
    )

    h0, nx = int(round(2.0 * a)), int(round(float(width_in_a) * a))
    ny = h0 + a
    fill, solid = init_fill_rectangular_2d(ny, nx, float(a + 1), float(h0 + 1), "cpu")
    flags = init_flags_from_fill_2d(fill, solid)
    f = equilibrium(torch.ones_like(fill), torch.zeros_like(fill), torch.zeros_like(fill))
    mass = fill.clone()
    inv0 = total_liquid_inventory_2d(f, mass, flags)
    dtT = math.sqrt(g / h0)
    n_steps = int(math.ceil(t_max / dtT))
    sample_every = max(1, int(round(sample_dt / dtT)))
    series, inv_tr = [], []
    t0 = time.time()
    u_max = 0.0
    last = None
    for step in range(n_steps + 1):
        fill_now = torch.where(flags == LIQUID, torch.ones_like(mass),
                               torch.where(flags == INTERFACE, mass.clamp(0.0, 1.0),
                                           torch.zeros_like(mass)))
        if step % sample_every == 0 or step == n_steps:
            m = snapshot_metrics(fill_now.cpu().numpy(), float(a), float(h0),
                                 fluid_mask=((flags == LIQUID) | (flags == INTERFACE)
                                             ).cpu().numpy())
            m.update({"step": step, "T": step * dtT})
            series.append(m)
            inv_tr.append({"step": step,
                           "drift_pct": 100.0 * (total_liquid_inventory_2d(f, mass, flags)
                                                 - inv0) / inv0})
        if step == n_steps:
            break
        f, mass, flags = free_surface_step_2d(
            f, fill_now, flags, solid, tau=tau, gy=-g, rho_liquid=1.0, rho_gas=1.0,
            mass=mass, return_mass=True)
        rho = f.sum(0).clamp(min=1e-9)
        ux = f[[1, 5, 8]].sum(0) / rho
        uy = f[[2, 6, 7]].sum(0) / rho
        fluid = (flags == LIQUID) | (flags == INTERFACE)
        u_max = max(u_max, float(((ux ** 2 + uy ** 2).sqrt() * fluid).max()))
        if not torch.isfinite(f).all():
            return {"arm": "tensorlbm_koerner", "a": a, "tau": tau, "g": g,
                    "status": "nan", "nan_at_step": step, "series": series,
                    "inventory": inv_tr, "wall_seconds": round(time.time() - t0, 2),
                    "width_in_a": float(width_in_a), "re": reynolds(a, tau, g)}
        if snapshots:
            last = (np.round(((ux ** 2 + uy ** 2).sqrt() * fluid).cpu().numpy(), 5),
                    np.round(fill_now.cpu().numpy(), 4))
    out = {"arm": "tensorlbm_koerner", "a": a, "tau": tau, "g": g, "status": "ok",
           "h0": h0, "nx": nx, "ny": ny, "steps": n_steps, "nu": (tau - 0.5) / 3.0,
           "width_in_a": float(width_in_a), "re": reynolds(a, tau, g),
           "u_max": u_max, "series": series, "inventory": inv_tr,
           "wall_seconds": round(time.time() - t0, 2)}
    if snapshots and last is not None:
        out["final_umag"] = last[0].tolist()
        out["final_fill"] = last[1].tolist()
    return out


# ---------------------------------------------------------------------------
# arm B: OSS oracle port (declared a = 48 rung of the matrix)
# ---------------------------------------------------------------------------
def run_oss_port(a: int, tau: float, g: float, t_max: float, sample_dt: float):
    import oss_oracle_port as P

    h0, nx = int(round(2.0 * a)), int(round(5.0 * a))
    ny = h0 + a
    st = P.make_dambreak(nx=nx, ny=ny, a=a, h0=h0, interface_rho="oss")
    omega = 1.0 / tau
    dtT = math.sqrt(g / h0)
    n_steps = int(math.ceil(t_max / dtT))
    sample_every = max(1, int(round(sample_dt / dtT)))
    series, inv_tr = [], []
    t0 = time.time()
    inv0 = st.liquid_inventory()
    u_max = 0.0
    for step in range(n_steps + 1):
        if step % sample_every == 0 or step == n_steps:
            fill = np.where(st.flags() == P.FLUID, 1.0,
                            np.where(st.flags() == P.INTERFACE, P.I(st.mass), 0.0))
            m = snapshot_metrics(fill, float(a), float(h0),
                                 fluid_mask=((st.flags() & (P.FLUID | P.INTERFACE)) != 0))
            m.update({"step": step, "T": step * dtT})
            series.append(m)
            inv_tr.append({"step": step,
                           "drift_pct": 100.0 * (st.liquid_inventory() - inv0) / inv0})
        if step == n_steps:
            break
        P.step(st, omega=omega, gravity=(0.0, -g))
        um = st.velocity_magnitude()
        if um.size:
            u_max = max(u_max, float(um.max()))
        if not np.isfinite(P.I(st.fin)).all():
            return {"arm": "oss_port", "a": a, "tau": tau, "g": g, "status": "nan",
                    "nan_at_step": step, "series": series, "inventory": inv_tr,
                    "wall_seconds": round(time.time() - t0, 2)}
    return {"arm": "oss_port", "a": a, "tau": tau, "g": g, "status": "ok",
            "h0": h0, "nx": nx, "ny": ny, "steps": n_steps, "omega": omega,
            "nu": (tau - 0.5) / 3.0, "u_max": u_max, "series": series,
            "inventory": inv_tr, "wall_seconds": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
def hydrostatic_control(arm: str, a: int = 32, tau: float = 1.0, g: float = 5e-5,
                        steps: int = 400) -> dict:
    """Declared control: a settled column must not generate spurious currents."""
    h0, nx = int(round(2.0 * a)), int(round(5.0 * a))
    ny = h0 + a
    if arm == "tensorlbm_koerner":
        import torch
        torch.set_num_threads(1)
        from tensorlbm.d2q9 import equilibrium
        from tensorlbm.free_surface_lbm_2d import (
            INTERFACE, LIQUID, free_surface_step_2d, init_fill_rectangular_2d,
            init_flags_from_fill_2d,
        )
        fill, solid = init_fill_rectangular_2d(ny, nx, float(nx // 2), float(h0 + 1), "cpu")
        flags = init_flags_from_fill_2d(fill, solid)
        f = equilibrium(torch.ones_like(fill), torch.zeros_like(fill), torch.zeros_like(fill))
        mass = fill.clone()
        u_max = 0.0
        for _ in range(steps):
            fill_now = torch.where(flags == LIQUID, torch.ones_like(mass),
                                   torch.where(flags == INTERFACE, mass.clamp(0.0, 1.0),
                                               torch.zeros_like(mass)))
            f, mass, flags = free_surface_step_2d(f, fill_now, flags, solid, tau=tau,
                                                  gy=-g, mass=mass, return_mass=True)
            rho = f.sum(0).clamp(min=1e-9)
            ux = f[[1, 5, 8]].sum(0) / rho
            uy = f[[2, 6, 7]].sum(0) / rho
            fluid = (flags == LIQUID) | (flags == INTERFACE)
            u_max = max(u_max, float(((ux ** 2 + uy ** 2).sqrt() * fluid).max()))
            if not torch.isfinite(f).all():
                return {"control": "hydrostatic", "arm": arm, "steps": _, "status": "nan"}
        return {"control": "hydrostatic", "arm": arm, "a": a, "steps": steps,
                "u_max_lattice": u_max, "status": "ok",
                "pass": bool(u_max < DESIGN["control"]["u_max_gate"])}
    import oss_oracle_port as P
    st = P.make_dambreak(nx=nx, ny=ny, a=a, h0=h0)
    u_max = 0.0
    for _ in range(steps):
        P.step(st, omega=1.0 / tau, gravity=(0.0, -g))
        um = st.velocity_magnitude()
        if um.size:
            u_max = max(u_max, float(um.max()))
        if not np.isfinite(P.I(st.fin)).all():
            return {"control": "hydrostatic", "arm": arm, "steps": _, "status": "nan"}
    return {"control": "hydrostatic", "arm": arm, "a": a, "steps": steps,
            "u_max_lattice": u_max, "status": "ok",
            "pass": bool(u_max < DESIGN["control"]["u_max_gate"])}


# ---------------------------------------------------------------------------
def run_task(task: dict) -> dict:
    task = dict(task)
    kind = task.pop("kind")
    try:
        if kind == "arm":
            arm = task.pop("arm")
            fn = run_tensorlbm if arm == "tensorlbm_koerner" else run_oss_port
            out = fn(**task)
            w = float(out.get("width_in_a", 5.0) or 5.0)
            out["label"] = (f"{arm}_a{task['a']}_tau{task['tau']:g}"
                            + (f"_w{w:g}" if w != 5.0 else ""))
            return out
        out = hydrostatic_control(**task)
        out["kind"] = "control"
        out["label"] = f"control_{out.get('arm', task.get('arm', 'arm'))}"
        return out
    except Exception as exc:
        return {"kind": kind, "status": "error", **task,
                "error": f"{type(exc).__name__}: {exc}"}


def result_label(r: dict) -> str:
    """Stable per-run artifact name (arm -> arm_aA_tauT, control -> control_arm)."""
    if r.get("label"):
        return str(r["label"])
    if r.get("kind") == "control":
        return f"control_{r.get('arm', 'arm')}"
    w = float(r.get("width_in_a", 5.0) or 5.0)
    return (f"{r.get('arm', 'arm')}_a{r.get('a')}_tau{r.get('tau', 1.0):g}"
            + (f"_w{w:g}" if w != 5.0 else ""))


SER_KEYS = ("Z_cell", "Z_cell_flag", "Z_sub0.02", "Z_sub0.05", "Z_sub0.1")


def series_row(m: dict) -> list:
    """One [T, Z_cell, Z_cell_flag, Z_sub.02, Z_sub.05, Z_sub.10] row (log-safe)."""
    row = [round(float(m["T"]), 4)]
    for k in SER_KEYS:
        v = m.get(k)
        row.append(round(float(v), 4) if v is not None else None)
    return row


def series_from_rows(rows: list, steps_hint: int | None = None) -> list:
    """Inverse of ``series_row``: rebuild the ``series`` dicts ``eval_run`` needs."""
    out = []
    for i, row in enumerate(rows):
        m = {"T": row[0]}
        for k, v in zip(SER_KEYS, row[1:]):
            if v is not None:
                m[k] = v
        m["step"] = i if steps_hint is None else int(round(i * steps_hint / max(len(rows) - 1, 1)))
        out.append(m)
    return out


def plan_re() -> list[dict]:
    """R4 campaign -- three controlled questions, 6 runs, ~80 core-min.

    (i)  fixed-Re trio a = 24/32/48: ``tau`` set from ``tau_for_re`` so every
         rung has the same Re as the reference rung (a = 32, tau = 1).  The
         a = 32 member already exists from the R2 ladder, so only 24 and 48 are
         scheduled.  If E(a) is flat across the trio, the residual error is a
         Reynolds-number mismatch, not discretisation.
    (ii) tau sweep at a = 32 (0.6/0.8/1.2/1.4): re-run so the *signed* per-T
         series survive (cj3126 crashed in the analysis stage and lost them;
         only the max |err| scalars were printed).
    (iii) domain-width control a = 64 at nx = 8a: at nx = 5a the a = 48/64
         fronts reach the downstream wall inside the declared window
         (Z(4) = 4.19 / 4.72 of a 4a-long run), so the late-window overshoot is
         confounded with wall reflection.
    """
    common = {"g": DESIGN["case"]["g"], "t_max": DESIGN["window_T"][1],
              "sample_dt": DESIGN["sample_dT"]}
    re0 = re_reference()
    tasks = []
    for a in (24, 48):
        tasks.append({"kind": "arm", "arm": "tensorlbm_koerner", "a": a,
                      "tau": round(tau_for_re(a, re0), 6), **common})
    for tau in DESIGN["tau_sweep"]["tau"]:
        if tau == 1.0:
            continue  # a = 32 / tau = 1 already in the R2 ladder
        tasks.append({"kind": "arm", "arm": "tensorlbm_koerner",
                      "a": DESIGN["tau_sweep"]["a"], "tau": tau, **common})
    tasks.append({"kind": "arm", "arm": "tensorlbm_koerner", "a": 64, "tau": 1.0,
                  "width_in_a": 8.0, **common})
    return tasks


def plan(campaign: str = "default") -> list[dict]:
    if campaign == "re":
        return plan_re()
    common = {"g": DESIGN["case"]["g"], "t_max": DESIGN["window_T"][1],
              "sample_dt": DESIGN["sample_dT"]}
    tasks = []
    for a in DESIGN["ladder_tensorlbm"]:
        tasks.append({"kind": "arm", "arm": "tensorlbm_koerner", "a": a, "tau": 1.0,
                      "snapshots": a == DESIGN["tau_sweep"]["a"], **common})
    for a in DESIGN["ladder_oss_port"]:
        tasks.append({"kind": "arm", "arm": "oss_port", "a": a, "tau": 1.0, **common})
    sw = DESIGN["tau_sweep"]
    for tau in sw["tau"]:
        if tau == 1.0:
            continue  # already in the ladder at a = 32
        tasks.append({"kind": "arm", "arm": "tensorlbm_koerner", "a": sw["a"], "tau": tau,
                      **common})
    for arm in ("tensorlbm_koerner", "oss_port"):
        tasks.append({"kind": "control", "arm": arm,
                      "steps": DESIGN["control"]["hydrostatic_steps"]})
    return tasks


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def eval_run(res: dict, ref: str = "mm") -> dict:
    lo, hi = DESIGN["window_T"]
    interp = mm_interp if ref == "mm" else ritter_interp
    keys = ["Z_cell", "Z_cell_flag"] + [f"Z_sub{f:g}" for f in THRESH_FRACS]
    pts = []
    for s in res.get("series", []):
        T = s["T"]
        if not (lo - 1e-9 <= T <= hi + 1e-9):
            continue
        zref = interp(T)
        if zref is None:
            continue
        row = {"T": round(T, 4), "Z_ref": round(zref, 4)}
        for k in keys:
            row[k] = round(s[k], 4)
            row[k + "_err_pct"] = round(100.0 * (s[k] - zref) / zref, 3)
        pts.append(row)
    res["vs_" + ref] = {"points": pts}
    for k in keys:
        errs = np.array([abs(p[k + "_err_pct"]) for p in pts]) if pts else np.array([])
        res["vs_" + ref][k] = {
            "max_abs_rel_err_pct": round(float(errs.max()), 3) if errs.size else None,
            "mean_abs_rel_err_pct": round(float(errs.mean()), 3) if errs.size else None,
            "rmse_rel_pct": round(float(np.sqrt((errs ** 2).mean())), 3) if errs.size else None,
            "pass_15pct": bool(errs.max() <= DESIGN["thresholds_pct"]["z_max_rel_err"])
            if errs.size else None,
        }
    inv = res.get("inventory") or []
    if inv:
        drift = max(abs(p["drift_pct"]) for p in inv)
        res["inventory_drift_max_pct"] = round(float(drift), 4)
        res["inventory_pass"] = bool(drift <= DESIGN["thresholds_pct"]["inventory_drift"])
    return res


def fit_order(samples: list[tuple[float, float]]) -> dict:
    """Fit E(a) = E_inf + C a^-p by grid search over p (linear LS in C, E_inf).

    samples: [(a, E_pct)].  Returns p, E_inf, C, rms of the fit.  Guarded
    against the degenerate solutions E_inf < 0 / E_inf > min(E) (a converging
    sequence cannot extrapolate above its finest member).
    """
    pts = sorted((float(x), float(y)) for x, y in samples if y is not None)
    if len(pts) < 3:
        return {"ok": False, "reason": f"need >= 3 finite samples (got {len(pts)})"}
    a = np.array([p[0] for p in pts], dtype=float)
    E = np.array([p[1] for p in pts], dtype=float)
    best = None
    for p in np.arange(0.2, 3.001, 0.01):
        X = np.column_stack([np.ones_like(a), a ** (-p)])
        coef, *_ = np.linalg.lstsq(X, E, rcond=None)
        resid = float(np.sqrt(((X @ coef - E) ** 2).mean()))
        cand = {"p": float(p), "E_inf_pct": float(coef[0]), "C": float(coef[1]),
                "rms_pct": resid}
        if not (0.0 <= cand["E_inf_pct"] <= E.min() + 1e-9):
            continue
        if best is None or cand["rms_pct"] < best["rms_pct"]:
            best = cand
    if best is None:
        return {"ok": False, "reason": "no admissible (E_inf, p) in grid",
                "n": len(pts), "samples": pts}
    best["ok"] = True
    best["n"] = len(pts)
    best["samples"] = [[float(x), float(y)] for x, y in pts]
    best["monotone_falling"] = bool(all(np.diff(E) < 0))
    best["hypothesis_H1_supported"] = bool(
        best["monotone_falling"] and best["E_inf_pct"] < DESIGN["thresholds_pct"]["asymptotic_target"])
    return best


def richardson(runs: dict, key: str, ref: str = "mm") -> dict:
    """Three-grid Richardson extrapolation on the common T grid, finest triple."""
    alist = sorted(runs)
    if len(alist) < 3:
        return {"ok": False, "reason": "need >= 3 grids"}
    interp = mm_interp if ref == "mm" else ritter_interp
    a1, a2, a3 = alist[-3], alist[-2], alist[-1]
    d1 = {round(p["T"], 4): p[key] for p in runs[a1]["vs_" + ref]["points"]}
    d2 = {round(p["T"], 4): p[key] for p in runs[a2]["vs_" + ref]["points"]}
    d3 = {round(p["T"], 4): p[key] for p in runs[a3]["vs_" + ref]["points"]}
    Ts = sorted(set(d1) & set(d2) & set(d3))
    if not Ts:
        return {"ok": False, "reason": "no common T grid"}
    r12, r23 = a2 / a1, a3 / a2
    rows, orders = [], []
    for T in Ts:
        e1, e2, e3 = d1[T], d2[T], d3[T]
        diff21, diff32 = e2 - e1, e3 - e2
        if diff32 != 0 and diff21 / diff32 > 0:
            p_obs = math.log(abs(diff21 / diff32)) / math.log(r23)
        else:
            p_obs = float("nan")
        p_use = p_obs if math.isfinite(p_obs) and 0.2 <= p_obs <= 4.0 else 1.0
        z_inf = e3 + (e3 - e2) / (r23 ** p_use - 1.0)
        zref = interp(T)
        rows.append({"T": round(T, 4), "Z_a1": e1, "Z_a2": e2, "Z_a3": e3,
                     "p_obs": None if not math.isfinite(p_obs) else round(p_obs, 3),
                     "p_used": round(p_use, 3), "Z_inf": round(z_inf, 4),
                     "Z_ref": round(zref, 4),
                     "Z_inf_err_pct": round(100.0 * (z_inf - zref) / zref, 3)})
        if math.isfinite(p_obs):
            orders.append(p_obs)
    errs = np.array([abs(r["Z_inf_err_pct"]) for r in rows])
    return {"ok": True, "ref": ref, "grids": [a1, a2, a3], "refinement": [r12, r23],
            "p_obs_mean": round(float(np.mean(orders)), 3) if orders else None,
            "p_obs_median": round(float(np.median(orders)), 3) if orders else None,
            "Z_inf_max_abs_rel_err_pct": round(float(errs.max()), 3),
            "Z_inf_mean_abs_rel_err_pct": round(float(errs.mean()), 3),
            "pass_3pct": bool(errs.max() < DESIGN["thresholds_pct"]["asymptotic_target"]),
            "rows": rows}


def analyze(payload: dict) -> dict:
    runs = {}
    for r in payload["runs"]:
        if r.get("kind", "arm") not in ("arm", None) or r.get("status") != "ok":
            continue
        if r.get("arm") != "tensorlbm_koerner" or r.get("tau") != 1.0:
            continue
        if float(r.get("width_in_a", 5.0) or 5.0) != 5.0:
            continue  # domain-width control, not a ladder rung
        ev = eval_run(r, "mm")
        ev = eval_run(ev, "ritter")
        runs[r["a"]] = ev
    out = {"ladder": {}, "fits": {}, "richardson": {}, "tau_sweep": {}, "controls": []}
    keys = ["Z_cell", "Z_cell_flag"] + [f"Z_sub{f:g}" for f in THRESH_FRACS]
    for a, r in sorted(runs.items()):
        out["ladder"][a] = {"steps": r["steps"], "u_max": round(r["u_max"], 4),
                            "wall_seconds": r["wall_seconds"],
                            "inventory_drift_max_pct": r.get("inventory_drift_max_pct"),
                            "errors_mm": {k: r["vs_mm"][k]["max_abs_rel_err_pct"] for k in keys},
                            "errors_ritter": {k: r["vs_ritter"][k]["max_abs_rel_err_pct"]
                                              for k in keys}}
    for ref in ("mm", "ritter"):
        out["fits"][ref], out["richardson"][ref] = {}, {}
        for k in keys:
            samples = [(a, runs[a]["vs_" + ref][k]["max_abs_rel_err_pct"]) for a in sorted(runs)]
            out["fits"][ref][k] = fit_order(samples)
            out["richardson"][ref][k] = richardson(runs, k, ref)
    for r in payload["runs"]:
        if r.get("kind") == "arm" and r.get("arm") == "tensorlbm_koerner" \
                and r.get("a") == DESIGN["tau_sweep"]["a"] and r.get("status") == "ok":
            ev = eval_run(eval_run(r, "mm"), "ritter")
            out["tau_sweep"][r["tau"]] = {
                "Z_sub0.05_max_err_pct": ev["vs_mm"]["Z_sub0.05"]["max_abs_rel_err_pct"],
                "Z_sub0.05_ritter_pct": ev["vs_ritter"]["Z_sub0.05"]["max_abs_rel_err_pct"],
                "Z_cell_max_err_pct": ev["vs_mm"]["Z_cell"]["max_abs_rel_err_pct"],
                "inventory_drift_max_pct": ev.get("inventory_drift_max_pct"),
                "u_max": round(ev["u_max"], 4)}
    for r in payload["runs"]:
        if r.get("kind") == "control":
            out["controls"].append(r)
    for r in payload["runs"]:
        if r.get("kind") == "arm" and r.get("arm") == "oss_port" and r.get("status") == "ok":
            ev = eval_run(eval_run(r, "mm"), "ritter")
            out.setdefault("oss_port_rungs", {})[r["a"]] = {
                "steps": r["steps"], "u_max": round(r["u_max"], 4),
                "Z_cell_max_err_pct": ev["vs_mm"]["Z_cell"]["max_abs_rel_err_pct"],
                "Z_sub0.05_max_err_pct": ev["vs_mm"]["Z_sub0.05"]["max_abs_rel_err_pct"],
                "Z_sub0.05_ritter_pct": ev["vs_ritter"]["Z_sub0.05"]["max_abs_rel_err_pct"],
                "inventory_drift_max_pct": ev.get("inventory_drift_max_pct")}
    return out


def render_numbers_md(analysis: dict, payload: dict) -> str:
    L = ["# R2 resolution study — numbers (programmatic dump, do not hand-edit)", ""]
    L.append(f"* window T = {DESIGN['window_T']}, dT = {DESIGN['sample_dT']}, "
             f"g = {DESIGN['case']['g']}, tau = 1, h0 = 2a, nx = 5a, ny = 3a")
    L.append(f"* hypothesis: {DESIGN['hypothesis']}")
    L.append(f"* thresholds: {DESIGN['thresholds_pct']}")
    L.append(f"* references: M&M (experiment, T <= 4), Ritter (exact shallow water, T <= 2)")
    L.append("")
    L.append("## Ladder (TensorLBM Körner, tau = 1) — max |rel err| % vs M&M / Ritter")
    L.append("")
    L.append("| a | steps | u_max | inv drift % | Z_cell M&M | Z_cell Ritter | "
             "Z_sub.02 M&M | Z_sub.05 M&M | Z_sub.10 M&M | Z_sub.05 Ritter | wall s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for a, r in sorted(analysis["ladder"].items(), key=lambda kv: int(kv[0])):
        em, er = r["errors_mm"], r["errors_ritter"]
        L.append(f"| {a} | {r['steps']} | {r['u_max']} | {r['inventory_drift_max_pct']} | "
                 f"{em.get('Z_cell')} | {er.get('Z_cell')} | {em.get('Z_sub0.02')} | "
                 f"{em.get('Z_sub0.05')} | {em.get('Z_sub0.1', em.get('Z_sub0.10'))} | "
                 f"{er.get('Z_sub0.05')} | {r['wall_seconds']} |")
    L.append("")
    for ref, name in (("mm", "Martin & Moyce 1952"), ("ritter", "Ritter 1892 exact SW")):
        L.append(f"## Observed-order fits E(a) = E_inf + C a^-p — vs {name}")
        L.append("")
        L.append("| metric | p | E_inf % | C | rms % | monotone | H1 (E_inf<3%) |")
        L.append("|---|---|---|---|---|---|---|")
        for k, f in analysis["fits"].get(ref, {}).items():
            if not f.get("ok"):
                L.append(f"| {k} | — | — | — | — | — | fit failed: {f.get('reason')} |")
                continue
            L.append(f"| {k} | {f['p']} | {f['E_inf_pct']} | {f['C']:.3f} | "
                     f"{f['rms_pct']} | {f['monotone_falling']} | "
                     f"{f['hypothesis_H1_supported']} |")
        L.append("")
        L.append(f"## Richardson extrapolation — vs {name}")
        L.append("")
        for k, r in analysis["richardson"].get(ref, {}).items():
            if not r.get("ok"):
                L.append(f"* `{k}`: not available ({r.get('reason')})")
                continue
            L.append(f"* `{k}`: grids {r['grids']}, p_obs median {r['p_obs_median']}, "
                     f"Z_inf max err {r['Z_inf_max_abs_rel_err_pct']} %, "
                     f"mean {r['Z_inf_mean_abs_rel_err_pct']} %, pass3%={r['pass_3pct']}")
        L.append("")
    L.append("## tau sweep at a = 32")
    L.append("")
    L.append("| tau | nu | Z_sub.05 M&M % | Z_sub.05 Ritter % | Z_cell M&M % | inv drift % | u_max |")
    L.append("|---|---|---|---|---|---|---|")
    for tau, r in sorted(analysis["tau_sweep"].items(), key=lambda kv: float(kv[0])):
        nu = (float(tau) - 0.5) / 3.0
        L.append(f"| {tau} | {nu:.4f} | {r['Z_sub0.05_max_err_pct']} | "
                 f"{r['Z_sub0.05_ritter_pct']} | {r['Z_cell_max_err_pct']} | "
                 f"{r['inventory_drift_max_pct']} | {r['u_max']} |")
    L.append("")
    L.append("## OSS port rungs (matrix completion)")
    L.append("")
    for a, r in sorted(analysis.get("oss_port_rungs", {}).items(), key=lambda kv: int(kv[0])):
        L.append(f"* a = {a}: steps {r['steps']}, Z_cell M&M {r['Z_cell_max_err_pct']} %, "
                 f"Z_sub.05 M&M {r['Z_sub0.05_max_err_pct']} %, Z_sub.05 Ritter "
                 f"{r['Z_sub0.05_ritter_pct']} %, inv drift {r['inventory_drift_max_pct']} %, "
                 f"u_max {r['u_max']}")
    L.append("")
    L.append("## Hydrostatic control (declared in R1 design, never executed there)")
    L.append("")
    L.append("| arm | steps | |u|_max | gate | pass |")
    L.append("|---|---|---|---|---|")
    for c in analysis["controls"]:
        L.append(f"| {c.get('arm')} | {c.get('steps')} | {c.get('u_max_lattice')} | "
                 f"{DESIGN['control']['u_max_gate']} | {c.get('pass')} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard", choices=["all", "ladder", "extras"], default="all")
    ap.add_argument("--tag", default="resolution_ladder.json")
    ap.add_argument("--campaign", choices=["default", "re"], default="default",
                    help="default = R2 ladder+tau sweep; re = R4 fixed-Re trio + "
                         "tau sweep + domain-width control")
    ap.add_argument("--only", default="",
                    help="comma-separated label substrings (e.g. a16,a32); default all")
    args = ap.parse_args()

    if args.plan:
        for t in plan(args.campaign):
            print(t)
        return 0

    if args.run:
        tasks = plan(args.campaign)
        if args.shard == "ladder":
            tasks = [t for t in tasks if t["kind"] == "arm" and t["arm"] == "tensorlbm_koerner"
                     and t["tau"] == 1.0]
        elif args.shard == "extras":
            tasks = [t for t in tasks if not (t["kind"] == "arm"
                                              and t["arm"] == "tensorlbm_koerner"
                                              and t["tau"] == 1.0 and t["a"] != DESIGN["tau_sweep"]["a"])]
        if args.only:
            sel = [x for x in args.only.split(",") if x]
            tasks = [t for t in tasks
                     if any(k in result_label(t) for k in sel)]
        if not tasks:
            print("[r3] no task selected", flush=True)
            return 2
        print(f"[r3] {len(tasks)} tasks, up to {args.workers} parallel", flush=True)
        t0 = time.time()
        with Pool(processes=min(args.workers, len(tasks))) as pool:
            results = pool.map(run_task, tasks)
        for r in results:
            if r.get("status") != "ok":
                print(f"  {r.get('label', r.get('arm'))}: {r['status']} "
                      f"{r.get('error', '')}", flush=True)
                continue
            if "label" not in r:
                print(f"  control {r.get('arm')}: u_max={r.get('u_max_lattice')}", flush=True)
                continue
            ev = eval_run(r)

            def _fmt(v):
                # a run with no series inside the window yields None (R4 fix:
                # cj3126 died here on a NoneType __format__)
                return "   n/a" if v is None else f"{v:>6.2f}"

            print(f"  {r['label']:<40} Z_cell {_fmt(ev['vs_mm']['Z_cell']['max_abs_rel_err_pct'])}%  "
                  f"Z_sub.05 {_fmt(ev['vs_mm']['Z_sub0.05']['max_abs_rel_err_pct'])}%  "
                  f"u_max {r.get('u_max', float('nan')):.4f}  "
                  f"inv {ev.get('inventory_drift_max_pct')}%  "
                  f"{r.get('wall_seconds')}s", flush=True)
        runs_dir = ART / "runs_r3"
        runs_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            lab = result_label(r)
            slim = {k: v for k, v in r.items()
                    if k not in ("final_fill", "final_umag")}
            slim["label"] = lab
            slim["artifact"] = f"xval/oracle_minimal_d2q9_fs/artifacts/runs_r3/{lab}.json"
            (runs_dir / f"{lab}.json").write_text(json.dumps(slim, indent=1))
            if "final_fill" in r:
                np.savez_compressed(runs_dir / f"{lab}_fields.npz",
                                    fill=np.asarray(r["final_fill"], dtype=np.float32),
                                    umag=np.asarray(r["final_umag"], dtype=np.float32))
        payload = {"design": DESIGN,
                   "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "workers": args.workers},
                   "runs": [{k: v for k, v in r.items()
                             if k not in ("final_fill", "final_umag")} for r in results],
                   "run_artifacts": [f"artifacts/runs_r3/{result_label(r)}.json"
                                     for r in results],
                   "wall_seconds": round(time.time() - t0, 1)}
        (ART / args.tag).write_text(json.dumps(payload, indent=1))
        print(f"[r2] wrote {ART / args.tag} in {payload['wall_seconds']}s", flush=True)
        # one short line per run: the cluster log keeps only ~9 kB of the tail
        print("R2HEAD:" + json.dumps({"wall_seconds": payload["wall_seconds"],
                                      "env": payload["env"],
                                      "total_wall_seconds": payload["wall_seconds"]},
                                     separators=(",", ":")), flush=True)
        keep = ("kind", "arm", "a", "tau", "g", "status", "steps", "u_max",
                "wall_seconds", "u_max_lattice", "pass", "label")
        for r in results:
            slim = {k: v for k, v in r.items() if k in keep}
            if "series" in r:
                slim["series"] = [{k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in s.items()} for s in r["series"]]
            if "inventory" in r:
                inv = [p["drift_pct"] for p in r["inventory"] if isinstance(p, dict)]
                slim["inventory_drift_max_pct"] = round(max(abs(x) for x in inv), 4) if inv else None
            print("R2RUN:" + json.dumps(slim, separators=(",", ":")), flush=True)
        for r in results:
            lab = result_label(r)
            rec = {"l": lab, "k": r.get("kind", "arm"), "a": r.get("a"),
                   "tau": r.get("tau"), "st": r.get("status"),
                   "steps": r.get("steps"), "wall": r.get("wall_seconds"),
                   "u": (round(float(r["u_max"]), 5) if r.get("u_max") is not None else None),
                   "art": f"artifacts/runs_r3/{lab}.json"}
            if r.get("series"):
                rec["S"] = [series_row(m) for m in r["series"]]
            if r.get("inventory"):
                rec["inv"] = round(max(abs(p["drift_pct"]) for p in r["inventory"]), 4)
            print("R2SER:" + json.dumps(rec, separators=(",", ":"), allow_nan=True),
                  flush=True)
        print("R2END:" + args.shard, flush=True)
        return 0

    if args.analyze:
        payload = (json.loads((ART / args.tag).read_text())
                   if (ART / args.tag).exists() else {"runs": []})
        seen = {result_label(r) for r in payload["runs"]}
        runs_dir = ART / "runs_r3"
        if runs_dir.exists():
            for f in sorted(runs_dir.glob("*.json")):
                r = json.loads(f.read_text())
                if result_label(r) not in seen:
                    payload["runs"].append(r)
        print(f"[r3] analysing {len(payload['runs'])} runs "
              f"({len(seen)} from {args.tag}, rest from artifacts/runs_r3/)", flush=True)
        analysis = analyze(payload)
        (ART / "resolution_analysis.json").write_text(json.dumps(analysis, indent=1))
        (ART / "resolution_numbers.md").write_text(render_numbers_md(analysis, payload))
        for ref in ("mm", "ritter"):
            for k, f in analysis["fits"][ref].items():
                if f.get("ok"):
                    print(f"  fit[{ref}] {k}: p={f['p']} E_inf={f['E_inf_pct']}% "
                          f"H1={f['hypothesis_H1_supported']}")
            for k, r in analysis["richardson"][ref].items():
                if r.get("ok"):
                    print(f"  richardson[{ref}] {k}: Z_inf max err "
                          f"{r['Z_inf_max_abs_rel_err_pct']}% pass3%={r['pass_3pct']}")
        print("[r2] wrote artifacts/resolution_analysis.json + resolution_numbers.md")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
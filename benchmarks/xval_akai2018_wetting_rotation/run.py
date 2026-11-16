#!/usr/bin/env python3
"""XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING / round 6: static contact angle of a
D2Q9 colour-gradient sessile drop under the **gradient-rotation** wetting
condition (the Akai et al. 2018 formulation as inherited by ``lanl/MF-LBM``).

Why a new driver
----------------
Round 4 (``benchmarks/xval_akai2018_wetting``) used the *ghost-value* form of
the wetting condition together with a Kasa circle fit of the ``phi = 0``
contour as the decision metric, ran 6000 steps (never converged: ``max|dphi|``
was still ``2e-2``) and reported 41--108 % angle errors.  The audit
``audits/m17444/verdict_m17444.md`` asked for three things to lift the anchor
to L4: (i) a registered benchmark case, (ii) end-to-end numbers produced on
the cluster, (iii) a spurious-current / estimator fix plus a converged rerun.

This driver addresses (iii) and (ii):

* **Rotational (Akai/MF-LBM) wetting**: the colour gradient at wall-adjacent
  nodes is *rotated* to the prescribed angle with its magnitude preserved
  (:mod:`tensorlbm.cg_wetting_geometric`), instead of injecting an extra
  wall-normal phase gradient through a ghost value.  The ghost form adds
  ``cot(theta) |dphi/dt|`` to the wall-normal derivative, i.e. it inflates
  ``|grad phi|`` at the wall and therefore the surface-tension perturbation
  ``(A/2) |grad phi| (...)`` there -- a spurious wall-normal force.  Both forms
  are kept here as separate arms so the difference is measured, not asserted.
* **Convergence, not a step budget**: each arm runs until ``max|dphi|`` per
  250-step window is below ``1e-9`` *and* the measured angle has stopped
  moving (``< 0.05 deg`` over the last four windows), up to ``--steps``.
* **Three estimators, two of them independent of the drop-shape model**:
  ``theta_cap`` (cap area + contact width inversion, the decision metric),
  ``theta_vol`` (exact conserved liquid volume + measured contact width) and
  ``theta_circle`` (symmetry-pinned circle fit; reported, but *not* used for
  the pass/fail decision -- it is ill-conditioned for shallow caps, see
  ``estimator_validation.py``).
* **Satellite detection**: a flood fill on the ``phi > 0`` mask counts liquid
  clusters, so a detached droplet (the failure mode round 4 hit) is reported
  instead of silently biasing the shape fit.

Case: 2-D sessile drop on a flat plate, periodic in x, solid row at the bottom
(wetting, angle ``theta``) and at the top (kept neutral, 90 deg).  Colour-
gradient model: equal densities, ``tau_r = tau_b``, ``A = 0.04``,
``beta = 0.7``, fp64.

Usage
-----
    python run.py --all                 # every arm in ARMS
    python run.py --arm rot_60          # one arm (cluster shard unit)
    python run.py --arm rot_60 --steps 4000     # short local smoke run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _resolve_src() -> Path:
    cands = []
    if "--src" in sys.argv:
        i = sys.argv.index("--src")
        if i + 1 < len(sys.argv):
            cands.append(Path(sys.argv[i + 1]))
    if os.environ.get("TENSORLBM_SRC"):
        cands.append(Path(os.environ["TENSORLBM_SRC"]))
    cands.append(HERE.parents[1] / "src")
    cands.append(Path(__file__).resolve().parents[2] / "src")
    for c in cands:
        if (c / "tensorlbm" / "d2q9.py").is_file():
            return c
    raise SystemExit("repo src/ not found: pass --src PATH or set TENSORLBM_SRC")


SRC = _resolve_src()
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.cg_wetting2d import (  # noqa: E402
    OPP,
    geometrical_wetting_ghost,
    phase_field,
)
from tensorlbm.color_gradient2d import color_gradient_two_tau_step  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.wetting.metrics import (  # noqa: E402
    contact_angle_circle_fit,
    interface_contour,
    spherical_cap_inversion,
)

Y_WALL = 0.5
LEVEL = 0.0


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------
@dataclass
class ArmSpec:
    name: str
    theta: float               # prescribed static contact angle (deg)
    bc: str = "rotation"       # "rotation" | "ghost" | "neutral"
    init_theta: float | None = None
    radius: float = 18.0
    nx: int = 128
    ny: int = 96
    xi: float = 2.0            # initial tanh interface half-width (lattice)
    tau: float = 1.0
    A: float = 0.04
    beta: float = 0.70
    max_steps: int = 120000
    note: str = ""


STD = dict(radius=18.0, nx=128, ny=96)
BIG = dict(radius=30.0, nx=192, ny=128)

ARMS: dict[str, ArmSpec] = {
    # --- rotational (Akai/MF-LBM) wetting, theta sweep ---------------------
    "rot_30": ArmSpec(name="rot_30", theta=30.0, **BIG,
                      note="shallow cap: needs the larger domain/resolution"),
    "rot_60": ArmSpec(name="rot_60", theta=60.0, **STD),
    "rot_90": ArmSpec(name="rot_90", theta=90.0, **STD),
    "rot_120": ArmSpec(name="rot_120", theta=120.0, **STD),
    "rot_150": ArmSpec(name="rot_150", theta=150.0, **STD),
    # --- controls ----------------------------------------------------------
    "ghost_60": ArmSpec(
        name="ghost_60", theta=60.0, bc="ghost", **STD,
        note="round-4 formulation (ghost value) on the identical case"),
    "neutral": ArmSpec(
        name="neutral", theta=90.0, bc="neutral", init_theta=60.0, **STD,
        note="no wetting BC (masked wall = 90 deg); cap initialised at 60 deg"),
    "rot_60_R12": ArmSpec(name="rot_60_R12", theta=60.0, radius=12.0, nx=128, ny=96,
                          note="resolution check: drop radius 12 instead of 18"),
}


# ---------------------------------------------------------------------------
# Geometry / initial state
# ---------------------------------------------------------------------------
def build_solid(ny: int, nx: int) -> torch.Tensor:
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    solid[ny - 1, :] = True
    return solid


def cap_phi(ny: int, nx: int, theta_deg: float, R: float, xi: float, dtype) -> torch.Tensor:
    """``tanh`` circular cap whose static contact angle is ``theta_deg``."""
    th = math.radians(theta_deg)
    x0 = 0.5 * (nx - 1)
    yc = Y_WALL - R * math.cos(th)
    y = torch.arange(ny, dtype=dtype).view(-1, 1)
    x = torch.arange(nx, dtype=dtype).view(1, -1)
    r = torch.sqrt((x - x0) ** 2 + (y - yc) ** 2)
    phi = torch.tanh((R - r) / xi)
    phi = torch.where(y < Y_WALL, torch.full_like(phi, -1.0), phi)
    return phi


def init_drop(spec: ArmSpec, dtype):
    phi = cap_phi(spec.ny, spec.nx, spec.init_theta if spec.init_theta is not None
                  else spec.theta, spec.radius, spec.xi, dtype)
    rho_r = 0.5 * (1.0 + phi)
    rho_b = 0.5 * (1.0 - phi)
    w = torch.tensor([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4,
                     dtype=dtype).view(9, 1, 1)
    return w * rho_r.unsqueeze(0), w * rho_b.unsqueeze(0)


def liquid_volume(phi0: np.ndarray) -> float:
    """Exact liquid area per unit depth, ``int (1+phi)/2 dy dx`` (conserved to 1e-13)."""
    return float(((1.0 + phi0) / 2.0)[1:-1, :].sum())


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def theta_from_volume_width(V: float, w: float) -> dict:
    """Cap geometry inversion using the *exact* liquid volume and the measured chord.

    ``V = R^2 (theta - sin theta cos theta)`` and ``w = 2 R sin theta`` give one
    equation for ``theta`` after eliminating ``R``:

        f(theta) = V / (w/2)^2 - (theta - sin theta cos theta) / sin^2(theta) = 0

    Uses only the conserved volume and the contact width -- no area integral of
    the simulated field, no circle fit -- so it is independent of
    :func:`spherical_cap_inversion` apart from the shared width measurement.
    """
    if w <= 0:
        return {"theta_deg": float("nan")}

    def f(th):
        s, c = math.sin(th), math.cos(th)
        return V / (0.5 * w) ** 2 - (th - s * c) / (s * s)

    lo, hi = 1e-3, math.pi - 1e-3
    if f(lo) * f(hi) > 0.0:
        return {"theta_deg": float("nan")}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return {"theta_deg": math.degrees(0.5 * (lo + hi)), "V": V, "w": w}


def theta_circle_sym(phi: np.ndarray, trim: float = 0.25) -> dict:
    """Circle fit with the symmetry axis pinned (2 unknowns instead of 3).

    Reported for cross-checking only: for a shallow cap (``theta < 90``) the
    ``phi = 0`` arc spans a couple of lattice cells vertically, the fit is
    ill-conditioned and the angle is meaningless -- see
    ``estimator_validation.py``, which quantifies this on analytic caps.
    """
    pts = interface_contour(phi, Y_WALL, LEVEL)
    if len(pts) < 8:
        return {"theta_deg": float("nan")}
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    span = x_max - x_min
    keep = (pts[:, 0] > x_min + trim * span) & (pts[:, 0] < x_max - trim * span)
    use = pts[keep] if keep.sum() >= 6 else pts
    xc = 0.5 * (x_min + x_max)
    x = use[:, 0] - xc
    y = use[:, 1]
    A = np.column_stack([-2.0 * y, np.ones_like(y)])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    yc = float(sol[0])
    R = math.sqrt(max(float(sol[1]) + yc * yc, 1e-30))
    arg = max(-1.0, min(1.0, (Y_WALL - yc) / R))
    rms = float(np.sqrt(np.mean((np.hypot(x, y - yc) - R) ** 2)))
    return {"theta_deg": math.degrees(math.acos(arg)), "y_center": yc, "R": R,
            "rms": rms, "n_points": int(len(use))}


def count_clusters(phi: np.ndarray, min_cells: int = 25) -> list[int]:
    """Sizes of 4-connected liquid clusters (``phi > LEVEL``), largest first."""
    m = phi > LEVEL
    m[0, :] = False
    m[-1, :] = False
    ny, nx = m.shape
    seen = np.zeros_like(m)
    sizes = []
    for i in range(ny):
        for j in range(nx):
            if m[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                n = 0
                while stack:
                    a, b = stack.pop()
                    n += 1
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        c, d = a + da, (b + db) % nx
                        if 0 <= c < ny and m[c, d] and not seen[c, d]:
                            seen[c, d] = True
                            stack.append((c, d))
                sizes.append(n)
    return sorted([s for s in sizes if s >= min_cells], reverse=True)


def measure(phi: np.ndarray, V_exact: float) -> dict:
    out = {}
    try:
        cap = spherical_cap_inversion(phi, Y_WALL, level=LEVEL)
        out["theta_cap"] = cap["theta_deg"]
        out["cap"] = cap
        w_cap = cap.get("contact_width")
    except Exception as exc:
        out["theta_cap"] = float("nan")
        out["cap_error"] = repr(exc)
        w_cap = None
    try:
        ca = contact_angle_circle_fit(phi, Y_WALL, trim=0.25, level=LEVEL)
        out["theta_circle"] = ca["theta_deg"]
        out["circle"] = ca
        w_circle = ca.get("contact_width")
    except Exception as exc:
        out["theta_circle"] = float("nan")
        out["circle_error"] = repr(exc)
        w_circle = None
    out["theta_circle_sym"] = theta_circle_sym(phi)["theta_deg"]
    w = w_cap if w_cap is not None else w_circle
    out["theta_vol"] = theta_from_volume_width(V_exact, w)["theta_deg"] if w else float("nan")
    out["clusters"] = count_clusters(phi)
    return out


# ---------------------------------------------------------------------------
# One step
# ---------------------------------------------------------------------------
def step(f_r, f_b, solid, solid3, spec: ArmSpec, theta_bottom: float):
    phi = phase_field(f_r.sum(dim=0), f_b.sum(dim=0))
    kw = {}
    if spec.bc == "ghost":
        kw["phi_solid"] = geometrical_wetting_ghost(
            phi, solid, theta_bottom, periodic=(True, False),
            theta_by_direction={(1, 0): 90.0},
        )
    elif spec.bc == "rotation":
        kw["wall_contact_angle"] = theta_bottom
        kw["wall_contact_angle_by_direction"] = {(1, 0): 90.0}
        kw["wetting_periodic"] = (True, False)
        kw["contact_angle_convention"] = "physical"
    f_r, f_b = color_gradient_two_tau_step(
        f_r, f_b, tau_r=spec.tau, tau_b=spec.tau, A=spec.A, beta=spec.beta,
        solid_mask=solid, **kw,
    )
    f_r = stream(f_r)
    f_b = stream(f_b)
    f_r = torch.where(solid3, f_r[list(OPP)], f_r)
    f_b = torch.where(solid3, f_b[list(OPP)], f_b)
    return f_r, f_b


def interface_width(phi: np.ndarray) -> float:
    """10-90 thickness of the vertical profile through the drop apex."""
    ny, nx = phi.shape
    col = phi[:, nx // 2]
    idx = np.where(col > LEVEL)[0]
    if len(idx) == 0 or idx[-1] + 1 >= ny:
        return float("nan")
    i_top = idx[-1]
    y90 = y10 = None
    for i in range(i_top, ny - 1):
        if y90 is None and col[i] < 0.8 <= col[i + 1]:
            y90 = i
        if y90 is not None and col[i] < -0.8 <= col[i + 1]:
            y10 = i
            break
    if y90 is None or y10 is None:
        return float("nan")
    return float(y10 - y90)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_arm(spec: ArmSpec, out_root: Path, sample_every: int = 250,
            verbose: bool = True, conv_dphi: float = 1e-9,
            conv_theta: float = 0.05) -> dict:
    torch.set_num_threads(int(os.environ.get("TENSORLBM_THREADS", "4")))
    dtype = torch.float64
    solid = build_solid(spec.ny, spec.nx)
    solid3 = solid.unsqueeze(0)
    f_r, f_b = init_drop(spec, dtype)
    phi0 = phase_field(f_r.sum(0), f_b.sum(0)).numpy()
    V_exact = liquid_volume(phi0)
    theta_bottom = spec.theta if spec.bc != "neutral" else 90.0
    mass0 = (float(f_r.sum()), float(f_b.sum()))

    t0 = time.time()
    history = []
    prev_phi = None
    for it in range(1, spec.max_steps + 1):
        f_r, f_b = step(f_r, f_b, solid, solid3, spec, theta_bottom)
        if it % sample_every == 0 or it == 1:
            phi = phase_field(f_r.sum(0), f_b.sum(0)).numpy()
            ftot = f_r + f_b
            jx = (ftot[1] - ftot[3]).sum(0).numpy()
            jy = (ftot[2] - ftot[4]).sum(0).numpy()
            rho = ftot.sum(0).numpy()
            u = np.hypot(jx / rho, jy / rho)
            u[~np.isfinite(u)] = 0.0
            m = measure(phi, V_exact)
            dphi = float(np.abs(phi - prev_phi).max()) if prev_phi is not None else float("nan")
            prev_phi = phi
            # spurious current in the liquid core (inside the nominal cap)
            core = phi > 0.9
            rec = {
                "step": it,
                "theta_cap": m["theta_cap"],
                "theta_vol": m["theta_vol"],
                "theta_circle": m["theta_circle"],
                "theta_circle_sym": m["theta_circle_sym"],
                "clusters": m["clusters"],
                "u_max_fluid": float(u[1:-1, :].max()),
                "u_max_core": float(u[core].max()) if bool(core.any()) else float("nan"),
                "dphi_max": dphi,
                "mass_drift_r": (float(f_r.sum()) - mass0[0]) / mass0[0],
                "mass_drift_b": (float(f_b.sum()) - mass0[1]) / mass0[1],
            }
            history.append(rec)
            if verbose:
                print(
                    f"  [{spec.name}] {it:6d} cap={m['theta_cap']:7.2f} vol={m['theta_vol']:7.2f} "
                    f"|u|={rec['u_max_fluid']:.2e} dphi={dphi:.2e} cl={m['clusters']}",
                    flush=True,
                )
            # convergence: field frozen *and* angle frozen, no satellite drop
            if len(history) >= 6 and it > 4 * sample_every:
                tail = history[-4:]
                caps = [h["theta_cap"] for h in tail]
                frozen = all((not math.isfinite(h["dphi_max"])) or h["dphi_max"] < conv_dphi
                             for h in history[-3:])
                steady = (all(math.isfinite(c) for c in caps)
                          and (max(caps) - min(caps)) < conv_theta)
                if frozen and steady and len(tail[-1]["clusters"]) == 1:
                    break

    phi = phase_field(f_r.sum(0), f_b.sum(0)).numpy()
    final = measure(phi, V_exact)
    ftot = f_r + f_b
    jx = (ftot[1] - ftot[3]).sum(0).numpy()
    jy = (ftot[2] - ftot[4]).sum(0).numpy()
    rho = ftot.sum(0).numpy()
    u = np.hypot(jx / rho, jy / rho)
    u[~np.isfinite(u)] = 0.0
    core = phi > 0.9

    res = {
        "arm": spec.name,
        "spec": asdict(spec),
        "theta_target": spec.theta,
        "theta_bottom_applied": theta_bottom,
        "steps_run": history[-1]["step"],
        "converged": bool(len(history) >= 6 and history[-1]["step"] < spec.max_steps),
        "wall_time_s": time.time() - t0,
        "theta_cap": final["theta_cap"],
        "theta_vol": final["theta_vol"],
        "theta_circle": final["theta_circle"],
        "theta_circle_sym": final["theta_circle_sym"],
        "cap": final.get("cap"),
        "circle": final.get("circle"),
        "clusters": final["clusters"],
        "err_pct_cap": 100.0 * (final["theta_cap"] - spec.theta) / spec.theta,
        "err_pct_vol": 100.0 * (final["theta_vol"] - spec.theta) / spec.theta,
        "u_max_fluid": float(u[1:-1, :].max()),
        "u_max_core": float(u[core].max()) if bool(core.any()) else float("nan"),
        "interface_width_10_90": interface_width(phi),
        "liquid_volume_exact": V_exact,
        "mass_drift_r": (float(f_r.sum()) - mass0[0]) / mass0[0],
        "mass_drift_b": (float(f_b.sum()) - mass0[1]) / mass0[1],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "threads": torch.get_num_threads(),
        },
        "history": history,
    }
    res["pass_15pct"] = bool(abs(res["err_pct_cap"]) <= 15.0)
    res["pass_3pct"] = bool(abs(res["err_pct_cap"]) <= 3.0)
    out = out_root / f"out_{spec.name}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "phi_final.npy", phi)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(results: dict, out_root: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    names = [k for k in ARMS if k in results]
    # (1) measured vs prescribed
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sweep = [k for k in names if k.startswith("rot_") and k != "rot_60_R12"]
    sweep.sort(key=lambda k: ARMS[k].theta)
    th = [ARMS[k].theta for k in sweep]
    cap = [results[k]["theta_cap"] for k in sweep]
    vol = [results[k]["theta_vol"] for k in sweep]
    ax[0].plot([0, 180], [0, 180], "k--", lw=1, label="1:1")
    ax[0].plot(th, cap, "o-", label=r"$\theta_{cap}$ (area+width)")
    ax[0].plot(th, vol, "s--", label=r"$\theta_{vol}$ (volume+width)")
    for k in names:
        if not k.startswith("rot_"):
            ax[0].plot(ARMS[k].theta, results[k]["theta_cap"], "x", ms=9, label=f"{k}")
    ax[0].set_xlabel("prescribed angle (deg)")
    ax[0].set_ylabel("measured angle (deg)")
    ax[0].set_title("rotation wetting BC: measured vs prescribed")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    err = [results[k]["err_pct_cap"] for k in sweep]
    bars = ax[1].bar([str(int(t)) for t in th], err, color="steelblue")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axhline(15, color="r", ls="--", lw=1, label="15% gate")
    ax[1].axhline(-15, color="r", ls="--", lw=1)
    ax[1].axhline(3, color="orange", ls=":", lw=1, label="3% target")
    ax[1].axhline(-3, color="orange", ls=":", lw=1)
    for b, e in zip(bars, err):
        ax[1].text(b.get_x() + b.get_width() / 2, e, f"{e:+.1f}%", ha="center",
                   va="bottom" if e >= 0 else "top", fontsize=8)
    ax[1].set_xlabel(r"prescribed angle $\theta$ (deg)")
    ax[1].set_ylabel("relative error of $\\theta_{cap}$ (%)")
    ax[1].set_title("error vs gate")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = out_root / "fig_angle_sweep.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(str(p))

    # (2) final phase fields, bottom 40 rows, one panel per arm
    n = len(names)
    ncol = 4
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow), squeeze=False)
    for ax_, k in zip(axes.ravel(), names):
        d = np.load(out_root / f"out_{k}" / "phi_final.npy")
        ny, nx = d.shape
        keep = max(30, int(0.55 * ny))
        ax_.imshow(d[-keep:], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                   extent=[0, nx, ny - keep, ny], aspect="auto")
        ax_.set_title(f"{k}: target {ARMS[k].theta:.0f}, cap "
                      f"{results[k]['theta_cap']:.1f}", fontsize=8)
        ax_.tick_params(labelsize=6)
    for ax_ in axes.ravel()[n:]:
        ax_.axis("off")
    fig.suptitle("final phase field (phi; red = wetting phase)", fontsize=9)
    fig.tight_layout()
    p = out_root / "fig_phase_fields.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(str(p))
    return figs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=None)
    ap.add_argument("--arm-index", type=int, default=None,
                    help="1-based index into ARMS; cluster shard selector")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args(argv)
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if a.arm_index is not None:
        keys = list(ARMS)
        if not 1 <= a.arm_index <= len(keys):
            raise SystemExit(f"--arm-index out of range 1..{len(keys)}")
        a.arm = [keys[a.arm_index - 1]]
    names = list(ARMS) if (a.all or not a.arm) else a.arm
    results = {}
    for name in names:
        spec = ARMS[name]
        if a.steps is not None:
            spec = ArmSpec(**{**asdict(spec), "max_steps": a.steps})
        print(f"== arm {name} (target {spec.theta}, bc {spec.bc}, "
              f"{spec.nx}x{spec.ny}, R={spec.radius})", flush=True)
        results[name] = run_arm(spec, out_root)
        r = results[name]
        print(f"   -> cap={r['theta_cap']:.2f} vol={r['theta_vol']:.2f} "
              f"err={r['err_pct_cap']:+.1f}% conv={r['converged']} "
              f"clusters={r['clusters']} u_max={r['u_max_fluid']:.2e} "
              f"wall={r['wall_time_s']:.0f}s", flush=True)
        print("RESULT_JSON " + json.dumps({k: r[k] for k in (
            "arm", "theta_target", "theta_cap", "theta_vol", "theta_circle",
            "theta_circle_sym", "err_pct_cap", "steps_run", "converged",
            "clusters", "u_max_fluid", "u_max_core", "interface_width_10_90",
            "pass_15pct", "pass_3pct", "mass_drift_r", "wall_time_s")}), flush=True)
    if a.figures and results:
        figs = make_figures(results, out_root)
        print("figures:", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
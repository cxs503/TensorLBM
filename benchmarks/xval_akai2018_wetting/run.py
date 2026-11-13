#!/usr/bin/env python3
"""XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING: static contact angle of a D2Q9
color-gradient droplet with the geometrical wetting condition.

Reference implementation (open source, read at source level):
``lanl/MF-LBM`` @ 11be324a97fe6331b2271102da370460715355e8,
``multiphase_3D/0.src/Phase_gradient.F90:209-265`` -- the geometrical wetting
condition of Akai et al., Phys. Rev. E 98, 043302 (2018): the color-gradient
normal at a wall-adjacent node is rotated about the wall normal so that it
meets the wall under the prescribed contact angle, and the phase value of the
solid neighbour is made consistent with that rotated normal.

Case
----
2-D sessile drop on a flat no-slip plate, height/depth periodic in x, solid
row at the top and the bottom (the top wall is kept neutral, 90 deg).  Equal
densities (rho = 1 in both phases, as in the reference solver), tau_r = tau_b,
A = 0.04, beta = 0.85.  The drop is initialised as the circular cap that
satisfies the *prescribed* angle, then relaxed; the angle is measured from the
settled field.

Estimators (the first is the decision metric):
* ``theta_circle``  -- Kasa circle fit of the phi = 0 contour (wetting.metrics)
* ``theta_cap``     -- independent cap-area / contact-width inversion
* ``theta_local``   -- angle implied by the imposed wall gradient (BC read-back)

Usage
-----
python run.py [--arms gk30,gk60,gk90] [--steps 40000] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
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
    wall_local_contact_angle,
)
from tensorlbm.color_gradient2d import color_gradient_two_tau_step  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.wetting.metrics import (  # noqa: E402
    contact_angle_circle_fit,
    spherical_cap_inversion,
)


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------
@dataclass
class ArmSpec:
    name: str
    theta_target: float          # prescribed static contact angle (deg)
    init_theta: float | None = None  # initial cap angle (defaults to theta_target)
    wetting: bool = True         # False -> neutral 90 deg wall (control arm)
    beta: float = 0.70
    A: float = 0.04
    tau: float = 1.0
    radius: float = 14.0
    nx: int = 96
    ny: int = 64
    xi: float = 2.0              # initial tanh interface half-width (lattice)
    steps: int = 40000
    note: str = ""


ARMS: dict[str, ArmSpec] = {
    "ctl60": ArmSpec(
        name="ctl60", theta_target=90.0, init_theta=60.0, wetting=False,
        note="control: no wetting BC, cap initialised at 60 deg must relax to 90 deg",
    ),
    "gk30": ArmSpec(name="gk30", theta_target=30.0),
    "gk60": ArmSpec(name="gk60", theta_target=60.0),
    "gk90": ArmSpec(name="gk90", theta_target=90.0),
    "gk120": ArmSpec(
        name="gk120", theta_target=120.0,
        note="beyond reference range (MF-LBM restricts theta<90 deg)",
    ),
    "gk60_b090": ArmSpec(name="gk60_b090", theta_target=60.0, beta=0.90,
                         note="sensitivity: recoloring/segregation beta (0.70 -> 0.90)"),
    "gk60_A006": ArmSpec(name="gk60_A006", theta_target=60.0, A=0.06,
                         note="sensitivity: surface-tension amplitude A (0.04 -> 0.06)"),
    "gk60_R10": ArmSpec(name="gk60_R10", theta_target=60.0, radius=10.0,
                        note="sensitivity: drop radius, i.e. resolution of the cap"),
    "gk60_xi3": ArmSpec(name="gk60_xi3", theta_target=60.0, xi=3.0,
                        note="sensitivity: initial interface width"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def build_solid(ny: int, nx: int) -> torch.Tensor:
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    solid[ny - 1, :] = True
    return solid


def init_drop(spec: ArmSpec, dtype=torch.float64):
    """Circular-cap initial state for the prescribed contact angle."""
    ny, nx = spec.ny, spec.nx
    y_wall = 0.5
    th = math.radians(spec.init_theta if spec.init_theta is not None else spec.theta_target)
    R = spec.radius
    x0 = nx / 2.0
    yc = y_wall - R * math.cos(th)          # centre of the cap (below the plate)
    y = torch.arange(ny, dtype=dtype).view(-1, 1)
    x = torch.arange(nx, dtype=dtype).view(1, -1)
    r = torch.sqrt((x - x0) ** 2 + (y - yc) ** 2)
    phi = torch.tanh((R - r) / spec.xi)      # +1 inside the (wetting) drop
    rho = torch.ones_like(phi)
    rho_r = 0.5 * rho * (1.0 + phi)
    rho_b = 0.5 * rho * (1.0 - phi)
    w = torch.tensor(
        [4.0 / 9.0, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
        dtype=dtype,
    ).view(9, 1, 1)
    f_r = w * rho_r.unsqueeze(0)
    f_b = w * rho_b.unsqueeze(0)
    return f_r, f_b


# ---------------------------------------------------------------------------
# One solver step
# ---------------------------------------------------------------------------
def step(f_r, f_b, solid, spec: ArmSpec, theta_bottom: float, solid3: torch.Tensor):
    rho_r = f_r.sum(dim=0)
    rho_b = f_b.sum(dim=0)
    phi = phase_field(rho_r, rho_b)
    theta_map = {(1, 0): 90.0}  # top wall neutral; bottom wall carries theta
    ghost = geometrical_wetting_ghost(
        phi, solid, theta_bottom, periodic=(True, False), theta_by_direction=theta_map
    )
    f_r, f_b = color_gradient_two_tau_step(
        f_r,
        f_b,
        tau_r=spec.tau,
        tau_b=spec.tau,
        A=spec.A,
        beta=spec.beta,
        solid_mask=solid if spec.wetting or True else None,
        phi_solid=ghost,
    )
    f_r = stream(f_r)
    f_b = stream(f_b)
    # halfway bounce-back through the solid cells
    f_r = torch.where(solid3, f_r[list(OPP)], f_r)
    f_b = torch.where(solid3, f_b[list(OPP)], f_b)
    return f_r, f_b


def measure(phi_np: np.ndarray, y_wall: float, trim: float = 0.05) -> dict:
    out = {}
    try:
        ca = contact_angle_circle_fit(phi_np, y_wall, trim=trim, level=0.0)
        out["theta_circle"] = ca["theta_deg"]
        out["circle"] = ca
    except Exception as exc:  # pragma: no cover - diagnostic path
        out["theta_circle"] = float("nan")
        out["circle_error"] = repr(exc)
    try:
        cap = spherical_cap_inversion(phi_np, y_wall, level=0.0)
        out["theta_cap"] = cap["theta_deg"]
        out["cap"] = cap
    except Exception as exc:  # pragma: no cover
        out["theta_cap"] = float("nan")
        out["cap_error"] = repr(exc)
    return out


def interface_width(phi_np: np.ndarray, column: int | None = None) -> float:
    """10-90 interface thickness of a vertical cut through the drop apex."""
    ny, nx = phi_np.shape
    j = nx // 2 if column is None else column
    col = phi_np[:, j]
    # use the top (vapour) branch: profile from -1 to +1 going down
    idx = np.where(col > 0.0)[0]
    if len(idx) == 0:
        return float("nan")
    i_top = idx[-1]
    if i_top + 1 >= ny:
        return float("nan")
    # walk upward from the topmost liquid node to the 10 % level
    y90 = None
    y10 = None
    for i in range(i_top, ny - 1):
        if y90 is None and col[i] < 0.8 and col[i + 1] >= 0.8:
            y90 = i
        if y10 is None and col[i] < -0.8 and col[i + 1] >= -0.8:
            y10 = i
            break
    if y90 is None or y10 is None:
        return float("nan")
    return float(y10 - y90)


# ---------------------------------------------------------------------------
# Arm driver
# ---------------------------------------------------------------------------
def run_arm(spec: ArmSpec, device: str = "cpu", sample_every: int = 500, verbose: bool = True) -> dict:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    dtype = torch.float64
    solid = build_solid(spec.ny, spec.nx)
    solid3 = solid.unsqueeze(0)
    f_r, f_b = init_drop(spec, dtype=dtype)
    if device != "cpu":
        solid = solid.to(device)
        solid3 = solid3.to(device)
        f_r, f_b = f_r.to(device), f_b.to(device)
    theta_bottom = spec.theta_target if spec.wetting else 90.0

    t0 = time.time()
    history = []
    prev_phi = None
    mass0 = None
    for it in range(1, spec.steps + 1):
        f_r, f_b = step(f_r, f_b, solid, spec, theta_bottom, solid3)
        if it % sample_every == 0 or it == 1:
            rho_r = f_r.sum(dim=0)
            rho_b = f_b.sum(dim=0)
            rho = rho_r + rho_b
            phi = phase_field(rho_r.cpu(), rho_b.cpu())
            jx = ((f_r + f_b)[1] - (f_r + f_b)[3]).sum(dim=0).cpu()
            jy = ((f_r + f_b)[2] - (f_r + f_b)[4]).sum(dim=0).cpu()
            u_mag = torch.sqrt((jx / rho.cpu()) ** 2 + (jy / rho.cpu()) ** 2)
            umax = float(u_mag[~solid.cpu()].max())
            if mass0 is None:
                mass0 = (float(rho_r.sum()), float(rho_b.sum()))
            dphi = float((phi - prev_phi).abs().max()) if prev_phi is not None else float("nan")
            prev_phi = phi
            phi_np = phi.numpy()
            m = measure(phi_np, 0.5)
            history.append(
                {
                    "step": it,
                    "theta_circle": m["theta_circle"],
                    "theta_cap": m["theta_cap"],
                    "u_max": umax,
                    "dphi_max": dphi,
                    "mass_r": float(rho_r.sum()),
                    "mass_b": float(rho_b.sum()),
                    "mass_drift_r": (float(rho_r.sum()) - mass0[0]) / mass0[0],
                    "mass_drift_b": (float(rho_b.sum()) - mass0[1]) / mass0[1],
                }
            )
            if verbose and (it % (sample_every * 4) == 0 or it == 1):
                print(
                    f"  [{spec.name}] step {it:6d}  theta={m['theta_circle']:6.2f} "
                    f"cap={m['theta_cap']:6.2f}  |u|max={umax:.3e}  dphi={dphi:.2e}",
                    flush=True,
                )
            if math.isfinite(dphi) and dphi < 1e-9 and it > spec.steps // 4:
                break

    rho_r = f_r.sum(dim=0)
    rho_b = f_b.sum(dim=0)
    phi = phase_field(rho_r.cpu(), rho_b.cpu()).numpy()
    # final estimators
    final = measure(phi, 0.5)
    final15 = measure(phi, 0.5, trim=0.15)
    phi_t = torch.from_numpy(phi)
    # BC read-back on the bottom wall faces only
    theta_map = {(1, 0): 90.0}
    ghost = geometrical_wetting_ghost(
        phi_t.to(f_r.dtype), solid.cpu(), theta_bottom,
        periodic=(True, False), theta_by_direction=theta_map,
    )
    face_bottom = torch.zeros(spec.ny, spec.nx, dtype=torch.bool)
    face_bottom[1, :] = True
    ang = wall_local_contact_angle(phi_t, ghost, solid.cpu(), face_subset=face_bottom)

    # spurious currents split: liquid core vs whole domain
    f_tot = (f_r + f_b)
    jx = (f_tot[1] - f_tot[3]).sum(dim=0).cpu()
    jy = (f_tot[2] - f_tot[4]).sum(dim=0).cpu()
    rho = rho_r.cpu() + rho_b.cpu()
    u = torch.sqrt((jx / rho) ** 2 + (jy / rho) ** 2)
    fluid = ~solid.cpu()
    u_core = u.clone()
    u_core[phi_t < 0.5] = 0.0
    u_max_fluid = float(u[fluid].max())

    last = [h for h in history if h["step"] > 0.75 * history[-1]["step"]] or history[-1:]
    theta_tail = float(np.nanmean([h["theta_circle"] for h in last]))
    theta_tail_std = float(np.nanstd([h["theta_circle"] for h in last]))

    res = {
        "arm": spec.name,
        "spec": spec.__dict__,
        "theta_target": spec.theta_target,
        "theta_bottom_applied": theta_bottom,
        "steps_run": history[-1]["step"],
        "wall_time_s": time.time() - t0,
        "theta_circle": final["theta_circle"],
        "theta_cap": final.get("theta_cap"),
        "theta_circle_trim15": final15["theta_circle"],
        "theta_local_median": float(ang.median()) if ang.numel() else float("nan"),
        "theta_local_n": int(ang.numel()),
        "theta_tail_mean": theta_tail,
        "theta_tail_std": theta_tail_std,
        "circle": final.get("circle"),
        "cap": final.get("cap"),
        "err_pct_circle": 100.0 * (final["theta_circle"] - spec.theta_target) / spec.theta_target,
        "err_deg_circle": final["theta_circle"] - spec.theta_target,
        "err_pct_cap": (
            100.0 * (final["theta_cap"] - spec.theta_target) / spec.theta_target
            if final.get("theta_cap") is not None and math.isfinite(final.get("theta_cap", float("nan")))
            else float("nan")
        ),
        "u_max_fluid": u_max_fluid,
        "u_max_liquid_core": float(u_core.max()),
        "interface_width_10_90": interface_width(phi),
        "mass_drift_r": history[-1]["mass_drift_r"],
        "mass_drift_b": history[-1]["mass_drift_b"],
        "converged_dphi": history[-1]["dphi_max"],
        "history": history,
        "phi_field_sha256": hashlib.sha256(phi.astype(np.float64).tobytes()).hexdigest(),
    }
    res["pass_15pct"] = bool(abs(res["err_pct_circle"]) <= 15.0)
    np.save(HERE / f"_last_phi_{spec.name}.npy", phi)
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(results: dict, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    primary = [k for k in ("ctl60", "gk30", "gk60", "gk90", "gk120") if k in results]

    # 1) measured vs prescribed
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    xs, ys, names = [], [], []
    for i, k in enumerate(primary):
        r = results[k]
        xs.append(i)
        ys.append(r["theta_circle"])
        names.append(k)
        ax.errorbar(
            i, r["theta_circle"],
            yerr=2.0 * r["theta_tail_std"] if r["theta_tail_std"] else 0.0,
            fmt="o", color="tab:blue", capsize=3, label="measured (circle fit)" if i == 0 else None,
        )
        ax.plot(i, r["theta_target"], "k+", ms=12, label="prescribed" if i == 0 else None)
        if math.isfinite(r.get("theta_cap", float("nan"))):
            ax.plot(i, r["theta_cap"], "s", mfc="none", color="tab:green",
                    label="cap inversion" if i == 0 else None)
        if math.isfinite(r.get("theta_local_median", float("nan"))):
            ax.plot(i, r["theta_local_median"], "^", mfc="none", color="tab:orange",
                    label="BC read-back" if i == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("contact angle [deg]")
    ax.set_title("static contact angle: measured vs prescribed")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    p = out / "fig1_angle_vs_prescribed.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p.name)

    # 2) relative error bars against the 15 % gate
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    keys = [k for k in results if not k.startswith("ctl")]
    labels = keys
    errs = [results[k]["err_pct_circle"] for k in keys]
    ax.bar(range(len(keys)), errs, color=["tab:blue" if abs(e) <= 15 else "tab:red" for e in errs])
    ax.axhline(15, ls="--", color="k", lw=1, label="+15 % gate")
    ax.axhline(-15, ls="--", color="k", lw=1)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("relative error of theta_circle [%]")
    ax.set_title("XVAL acceptance: |err| <= 15 % per arm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = out / "fig2_error_gate.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p.name)

    # 3) relaxation history
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for k in primary:
        h = results[k]["history"]
        ax.plot([x["step"] for x in h], [x["theta_circle"] for x in h], label=k, lw=1.2)
        ax.axhline(results[k]["theta_target"], ls=":", lw=0.8, color="grey")
    ax.set_xlabel("iteration")
    ax.set_ylabel("theta_circle [deg]")
    ax.set_title("relaxation of the measured angle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig3_relaxation.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p.name)

    # 4) sensitivity panel
    sens = [k for k in results if k in ("gk60", "gk60_b090", "gk60_A006", "gk60_R10", "gk60_xi3")]
    if len(sens) > 1:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        vals = [results[k]["theta_circle"] for k in sens]
        ax.bar(range(len(sens)), vals, color="tab:purple", alpha=0.75)
        ax.axhline(60.0, ls="--", color="k", lw=1, label="prescribed 60 deg")
        ax.set_xticks(range(len(sens)))
        ax.set_xticklabels(sens, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("theta_circle [deg]")
        ax.set_title("sensitivity: beta / A / drop radius")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = out / "fig4_sensitivity.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append(p.name)
    return figs


def phase_contour_figure(fields: dict, out: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(fields)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.2), squeeze=False)
    for ax, k in zip(axes[0], keys):
        phi = fields[k]
        im = ax.imshow(phi, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.contour(phi, levels=[0.0], colors="k", linewidths=0.8)
        ax.set_title(f"{k} (t={-0.5:.0f})" if False else k, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="phi")
    p = out / "fig0_phase_fields.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p.name


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ctl60,gk30,gk60,gk90")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--src", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample-every", type=int, default=500)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, fields = {}, {}
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        spec = ARMS[name]
        spec.steps = args.steps
        print(f"== arm {name}: theta_target={spec.theta_target} wetting={spec.wetting}", flush=True)
        res = run_arm(spec, device=args.device, sample_every=args.sample_every)
        fields[name] = np.load(HERE / f"_last_phi_{name}.npy")
        results[name] = res
        print(
            f"   -> theta_circle={res['theta_circle']:.2f} (target {spec.theta_target}) "
            f"err={res['err_pct_circle']:+.2f}%  |u|max={res['u_max_fluid']:.2e}  "
            f"pass15={res['pass_15pct']}",
            flush=True,
        )

    figs = []
    if fields:
        figs.append(phase_contour_figure(fields, out))
    figs += make_figures(results, out)

    payload = {
        "anchor": "XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING",
        "reference": {
            "repo": "lanl/MF-LBM",
            "commit": "11be324a97fe6331b2271102da370460715355e8",
            "files": ["multiphase_3D/0.src/Phase_gradient.F90:209-265"],
            "license": "BSD-3-Clause",
        },
        "engine": {
            "module": "tensorlbm.cg_wetting2d",
            "sha256": _sha256(SRC / "tensorlbm" / "cg_wetting2d.py"),
            "cg_step_sha256": _sha256(SRC / "tensorlbm" / "color_gradient2d.py"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "threads": torch.get_num_threads(),
        },
        "criteria": {
            "gate": "|theta_circle - theta_target| / theta_target <= 15 %",
            "secondary": "|theta_cap - theta_circle| <= 5 deg; mass drift <= 1e-8",
        },
        "results": results,
        "figures": figs,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
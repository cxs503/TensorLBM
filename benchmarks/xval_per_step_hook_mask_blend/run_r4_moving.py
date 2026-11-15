"""Round 4 of the mask-blend cross-validation: the *moving* sub-cell wall.

R3 validated a stationary partially saturated cell (Noble & Torczynski 1998) and
explicitly left the moving case -- the momentum term ``6 w_i (c_i . u_w)`` that
``partial_cell_layer(..., velocity=...)`` is supposed to add -- unmeasured: the
only test was a smoke assertion that "moving differs from stationary".  R4
measures it against a closed-form Couette solution, and measures the per-step
setup hook *together with* a sub-cell mask (the anchor's own subject).

Design frozen before measurement: ``predeclared_r4_moving.json``.

Case: plane Couette with the *bottom* wall inside the cell (row 1, solid
fraction ``eps_s``, moving at ``U``) and the top wall stationary, so the analytic
solution is ``u(y) = U (y_t - y) / H`` with ``y_b = 0.5 + eps_s``, ``y_t = ny-1.5``.

Run: ``python benchmarks/xval_per_step_hook_mask_blend/run_r4_moving.py``
(full sweep is submitted to the scheduler; the pytest regression in
``tests/test_moving_subcell_wall.py`` keeps the claims under CI).
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks" / "xval_per_step_hook_mask_blend"
sys.path.insert(0, str(BENCH))

import run_xval as R  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
import tensorlbm.step_hooks as SH  # noqa: E402

DT = torch.float64
OUT = ROOT / "artifacts" / "xval_moving_subcell_wall"
ANCHOR = "XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND"
MISSION = 21235
ARMS = ("naive_eps", "nt_pre", "nt_post")
U_WALL = 0.02


# --------------------------------------------------------------------- helpers
def _geometry(ny: int, nx: int, eps_s: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    solid = torch.zeros(ny, nx, dtype=DT)
    solid[0, :] = 1.0
    part = torch.zeros(ny, nx, dtype=DT)
    part[1, :] = float(eps_s)
    top = torch.zeros(ny, nx, dtype=DT)
    top[-1, :] = 1.0
    return solid, part, top


def _partial(arm: str, part: torch.Tensor, tau: float, velocity):
    if arm == "naive_eps":
        return SH.BoundaryLayer(name="partial", mask=part, kind="moving", velocity=velocity)
    if arm in ("nt_pre", "nt_post"):
        return SH.partial_cell_layer(
            "partial", part, tau, velocity, reflect=arm.split("_", 1)[1]
        )
    raise ValueError(f"unknown arm {arm!r}")


def run_moving_wall(
    ny: int,
    nx: int,
    tau: float,
    eps_s: float,
    arm: str,
    *,
    u_wall: float = U_WALL,
    setup_step=None,
    save_step=None,
    save_interval: int = 0,
    steps: int | None = None,
) -> dict:
    """Plane Couette, bottom sub-cell wall moving at ``u_wall``, top wall at rest."""
    y_b = 0.5 + float(eps_s)
    y_t = ny - 1.5
    H = y_t - y_b
    if steps is None:
        steps = R.steps_for_couette(H, (tau - 0.5) / 3.0)
    solid, part, top = _geometry(ny, nx, eps_s)
    f0 = R.rest_field(ny, nx)
    tables = SH.tables_for(f0)
    velocity = (lambda st, like: st["wall_velocity"]) if setup_step is not None else (u_wall, 0.0)
    layers = [
        SH.stationary_bounce_layer("solid_row0", solid),
        _partial(arm, part, tau, velocity),
        SH.stationary_bounce_layer("top_wall", top),
    ]
    op = SH.masked_step(tau=tau, layers=layers, tables=tables)
    t0 = time.perf_counter()
    with torch.no_grad():
        state, log = SH.solve_with_hooks(
            {"f": f0.clone()}, steps, operator=op, setup_step=setup_step,
            save_step=save_step, save_interval=save_interval,
        )
        f = state["f"]
        mass_rel = abs(float(f.sum()) - float(f0.sum())) / float(f0.sum())
    wall = time.perf_counter() - t0
    rho, ux, _uy = R.macroscopic(f)
    u = ux.detach().numpy()[:, 0]

    # fluid rows only: the partially saturated row is a mixture, not a velocity
    ys = np.arange(2, ny - 1, dtype=float)
    uu = np.array([u[int(k)] for k in range(2, ny - 1)])
    ref = u_wall * (y_t - ys) / H
    l2 = 100.0 * float(np.linalg.norm(uu - ref) / np.linalg.norm(ref))
    max_err = 100.0 * float(np.max(np.abs(uu - ref)) / u_wall)
    slope, icept = np.polyfit(ys, uu, 1)
    u_at_wall = icept + slope * y_b
    y_at_u = (u_wall - icept) / slope if slope != 0.0 else float("nan")
    return {
        "arm": arm,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "eps_s": float(eps_s),
        "y_b": y_b,
        "y_t": y_t,
        "H": H,
        "u_wall": u_wall,
        "steps": int(steps),
        "profile_l2_pct": l2,
        "max_err_pct": max_err,
        "u_wall_recovered": float(u_at_wall / u_wall),
        "offset_err_dx": float(y_at_u - y_b),
        "mass_rel_drift": mass_rel,
        "finite": bool(np.isfinite(f.detach().numpy()).all()),
        "hooks_setup": len(log.setup),
        "wall_seconds": wall,
        "profile": [float(v) for v in uu],
        "ys": [float(v) for v in ys],
        "analytic": [float(v) for v in ref],
    }


def _fit_ok(r: dict, tol: float = 0.03) -> bool:
    return abs(r["u_wall_recovered"] - 1.0) <= tol


# ------------------------------------------------------------------- limit arms
def solid_limit(ny: int, nx: int, tau: float, *, moving: bool) -> dict:
    """eps_s = 1 must collapse to the binary layer, bit-wise, with and without velocity."""
    solid, part, top = _geometry(ny, nx, 1.0)
    f0 = R.rest_field(ny, nx)
    tables = SH.tables_for(f0)
    vel = (U_WALL, 0.0) if moving else None
    sub = SH.partial_cell_layer("partial", part, tau, vel, reflect="post")
    binary_vel = SH.BoundaryLayer(
        name="partial",
        mask=part,
        kind="moving" if moving else "bounce",
        velocity=(U_WALL, 0.0) if moving else None,
    )
    # one composed step is enough: the operators must agree cell by cell
    f_post = R.equilibrium(torch.ones(ny, nx, dtype=DT), torch.full((ny, nx), 0.01, dtype=DT),
                           torch.full((ny, nx), 0.004, dtype=DT))
    nudge = 1e-3 * torch.randn_like(f_post)
    f_post = f_post + nudge
    f_pre = R.equilibrium(torch.ones(ny, nx, dtype=DT), torch.full((ny, nx), 0.008, dtype=DT),
                          torch.full((ny, nx), 0.002, dtype=DT)) + 0.5 * nudge
    a = SH.compose_masked_boundaries(f_post, f_pre, [sub], tables=tables)
    b = SH.compose_masked_boundaries(f_post, f_pre, [binary_vel], tables=tables)
    scale = float(b.abs().max())
    rel = float((a - b).abs().max()) / scale
    # and the pre-fix operator (eps_s = 1 - 1e-6 still takes the post-collision base)
    part_lo = part.clone()
    part_lo[1, :] = 1.0 - 1e-6
    lo = SH.compose_masked_boundaries(
        f_post, f_pre, [SH.partial_cell_layer("partial", part_lo, tau, vel, reflect="post")],
        tables=tables,
    )
    rel_lo = float((lo - b).abs().max()) / scale
    return {
        "moving": moving,
        "max_rel_vs_binary": rel,
        "bitwise": rel <= 1e-12,
        "eps_1_minus_1e6_rel_vs_binary": rel_lo,
    }


def fractional_run_stationary(ny: int, nx: int, tau: float, eps_s: float, arm: str) -> dict:
    """R3's stationary arm, re-run so the report can show the fix moved nothing else."""
    return R.run_fractional_wall("couette", ny, nx, tau, eps_s, arm)


# ------------------------------------------------------------------- hook cases
def hook_cases(ny: int, nx: int, tau: float, eps_s: float) -> dict:
    """Per-step setup hook x sub-cell mask (the anchor's own subject)."""
    direct = run_moving_wall(ny, nx, tau, eps_s, "nt_post")

    def make(value: float):
        def setup(state, t, i):
            state["wall_velocity"] = torch.tensor([value, 0.0], dtype=state["f"].dtype)
            return state

        return setup

    hooked = run_moving_wall(ny, nx, tau, eps_s, "nt_post", setup_step=make(U_WALL))
    still = run_moving_wall(ny, nx, tau, eps_s, "nt_post", setup_step=make(0.0))

    ramp_steps = 200
    ramp_hook = SH.velocity_ramp_hook("wall_velocity", (U_WALL, 0.0), ramp_steps, dim=2)
    trace: list[dict] = []

    def recorder(state, t, i):
        # called every step (save_interval=1); keep a decimated trace
        if int(i) % 50 == 0:
            _rho, ux, _uy = R.macroscopic(state["f"])
            trace.append({
                "step": int(i),
                "t": float(t),
                "u_mid": float(ux[ny // 2, 0].item()),
                # save_step runs *before* setup_step on the first step, so the
                # hook output does not exist yet there (documented order)
                "u_wall_hook": float(state.get("wall_velocity", torch.zeros(2))[0].item()),
            })
        return state

    ramped = run_moving_wall(
        ny, nx, tau, eps_s, "nt_post",
        setup_step=ramp_hook, save_step=recorder, save_interval=1,
    )

    d_direct_hook = float(
        np.max(np.abs(np.array(hooked["profile"]) - np.array(direct["profile"]))) / U_WALL
    )
    still_peak = float(np.max(np.abs(np.array(still["profile"]))) / U_WALL)
    ramp_l2 = 100.0 * float(
        np.linalg.norm(np.array(ramped["profile"]) - np.array(direct["profile"]))
        / np.linalg.norm(np.array(direct["profile"]))
    )
    return {
        "direct": direct,
        "hook_constant": hooked,
        "hook_zero": still,
        "hook_ramp": ramped,
        "max_rel_diff_direct_vs_hook_pct": 100.0 * d_direct_hook,
        "hook_zero_peak_rel": still_peak,
        "ramp_vs_constant_l2_pct": ramp_l2,
        "ramp_steps": ramp_steps,
        "trace": trace,
    }


# ---------------------------------------------------------------------- figures
def make_figures(res: dict, out_dir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []

    rows = res["sweep"]
    base = [r for r in rows if r["tau"] == 0.8 and r["ny"] == 16]

    # 1. profiles
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, eps in zip(axes, (0.25, 0.5, 0.75)):
        sub = [r for r in base if abs(r["eps_s"] - eps) < 1e-12]
        if not sub:
            continue
        ax.plot(sub[0]["analytic"], sub[0]["ys"], "k--", lw=1.2, label="analytic")
        for r in sub:
            ax.plot(r["profile"], r["ys"], "o-", ms=3, lw=1.1, label=r["arm"])
        ax.axhline(0.5 + eps, color="0.6", lw=0.8, ls=":")
        ax.set_title(f"$\\varepsilon_s$={eps:.2f}, Ny=16, $\\tau$=0.8")
        ax.set_xlabel("$u/U$")
    axes[0].set_ylabel("$y$ [dx]")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Moving sub-cell wall (bottom moving at U): measured vs analytic Couette", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "fig1_moving_subcell_profiles.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig1_moving_subcell_profiles.png", "path": str(p),
                 "caption": "u/U profiles, bottom sub-cell wall moving at U, eps_s = 0.25/0.5/0.75."})

    # 2. profile L2 vs eps
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for arm in ARMS:
        for ny, ls in ((16, "-"), (24, "--")):
            sub = sorted(
                [r for r in rows if r["arm"] == arm and r["tau"] == 0.8 and r["ny"] == ny],
                key=lambda r: r["eps_s"],
            )
            ax.plot([r["eps_s"] for r in sub], [r["profile_l2_pct"] for r in sub],
                    ls, marker="o", ms=3.5, label=f"{arm}, Ny={ny}")
    ax.axhline(res["thresholds"]["profile_l2_internal_target_pct"], color="g", lw=1, ls=":", label="3 % target")
    ax.axhline(res["thresholds"]["owner_ceiling_pct"], color="r", lw=1, ls=":", label="15 % owner ceiling")
    ax.set_xlabel("$\\varepsilon_s$")
    ax.set_ylabel("profile $L_2$ vs analytic [%]")
    ax.set_yscale("log")
    ax.legend(fontsize=6.5)
    ax.set_title("Moving sub-cell wall accuracy", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "fig2_l2_vs_eps.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig2_l2_vs_eps.png", "path": str(p),
                 "caption": "Profile L2 vs solid fraction for the three arms (Ny=16/24)."})

    # 3. recovered wall velocity
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    width = 0.26
    eps_list = sorted({r["eps_s"] for r in base})
    for k, arm in enumerate(ARMS):
        vals = []
        for eps in eps_list:
            hit = [r for r in base if r["arm"] == arm and abs(r["eps_s"] - eps) < 1e-12]
            vals.append(hit[0]["u_wall_recovered"] if hit else np.nan)
        ax.bar([i + (k - 1) * width for i in range(len(eps_list))], vals, width, label=arm)
    ax.axhline(1.0, color="k", lw=1, ls="--", label="target $u(y_b)=U$")
    ax.axhspan(0.97, 1.03, color="g", alpha=0.15, label="+/-3 %")
    ax.set_xticks(range(len(eps_list)))
    ax.set_xticklabels([f"{e:.2f}" for e in eps_list])
    ax.set_xlabel("$\\varepsilon_s$")
    ax.set_ylabel("$u(y_b)/U$ (fit)")
    ax.legend(fontsize=7)
    ax.set_title("Is the wall velocity reproduced?", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "fig3_wall_velocity_recovery.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig3_wall_velocity_recovery.png", "path": str(p),
                 "caption": "Fitted wall velocity at y_b, normalised by U (1.0 = correct)."})

    # 4. hook ramp trace
    hk = res["hooks"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot([r["step"] for r in hk["trace"]], [r["u_wall_hook"] / U_WALL for r in hk["trace"]],
            label="hook wall velocity $u_w/U$")
    ax.plot([r["step"] for r in hk["trace"]], [r["u_mid"] / U_WALL for r in hk["trace"]],
            label="$u$ at mid-channel / U")
    ax.set_xlabel("step")
    ax.set_ylabel("normalised")
    ax.legend(fontsize=7)
    ax.set_title("Per-step setup hook driving a sub-cell wall (ramp 0->U)", fontsize=10)
    fig.tight_layout()
    p = fig_dir / "fig4_hook_ramp.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig4_hook_ramp.png", "path": str(p),
                 "caption": "Hook-written wall velocity and mid-channel response during the ramp."})

    return figs


# ------------------------------------------------------------------------- main
def main() -> int:
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    sweep: list[dict] = []
    for ny in (16, 24):
        for eps in (0.25, 0.5, 0.75, 1.0):
            for arm in ARMS:
                sweep.append(run_moving_wall(ny, 8, 0.8, eps, arm))
    tau_sens: list[dict] = []
    for tau in (0.7, 0.9):
        for arm in ARMS:
            tau_sens.append(run_moving_wall(16, 8, tau, 0.5, arm))

    limits = {
        "stationary": solid_limit(16, 8, 0.8, moving=False),
        "moving": solid_limit(16, 8, 0.8, moving=True),
    }
    stationary_regression = [
        fractional_run_stationary(16, 8, 0.8, eps, arm)
        for eps in (0.25, 0.5, 0.75, 1.0)
        for arm in ARMS
    ]
    hooks = hook_cases(16, 8, 0.8, 0.5)

    preds = {
        "P1_hook_drives_sub_cell_wall": bool(
            hooks["max_rel_diff_direct_vs_hook_pct"] <= 1e-10 and hooks["hook_zero_peak_rel"] < 1e-9
        ),
        "P2_hook_ramp_equivalence": bool(hooks["ramp_vs_constant_l2_pct"] <= 1.0),
        "P3_solid_limit_moving_bitwise": bool(limits["moving"]["bitwise"]),
        "P4_wall_velocity_recovered": bool(
            all(_fit_ok(r) for r in sweep if r["arm"] == "nt_post")
        ),
        "P5_profile_accuracy": bool(
            all(r["profile_l2_pct"] < 3.0 for r in sweep if r["arm"] == "nt_post")
        ),
        "P6_error_monotone_in_eps": bool(
            all(
                all(b["profile_l2_pct"] <= a["profile_l2_pct"] + 1e-12
                    for a, b in zip(series, series[1:]))
                for series in (
                    sorted((r for r in sweep if r["arm"] == arm and r["ny"] == ny),
                           key=lambda r: r["eps_s"])
                    for arm in ARMS
                    for ny in (16, 24)
                )
            )
        ),
    }

    worst_nt_post = max(
        (r for r in sweep if r["arm"] == "nt_post"), key=lambda r: r["profile_l2_pct"]
    )
    res = {
        "anchor": ANCHOR,
        "round": 4,
        "mission": MISSION,
        "predeclared": json.loads((BENCH / "predeclared_r4_moving.json").read_text()),
        "thresholds": {"profile_l2_internal_target_pct": 3.0, "owner_ceiling_pct": 15.0},
        "sweep": sweep,
        "tau_sensitivity": tau_sens,
        "limits": limits,
        "stationary_regression": [
            {k: v for k, v in r.items() if k not in ("profile",)} for r in stationary_regression
        ],
        "hooks": {k: v for k, v in hooks.items() if k != "trace"} | {"trace": hooks["trace"]},
        "predictions": preds,
        "worst_nt_post": {k: worst_nt_post[k] for k in
                          ("arm", "ny", "tau", "eps_s", "profile_l2_pct", "u_wall_recovered")},
        "all_below_owner_ceiling": bool(all(r["profile_l2_pct"] < 15.0 for r in sweep)),
        "all_finite": bool(all(r["finite"] for r in sweep)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cwd": str(ROOT),
            "git_commit": R.git_commit(),
            "seconds": time.perf_counter() - t0,
        },
    }

    # table.csv
    with (OUT / "table.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["arm", "ny", "nx", "tau", "eps_s", "y_b", "H", "steps", "profile_l2_pct",
                "max_err_pct", "u_wall_recovered", "offset_err_dx", "mass_rel_drift", "finite"]
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in sweep + tau_sens:
            wr.writerow({k: r[k] for k in cols})

    # profiles.npz
    np.savez(
        OUT / "profiles.npz",
        **{
            f"{r['arm']}_eps{r['eps_s']:.2f}_ny{r['ny']}_tau{r['tau']:.2f}_u": np.array(r["profile"])
            for r in sweep
        },
        ys=np.array(sweep[0]["ys"]),
        analytic_eps0_50=np.array(
            [r["analytic"] for r in sweep if abs(r["eps_s"] - 0.5) < 1e-12][0]
        ),
    )

    # numbers first: figures are optional (a compute node has no matplotlib)
    (OUT / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    try:
        res["figures"] = make_figures(res, OUT)
    except ImportError as exc:  # pragma: no cover - node without matplotlib
        res["figures"] = []
        res["figures_error"] = f"{exc}; figures are rendered by generate_report_r4.py"
    (OUT / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    summary = {
        "predictions": preds,
        "worst_nt_post": res["worst_nt_post"],
        "all_below_15pct": res["all_below_owner_ceiling"],
        "limits": limits,
        "hooks": {k: v for k, v in hooks.items() if k not in ("trace", "direct", "hook_constant",
                                                              "hook_zero", "hook_ramp")},
        "artifact": str(OUT / "result.json"),
    }
    print("RESULT_SUMMARY " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
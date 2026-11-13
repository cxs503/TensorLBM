#!/usr/bin/env python
"""Verification study for the anti-bounceback (Inamuro) wet-node BC.

Runs the plane-channel benchmarks defined in
:mod:`tensorlbm.boundary.antibb_channel_benchmark`, compares against the exact
Poiseuille/Couette solutions, measures the *effective wall position* of the
boundary rule, and writes

* ``artifacts/antibb_wetnode_verification.json``  (all numbers + provenance)
* ``artifacts/antibb_wetnode_verification.png``   (report figure)
* ``docs/antibb_wetnode_verification_20261113.md`` tables are generated from
  the same numbers (see ``--emit-markdown``).

Usage
-----
    PYTHONPATH=src python scripts/run_antibb_wetnode_verification.py [--quick]

``--quick`` trades resolution for runtime (~10 s instead of ~6 min) and is what
the unit tests exercise.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

from tensorlbm.boundary.antibb_channel_benchmark import (  # noqa: E402
    ChannelSetup,
    couette_profile,
    l2_relative_error,
    poiseuille_profile,
    run_channel,
)

NU = 0.1          # kinematic viscosity (lattice units)
NX = 4            # streamwise cells (flow is x-invariant)
UMAX = 0.02       # target Poiseuille centre-line velocity (low Mach)
UWALL = 0.05      # Couette wall velocity


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def analytic_profile(setup: ChannelSetup, delta: float) -> torch.Tensor:
    """Exact solution with the bottom no-slip surface placed at ``y = -delta``."""
    ny = setup.ny
    H = (ny - 1) + 2.0 * delta
    y = torch.arange(ny, dtype=torch.float64) + delta
    if setup.force != 0.0:
        return setup.force * y * (H - y) / (2.0 * setup.nu * setup.rho0)
    ub, ut = float(setup.u_wall_bottom), float(setup.u_wall_top)
    return ub + (ut - ub) * (y / H)


def fit_wall_offset(
    profile: torch.Tensor, setup: ChannelSetup, tol: float = 1e-6
) -> tuple[float, float]:
    """Least-squares wall offset ``delta`` (lattice units) and residual error.

    The optimum of the relative L2 error over ``delta in [-0.75, 1.25]`` is
    found by a coarse scan followed by golden-section refinement.  ``delta = 0``
    means the no-slip surface coincides with the boundary-node plane (the
    anti-bounceback claim); ``delta = 0.5`` is the halfway bounce-back
    convention.
    """
    ny = setup.ny
    inner = slice(1, ny - 1)
    target = profile[inner].double()

    def err(delta: float) -> float:
        return l2_relative_error(target, analytic_profile(setup, delta)[inner])

    lo, hi = -0.75, 1.25
    grid = torch.linspace(lo, hi, 41, dtype=torch.float64)
    deltas = [float(d) for d in grid]
    errs = [err(d) for d in deltas]
    best = min(range(len(errs)), key=lambda i: errs[i])
    lo = deltas[max(best - 1, 0)]
    hi = deltas[min(best + 1, len(deltas) - 1)]
    # golden-section search
    phi = (5 ** 0.5 - 1) / 2
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = err(c), err(d)
    while hi - lo > tol:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = err(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = err(d)
    delta = 0.5 * (lo + hi)
    return delta, err(delta)


def run_case(
    *,
    ny: int,
    mode: str,
    wall_mode: str = "antibb",
    rho_mode: str = "one",
    closure: str = "exact",
    steady_tol: float = 1e-12,
    max_steps: int = 600000,
) -> dict:
    """One channel run + all verification metrics."""
    H = (ny - 1) if wall_mode == "antibb" else ny
    if mode == "poiseuille":
        force = 8.0 * NU * UMAX / H**2
        setup = ChannelSetup(
            ny=ny, nx=NX, tau=3 * NU + 0.5, force=force,
            steps=max_steps, steady_tol=steady_tol, check_every=500,
            wall_mode=wall_mode, rho_mode=rho_mode,
        )
        exact = poiseuille_profile(setup)
    elif mode == "couette":
        setup = ChannelSetup(
            ny=ny, nx=NX, tau=3 * NU + 0.5, u_wall_bottom=UWALL,
            steps=max_steps, steady_tol=steady_tol, check_every=500,
            wall_mode=wall_mode, rho_mode=rho_mode,
        )
        exact = couette_profile(setup)
    else:
        raise ValueError(mode)

    if closure != "exact":
        # patch the BC closure for the comparison runs
        import tensorlbm.boundary.antibb_channel_benchmark as bench

        original = bench.apply_antibb_wetnode

        def patched(*args, **kwargs):
            kwargs["closure"] = closure
            return original(*args, **kwargs)

        bench.apply_antibb_wetnode = patched
        try:
            result = run_channel(setup)
        finally:
            bench.apply_antibb_wetnode = original
    else:
        result = run_channel(setup)

    prof = result.ux[:, NX // 2].double()
    inner = slice(1, ny - 1)
    delta, err_fit = fit_wall_offset(prof, setup)
    wall_ux = float(result.ux[0, NX // 2])
    u_target = float(setup.u_wall_bottom)
    scale = max(float(prof.abs().max()), 1e-30)
    return {
        "mode": mode,
        "ny": ny,
        "wall_mode": wall_mode,
        "rho_mode": rho_mode,
        "closure": closure,
        "tau": setup.tau,
        "nu": setup.nu,
        "hydraulic_width_nominal": setup.hydraulic_width,
        "steps_used": result.steps_used,
        "steady_residual": result.steady_residual,
        "err_delta0": l2_relative_error(prof[inner], exact[inner]),
        "wall_offset_fit": delta,
        "err_at_fit": err_fit,
        "u_max_numeric": float(prof[inner].max()),
        "u_max_analytic": float(exact[inner].max()),
        "wall_node_ux": wall_ux,
        "wall_node_ux_error_rel": abs(wall_ux - u_target) / scale,
        "normal_velocity_ratio": float(result.uy.abs().max()) / scale,
        "mass_drift": result.mass_drift,
        "profile": [float(v) for v in prof],
    }


def uniform_flow_fixed_point() -> dict:
    """A uniform flow with matching wall velocities must be an exact fixed point."""
    from tensorlbm.d2q9 import C, equilibrium
    from tensorlbm.boundary.antibb_wetnode import (
        antibb_unknown_mask,
        apply_antibb_wetnode,
        bottom_top_normals,
    )

    out = {}
    ny, nx = 5, 4
    U = 0.1
    rho = torch.ones(ny, nx, dtype=torch.float64)
    ux = torch.full((ny, nx), U, dtype=torch.float64)
    uy = torch.zeros_like(ux)
    f = equilibrium(rho, ux, uy)
    wet = torch.zeros((ny, nx), dtype=torch.bool)
    wet[0, :] = wet[-1, :] = True
    normals = bottom_top_normals(ny, nx, dtype=torch.float64)
    uw = torch.zeros((ny, nx, 2), dtype=torch.float64)
    uw[..., 0] = U
    f_str = torch.stack(
        [torch.roll(f[i], shifts=(int(C[i, 1]), int(C[i, 0])), dims=(0, 1)) for i in range(9)]
    )
    mask = antibb_unknown_mask(normals, wet)
    f_str = f_str.clone()
    f_str[mask] += 0.3  # garbage in the unknown directions must be overwritten
    for closure in ("exact", "linear", "none"):
        fb = apply_antibb_wetnode(f, f_str, wet, normals, uw, 1.0, closure=closure)
        r = fb.sum(0)
        uxx = (fb * C[:, 0].view(-1, 1, 1)).sum(0) / r
        uyy = (fb * C[:, 1].view(-1, 1, 1)).sum(0) / r
        out[closure] = {
            "max_rho_error": float((r - 1).abs().max()),
            "max_ux_error": float((uxx - U).abs().max()),
            "max_uy": float(uyy.abs().max()),
        }
    out["unknown_directions_bottom"] = mask[:, 0, 0].nonzero().flatten().tolist()
    out["unknown_directions_top"] = mask[:, -1, 0].nonzero().flatten().tolist()
    return out


def no_bc_control() -> dict:
    """Negative control: without a wall rule the forced channel never equilibrates.

    The boundary rows are left as plain periodic streaming nodes, so the body
    force has nothing to balance and the streamwise momentum grows linearly in
    time.  A verification test that does not separate this from the ABB result
    would not be measuring the boundary condition at all.
    """
    H = 16
    force = 8.0 * NU * UMAX / H**2
    setup = ChannelSetup(
        ny=17, nx=NX, tau=3 * NU + 0.5, force=force, steps=2000, wall_mode="none"
    )
    res = run_channel(setup)
    prof = res.ux[:, NX // 2].double()
    exact = poiseuille_profile(setup)
    return {
        "err_delta0": l2_relative_error(prof[1:-1], exact[1:-1]),
        "u_max_numeric": float(prof[1:-1].max()),
        "u_max_analytic": float(exact[1:-1].max()),
        "steps": 2000,
    }


def collision_ablation() -> dict:
    """Ablation: do not prescribe the wall velocity at the collision step.

    The anti-bounceback rule only fixes the populations streaming in from
    outside the domain; if the boundary node is then collided with its *own*
    (unprescribed) velocity, the Dirichlet value is never enforced and the
    boundary-node velocity drifts away without bound.  Measured here.
    """
    H = 16
    force = 8.0 * NU * UMAX / H**2
    setup = ChannelSetup(
        ny=17, nx=NX, tau=3 * NU + 0.5, force=force, steps=6000,
        prescribe_wall_in_collision=False,
    )
    res = run_channel(setup)
    prof = res.ux[:, NX // 2].double()
    exact = poiseuille_profile(setup)
    return {
        "steps": 6000,
        "wall_node_ux": float(prof[0]),
        "u_max_numeric": float(prof[1:-1].max()),
        "u_max_analytic": float(exact[1:-1].max()),
        "err_delta0": l2_relative_error(prof[1:-1], exact[1:-1]),
    }


def tau_sweep(resolutions=(17,), taus=(0.6, 0.7, 0.8, 1.0, 1.2, 1.5)) -> list[dict]:
    """Effective no-slip offset of each wall rule as a function of tau.

    Halfway bounce-back is known to place the wall exactly at the link midpoint
    only for ``tau = 1``; this sweep measures the same effect for the
    anti-bounceback rule and is the quantitative statement behind the
    "wall on the boundary node" claim.
    """
    rows = []
    for ny in resolutions:
        H = ny - 1
        force = 8.0 * NU * UMAX / H**2
        for tau in taus:
            for wall_mode in ("antibb", "halfway"):
                setup = ChannelSetup(
                    ny=ny, nx=NX, tau=tau, force=force, steps=200000,
                    steady_tol=1e-12, check_every=500, wall_mode=wall_mode,
                )
                res = run_channel(setup)
                prof = res.ux[:, NX // 2].double()
                delta, err = fit_wall_offset(prof, setup)
                rows.append({
                    "ny": ny, "tau": tau, "wall_mode": wall_mode, "nu": setup.nu,
                    "wall_offset_fit": delta, "err_at_fit": err,
                    "err_delta0": l2_relative_error(prof[1:-1], analytic_profile(setup, 0.0)[1:-1]),
                })
                print(
                    f"tau={tau:4.2f} {wall_mode:8s} delta={delta:+.4f} "
                    f"err@fit={err:.2e} err@0={(rows[-1]['err_delta0'])*100:6.3f}%",
                    flush=True,
                )
    return rows


def _unused_wrong_normal_control() -> dict:
    """Unused: a flipped normal is *not* a discriminating control for this rule."""
    from tensorlbm.boundary.antibb_wetnode import bottom_top_normals

    normals = bottom_top_normals(5, 4)
    import tensorlbm.boundary.antibb_channel_benchmark as bench

    original = bench.bottom_top_normals
    bench.bottom_top_normals = lambda ny, nx, **kw: -original(ny, nx, **kw)
    try:
        H = 16
        force = 8.0 * NU * UMAX / H**2
        setup = ChannelSetup(
            ny=17, nx=NX, tau=3 * NU + 0.5, force=force, steps=2000, steady_tol=0.0
        )
        res = run_channel(setup)
    finally:
        bench.bottom_top_normals = original
    prof = res.ux[:, NX // 2].double()
    exact = poiseuille_profile(setup)
    return {
        "err_delta0": l2_relative_error(prof[1:-1], exact[1:-1]),
        "u_max_numeric": float(prof[1:-1].max()),
        "finite": bool(torch.isfinite(prof).all()),
    }


def _finite(value):
    """NaN/inf -> None so the artifact stays strict JSON."""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def make_figure(results: list[dict], path: Path, tau_rows: list[dict] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) Poiseuille profile at the finest resolution
    ax = axes[0]
    poiseuille = [r for r in results if r["mode"] == "poiseuille" and r["closure"] == "exact"]
    if poiseuille:
        finest = max(poiseuille, key=lambda r: r["ny"])
        ny = finest["ny"]
        prof = torch.tensor(finest["profile"], dtype=torch.float64)
        setup = ChannelSetup(
            ny=ny, nx=NX, tau=3 * NU + 0.5,
            force=8.0 * NU * UMAX / ((ny - 1) if finest["wall_mode"] == "antibb" else ny) ** 2,
            wall_mode=finest["wall_mode"],
        )
        y = torch.arange(ny, dtype=torch.float64)
        ax.plot(prof.numpy(), y.numpy(), "o", ms=4, label=f"ABB wet node (ny={ny})")
        ax.plot(poiseuille_profile(setup).numpy(), y.numpy(), "-", lw=1.2, label="exact Poiseuille")
        ax.set_xlabel("$u_x$ [lu]")
        ax.set_ylabel("y [lu]")
        ax.set_title(f"(a) Poiseuille, err={finest['err_delta0']*100:.2f}%")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # (b) relative error vs resolution
    ax = axes[1]
    for mode, marker in (("poiseuille", "o"), ("couette", "s")):
        for wall_mode, ls in (("antibb", "-"), ("halfway", "--")):
            sel = [
                r for r in results
                if r["mode"] == mode and r["wall_mode"] == wall_mode and r["closure"] == "exact"
            ]
            if not sel:
                continue
            sel.sort(key=lambda r: r["ny"])
            ax.loglog(
                [r["ny"] for r in sel],
                [r["err_delta0"] for r in sel],
                marker + ls,
                label=f"{mode} / {wall_mode}",
            )
    ax.set_xlabel("ny (hydraulic width H = ny-1)")
    ax.set_ylabel("relative L2 error (wall at node plane)")
    ax.set_title("(b) resolution study")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    # (c) measured effective wall offset
    ax = axes[2]
    if tau_rows:
        for wall_mode, marker in (("antibb", "o"), ("halfway", "s")):
            sel = [r for r in tau_rows if r["wall_mode"] == wall_mode]
            if not sel:
                continue
            ax.plot(
                [r["tau"] for r in sel],
                [r["wall_offset_fit"] for r in sel],
                marker + "-",
                label=wall_mode,
            )
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"fitted no-slip offset $\delta$ [lu]")
        ax.set_title(r"(c) effective wall position vs $\tau$")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axhline(0.5, color="r", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    labels, values, colors = [], [], []
    for wall_mode, color in (("antibb", "tab:blue"), ("halfway", "tab:orange")):
        sel = [r for r in results if r["mode"] == "poiseuille" and r["wall_mode"] == wall_mode]
        sel.sort(key=lambda r: r["ny"])
        if not sel:
            continue
        labels += [f"{wall_mode}\nny={r['ny']}" for r in sel]
        values += [r["wall_offset_fit"] for r in sel]
        colors += [color] * len(sel)
    if not tau_rows:
        ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_ylabel(r"fitted no-slip offset $\delta$ [lu]")
        ax.set_title(r"(c) effective wall position ($\delta$=0: node, 0.5: link mid)")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axhline(0.5, color="r", ls=":", lw=0.8)
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast low-resolution study")
    parser.add_argument("--out", default="artifacts/antibb_wetnode_verification.json")
    parser.add_argument("--fig", default="artifacts/antibb_wetnode_verification.png")
    args = parser.parse_args(argv)

    resolutions = (17, 25) if args.quick else (17, 33, 65)
    tol = 1e-9 if args.quick else 1e-12

    results = []
    for ny in resolutions:
        for mode in ("poiseuille", "couette"):
            for wall_mode in ("antibb", "halfway"):
                if args.quick and wall_mode == "halfway":
                    continue
                r = run_case(ny=ny, mode=mode, wall_mode=wall_mode, steady_tol=tol)
                results.append(r)
                print(
                    f"{mode:10s} {wall_mode:8s} ny={ny:3d} err={r['err_delta0']*100:7.3f}% "
                    f"delta={r['wall_offset_fit']:+.3f} err@fit={r['err_at_fit']*100:6.3f}% "
                    f"steps={r['steps_used']}",
                    flush=True,
                )

    # closure comparison (Couette is the discriminator: it exercises u_w)
    closure_rows = []
    for closure in ("linear", "none"):
        r = run_case(ny=33 if not args.quick else 17, mode="couette", closure=closure,
                     steady_tol=tol)
        closure_rows.append(r)
        print(
            f"couette    antibb   closure={closure:6s} err={r['err_delta0']*100:7.3f}% "
            f"wall_ux={r['wall_node_ux']:.5f} (target {UWALL})",
            flush=True,
        )

    fixed = uniform_flow_fixed_point()
    no_bc = no_bc_control()
    ablation = collision_ablation()
    tau_rows = tau_sweep() if not args.quick else []
    if args.quick:
        no_bc = {"skipped": True}

    payload = {
        "study": "antibb_wetnode_inamuro",
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provenance": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "settings": {"nu": NU, "nx": NX, "u_max_poiseuille": UMAX, "u_wall_couette": UWALL},
        },
        "channel_runs": results,
        "closure_comparison": closure_rows,
        "uniform_flow_fixed_point": fixed,
        "no_bc_control": no_bc,
        "collision_ablation": ablation,
        "tau_sweep": tau_rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_finite(payload), indent=2))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"\nwrote {out} sha256={digest[:16]}")

    make_figure(results + closure_rows, Path(args.fig), tau_rows)
    print(f"wrote {args.fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Audit the Spalding exchange wall model source moments (ladder rung 1).

Produces, for D3Q19 and D3Q27:

1. measured residual of the mass source (zeroth moment) and the error of the
   momentum source against the analytic ``rho (u_target - u)`` prediction,
   both with and without the weight renormalization applied in
   :func:`tensorlbm.spalding_wall_model._lattice`;
2. a manufactured-state sweep of the friction-velocity inversion accuracy
   (``u_tau`` recovered from a Spalding state vs the manufactured value);
3. figures and a JSON record under ``artifacts/spalding_wall_moments_2026-11-03``.

Usage::

    python scripts/verify_spalding_wall_moments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The repository may be co-installed with other worktrees; audit *this* tree.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

import tensorlbm.spalding_wall_model as swm  # noqa: E402

RESOLVED = Path(swm.__file__).resolve()
if SRC not in RESOLVED.parents:
    raise SystemExit(f"auditing the wrong tree: {RESOLVED}")

from tensorlbm.spalding_wall_model import (  # noqa: E402
    apply_spalding_exchange_wall_model,
    sample_wall_exchange_velocity,
    solve_spalding_friction_velocity,
    spalding_u_plus_from_y_plus,
)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "spalding_wall_moments_2026-11-03"

# Standard log-law intercept, plotted only as an asymptote reference for the
# Spalding law: the model itself uses the unified law, not this fit.
B_LOG_LAW = 5.2


def _lattice(q: int, *, renormalized: bool):
    if q == 19:
        from tensorlbm.d3q19 import C, W, macroscopic3d

        macro = macroscopic3d
    else:
        from tensorlbm.d3q27 import C, W, macroscopic27

        macro = macroscopic27
    weights = W.double()
    if renormalized:
        weights = weights / weights.sum()
    return C.double(), weights, macro


def _equilibrium(rho, ux, uy, uz, c, w):
    cx, cy, cz = (c[:, axis, None, None, None] for axis in range(3))
    cu = cx * ux + cy * uy + cz * uz
    u2 = ux.square() + uy.square() + uz.square()
    return w[:, None, None, None] * rho * (1.0 + 3.0 * cu + 4.5 * cu.square() - 1.5 * u2)


def _wall_state(q: int, seed: int = 26, shape=(5, 7, 9)):
    c, _, _ = _lattice(q, renormalized=True)
    generator = torch.Generator().manual_seed(seed + q)
    mask = torch.zeros((q,) + shape, dtype=torch.bool)
    cell = (2, 3, 4)
    for direction in range(q):
        if c[direction, 1] < 0.0:
            mask[(direction,) + cell] = True
    normals = (
        torch.zeros(shape, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
    )
    normals[1][cell] = 1.0
    q_field = torch.full((q,) + shape, 0.5, dtype=torch.float64)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.06, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    _, w, _ = _lattice(q, renormalized=True)
    f = _equilibrium(rho, ux, zero, zero, c, w)
    f = f + 0.02 * torch.randn((q,) + shape, generator=generator, dtype=torch.float64)
    return c, mask, normals, q_field, cell


def _analytic_source(q, c, f, mask, normals, q_field, nu, exchange_distance, out):
    _, _, macro = _lattice(q, renormalized=True)
    _, ux, uy, uz = macro(f)
    samples = sample_wall_exchange_velocity(
        (ux, uy, uz),
        mask,
        q_field,
        normals,
        exchange_distance=exchange_distance,
    )
    boundary = samples.boundary
    normal = torch.stack((samples.normal_x, samples.normal_y, samples.normal_z))
    sampled = torch.stack(
        (samples.velocity_x, samples.velocity_y, samples.velocity_z),
    )
    u_old = torch.stack((ux[boundary], uy[boundary], uz[boundary]))
    rho_old = f.sum(dim=0)[boundary]
    normal_flux = (sampled * normal).sum(dim=0)
    tangential = sampled - normal_flux[None] * normal
    speed = tangential.norm(dim=0).clamp_min(1e-20)
    u_tau = solve_spalding_friction_velocity(speed, samples.y2, nu)
    u1 = spalding_u_plus_from_y_plus(samples.y1 * u_tau / nu) * u_tau
    target = (u_old * normal).sum(dim=0)[None] * normal + u1[None] * (tangential / speed)
    injected = torch.stack(
        [(c[:, axis, None, None, None] * (out - f)).sum(dim=0) for axis in range(3)],
    )[:, boundary]
    return (
        float((out - f).sum(dim=0)[boundary].abs().max()),
        float((injected - rho_old[None] * (target - u_old)).abs().max()),
        float((rho_old[None] * (target - u_old)).abs().max()),
        float(u_tau.mean()),
        float(samples.y1.mean()),
    )


def moment_table(renormalized: bool) -> list[dict]:
    """Measure source moments with the model's weights normalized or not."""
    module_lattice = swm._lattice
    if not renormalized:

        def raw_lattice(q, device, dtype):
            c, _, macro = module_lattice(q, device, dtype)
            module = __import__(f"tensorlbm.d3q{q}", fromlist=["W"])
            return c, module.W.to(device=device, dtype=dtype), macro

        swm._lattice = raw_lattice
    try:
        return _moment_table(renormalized)
    finally:
        swm._lattice = module_lattice


def _moment_table(renormalized: bool) -> list[dict]:
    rows = []
    for q in (19, 27):
        c, mask, normals, q_field, _ = _wall_state(q)
        _, eq_w, _ = _lattice(q, renormalized=renormalized)
        rho = torch.ones((5, 7, 9), dtype=torch.float64)
        ux = torch.full((5, 7, 9), 0.06, dtype=torch.float64)
        zero = torch.zeros((5, 7, 9), dtype=torch.float64)
        generator = torch.Generator().manual_seed(26 + q)
        f = _equilibrium(rho, ux, zero, zero, c, eq_w)
        f = f + 0.02 * torch.randn((q,) + (5, 7, 9), generator=generator, dtype=torch.float64)
        out, diagnostics = apply_spalding_exchange_wall_model(
            f,
            mask,
            q_field,
            normals,
            nu=1.0e-4,
            exchange_distance=2.0,
        )
        mass, momentum, magnitude, u_tau, y1 = _analytic_source(
            q,
            c,
            f,
            mask,
            normals,
            q_field,
            1.0e-4,
            2.0,
            out,
        )
        rows.append(
            {
                "q": q,
                "renormalized": renormalized,
                "weight_sum_minus_one": float(eq_w.sum() - 1.0),
                "mass_source_abs": mass,
                "momentum_source_abs_error": momentum,
                "momentum_source_magnitude": magnitude,
                "momentum_source_rel_error": momentum / magnitude,
                "u_tau_diagnostic": diagnostics.mean_u_tau,
                "y1": y1,
            },
        )
    return rows


def inversion_sweep() -> list[dict]:
    """Recover u_tau from manufactured Spalding states over a y+/u_tau range."""
    rows = []
    nu = 1.0e-4
    u_tau_values = [1.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2, 2.0e-2]
    y2_values = [0.5, 1.0, 2.0, 3.0, 8.0]
    for u_tau in u_tau_values:
        for y2 in y2_values:
            y2_t = torch.tensor([y2], dtype=torch.float64)
            u_plus = spalding_u_plus_from_y_plus(y2_t * u_tau / nu)
            speed = u_plus * u_tau
            recovered = solve_spalding_friction_velocity(speed, y2_t, nu)
            rows.append(
                {
                    "u_tau_manufactured": u_tau,
                    "y2": y2,
                    "y_plus": float(y2 * u_tau / nu),
                    "u_tau_recovered": float(recovered[0]),
                    "rel_error": float((recovered[0] - u_tau) / u_tau),
                },
            )
    return rows


def figures(rows: list[dict], sweep: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    labels = [f"D3Q{r['q']}\n{'renorm' if r['renormalized'] else 'float32 W'}" for r in rows]
    mass = [max(r["mass_source_abs"], 1e-18) for r in rows]
    mom = [max(r["momentum_source_abs_error"], 1e-18) for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, mass, 0.4, label="|mass source| (zeroth moment)")
    ax.bar(x + 0.2, mom, 0.4, label="|momentum source error|")
    ax.axhline(1e-16, color="k", ls=":", lw=1, label="double-precision floor")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("absolute residual (lattice units)")
    ax.set_title("(a) Source moments vs analytic identity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    y2s = sorted({s["y2"] for s in sweep})
    for y2 in y2s:
        pts = [s for s in sweep if s["y2"] == y2]
        ax.plot(
            [p["y_plus"] for p in pts],
            [abs(p["rel_error"]) for p in pts],
            "o-",
            ms=3,
            label=f"y2 = {y2} cells",
        )
    ax.axhline(0.03, color="r", ls="--", lw=1, label="3% acceptance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("manufactured $y^+$ at exchange location")
    ax.set_ylabel("relative error of recovered $u_\\tau$")
    ax.set_title("(b) Spalding friction-velocity inversion")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    # Asymptote cross-check of the Spalding law actually used by the model: the
    # viscous sublayer (u+ = y+) and the log law (u+ = ln(y+)/kappa + B) must be
    # recovered in their own ranges.  (An earlier version of this panel plotted
    # the same call twice as "solid vs dashed", which proved nothing.)
    y_plus = np.logspace(-1, 3, 400)
    u_plus = spalding_u_plus_from_y_plus(torch.tensor(y_plus, dtype=torch.float64)).numpy()
    ax.plot(y_plus, u_plus, "k-", lw=1.6, label="Spalding law (model)")
    ax.plot(y_plus, y_plus, "b--", lw=1.0, label="viscous sublayer $u^+=y^+$")
    ax.plot(
        y_plus,
        np.log(y_plus) / swm.SPALDING_KAPPA + B_LOG_LAW,
        "r:",
        lw=1.2,
        label=f"log law ($\\kappa$={swm.SPALDING_KAPPA}, B={B_LOG_LAW})",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("$u^+$")
    ax.set_title("(c) Spalding law vs sublayer / log-law asymptotes")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "spalding_wall_moments.png", dpi=125)
    print(f"wrote {OUT / 'spalding_wall_moments.png'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = moment_table(renormalized=False) + moment_table(renormalized=True)
    sweep = inversion_sweep()
    record = {
        "date": "2026-11-03",
        "ladder_rung": 1,
        "moment_table": rows,
        "inversion_sweep": sweep,
        "max_abs_rel_error_u_tau": max(abs(r["rel_error"]) for r in sweep),
    }
    (OUT / "moments.json").write_text(json.dumps(record, indent=2))
    for row in rows:
        print(
            f"q={row['q']:2d} renorm={row['renormalized']!s:5s} "
            f"sum(W)-1={row['weight_sum_minus_one']:+.3e} "
            f"mass={row['mass_source_abs']:.3e} "
            f"mom_err={row['momentum_source_abs_error']:.3e} "
            f"rel={row['momentum_source_rel_error']:.3e}",
        )
    print(f"max |rel error| u_tau inversion = {record['max_abs_rel_error_u_tau']:.3e}")
    figures(rows, sweep)


if __name__ == "__main__":
    main()

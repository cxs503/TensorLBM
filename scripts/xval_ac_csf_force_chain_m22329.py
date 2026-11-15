#!/usr/bin/env python3
"""XVAL: Allen-Cahn / CSF force chain - which closure holds a static droplet?

Anchor: feature ledger ``XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN``
(``lssfau/lbmpy`` Allen-Cahn CSF force chain), acceptance ``< 15%``.

XVAL round 1 (``docs/XVAL_M17410_ALLEN_CAHN_CSF_FORCE_CHAIN.md``) proved the
*force evaluation* ``F_st = mu grad(phi)`` is bit-identical to lbmpy's and
deferred the *dynamic* chain.  This script closes that round with a controlled
ablation on a static droplet:

  formulation  in {density_deviation, full_density}
  chain        in {st, st+p, st+p+v}      (lbmpy: F_st, F_p, F_v)
  momentum     in {reference, physical}   (velocity normalisation)

and grades every run against the analytic Young-Laplace law ``dp = 2 sigma/R``
(sphere; two equal principal curvatures).  ``dp`` is sampled during the run and
reported both as a time series and as the mean of the second half, because the
initial transient is an acoustic ring that must not be mistaken for a steady
bias.

Every number written to JSON is produced here; nothing is hand-transcribed.

Single point:
    PYTHONPATH=src python3 scripts/xval_ac_csf_force_chain_m22329.py \
        --grid 48 --radius 14 --width 3.5 --ratio 10 --steps 2000

Full matrix (parallel):
    PYTHONPATH=src python3 scripts/xval_ac_csf_force_chain_m22329.py \
        --matrix --steps 2000 --workers 12
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.phasefield.ac_csf_force_chain import (  # noqa: E402
    ForceChainConfig,
    ac_csf_step,
    chemical_potential_isotropic,
    laplace_pressure_jump_2phase,
)
from tensorlbm.phasefield.free_energy import force_mu_grad_phi  # noqa: E402
from tensorlbm.phasefield.isotropic_fd import isotropic_gradient_3d  # noqa: E402
from tensorlbm.phasefield.operators import laplacian_3d  # noqa: E402

OUT_DEFAULT = REPO / "artifacts" / "xval_ac_csf_force_chain_m22329"


def droplet(n: int, radius: float, width: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Bulk ``tanh((R - r)/W)`` sphere on a ``(z, y, x)`` grid (periodic box)."""
    axis = torch.arange(n, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    r = torch.sqrt((zz - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2 + (xx - n / 2 + 0.5) ** 2)
    return torch.tanh((radius - r) / width), r


def initial_state(phi: torch.Tensor, rho_h: float, rho_l: float, formulation: str) -> torch.Tensor:
    """Populations at rest for the two formulations (deviation starts at zero)."""
    from tensorlbm.phasefield.ac_csf_force_chain import (
        _equilibrium_density_deviation,
        _equilibrium_full_density,
        density_from_phase_field,
    )

    zero = torch.zeros_like(phi)
    if formulation == "density_deviation":
        return _equilibrium_density_deviation(zero, zero, zero, zero)
    rho = density_from_phase_field(phi, rho_h, rho_l)
    return _equilibrium_full_density(rho, zero, zero, zero)


def run_point(
    *,
    grid: int,
    radius: float,
    width: float,
    ratio: float,
    steps: int,
    formulation: str,
    terms: tuple[str, ...],
    momentum_density: str = "reference",
    sigma: float = 1e-2,
    mobility: float = 5e-3,
    nu_h: float = 0.1,
    nu_l: float = 0.02,
    advance_phase_field: bool = True,
    sample_every: int = 25,
) -> dict:
    """Integrate one configuration and return the graded diagnostics."""
    torch.set_num_threads(1)
    rho_h, rho_l = 1.0, 1.0 / ratio
    phi, _ = droplet(grid, radius, width)
    f = initial_state(phi, rho_h, rho_l, formulation).to(torch.float64)
    phi = phi.to(torch.float64)
    config = ForceChainConfig(
        terms=terms,  # type: ignore[arg-type]
        formulation=formulation,  # type: ignore[arg-type]
        momentum_density=momentum_density,  # type: ignore[arg-type]
    )

    samples: list[dict] = []
    trace: list[dict] = []
    finite = True
    for step in range(1, steps + 1):
        try:
            f, phi, diag = ac_csf_step(
                f,
                phi,
                rho_h=rho_h,
                rho_l=rho_l,
                nu_h=nu_h,
                nu_l=nu_l,
                sigma=sigma,
                W=width,
                mobility=mobility,
                config=config,
                advance_phase_field=advance_phase_field,
            )
        except Exception as exc:  # noqa: BLE001 - report, never mask
            finite = False
            trace.append({"step": step, "error": f"{type(exc).__name__}: {exc}"})
            break
        if not (torch.isfinite(f).all() and torch.isfinite(phi).all()):
            finite = False
            trace.append(
                {
                    "step": step,
                    "max_abs_u": diag["max_abs_u"],
                    "max_abs_rho_dev": float(diag["rho_dev"].abs().max()),
                    "blowup": True,
                }
            )
            break
        if step % sample_every == 0:
            jump = laplace_pressure_jump_2phase(
                f,
                phi,
                radius=radius,
                sigma=sigma,
                rho_h=rho_h,
                rho_l=rho_l,
                formulation=formulation,  # type: ignore[arg-type]
            )
            samples.append(
                {
                    "step": step,
                    "max_abs_u": diag["max_abs_u"],
                    "max_abs_force": diag["max_abs_force"],
                    "max_abs_rho_dev": float(diag["rho_dev"].abs().max()),
                    "phi_center": float(phi[grid // 2, grid // 2, grid // 2]),
                    "phi_max": float(phi.max()),
                    "dp": jump["dp_measured"],
                    "dp_rel_error": jump["relative_error"],
                }
            )

    result = {
        "grid": grid,
        "radius": radius,
        "width": width,
        "radius_over_interface": radius / width,
        "density_ratio": ratio,
        "sigma": sigma,
        "mobility": mobility,
        "nu_h": nu_h,
        "nu_l": nu_l,
        "formulation": formulation,
        "terms": list(terms),
        "momentum_density": momentum_density,
        "steps": steps,
        "advance_phase_field": advance_phase_field,
        "finite": finite,
        "trace": trace,
        "samples": samples,
    }
    if samples:
        half = samples[len(samples) // 2 :]
        dp_mean = sum(s["dp"] for s in half) / len(half)
        u_mean = sum(s["max_abs_u"] for s in half) / len(half)
        expected = 2.0 * sigma / radius
        result["steady"] = {
            "n_samples": len(half),
            "dp_mean": dp_mean,
            "dp_min": min(s["dp"] for s in half),
            "dp_max": max(s["dp"] for s in half),
            "dp_rel_error_mean": abs(dp_mean - expected) / expected,
            "max_abs_u_mean": u_mean,
            "max_abs_u_max": max(s["max_abs_u"] for s in half),
            "dp_laplace_sphere": expected,
        }
        result["final"] = samples[-1]
    return result


def legacy_reference(n: int, radius: float, width: float, sigma: float = 1e-2) -> dict:
    """Static grading of ``F_st`` alone (no time integration).

    Separates the *force evaluation* (round 1) from the *dynamic chain*: both
    operator families are graded on the initial field, including the net force
    and the spurious tangential share that drives parasitic currents.
    """
    phi, _ = droplet(n, radius, width)
    from tensorlbm.allen_cahn_lbm import free_energy_parameters

    p = free_energy_parameters(sigma, width)
    mu7 = p.beta * phi * (phi * phi - 1.0) - p.kappa * laplacian_3d(phi, boundary="periodic")
    f7 = force_mu_grad_phi(phi, mu7, boundary="periodic")
    mu_i = chemical_potential_isotropic(phi, p.beta, p.kappa)
    g = isotropic_gradient_3d(phi, boundary="periodic")
    fi = tuple(mu_i * gi for gi in g)

    def stats(field):
        mag = torch.sqrt(sum(c * c for c in field))
        gn = torch.stack(g)
        gmag = torch.sqrt(sum(c * c for c in g)).clamp(min=1e-30)
        dot = sum(field[i] * gn[i] for i in range(3))
        tang = torch.sqrt((mag**2 - (dot / gmag) ** 2).clamp(min=0.0))
        band = mag > 0.05 * float(mag.max())
        return {
            "max_abs_force": float(mag.max()),
            "net_force": float(torch.stack([s.sum() for s in field]).abs().max()),
            "tangential_share": float(tang[band].sum() / mag[band].sum()),
            "radial_force_integral": float(dot[band].sum() / gmag[band].sum()),
        }

    return {"seven_point": stats(f7), "isotropic": stats(fi)}


def matrix_configs(geometries, ratios, formulations, term_sets, momenta):
    return [
        {
            "grid": g[0],
            "radius": g[1],
            "width": g[2],
            "ratio": r,
            "formulation": fo,
            "terms": ts,
            "momentum_density": mo,
        }
        for g, r, fo, ts, mo in itertools.product(
            geometries, ratios, formulations, term_sets, momenta
        )
    ]


def _run_kwargs(cfg: dict, steps: int) -> dict:
    return {**cfg, "steps": steps}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--radius", type=float, default=14.0)
    ap.add_argument("--width", type=float, default=3.5)
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--formulation", default="density_deviation")
    ap.add_argument("--terms", default="st,p,v")
    ap.add_argument("--momentum", default="reference")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--no-phase-field", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    advance = not args.no_phase_field

    if args.matrix:
        geometries = ((32, 10.0, 3.0), (48, 14.0, 3.5), (64, 18.0, 4.0))
        static = {f"n{n}_R{r}_W{w}": legacy_reference(n, r, w) for n, r, w in geometries}
        configs = matrix_configs(
            geometries=((32, 10.0, 3.0), (48, 14.0, 3.5)),
            ratios=(1.0, 10.0, 100.0),
            formulations=("density_deviation", "full_density"),
            term_sets=(("st",), ("st", "p"), ("st", "p", "v")),
            momenta=("reference",),
        )
        # convergence ladder at the working point (10:1, full chain)
        configs += [
            {
                "grid": n,
                "radius": r,
                "width": w,
                "ratio": 10.0,
                "formulation": "density_deviation",
                "terms": ("st", "p", "v"),
                "momentum_density": "reference",
            }
            for n, r, w in geometries
        ]
        # the physical-normalisation variant: measured, reported, expected to die
        configs += [
            {
                "grid": 32,
                "radius": 10.0,
                "width": 3.0,
                "ratio": 10.0,
                "formulation": "density_deviation",
                "terms": ("st", "p", "v"),
                "momentum_density": "physical",
            }
        ]

        def work(cfg):
            return run_point(**_run_kwargs(cfg, args.steps), advance_phase_field=advance)

        if args.workers > 1:
            from multiprocessing import Pool

            with Pool(args.workers) as pool:
                rows = pool.map(work, configs)
        else:
            rows = []
            for cfg in configs:
                row = work(cfg)
                rows.append(row)
                print(_line(row), flush=True)
        payload = {"static_chain_grading": static, "runs": rows}
        (out / "matrix.json").write_text(json.dumps(payload, indent=1))
        for row in rows:
            print(_line(row), flush=True)
        print(f"wrote {out / 'matrix.json'}")
        return 0

    terms = tuple(t for t in args.terms.split(",") if t)
    row = run_point(
        grid=args.grid,
        radius=args.radius,
        width=args.width,
        ratio=args.ratio,
        steps=args.steps,
        formulation=args.formulation,
        terms=terms,  # type: ignore[arg-type]
        momentum_density=args.momentum,
        advance_phase_field=advance,
    )
    (out / "point.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k not in ("samples", "trace")}, indent=1))
    return 0


def _line(row: dict) -> str:
    steady = row.get("steady", {})
    return (
        f"n={row['grid']:2d} R={row['radius']:4.1f} W={row['width']:3.1f} "
        f"ratio={row['density_ratio']:6.1f} {row['formulation']:17s} "
        f"terms={'+'.join(row['terms']):6s} mom={row['momentum_density']:9s} "
        f"finite={str(row['finite']):5s} steps={len(row['samples']) * 25:5d} "
        f"maxu={steady.get('max_abs_u_mean', float('nan')):.2e} "
        f"dp={steady.get('dp_mean', float('nan')):.3e} "
        f"ref={2e-2 / row['radius']:.3e} "
        f"err={steady.get('dp_rel_error_mean', float('nan')):.2%}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
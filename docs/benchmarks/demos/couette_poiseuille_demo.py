#!/usr/bin/env python3
"""Field visualisation for the Couette-Poiseuille tutorial.

Runs the documented case (``benchmarks/verified/couette_poiseuille/run.py``)
and produces the three-panel figure used in
``docs/benchmarks/tutorials/couette_poiseuille.md``:

1. ``u_x`` field (walls at y = 0.5 / y = ny-1.5 drawn), the sliding top plate
   drags a shear while the body force bulges the profile -- the extremum is
   off-centre.
2. Wall-normal profile: numerical markers vs the analytic superposition
   ``g/(2 nu) y (H-y) + u_wall y/H``, plus the pure-Couette control (g = 0).
3. Point-wise deviation from the analytic profile and the L2 gate.

Usage
-----
    python docs/benchmarks/demos/couette_poiseuille_demo.py \
        --steps 12000 --out docs/benchmarks/tutorials/figs/couette_poiseuille/fields.png

The figure is diagnostic only: the numbers quoted in the tutorial come from the
archived ``benchmarks/verified/couette_poiseuille/result.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_RUN_PY = Path(__file__).resolve().parents[3] / "benchmarks/verified/couette_poiseuille/run.py"


def load_case_module():
    """Import ``benchmarks/verified/couette_poiseuille/run.py`` by path."""
    spec = importlib.util.spec_from_file_location("couette_poiseuille_run", _RUN_PY)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {_RUN_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plot(res: dict, ux_field: np.ndarray, out: Path) -> None:
    ny, nx = res["params"]["ny"], res["params"]["nx"]
    y_star = np.asarray(res["profile"]["y_star"])
    u_num = np.asarray(res["profile"]["u_num"])
    u_ex = np.asarray(res["profile"]["u_exact"])
    H = res["params"]["H"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # 1. field
    ax = axes[0]
    fluid = ux_field[1 : ny - 1, :]
    im = ax.imshow(
        fluid,
        origin="lower",
        extent=(0, nx, 0.5, ny - 1.5),
        aspect="auto",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="$u_x$")
    ax.set_xlabel("x / lattice units (periodic)")
    ax.set_ylabel("y")
    ax.set_title(f"$u_x$ field (ny={ny}, nx={nx}, {res['params']['steps']} steps)")

    # 2. profile vs analytic
    ax = axes[1]
    ax.plot(u_ex, y_star, "-", color="C0", lw=2, label="analytic (Couette+Poiseuille)")
    ax.plot(u_num, y_star, "o", ms=4, color="C1", label="LBM")
    u_couette = res["params"]["u_wall"] * y_star / H
    ax.plot(u_couette, y_star, "--", color="0.5", lw=1.2, label="pure Couette ($g=0$)")
    y_v = res["metrics"]["peak_y_star_exact"]
    ax.axhline(y_v, color="C3", ls=":", lw=1.2)
    ax.annotate(f"analytic peak $y_v/H$={y_v / H:.3f}", xy=(u_num.max(), y_v),
                xytext=(0.05, y_v + 0.10 * H), color="C3", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
    ax.set_xlabel("$u_x$")
    ax.set_ylabel("$y$ (walls 0.5 / ny-1.5)")
    ax.set_title("profile: LBM vs analytic")
    ax.legend(fontsize=8, loc="upper left")

    # 3. deviation
    ax = axes[2]
    ax.plot((u_num - u_ex), y_star, "o-", ms=3, color="C2")
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("$u_x^{LBM} - u_x^{analytic}$")
    ax.set_ylabel("$y$")
    ax.set_title(f"L2 rel err = {res['metrics']['profile_l2']:.2e} (gate 2%)")

    fig.suptitle(
        f"Couette-Poiseuille mixed-driven channel: tau={res['params']['tau']}, "
        f"nu={res['params']['nu']:.3f}, u_wall={res['params']['u_wall']}, "
        f"g={res['params']['accel']:.3g}, Re={res['params']['Re']:.1f}"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ny", type=int, default=34)
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--u-wall", type=float, default=0.02)
    ap.add_argument("--accel", type=float, default=None, help="default: self-similar 4*nu*u_wall/H^2")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/benchmarks/tutorials/figs/couette_poiseuille/fields.png"),
    )
    args = ap.parse_args(argv)

    cp = load_case_module()
    res = cp.run_case(
        ny=args.ny,
        nx=args.nx,
        tau=args.tau,
        u_wall=args.u_wall,
        accel=args.accel,
        steps=args.steps,
        device=args.device,
        return_field=True,
    )
    ux = np.asarray(res.pop("field_ux"))
    cp._print_summary(res)
    plot(res, ux, args.out)
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
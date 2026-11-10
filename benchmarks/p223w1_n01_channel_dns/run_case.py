#!/usr/bin/env python3
"""P223W1-N01 — DNS-arm plane-channel case runner (TensorLBM).

Runs one case of the Re_tau = 180/395/590 ladder with the native OpenMP kernel
(:mod:`tensorlbm.turbulence.les_chan`) in **DNS mode**: the Smagorinsky closure
is switched off bit-for-bit (``cs = 0``, the kernel's documented negative-control
arm) and the plain TRT collision (``--reg 0``, magic parameter ``Lambda = 3/16``)
is used.

Why ``--reg 0`` for this arm: the regularised (Hermite-projected) collision used
by the LES campaign discards the ghost modes, and at ``tau0 - 0.5 ~ 4e-3`` that
projection is *slowly unstable* -- measured here: identical grid, ``u_tau =
0.0125``, ``Re_tau = 180``, ``--reg 1`` reaches ``umax ~ 1e6`` within 600 steps
while ``--reg 0`` holds ``umax ~ 0.25`` and ``rho_min ~ 0.999``.  The TRT arm is
the stable one for a wall-resolved (no SGS dissipation) run; the instability is
recorded in the run report as a solver defect.

Usage
-----
    python3 benchmarks/p223w1_n01_channel_dns/run_case.py --list
    python3 benchmarks/p223w1_n01_channel_dns/run_case.py \
        --case R180_h60 --out-dir runs/p223w1_n01_channel_dns --threads 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.turbulence.les_channel import LesChannelConfig, run_native  # noqa: E402


@dataclass(frozen=True)
class Case:
    """One ladder leg: Reynolds anchor + grid + averaging budget."""

    name: str
    re_tau: float
    mkm_anchor: int
    nx: int
    ny: int
    nz: int
    steps: int
    warmup: int
    sample: int
    u_tau: float = 0.0125
    cs: float = 0.0
    perturb: float = 0.25
    perturb_modes: int = 8
    seed: int = 20261110
    regularized: int = 0
    spec_every: int = 500
    spec_y_plus: tuple[float, ...] = field(default_factory=tuple)

    @property
    def half_height(self) -> float:
        return 0.5 * (self.ny - 2)

    @property
    def nu(self) -> float:
        return self.u_tau * self.half_height / self.re_tau

    @property
    def spacing_plus(self) -> float:
        return self.u_tau / self.nu

    @property
    def spec_rows(self) -> tuple[int, ...]:
        rows = []
        for target in self.spec_y_plus:
            row = int(round(target / self.spacing_plus + 0.5))
            row = min(max(row, 1), self.ny - 2)
            if row not in rows:
                rows.append(row)
        return tuple(rows)

    @property
    def eddy_turnovers(self) -> float:
        return (self.steps - self.warmup) * self.u_tau / self.half_height

    def config(self, threads: int) -> LesChannelConfig:
        return LesChannelConfig(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            steps=self.steps,
            warmup=self.warmup,
            sample=self.sample,
            re_tau=self.re_tau,
            u_tau=self.u_tau,
            cs=self.cs,
            perturb=self.perturb,
            perturb_modes=self.perturb_modes,
            seed=self.seed,
            threads=threads,
            regularized=self.regularized,
            spec_every=self.spec_every,
            spec_rows=self.spec_rows,
        )

    def describe(self) -> dict:
        return {
            **asdict(self),
            "half_height": self.half_height,
            "nu": self.nu,
            "spacing_plus": self.spacing_plus,
            "spec_rows": list(self.spec_rows),
            "eddy_turnovers": self.eddy_turnovers,
            "cells": self.nx * self.ny * self.nz,
        }


#: The ladder.  Every leg keeps the *same* cost (cells x steps) so the runs are
#: comparable in statistical error; the wall-unit resolution therefore degrades
#: with Re_tau -- which is exactly the honest statement this node has to make
#: (see ``channel_dns.dns_cells_required`` for what DNS would cost).
CASES: dict[str, Case] = {
    c.name: c
    for c in (
        Case(
            name="R180_h60",
            mkm_anchor=180,
            re_tau=178.12,
            nx=126,
            ny=122,
            nz=63,
            steps=40000,
            warmup=12000,
            sample=100,
            spec_y_plus=(25.0, 45.0, 80.0),
        ),
        Case(
            name="R180_h30",
            mkm_anchor=180,
            re_tau=178.12,
            nx=63,
            ny=62,
            nz=32,
            steps=30000,
            warmup=9000,
            sample=100,
            spec_y_plus=(25.0, 45.0, 80.0),
        ),
        Case(
            name="R395_h60",
            mkm_anchor=395,
            re_tau=395.0,
            nx=126,
            ny=122,
            nz=63,
            steps=40000,
            warmup=12000,
            sample=100,
            spec_y_plus=(30.0, 60.0, 120.0),
        ),
        Case(
            name="R590_h60",
            mkm_anchor=590,
            re_tau=590.0,
            nx=126,
            ny=122,
            nz=63,
            steps=40000,
            warmup=12000,
            sample=100,
            spec_y_plus=(40.0, 80.0, 160.0),
        ),
    )
}

#: Legs that must be present for the study to be graded at all.
REQUIRED = ("R180_h60", "R395_h60", "R590_h60")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", help="case name (see --list)")
    ap.add_argument("--list", action="store_true", help="print the ladder and exit")
    ap.add_argument("--out-dir", default=str(_REPO / "runs" / "p223w1_n01_channel_dns"))
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--name", help="artefact name (default: case name)")
    ap.add_argument("--cs", type=float, default=None, help="override the SGS constant")
    ap.add_argument("--u-tau", type=float, default=None, help="override u_tau")
    ap.add_argument("--reg", type=int, default=None,
                    help="override the collision arm (1 = regularised, 0 = TRT)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="run every case sequentially")
    args = ap.parse_args(argv)

    if args.list:
        for name, case in CASES.items():
            print(json.dumps({"case": name, **case.describe()}, indent=2))
        return 0

    names = list(CASES) if args.all else [args.case]
    if not any(names) or (names and names[0] is None):
        ap.error("--case or --all required")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        case = CASES[name]
        if (
            args.cs is not None
            or args.steps is not None
            or args.warmup is not None
            or args.u_tau is not None
            or args.reg is not None
        ):
            from dataclasses import replace

            case = replace(
                case,
                cs=case.cs if args.cs is None else args.cs,
                steps=case.steps if args.steps is None else args.steps,
                warmup=case.warmup if args.warmup is None else args.warmup,
                u_tau=case.u_tau if args.u_tau is None else args.u_tau,
                regularized=case.regularized if args.reg is None else args.reg,
            )
        artefact = args.name or case.name
        cfg = case.config(args.threads)
        cfg.validate()
        started = time.time()
        result = run_native(cfg, out_dir, artefact, overwrite=args.overwrite)
        payload = {
            "case": case.describe(),
            "elapsed_seconds": time.time() - started,
            "derived": result.get("derived"),
            "provenance": result.get("provenance"),
        }
        (out_dir / f"{artefact}_meta.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
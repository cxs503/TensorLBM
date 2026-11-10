#!/usr/bin/env python3
"""Experiment plan for campaign node P295W1R2-N04.

Implements the *predeclared* design (see ``REPORT.md`` section 2): the case
matrix, the matching rule that puts every flow-index group at the same
generalised Reynolds number, the repetition/averaging budget and the
acceptance thresholds.  Nothing here depends on the results.

``python plan.py --write plan.json``   emit the case list
``python plan.py --shard i --nshards n``  run the i-th shard (cluster-friendly)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ART = REPO / "artifacts" / "p295w1r2_n04"

# ---------------------------------------------------------------------------
# predeclared thresholds (fixed before any production run)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "B1_laminar_bulk_rel_err": 0.03,
    "B2_newtonian_f_rel_diff": 0.02,
    "B3_collapse_max_spread": 0.10,
    "B4_monotone_tolerance": 0.02,
    "B4_Yc_rel_dev": 0.40,
    "B5_lattice_rel_diff": 0.03,
    "B5_gci_pct": 5.0,
}

# Re_tau = 180 operating point: u_tau is a free scale (it only fixes the time
# step); 0.008 keeps u_max ~ 0.16 (Ma ~ 0.28) while giving an eddy-turnover
# time of h/u_tau = 15/0.008 = 1875 steps.
RE180 = {"re_tau": 180.0, "u_tau": 0.008}
RE395 = {"re_tau": 395.0, "u_tau": 0.010}

BASE_GRID = {"nx": 32, "ny": 32, "nz": 32}  # production (coarse LES)
MID_GRID = {"nx": 48, "ny": 40, "nz": 32}  # GCI mid
COARSE_GRID = {"nx": 24, "ny": 24, "nz": 24}  # GCI coarse
FINE_GRID = {"nx": 64, "ny": 56, "nz": 48}  # GCI fine
RE395_GRID = {"nx": 64, "ny": 56, "nz": 40}


def _case(
    name: str,
    *,
    mode: str,
    grid: dict,
    op: dict,
    n: float,
    Y: float,
    steps: int,
    warmup: int,
    lattice: str = "d3q19",
    K: float | None = None,
    re_mr_target: float | None = None,
    nu_ref: float | None = None,
    u_bulk_ref: float | None = None,
    cs: float = 0.1,
    sample: int = 20,
    nu_update: int = 8,
    threads: int = 1,
) -> dict:
    args = [
        "--name",
        name,
        "--mode",
        mode,
        "--nx",
        str(grid["nx"]),
        "--ny",
        str(grid["ny"]),
        "--nz",
        str(grid["nz"]),
        "--re-tau",
        str(op["re_tau"]),
        "--u-tau",
        str(op["u_tau"]),
        "--n",
        str(n),
        "--y",
        str(Y),
        "--smagorinsky-cs",
        str(cs),
        "--lattice",
        lattice,
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--sample-interval",
        str(sample),
        "--nu-update-interval",
        str(nu_update),
        "--threads",
        str(threads),
    ]
    if K is not None:
        args += ["--K", f"{K:.10e}"]
    if nu_ref is not None:
        args += ["--nu-ref", f"{nu_ref:.6e}"]
    if re_mr_target is not None:
        args += ["--re-mr-target", f"{re_mr_target:.6e}"]
    if u_bulk_ref is not None:
        args += ["--u-bulk-ref", f"{u_bulk_ref:.6e}"]
    return {"name": name, "args": args}


def build_plan() -> dict:
    cases: list[dict] = []

    # -- B1: laminar pressure-drop/flow parity ---------------------------
    # tiny periodic box, high viscosity so the laminar state is reached in
    # O(3 h^2/nu) steps; two flow rates per flow index.
    for u_tau in (0.002, 0.004):
        for n in (0.6, 0.8, 1.0):
            for Y in (0.0, 0.3):
                cases.append(
                    _case(
                        f"lam_u{u_tau:.3f}_n{n:g}_Y{Y:g}",
                        mode="laminar",
                        grid={"nx": 8, "ny": 46, "nz": 8},
                        op={"re_tau": 50.0, "u_tau": u_tau},
                        n=n,
                        Y=Y,
                        steps=12000,
                        warmup=8000,
                        cs=0.0,
                        sample=50,
                        nu_update=1,
                        nu_ref=0.2,
                    )
                )

    # -- B2/B3/B4: turbulent matrix at Re_tau = 180 ----------------------
    U_REF = 19.0 * RE180["u_tau"]  # U+ ~ 19 estimate for matching
    RE_MR_TARGET = 8.0e3  # inside [4e3, 4e4]
    for n in (0.6, 0.8, 1.0):
        cases.append(
            _case(
                f"turb_re180_n{n:g}_Y0",
                mode="turbulent",
                grid=BASE_GRID,
                op=RE180,
                n=n,
                Y=0.0,
                steps=12000,
                warmup=4000,
                re_mr_target=RE_MR_TARGET,
                u_bulk_ref=U_REF,
            )
        )
        cases.append(
            _case(
                f"turb_re180_n{n:g}_Y0.3",
                mode="turbulent",
                grid=BASE_GRID,
                op=RE180,
                n=n,
                Y=0.3,
                steps=12000,
                warmup=4000,
                re_mr_target=RE_MR_TARGET,
                u_bulk_ref=U_REF,
            )
        )
    # B2: Y -> 0 limit (n = 1) and B4: yield-number series
    for Y in (0.02, 0.05, 0.15, 0.6, 0.9, 1.2, 1.6):
        cases.append(
            _case(
                f"turb_re180_n1_Y{Y:g}",
                mode="turbulent",
                grid=BASE_GRID,
                op=RE180,
                n=1.0,
                Y=Y,
                steps=12000 if Y < 1.0 else 6000,
                warmup=4000 if Y < 1.0 else 2000,
                re_mr_target=RE_MR_TARGET,
                u_bulk_ref=U_REF,
            )
        )

    # -- B5: lattice and grid convergence (same operating point) ---------
    for lattice, grid, gname in (
        ("d3q19", COARSE_GRID, "c"),
        ("d3q19", BASE_GRID, "b"),
        ("d3q19", MID_GRID, "m"),
        ("d3q27", BASE_GRID, "b"),
    ):
        tag = f"gci_{lattice}"
        cases.append(
            _case(
                f"{tag}_{gname}",
                mode="turbulent",
                grid=grid,
                op=RE180,
                n=1.0,
                Y=0.0,
                steps=12000,
                warmup=4000,
                lattice=lattice,
                re_mr_target=RE_MR_TARGET,
                u_bulk_ref=U_REF,
            )
        )

    # -- Re_tau = 395 (higher-Reynolds anchor for the collapse) ----------
    for n in (0.8, 1.0):
        cases.append(
            _case(
                f"turb_re395_n{n:g}_Y0",
                mode="turbulent",
                grid=RE395_GRID,
                op=RE395,
                n=n,
                Y=0.0,
                steps=20000,
                warmup=6000,
                re_mr_target=3.0e4,
                u_bulk_ref=21.0 * RE395["u_tau"],
            )
        )
    cases.append(
        _case(
            "turb_re395_n1_Y0.3",
            mode="turbulent",
            grid=RE395_GRID,
            op=RE395,
            n=1.0,
            Y=0.3,
            steps=20000,
            warmup=6000,
            re_mr_target=3.0e4,
            u_bulk_ref=21.0 * RE395["u_tau"],
        )
    )
    return {"node": "P295W1R2-N04", "thresholds": THRESHOLDS, "cases": cases}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--run", action="store_true", help="execute this shard")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--out-dir", default=str(ART))
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--group", default="all", help="all | laminar | turb | gci (case-name prefix filter)"
    )
    args = ap.parse_args()

    plan = build_plan()
    if args.write:
        out = Path(args.write)
        out.write_text(json.dumps(plan, indent=2))
        print(f"wrote {out} ({len(plan['cases'])} cases)")
    if not args.run:
        return 0

    cases = plan["cases"]
    if args.group == "laminar":
        cases = [c for c in cases if c["name"].startswith("lam")]
    elif args.group == "turb":
        cases = [c for c in cases if c["name"].startswith("turb")]
    elif args.group == "core":
        keep = (
            "turb_re180_n1_Y0",
            "turb_re180_n0.6_Y0",
            "turb_re180_n0.8_Y0",
            "turb_re180_n1_Y0.05",
            "turb_re180_n1_Y0.3",
            "gci_d3q19_c",
            "gci_d3q19_m",
            "gci_d3q27_b",
        )
        cases = [c for c in cases if c["name"] in keep]
    elif args.group == "extra":
        skip = (
            "turb_re180_n1_Y0",
            "turb_re180_n0.6_Y0",
            "turb_re180_n0.8_Y0",
            "turb_re180_n1_Y0.05",
            "turb_re180_n1_Y0.3",
            "gci_d3q19_c",
            "gci_d3q19_m",
            "gci_d3q27_b",
        )
        cases = [c for c in cases if c["name"] not in skip and not c["name"].startswith("lam")]
    elif args.group == "gci":
        cases = [c for c in cases if c["name"].startswith("gci")]
    todo = cases[args.shard :: max(1, args.nshards)]
    runner = HERE / "run_case.py"
    for case in todo:
        target = Path(args.out_dir) / f"{case['name']}.json"
        if args.skip_existing and target.exists():
            print(f"skip {case['name']}", flush=True)
            continue
        cmd = [sys.executable, str(runner), *case["args"], "--out-dir", args.out_dir]
        print("RUN " + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"FAILED {case['name']} rc={rc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

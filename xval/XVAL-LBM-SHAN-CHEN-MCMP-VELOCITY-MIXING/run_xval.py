"""Matrix runner for XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING.

Everything the report shows comes from ``artifacts/result.json`` produced here;
no number is typed by hand.

Usage::

    python xval/XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING/run_xval.py --only W1
    python xval/XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING/run_xval.py --only W2
    python xval/XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING/run_xval.py --only W3
    python xval/XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING/run_xval.py --merge

``--only`` writes ``artifacts/part_<W>.json`` so the workloads can run in
parallel processes; ``--merge`` assembles ``artifacts/result.json`` and stamps
the environment.  The matrix itself is declared in :data:`MATRIX` below and
mirrors ``predeclared.json`` (deviations are recorded in ``DEVIATIONS``).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import velocity_mixing_arms as V  # noqa: E402

ART = HERE / "artifacts"

# Small grids, many small ops: one to four torch threads is ~18x faster than the
# container default of 128 (measured: 600 static-droplet steps took 58 s at 128
# threads and 3.3 s at 1 thread).
torch.set_num_threads(int(__import__("os").environ.get("XVAL_THREADS", "2")))

# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------
PRIMARY_ARMS = ["GOT-MIX", "GOT-SELF", "LBM4", "LBM3", "CTRL-NORM"]
DROPLET_STEPS = 2500       # equal-time window, see DEVIATIONS (GOT-SELF converges ~2250)
POISEUILLE_STEPS = 12000   # ~3.5 diffusion times H^2/nu = 23^2/(1/6) = 3170

MATRIX = {
    "W1_droplet_coupling": {
        "protocol": "n=64, R=12, tau=1, sc_form=standard, {} steps, equal-time window".format(DROPLET_STEPS),
        "runs": [
            {"arm": arm, "G_12": G, "n": 64, "radius": 12.0, "steps": DROPLET_STEPS}
            for G in (-1.5, -2.5)
            for arm in PRIMARY_ARMS
        ],
    },
    "W1_droplet_radius": {
        "protocol": "n=64, G_12=-2.0, tau=1, R in {8,12,16} for Young-Laplace consistency",
        "runs": [
            {"arm": arm, "G_12": -2.0, "n": 64, "radius": R, "steps": DROPLET_STEPS}
            for arm in ("GOT-MIX", "LBM4", "GOT-SELF")
            for R in (8.0, 12.0, 16.0)
        ],
    },
    "W2_poiseuille_primary": {
        "protocol": "ny=25, tau=(1,1), G_12=-2.5, G_x=5e-5, {} steps".format(POISEUILLE_STEPS),
        "runs": [
            {"arm": arm, "ny": 25, "tau_upper": 1.0, "G_12": -2.5, "steps": POISEUILLE_STEPS}
            for arm in PRIMARY_ARMS + ["LIB-ENTRY"]
        ],
    },
    "W2_poiseuille_shift_factor": {
        "protocol": "ny=25, tau_upper=0.75 (vupper=1/12), G_12=-2.5 - separates the "
                    "shift-coefficient factor from the mixing-rule factor",
        "runs": [
            {"arm": arm, "ny": 25, "tau_upper": 0.75, "G_12": -2.5, "steps": POISEUILLE_STEPS}
            for arm in ("GOT-MIX", "LBM4", "LBM4-TAU")
        ],
    },
    "W2_poiseuille_resolution": {
        "protocol": "ny=41 vs ny=25, tau=(1,1), G_12=-2.5 (grid sensitivity)",
        "runs": [
            {"arm": arm, "ny": 41, "tau_upper": 1.0, "G_12": -2.5, "steps": POISEUILLE_STEPS}
            for arm in ("GOT-MIX", "LBM4")
        ],
    },
    "W2_poiseuille_coupling": {
        "protocol": "G_12=-2.0 vs -2.5, ny=25, tau=(1,1) (coupling sensitivity)",
        "runs": [
            {"arm": arm, "ny": 25, "tau_upper": 1.0, "G_12": -2.0, "steps": POISEUILLE_STEPS}
            for arm in ("GOT-MIX", "LBM4")
        ],
    },
    "W3_momentum_ledger": {
        "protocol": "n=48 periodic, no body force, G_12=-2.5, tau=1, 400 steps",
        "runs": [{"arm": arm, "n": 48, "G_12": -2.5, "steps": 400} for arm in PRIMARY_ARMS + ["LBM4-TAU"]],
    },
}

# Honest record of what changed after the design was frozen (indices into
# predeclared.json; no threshold was moved).
DEVIATIONS = [
    {
        "where": "workloads.W1_static_droplet.protocol",
        "predeclared": "relaxed until |u_mix|_max plateaus (convergence-based stopping)",
        "actual": "fixed equal-time window of 3000 steps for every arm",
        "reason": "the OSS/Barycentric arms do not meet the plateau criterion at all "
                  "(|u|max still ~0.05 at 8000 steps with tau=1, G_12=-2.0), so a "
                  "convergence-based stopping rule would compare an equilibrated 'self' "
                  "run with a non-equilibrated mixture run. An equal-time window keeps "
                  "the comparison defined; both the plateau flag and the history are "
                  "reported per run so the reader can see which arms did converge.",
        "threshold_change": False,
    },
    {
        "where": "arms",
        "predeclared": "6 arms",
        "actual": "7 arms (LIB-ENTRY added to the Poiseuille primary block as the harness control)",
        "reason": "the control was described in predeclared.json under arms.LIB-ENTRY but "
                  "not listed in the primary block; it runs at the primary configuration.",
        "threshold_change": False,
    },
]


def _env() -> dict:
    import numpy

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "machine": platform.machine(),
        "cuda": bool(torch.cuda.is_available()),
        "torch_threads": torch.get_num_threads(),
    }


def run_block(name: str, block: dict) -> dict:
    out = {"protocol": block["protocol"], "runs": []}
    t_block = time.time()
    for spec in block["runs"]:
        t0 = time.time()
        spec = dict(spec)
        try:
            if name.startswith("W1"):
                res = V.run_static_droplet(
                    spec["arm"], n=spec["n"], radius=spec["radius"],
                    G_12=spec["G_12"], steps=spec["steps"], check_every=250,
                )
                res["laplace"] = V.laplace_metrics(res)
            elif name.startswith("W2"):
                res = V.run_stratified_poiseuille(
                    None if spec["arm"] == "LIB-ENTRY" else spec["arm"],
                    ny=spec["ny"], tau_upper=spec["tau_upper"], G_12=spec["G_12"],
                    steps=spec["steps"], sample_every=1500,
                )
            else:
                res = V.momentum_ledger(spec["arm"], n=spec["n"], G_12=spec["G_12"], steps=spec["steps"])
            res["wall_s"] = round(time.time() - t0, 2)
        except Exception as exc:  # an arm that cannot run is SKIPPED, never a zero
            res = {"arm": spec["arm"], "spec": spec, "status": "SKIPPED", "reason": repr(exc)}
        out["runs"].append(res)
        print(f"  [{name}] {spec['arm']:<10} G={spec.get('G_12')} ny={spec.get('ny')} "
              f"R={spec.get('radius')} -> {res.get('wall_s')}s", flush=True)
    out["wall_s"] = round(time.time() - t_block, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=["W1", "W2", "W3"])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny matrix, for tests")
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)

    if args.merge:
        result = {"meta": {"env": _env(), "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "predeclared": json.loads((HERE / "predeclared.json").read_text()),
                           "deviations": DEVIATIONS},
                  "workloads": {}}
        for w in ("W1", "W2", "W3"):
            p = ART / f"part_{w}.json"
            if p.exists():
                result["workloads"].update(json.loads(p.read_text()))
        w4 = V.near_vacuum_guard()
        result["workloads"]["W4_near_vacuum_guard"] = {
            "protocol": "one collision on a 16x16 droplet field with one 2x2 cell depleted to rho1=1e-8",
            "result": w4,
        }
        (ART / "result.json").write_text(json.dumps(result, indent=1))
        print("wrote", ART / "result.json", "workloads:", sorted(result["workloads"]))
        return 0

    if args.quick:
        DROPLET = 300
        POIS = 600
        quick = {
            "W1_quick": {"protocol": "quick", "runs": [
                {"arm": a, "G_12": -2.5, "n": 32, "radius": 6.0, "steps": DROPLET}
                for a in ("GOT-MIX", "GOT-SELF", "LBM4", "CTRL-NORM")]},
            "W2_quick": {"protocol": "quick", "runs": [
                {"arm": a, "ny": 25, "tau_upper": 1.0, "G_12": -2.5, "steps": POIS}
                for a in ("GOT-MIX", "LBM4", "CTRL-NORM")]},
        }
        payload = {}
        for name, block in quick.items():
            key = "W1" if name.startswith("W1") else "W2"
            res = run_block(name, block)
            payload.setdefault(key, {})[name] = res
        for key, val in payload.items():
            (ART / f"part_{key}.json").write_text(json.dumps(val, indent=1))
        print("quick run ok")
        return 0

    blocks = {k: v for k, v in MATRIX.items() if k.startswith(args.only)}
    payload = {}
    print(f"running {len(blocks)} block(s) for {args.only}:", flush=True)
    for name, block in blocks.items():
        payload[name] = run_block(name, block)
    (ART / f"part_{args.only}.json").write_text(json.dumps(payload, indent=1))
    print("wrote", ART / f"part_{args.only}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
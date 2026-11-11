"""Round-3 matrix driver for GAP-LBM-MCMP-POISEUILLE.

Runs the *final* pre-declared cell list and writes one JSON record per cell, so
that every number in the report / benchmark is generated (nothing hand-typed).

Pre-declared cells (fixed before the runs; see
``docs/MCMP_STRATIFIED_POISEUILLE_R3.md`` §2):

    sc_form in {"standard", "edm"}            # interaction-force discretisation
    ny      in {25, 33, 49}                   # acceptance / acceptance / hold-out
    kind    in {"driven", "static"}

* ``driven``  (``G_x = 5e-5``): criteria 1, 2, 4 — interface position, two-phase
  flow rate, resolution trend.  Runner: ``run_sweep.run_case`` (two_phase mode).
* ``static``  (``G_x = 0``):    criterion 3 — the parasitic-current floor of the
  model, measured inside the interface band with the time-resolved diagnostic
  ``spurious_diagnostic.run``.

All cells use the calibrated halfway wall convention (``H = ny - 2``), the
neutral-wetting (``mirror``) SC wall, ``u_eq="viscous"`` and
``force_scheme="guo"`` — i.e. the round-2 repair configuration, unchanged.

Usage (whole matrix, serial)::

    PYTHONPATH=src python cases/mcmp_stratified_poiseuille/run_final_matrix.py

Shard (e.g. 4-way, for the cluster queue)::

    PYTHONPATH=src python cases/mcmp_stratified_poiseuille/run_final_matrix.py \
        --shard 0/4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

OUT_DIR = os.path.join("artifacts", "mcmp_stratified_poiseuille", "final")

# (tag, kind, ny, steps, sc_form)
CELLS = [
    (f"{kind}_ny{ny}_{sc}", kind, ny, steps, sc)
    for sc in ("standard", "edm")
    for ny, steps in ((25, 20000), (33, 30000), (49, 45000))
    for kind in ("driven", "static")
]

COMMON = dict(
    nx=4,
    tau_lower=1.0,
    tau_upper=0.75,
    G_x=5e-5,
    G_12=-2.5,
    rho_heavy=0.7,
    rho_light=0.3,
    interface_width=3.0,
    u_eq="viscous",
    force_scheme="guo",
    wall_mode="mirror",
)


def run_cell(tag: str, kind: str, ny: int, steps: int, sc_form: str) -> dict:
    t0 = time.time()
    if kind == "driven":
        from cases.mcmp_stratified_poiseuille.run_sweep import run_case

        rec = run_case(
            ny=ny, nx=COMMON["nx"], tau_lower=COMMON["tau_lower"],
            tau_upper=COMMON["tau_upper"], G_x=COMMON["G_x"], steps=steps,
            G_12=COMMON["G_12"], u_eq=COMMON["u_eq"],
            force_scheme=COMMON["force_scheme"], rho_heavy=COMMON["rho_heavy"],
            rho_light=COMMON["rho_light"], interface_width=COMMON["interface_width"],
            mode="two_phase", wall_mode=COMMON["wall_mode"], sc_form=sc_form,
        )
    else:
        from cases.mcmp_stratified_poiseuille.spurious_diagnostic import run

        rec = run(
            ny=ny, nx=COMMON["nx"], tau_lower=COMMON["tau_lower"],
            tau_upper=COMMON["tau_upper"], gx=0.0, g12=COMMON["G_12"],
            steps=steps, u_eq=COMMON["u_eq"], force_scheme=COMMON["force_scheme"],
            wall_mode=COMMON["wall_mode"], sc_form=sc_form,
            rho_heavy=COMMON["rho_heavy"], rho_light=COMMON["rho_light"],
            width=COMMON["interface_width"], sample_every=max(200, steps // 100),
        )
        # A static control is the same stratification without the body force;
        # U_ref of the driven case at the same resolution is attached by the
        # benchmark assembler (it is an analytic number, not a simulation one).
        rec["gx"] = 0.0
    rec["tag"] = tag
    rec["kind"] = kind
    rec["sc_form"] = sc_form
    rec["wall_seconds_including_import"] = time.time() - t0
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="i/n — run cell i, i+n, i+2n, ...")
    ap.add_argument("--only", default=None, help="comma-separated tags")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)

    cells = list(CELLS)
    if args.only:
        want = {t.strip() for t in args.only.split(",")}
        cells = [c for c in cells if c[0] in want]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cells = [c for k, c in enumerate(cells) if k % n == i]

    os.makedirs(args.out_dir, exist_ok=True)
    for tag, kind, ny, steps, sc_form in cells:
        path = os.path.join(args.out_dir, f"{tag}.json")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"SKIP {tag} (exists)", flush=True)
            continue
        print(f"RUN  {tag} ny={ny} steps={steps} kind={kind} sc_form={sc_form}", flush=True)
        rec = run_cell(tag, kind, ny, steps, sc_form)
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"DONE {tag} in {rec['wall_seconds_including_import']:.1f}s -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
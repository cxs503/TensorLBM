"""Cluster driver: full IN101-N01 campaign + criteria + figures."""
import json
import sys
from pathlib import Path

from tensorlbm.ice_keel_drag import campaign, cases
from tensorlbm.ice_keel_drag.report import build_artifact

RUN = Path("artifacts/IN101_N01/run")
OUT = Path("artifacts/IN101_N01")

which = sys.argv[1] if len(sys.argv) > 1 else "all"
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 64

table = []
if which in ("all", "smooth"):
    table += cases.smooth_ladder(steps=6000, warmup=2000, threads=threads)
if which in ("all", "keel"):
    table += cases.keel_sweep(steps=12000, warmup=4000, threads=threads)
if which in ("all", "turb"):
    table.append(cases.turbulent_attempt(threads=threads))

campaign.run_campaign(table, RUN, backend="native", skip_existing=True)

if which == "all":
    art = build_artifact(RUN, OUT, make_figures=True)
    print(json.dumps({"n_cases": art["n_cases"], "verdicts": art["verdicts"],
                      "n_passed": art["n_passed"], "all_passed": art["all_passed"],
                      "figures": art.get("figures"), "figures_error": art.get("figures_error")},
                     indent=1))
    for cid, row in art["cases"].items():
        print(f"{cid:28s} cf={row['cf']:.6g} cd={row['cd_total']:.6g} "
              f"form={row['cd_form']:.6g} skin={row['cd_skin']:.6g} "
              f"Sfr={row['S_fr_over_plan']:.4f} drift={row['stationarity_drift']:.3g} "
              f"clo={row['closure_residual']:.3g}")
print("ARTIFACT_FILES")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"{p} {p.stat().st_size}")
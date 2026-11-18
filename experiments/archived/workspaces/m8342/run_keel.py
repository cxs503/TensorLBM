"""Cluster driver: keel sweep (resolved geometry) + artifact rebuild."""
import json
from pathlib import Path

from tensorlbm.ice_keel_drag import campaign, cases
from tensorlbm.ice_keel_drag.report import build_artifact

RUN = Path("artifacts/IN101_N01/run")
OUT = Path("artifacts/IN101_N01")
for stale in RUN.glob("keel_*"):
    for f in stale.glob("*"):
        f.unlink()
    stale.rmdir()

table = cases.keel_sweep(steps=12000, warmup=4000, threads=64)
campaign.run_campaign(table, RUN)
art = build_artifact(RUN, OUT, make_figures=False)
print(json.dumps({"verdicts": art["verdicts"], "n_passed": art["n_passed"]}, indent=1))
for cid, row in art["cases"].items():
    print("%-24s cd=%.5g form=%.5g skin=%.5g Sfr=%.3f clo=%.3g drift=%.3g tau_form=%.3g tau_bal=%.3g"
          % (cid, row["cd_total"], row["cd_form"], row["cd_skin"], row["S_fr_over_plan"],
             row["closure_residual"] or float("nan"), row["stationarity_drift"] or float("nan"),
             row["tau_form"], row["tau_total"]))
b = art["criteria"]["b_form_scaling"]
print("fit", (b.get("fit") or {}).get("exponent"), "err", b.get("exponent_error"))
print("ARTIFACT_FILES")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(p, p.stat().st_size)
"""Full campaign at the stable operating points + artifact (no figures)."""
import json
import sys
from pathlib import Path

from tensorlbm.ice_keel_drag import campaign, cases
from tensorlbm.ice_keel_drag.report import build_artifact

RUN = Path("artifacts/IN101_N01/run")
OUT = Path("artifacts/IN101_N01")
which = sys.argv[1] if len(sys.argv) > 1 else "keel"

table = []
if which in ("all", "smooth"):
    table += cases.smooth_ladder(steps=6000, warmup=2000, threads=64)
    table.append(cases.smooth_reference(threads=64))
if which in ("all", "keel"):
    table += cases.keel_sweep(steps=12000, warmup=4000, threads=64)

campaign.run_campaign(table, RUN)
if which == "all":
    art = build_artifact(RUN, OUT, make_figures=False)
    print(json.dumps({"verdicts": art["verdicts"], "n_passed": art["n_passed"]}, indent=1))
    for cid, row in art["cases"].items():
        print("%-26s re_tau=%-5.0f cd=%.5g form=%.5g skin=%.5g Sfr=%.3f "
              "tau_form=%.3g tau_bal=%.3g clo=%.3g drift=%.3g re_h=%.3g"
              % (cid, row["re_tau"], row["cd_total"], row["cd_form"], row["cd_skin"],
                 row["S_fr_over_plan"], row["tau_form"], row["tau_total"],
                 row["closure_residual"], row["stationarity_drift"], row["re_h"]))
    b = art["criteria"]["b_form_scaling"]
    print("fit exponent", (b.get("fit") or {}).get("exponent"), "err", b.get("exponent_error"),
          "r2", (b.get("fit") or {}).get("r2"), "geo_err", b.get("geometry_rel_error_max"))
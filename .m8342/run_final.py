"""Final campaign step: turbulent-branch attempt + artifact rebuild (no figures)."""
import json
from pathlib import Path

from tensorlbm.ice_keel_drag import campaign, cases
from tensorlbm.ice_keel_drag.report import build_artifact

RUN = Path("artifacts/IN101_N01/run")
OUT = Path("artifacts/IN101_N01")
campaign.run_campaign([cases.turbulent_attempt(threads=64)], RUN)
art = build_artifact(RUN, OUT, make_figures=False)
print(json.dumps({"verdicts": art["verdicts"], "n_cases": art["n_cases"]}, indent=1))
b = art["criteria"]["b_form_scaling"]
print("exp", (b.get("fit") or {}).get("exponent"), "err", b.get("exponent_error"))
print("exp_excess", b.get("exponent_excess"), "err", b.get("exponent_excess_error"),
      "r2", (b.get("fit_excess") or {}).get("r2"))
print("a:", json.dumps(art["criteria"]["a_smooth_cf"], indent=1)[:1200])
print("d:", json.dumps({k: v for k, v in art["criteria"]["d_observed_band"].items() if k != "rows"}))
print("ARTIFACT_FILES")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(p, p.stat().st_size)
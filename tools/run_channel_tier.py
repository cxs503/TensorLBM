"""Run a single channel tier and emit a campaign-compatible results payload."""
import json
import os
import sys
from pathlib import Path

import numpy as np

from tensorlbm.turbulence_closure import cases
from tensorlbm.turbulence_closure.campaign import _json_safe, _verdicts

re_tau = int(os.environ.get("RE_TAU", "180"))
steps = int(os.environ.get("STEPS", "25000"))
out = Path(os.environ.get("OUT", "runs/tvf_channel_single"))
cfgs = {
    180: dict(re_tau=180, ny=102, nu=0.0022, length_wall_units=300.0,
              span_wall_units=108.0, cs=0.16),
    395: dict(re_tau=395, ny=182, nu=0.0021, length_wall_units=300.0,
              span_wall_units=110.0, cs=0.16),
    590: dict(re_tau=590, ny=242, nu=0.0018, length_wall_units=300.0,
              span_wall_units=108.0, cs=0.16),
}
cfg = cases.ChannelConfig(**cfgs[re_tau], steps=steps, warmup_fraction=0.3,
                          threads=int(os.environ.get("THREADS", "40")))
out.mkdir(parents=True, exist_ok=True)
res = cases.run_channel(cfg, out / "channel")
payload = {
    "campaign": "P152W1-N02 turbulence validation face",
    "profile": "compact-single",
    "cases": {"channel": [res]},
    "artefacts": {"results_json": str((out / "results.json").resolve())},
}
payload["verdicts"] = _verdicts(payload)
(out / "results.json").write_text(
    json.dumps(_json_safe(payload), indent=2, allow_nan=False, default=str),
    encoding="utf-8")
d = res["reference"]["dns"]
print(json.dumps({"re_tau": re_tau, "max_rel_error": d.get("max_rel_error"),
                  "peak_u_rms_plus": res["measured"].get("peak_u_rms_plus"),
                  "kappa_fit": d.get("kappa_fit"), "passed": d.get("passed")}, indent=2))
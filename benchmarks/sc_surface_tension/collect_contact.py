#!/usr/bin/env python
"""Collect the per-point contact-angle probes (logs/st_cal_rw*.log) into artifacts.

No number is typed in: each row is parsed from the probe's own JSON line and
re-emitted with its calibration/hold-out role attached.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "sc_surface_tension"
CAL = {0.70, 0.85, 1.00, 1.15, 1.30}
HOLD = {0.775, 1.225}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(str(REPO / "logs" / "st_cal_rw*.log"))):
        for line in Path(f).read_text().splitlines():
            if not line.startswith("RESULT "):
                continue
            d = json.loads(line[len("RESULT "):])
            rho_w = round(float(d["rho_w"]), 3)
            d["kind"] = "cal" if rho_w in CAL else ("holdout" if rho_w in HOLD else "other")
            # the probe stored the *supplementary* angle (old estimator convention
            # cos(theta) = z_c/R); the corrected sessile convention is
            # cos(theta) = -z_c/R, an exact one-to-one map theta -> 180 - theta.
            for key in ("theta_mask_deg", "theta_mask_corrected_deg"):
                if key in d and d[key] == d[key]:
                    d[key] = 180.0 - d[key]
            d["source_log"] = Path(f).name
            rows.append(d)
            (OUT / f"contact_sweep_rho{rho_w:.3f}.json").write_text(json.dumps(d, indent=1))
    (OUT / "contact_sweep_rows.json").write_text(json.dumps(rows, indent=1))
    print(f"collected {len(rows)} rows")
    for d in rows:
        print(f"  rho_w={d['rho_w']:.3f} kind={d['kind']} young={d['theta_young_deg']:.1f} "
              f"mask={d['theta_mask_deg']} status={d['status_mask']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

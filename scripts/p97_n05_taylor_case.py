"""Single Taylor–Aris dispersion case (P97-N05 shard worker).

Usage::

    python scripts/p97_n05_taylor_case.py --case '{"ny": 9, "nx": 64, ...}' --out out.json

The case dict is unpacked into :class:`tensorlbm.taylor_aris.DispersionConfig`.
Writes a JSON dict in the same format as the campaign collector expects.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.taylor_aris import DispersionConfig, simulate_channel_dispersion  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="JSON dict of DispersionConfig fields")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    case = json.loads(args.case)
    name = case.pop("name", "case")
    cfg = DispersionConfig(**case)
    t0 = time.time()
    res = simulate_channel_dispersion(cfg)
    res["name"] = name
    res["elapsed_s"] = round(time.time() - t0, 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(
        f"{name}: D_eff={res['d_eff']:.5f} ref={res['d_eff_discrete']:.5f} "
        f"rel={100 * res['rel_err_vs_discrete']:+.2f}%  r2={res['r2']:.5f} "
        f"Pe={res['pe']:.2f} ({res['elapsed_s']}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Shard driver: run one case from ``cases.json`` (used by the cluster queue).

    python3 run_shard.py --index 3          # run case 3
    python3 run_shard.py --list             # print the case table

Each case writes ``runs/<tag>.json``; the comparison step
(``make_comparison.py``) consumes whatever is present, so a partially finished
sweep still yields a complete report with the missing rows marked ``not_run``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list or a.index is None:
        for i, c in enumerate(CASES):
            print(i, json.dumps(c))
        return

    c = CASES[a.index]
    tag = c["tag"]
    out = HERE / "runs" / f"{tag}.json"
    (HERE / "runs").mkdir(exist_ok=True)

    if c["party"] == "A":
        from party_a_tensorlbm import run_case
        from cavity_spec import CavitySpec
        import torch

        torch.set_num_threads(int(c.get("threads", 4)))
        spec = CavitySpec(
            re=c["re"], n=c["n_nodes"] - 2, u_lid=c["u_lid"],
            max_steps=c["max_steps"], resid_tol=c["resid_tol"],
        )
        res = run_case(spec, c["n_nodes"])
    else:
        from party_b_pylbm import run_case
        from cavity_spec import CavitySpec

        spec = CavitySpec(
            re=c["re"], n=c["n"], u_lid=c["u_lid"],
            max_steps=c["max_steps"], resid_tol=c["resid_tol"],
        )
        res = run_case(spec, c["n"], generator=c.get("generator", "cython"))

    res["case"] = c
    out.write_text(json.dumps(res, indent=1))
    m = res["metrics"]
    print(
        f"[{tag}] steps={res['steps']} resid={res['final_resid']:.2e} "
        f"u_min={m['u_min']:+.4f} v_max={m['v_max']:+.4f} "
        f"rmse_u={m['rmse_u']:.4f} max_dev={m['max_abs_dev_pct']:.2f}% -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
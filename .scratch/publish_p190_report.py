#!/usr/bin/env python
"""Assemble and publish the P190-N01 progress report to the platform API.

Reads the generated document + campaign JSON + figures and POSTs a paper-style
report to ``$TENSOR_API_BASE/tensor/reports``.  Figures are attached as base64
PNGs; the markdown is the generated document with its relative image links
removed (the figures travel in the ``figures`` array instead).
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys

import requests

API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 9922
TITLE = (
    "P190-N01 rev.2 (supersedes report 11983): LBM free-surface fill transport on the Zalesak slotted-cylinder "
    "rotation benchmark (D2Q9/D3Q19, mass ledger, sharpness, grid convergence)"
)
SCENARIO = "P190-N01"

CAPTIONS = {
    "fig01_fields.png": "Zalesak slotted cylinder, one revolution, N=64: (a) initial "
    "fill fraction, (b) proposed LBM fill transport (tau=0.504 + conservative mass "
    "ledger), (c) finite-volume VOF reference, (d) pointwise error of the LBM run.",
    "fig02_convergence.png": "Grid convergence of the L1 shape error, the Jaccard "
    "overlap and the 10-90% interface thickness, LBM fill transport vs the VOF "
    "reference (N = 64/128/256).",
    "fig03_mass_and_tau.png": "Relative volume drift per step for both operators "
    "(left) and the BGK relaxation-rate scan at N=64 (right).",
    "fig04_3d_cross_section.png": "D3Q19 extruded (y-invariant) body vs the D2Q9 "
    "solution started from the same field, slice y = 0.5, plus their difference.",
}


def token():
    r = requests.post(f"{API}/auth/login", json={"username": USER, "password": PASS}, timeout=30)
    r.raise_for_status()
    d = r.json()
    return d.get("access_token") or d.get("token") or d["data"]["access_token"]


def main():
    art = "artifacts/zalesak_p190_n01"
    doc = "docs/P190_N01_ZALESAK_LBM.md"
    with open(doc) as fh:
        md = fh.read()
    md = re.sub(r"^!\[.*\]\(figures/.*\)\s*$", "", md, flags=re.M)
    with open(os.path.join(art, "zalesak_lbm_all.json")) as fh:
        res = json.load(fh)
    crit = res.get("criteria", {})
    verdicts = ", ".join(
        f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in crit.items() if v["pass"] is not None
    )
    g = res["gci"]
    lbm = {r["n"]: r for r in g["lbm"]}
    vof = {r["n"]: r for r in g["vof"]}
    ref = lbm.get(128) or lbm[max(lbm)]
    summary = (
        "[rev.2 corrects one wrong superlative in doc 3.1 of report 11983; all numbers "
        "identical] New capability: TensorLBM now carries the free-surface liquid fraction on the "
        "D2Q9/D3Q19 lattice populations (BGK scalar transport + conservative mass ledger + "
        "Korner flag layer) and grades it on the Zalesak rotation benchmark. Mass is "
        f"conserved to round-off ({ref['volume_abs_rel_change']:.1e} at N={ref['n']}), the "
        f"field is divergence-free ({ref['div_u_max']:.0e}), the L1 shape error converges "
        f"(order {res['gci']['orders_lbm']['order_finest_pair']:.2f}), but sharpness "
        f"({ref['interface_thickness_mean_cells']:.1f} cells vs <3) and the N={ref['n']} "
        f"overlap ({ref['shape_overlap']:.3f} vs >0.95) FAIL. Verdict: {verdicts}."
    )[:500]
    figs = []
    fig_dir = os.path.join(art, "figures")
    for name in sorted(os.listdir(fig_dir)):
        if not name.endswith(".png"):
            continue
        with open(os.path.join(fig_dir, name), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        figs.append({"name": name, "caption": CAPTIONS.get(name, name), "png_base64": b64})
    payload = {
        "title": TITLE,
        "summary": summary,
        "scenario_key": SCENARIO,
        "mission_id": MISSION_ID,
        "tags": ["TensorLBM", "free-surface", "Zalesak", "LBM", "D2Q9", "D3Q19",
                 "grid-convergence", "interface-capturing", "P190-N01"],
        "markdown": md,
        "figures": figs,
    }
    tok = token()
    r = requests.post(
        f"{API}/tensor/reports",
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    print("status", r.status_code)
    print(r.text[:1500])
    r.raise_for_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
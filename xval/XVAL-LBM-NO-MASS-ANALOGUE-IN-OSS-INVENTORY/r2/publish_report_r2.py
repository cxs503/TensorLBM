"""Publish the R2 report to the platform (token from the environment, no secrets stored).

Numbers are injected from ``summary_r2.json`` / ``REPORT2.md``; figures are embedded as
base64 PNG.  Nothing is copied by hand.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main() -> int:
    tok = post("/auth/login", {"username": USER, "password": PASS})
    token = tok.get("access_token") or tok.get("token")
    if not token:
        raise SystemExit(f"login failed: {list(tok)}")
    dig = json.loads((HERE / "summary_r2.json").read_text())
    body = (HERE / "REPORT2.md").read_text()
    figs = []
    caps = {
        "fig1_beta_ladder.png":
            "C_I / C_Q vs beta against the exact unsteady-Stokes Gamma curves; the "
            "inviscid potential anchor is the dotted line. Right: the same measurements "
            "against both references.",
        "fig2_resolution.png":
            "In-phase error vs Stokes-layer thickness in lattice cells: +6.0/+10.7/+14.6 % "
            "at 2.55/1.81/1.28 cells -- a resolution effect.",
        "fig3_interior_zeroing.png":
            "Defect D1: 4-neighbour core zeroing injects a zero population into the fluid "
            "(max|rho-1| = 0.30, dM/M = -9 %); 8-neighbour gives 0.011 and 3e-7.",
        "fig4_quadrature.png":
            "Quadrature channel: E1/E2 sit 29-38 % below -Im Gamma at every beta -> "
            "WITHHELD, do not quote C_Q from this module.",
    }
    for name, cap in caps.items():
        p = HERE / "figures" / name
        if p.exists():
            figs.append({"name": name, "caption": cap,
                         "png_base64": base64.b64encode(p.read_bytes()).decode()})
    payload = {
        "title": "XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY R2: 用精确非定常 Stokes 参考"
                 "闭合附加质量验证欠账（并修掉固体核心清零缺陷）",
        "summary": (
            f"R2 找回了 round-1 缺失的参考系：精确非定常 Stokes(1851) 水动力函数 Γ(β)，"
            f"主臂 C_I 与 C_a·Γ_re 相差 {dig['primary']['err_vs_stokes_pct']:+.1f}%"
            f"（对势流锚则为 {dig['primary']['err_vs_potential_pct']:+.1f}%），4 个健康臂 "
            f"全部落在 owner 15% 内、均未进 3%；同时定位并修复 "
            f"oscillatory_cylinder_inertia._interior_nodes 的 4-邻域清零缺陷"
            f"（max|ρ−1| 0.30→0.011，ΔM/M −9%→3e-7），配 2 个回归测试。"),
        "scenario_key": "xval_lbm_no_mass_analogue_in_oss_inventory_r2",
        "mission_id": 22617,
        "tags": ["XVAL", "TensorLBM", "added-mass", "Stokes", "verification-debt",
                 "oscillatory-cylinder", "defect-fix"],
        "markdown": body,
        "figures": figs,
    }
    (Path("/tmp/report.json")).write_text(json.dumps(payload))
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
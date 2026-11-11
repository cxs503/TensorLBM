#!/usr/bin/env python
"""Build the platform progress report (six elements) for the F/2 audit, pass 2.

Reads the sealed audit JSON + figures and writes ``/tmp/report.json`` for

    curl -X POST $TENSOR_API_BASE/tensor/reports ...

so that every number in the report is injected programmatically and nothing is
transcribed by hand.

    python tools/make_f2_v2_platform_report.py --out runs/f2_audit_v2 \
        --json /tmp/report.json --mission 11525
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import make_f2_v2_report as R


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/f2_audit_v2")
    ap.add_argument("--json", default="/tmp/report.json")
    ap.add_argument("--mission", type=int, default=11525)
    ap.add_argument("--scenario-key", default="ice_roughness_f2_audit")
    args = ap.parse_args()

    out = Path(args.out)
    payload = json.loads((out / "f2_contamination_audit_v2.json").read_text())
    prov_path = out / "provenance.json"
    if prov_path.is_file():
        payload["_provenance"] = json.loads(prov_path.read_text())
    figdir = out / "figures"
    payload["figures"] = sorted(p.name for p in figdir.glob("*.png")) if figdir.is_dir() else []
    seal_path = out / "seal.json"
    if seal_path.is_file():
        payload["aggregate_seal"] = json.loads(seal_path.read_text()).get("aggregate")

    md = R.build(payload, fig_dir=str(figdir))
    crit = payload["machine_criteria"]
    passed = [k for k, v in crit.items() if v.get("pass")]
    verdict = payload["recalibration"]["verdict"]

    figures = []
    captions = {
        "fig1_defect_signature.png": "缺陷签名 D(tau)：闭式驱动的动量占比，圆点为战役实测松弛时间",
        "fig2_reported_uplus.png": "同一输入 k_s+ 下两台引擎报出的 U_b+（种子均值±sd）",
        "fig3_paired_delta.png": "配对相对 Δ 与 95% Student-t CI，虚线为 ±1% 判定带",
        "fig4_laminar_anchor.png": "层流锚：各引擎/网格的实测/解析比（1.0 精确，0.5 为 F/2 签名）",
    }
    for name in payload["figures"]:
        figures.append({"name": name, "caption": captions.get(name, name),
                        "png_base64": _b64(figdir / name)})

    title = (f"F/2 强制项标度污染审计 pass2：{len(passed)}/6 判据通过，"
             f"重标结论 {verdict}")
    summary = (
        "在缺陷窗口源码上复现 (1-s_+/2) 外力权重导致的驱动塌缩 "
        f"(闭式 D(tau*)=0.5，绝对误差 {crit['1_defect_signature_F_half'].get('closed_form_abs_err'):.1e})，"
        f"修复引擎层流锚偏差 {100 * crit['2_fixed_laminar_anchor_le_0p4pct'].get('converged_abs_err', float('nan')):.3f} %；"
        f"以单因子引擎做 3 档 k_s+ × {len(payload['seeds'])} 种子的配对重算，给出 C_w 的 Δ 与 95% t-CI，"
        f"并按 |Δ|>1% 给出重标表（{verdict}）。"
    )
    report = {
        "title": title,
        "summary": summary,
        "scenario_key": args.scenario_key,
        "mission_id": args.mission,
        "tags": ["TensorLBM", "ice_roughness", "TRT forcing", "contamination audit",
                 "recalibration", "MOCK-audit", "numerical"],
        "markdown": md,
        "figures": figures,
    }
    Path(args.json).write_text(json.dumps(report, ensure_ascii=False))
    print(json.dumps({"json": args.json, "bytes": Path(args.json).stat().st_size,
                      "figures": [f["name"] for f in figures],
                      "criteria_passed": passed}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
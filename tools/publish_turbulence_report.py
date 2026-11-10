"""Build the ``/tmp/report.json`` payload for the TensorPlatform report endpoint.

Reads the campaign artefacts (``results.json`` + figures) and emits a
publication-grade payload: title / summary / scenario_key / mission_id / tags /
markdown / figures (name, caption, png_base64).

Usage::

    python tools/publish_turbulence_report.py --results runs/turbulence_validation_face/results.json \\
        --report docs/turbulence_validation_face.md --out /tmp/report.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def build(results_path: Path, report_path: Path, out_path: Path,
          mission_id: int = 9008) -> dict:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    markdown = report_path.read_text(encoding="utf-8")
    fig_dir = results_path.parent / "figures"
    figures = []
    for path in sorted(fig_dir.glob("*.png")):
        name = path.stem
        caption = name.replace("_", " ")
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        })
    verdicts = results.get("verdicts", {})
    passed = sum(1 for k, v in verdicts.items() if k != "all_passed" and v.get("passed"))
    total = sum(1 for k in verdicts if k != "all_passed")
    payload = {
        "title": "TensorLBM 湍流验证面：MRT 碰撞器 + 非平衡壁模型驱动槽道/台阶/圆柱三案例三档 Re 闭环",
        "summary": (
            f"新增 MRT 碰撞器（正交归一矩基自动生成）+ NEEM 边界 + Reichardt 壁面应力模型；"
            f"槽道 u⁺(y⁺) vs MKM1999 DNS、后向台阶 x_R/h vs Armaly 实验、圆柱 St vs 实验，"
            f"各三档 Reynolds 数扫描；判据 {passed}/{total} 通过。"
        ),
        "scenario_key": "tensorlbm.turbulence_validation_face",
        "mission_id": mission_id,
        "tags": ["TensorLBM", "turbulence", "MRT", "wall-model", "channel-flow",
                 "backward-facing-step", "cylinder", "Strouhal", "DNS-closure"],
        "markdown": markdown,
        "figures": figures,
        "verdicts": verdicts,
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return {"path": str(out_path), "n_figures": len(figures),
            "verdicts": verdicts, "bytes": out_path.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", default="/tmp/report.json")
    ap.add_argument("--mission-id", type=int, default=9008)
    args = ap.parse_args()
    info = build(Path(args.results), Path(args.report), Path(args.out), args.mission_id)
    print(json.dumps(info, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
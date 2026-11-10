"""Build `/tmp/report.json` for the platform report API from the run artefacts.

Reads the machine-generated `results.json` (+ the markdown section + figures) of
a DNS run and emits the payload the `/tensor/reports` endpoint expects, with the
figures inlined as base64.  No number is typed by hand: everything comes from
`results.json`.

Usage::

    python tools/elbm_report_payload.py --results artifacts/elbm_dns/re180/results.json \
        --markdown docs/reports/elbm_dns_re180.md --out /tmp/report.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

TITLE = "ELBM 熵格点 Boltzmann 槽道湍流：可续算 DNS 驱动 + 两处熵条件缺陷修复"


def _fig_caption(name: str) -> str:
    if "uplus" in name:
        return "平均速度剖面 U+(y+)：ELBM 算例 vs MKM-1999 谱方法 DNS，含 κ=0.41 对数律参照"
    if "stresses" in name:
        return "雷诺应力：u_rms+ 与 -<u'v'>+ vs DNS（判据 b 用 u_rms+ 内层峰位）"
    if "stability" in name:
        return "稳定性轨迹：max|u|、最小分布函数（正性）与 <alpha>（熵限幅器工作点）"
    return name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--out", default="/tmp/report.json")
    ap.add_argument("--extra-markdown", default=None,
                    help="optional file prepended to the markdown section")
    args = ap.parse_args(argv)

    res = json.loads(Path(args.results).read_text())
    md = Path(args.markdown).read_text()
    if args.extra_markdown and Path(args.extra_markdown).exists():
        md = Path(args.extra_markdown).read_text() + "\n\n" + md

    g = res["gates"]
    verdict = res["verdict"]
    figures = []
    for path in res.get("figures", []):
        p = Path(path)
        if p.exists():
            figures.append({
                "name": p.stem,
                "caption": _fig_caption(p.stem),
                "png_base64": base64.b64encode(p.read_bytes()).decode("ascii"),
            })

    summary = (
        f"ELBM(D3Q27) 槽道湍流：新增可续算 DNS 驱动与报告工具；修复两处熵条件缺陷"
        f"（精确平衡态 α→0：fp32 f^eq 在镜面点的虚假 ΔH=+1.1e-5；Guo 半步力平移污染熵条件"
        f" ΔH≈ρ|u|F/2）。本次算例 Re_tau={res['re_tau_target']:.0f}, 网格 {res['grid']}, "
        f"Δy+={res['delta_plus']:.2f}, {res['steps']} 步/{res['samples']} 样本；"
        f"判据 (a)={verdict['a']} (b)={verdict['b']} (c)={verdict['c']} (d)={verdict['d']}"
        f"（Cf 相对误差 {100 * g['c_friction_coefficient']['cf_rel_error']:.2f}%，"
        f"对数斜率相对误差 {100 * g['a_log_slope'].get('relative_error', float('nan')):.2f}%）"
    )

    payload = {
        "title": TITLE,
        "summary": summary,
        "scenario_key": "elbm_channel_dns_re180",
        "mission_id": 10817,
        "tags": ["TensorLBM", "ELBM", "entropic-lattice-boltzmann", "channel-flow", "DNS",
                 "p318w3-n03", "defect-fix", "resumable-driver"],
        "markdown": md,
        "figures": figures,
    }
    Path(args.out).write_text(json.dumps(payload))
    print(f"wrote {args.out}: {len(md)} chars markdown, {len(figures)} figures, "
          f"{Path(args.out).stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
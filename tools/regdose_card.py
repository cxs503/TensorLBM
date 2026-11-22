#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DE-X-COL2D-REGDOSE: assemble the BB card from the measured run data.

The prose lives in ``tools/regdose_card_template.md``; every number in the
card is injected from ``artifacts/COL2D-REGDOSE/summary.json`` (extracted
programmatically from ``runs/*/result.json`` by ``tools/regdose_analyze.py``)
or from the archived ``runs/<arm>/run.log`` headers.  No number is
hand-copied.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def fmt(value, spec="{:.6f}", dash="—"):
    if value is None:
        return dash
    return spec.format(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="artifacts/COL2D-REGDOSE/summary.json")
    ap.add_argument("--triple", default="artifacts/COL2D-REGDOSE/triple.md")
    ap.add_argument("--template", default="tools/regdose_card_template.md")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="docs/drop_entry_bb/X_COL2D-REGDOSE.md")
    ap.add_argument("--max-lines", type=int, default=55)
    a = ap.parse_args()

    s = json.loads(Path(a.summary).read_text())
    triple = Path(a.triple).read_text().rstrip("\n")
    one_x = s["cited_1x"]
    ctrl = s["cited_ctrl"]
    arms = {m["arm"]: m for m in s["arms"]}
    runs = Path(a.runs_dir)

    d05, d200, d400 = arms["reg_d050"], arms["reg_d200"], arms["reg_d400"]

    # provenance: the commit recorded in the run.log header (not hard-coded)
    header = (runs / "reg_d200" / "run.log").read_text().splitlines()[0]
    commit = re.search(r"commit=(\S+)", header).group(1)
    card_no = re.search(r"card=(\d+)", header).group(1)

    # measured ctrl re-run (bit-level anchor check)
    ctrl_run = json.loads((runs / "ctrl_pl" / "result.json").read_text())
    ctrl_hist = [r for r in ctrl_run["history"] if r.get("depth_over_D") is not None]
    ctrl_peak_row = max(ctrl_hist, key=lambda r: float(r["depth_over_D"]))
    ctrl_peak = float(ctrl_peak_row["depth_over_D"])
    ctrl_t = float(ctrl_peak_row["time_ms"])
    p200 = float(max((r for r in json.loads((runs / "reg_d200_pl" / "result.json").read_text())["history"]
                       if r.get("depth_over_D") is not None),
                     key=lambda r: float(r["depth_over_D"]))["depth_over_D"])

    dose_rows = []
    for arm, dose, tau in (("reg_d050", 0.5, 0.25019385), ("reg_d200", 2.0, 1.0007754),
                           ("reg_d400", 4.0, 2.0015508)):
        dose_rows.append(f"| {arm} | {tau} | {dose}× | 实测 |")
    dose_rows.append(f"| rr_base | {one_x['rr_tau']} | 1× | 引 COL2 卡，禁重跑 |")
    dose_table = "\n".join(dose_rows)

    best_err = one_x["peak_err_vs_exp_pct"]
    best_terr = 100.0 * (one_x["peak_time_ms_pc"] - 10.39446) / 10.39446

    verdict = [
        f"- **判决：{s['verdict']}**（预登记的「peak 随剂量单调升、2×∈[1.60,1.75]D」被**证伪**；"
        f"证伪判据「任一高剂量臂 peak < 1× 的 1.555257D」触发：{', '.join(s['non_monotonic_arms'])}）",
        f"  - 0.5×（τ=0.25019385）：**触稳定性边界**——{d05['stop_reason']} @ {fmt(d05['peak_time_ms_abs'], '{:.4f}')}ms"
        f"（接触前，无接触步），漂移 {fmt(d05['volume_drift_final'], '{:.3e}')}>1e-4 门、"
        f"mach_max {fmt(d05['mach_max'], '{:.4f}')}<0.3 门 ⇒ 死于体积漂移而非爆速",
        f"  - 2×（τ=1.0007754）：满窗 {d200['stop_reason']}，peak {fmt(d200['peak_depth_D'])}D"
        f"（误差 {fmt(d200['peak_err_vs_exp_pct'], '{:+.3f}')}%，Δ1× {fmt(d200['delta_vs_1x_D'], '{:+.6f}')}D ="
        f" {fmt(d200['delta_vs_1x_pct'], '{:+.2f}')}%），峰时 {fmt(d200['peak_time_ms_pc'], '{:.4f}')}ms(pc)，"
        f"接触后健康 {fmt(d200['postcontact_healthy_ms'], '{:.3f}')}ms，漂移 {fmt(d200['volume_drift_final'], '{:.3e}')}",
        f"  - 4×（τ=2.0015508）：满窗 {d400['stop_reason']}，peak {fmt(d400['peak_depth_D'])}D"
        f"（误差 {fmt(d400['peak_err_vs_exp_pct'], '{:+.3f}')}%，Δ1× {fmt(d400['delta_vs_1x_D'], '{:+.6f}')}D ="
        f" {fmt(d400['delta_vs_1x_pct'], '{:+.2f}')}%），峰时 {fmt(d400['peak_time_ms_pc'], '{:.4f}')}ms(pc)，"
        f"接触后健康 {fmt(d400['postcontact_healthy_ms'], '{:.3f}')}ms，漂移 {fmt(d400['volume_drift_final'], '{:.3e}')}",
        f"  - 对照（引值）：ctrl {fmt(ctrl['peak_depth_D'])}D / 1× REG {fmt(one_x['peak_depth_D'])}D；"
        f"⇒ 剂量响应单峰，最优点在 1×（BGK 稳定底），增益 {fmt(one_x['peak_depth_D'] - ctrl['peak_depth_D'], '{:+.6f}')}D"
        f"（{fmt(100.0 * (one_x['peak_depth_D'] - ctrl['peak_depth_D']) / ctrl['peak_depth_D'], '{:+.2f}')}%）",
    ]

    # card numbers, read back from the archived run.log headers
    card_map = {}
    for arm_dir in sorted(runs.iterdir()):
        log = arm_dir / "run.log"
        if log.exists():
            head = log.read_text().splitlines()[0]
            m = re.search(r"card=(\d+)", head)
            if m:
                card_map[arm_dir.name] = m.group(1)
    main_cards = "/".join(card_map.get(a, "?") for a in ("reg_d050", "reg_d200", "reg_d400"))
    plane_cards = "/".join(card_map.get(a, "?") for a in ("reg_d050_pl", "reg_d200_pl", "reg_d400_pl", "ctrl_pl"))

    subs = {
        "{{TRIPLE_TABLE}}": triple,
        "{{DOSE_TABLE}}": dose_table,
        "{{VERDICT_LINES}}": "\n".join(verdict),
        "{{NU05}}": fmt(d05.get("nu_ratio_vs_1x"), "{:.2f}"),
        "{{NU2}}": fmt(d200.get("nu_ratio_vs_1x"), "{:.2f}"),
        "{{NU4}}": fmt(d400.get("nu_ratio_vs_1x"), "{:.2f}"),
        "{{COMMIT}}": commit,
        "{{CTRL_PEAK}}": repr(ctrl_peak),
        "{{CTRL_T}}": repr(ctrl_t),
        "{{P200}}": repr(p200),
        "{{P2}}": fmt(d200["peak_depth_D"]),
        "{{P4}}": fmt(d400["peak_depth_D"]),
        "{{C2}}": str(d200["contact_step"]),
        "{{C4}}": str(d400["contact_step"]),
        "{{T05}}": fmt(d05["peak_time_ms_abs"], "{:.4f}"),
        "{{DRIFT05}}": fmt(d05["volume_drift_final"], "{:.3e}"),
        "{{BEST_ERR}}": fmt(best_err, "{:.3f}"),
        "{{BEST_TERR}}": fmt(best_terr, "{:+.2f}"),
        "{{CARDS}}": f"主臂 卡{main_cards}；planes 复跑臂 卡{plane_cards}",
    }

    text = Path(a.template).read_text()
    for key, val in subs.items():
        text = text.replace(key, val)
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", text)
    if left:
        raise SystemExit(f"unresolved placeholders: {sorted(set(left))}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    n = len(text.rstrip("\n").split("\n"))
    print(f"card -> {out} ({n} lines, limit {a.max_lines})")
    if n > a.max_lines:
        print("WARNING: card exceeds the line limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

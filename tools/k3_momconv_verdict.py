#!/usr/bin/env python3
"""K3-MOMCONV verdict: momentum-carrying I->G interface conversion.

All numbers are injected programmatically from the arm artefacts in
``runs/k3_*/result.json``; nothing is hand-copied.  The decision gates are
the ones written in ``docs/audits/k3_momconv_20261121/PREREG.md`` (written
before the runs).

Usage::

    python3 tools/k3_momconv_verdict.py --tree .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# --- pre-registered constants (PREREG.md) ---------------------------------
EXP_DEPTH_D = 1.857434          # experiment, Santini case-3
EXP_TIME_MS = 10.394
ANCHOR_DEPTH_D = 1.4352204748   # campaign anchor (ctrl)
ANCHOR_TIME_MS = 11.090571428571428
HIT_ERR = 0.12                  # |err| < 12 % and alive > 10.4 ms
FALSIFY_ERR = 0.15              # |err| > 15 %  (or shallower / dead early)
SURVIVE_MS = 10.4

ARMS = ("ctrl", "inherit", "redistribute")
REPAIR_ARMS = ("inherit_fix", "redistribute_liq")


def load_arm(tree: Path, arm: str) -> dict:
    path = tree / "runs" / f"k3_{arm}" / "result.json"
    if not path.exists():
        raise SystemExit(f"missing arm artefact: {path}")
    return json.loads(path.read_text())


def peak_depth(result: dict) -> tuple[float, float, int]:
    """(peak depth over D, time of the peak, step of the peak)."""
    best = None
    for row in result["history"]:
        value = row.get("depth_over_D")
        if value is None:
            continue
        if best is None or value > best[0]:
            best = (value, row["time_ms"], row["step"])
    if best is None:
        return (float("nan"), float("nan"), -1)
    return best


def budget_window(result: dict, t_lo_ms: float, t_hi_ms: float) -> dict:
    """Cumulative budget terms between two times (contact -> peak)."""
    rows = (result.get("budget_diag") or {}).get("rows") or []
    keys = (
        "conv_loss",
        "conv_gain",
        "ke_liq",
        "ke_axial",
        "ke_lat",
        "ke_drop",
        "diss_rate",
        "gas_power_free",
        "grav_power",
        "area_total",
    )
    out = {key: 0.0 for key in keys}
    e_scale = None
    n = 0
    for row in rows:
        if not (t_lo_ms <= row["time_ms"] <= t_hi_ms):
            continue
        if e_scale is None:
            e_scale = row["ke_liq"]
        for key in keys:
            out[key] += float(row.get(key, 0.0))
        n += 1
    out["rows"] = n
    out["e_scale"] = e_scale or 0.0
    out["conv_net"] = out["conv_loss"] - out["conv_gain"]
    out["conv_loss_share"] = out["conv_loss"] / out["e_scale"] if out["e_scale"] else float("nan")
    out["conv_gain_share"] = out["conv_gain"] / out["e_scale"] if out["e_scale"] else float("nan")
    out["conv_net_share"] = out["conv_net"] / out["e_scale"] if out["e_scale"] else float("nan")
    out["mass_blowup"] = bool(abs(out["conv_gain"]) > 1e3 * max(out["e_scale"], 1e-30))
    out["ke_axial_at_peak"] = 0.0
    for row in rows:
        if abs(row["time_ms"] - t_hi_ms) < 1e-9:
            out["ke_axial_at_peak"] = float(row.get("ke_axial", 0.0))
            out["ke_liq_at_peak"] = float(row.get("ke_liq", 0.0))
            out["conv_loss_at_peak"] = float(row.get("conv_loss", 0.0))
            out["conv_gain_at_peak"] = float(row.get("conv_gain", 0.0))
    return out


def summarise(tree: Path) -> dict:
    arms = {}
    for arm in ARMS + REPAIR_ARMS:
        if not (tree / "runs" / f"k3_{arm}" / "result.json").exists():
            continue
        result = load_arm(tree, arm)
        depth, t_peak, step_peak = peak_depth(result)
        history = result["history"]
        alive_ms = history[-1]["time_ms"] if history else 0.0
        err = (depth - EXP_DEPTH_D) / EXP_DEPTH_D
        budget = budget_window(result, 0.0, t_peak)
        arms[arm] = {
            "stop_reason": result.get("stop_reason"),
            "momconv": result.get("momconv", ""),
            "peak_depth_D": depth,
            "peak_time_ms": t_peak,
            "peak_step": step_peak,
            "alive_ms": alive_ms,
            "rel_err_vs_experiment": err,
            "abs_rel_err_vs_experiment": abs(err),
            "alive_past_experiment": alive_ms > SURVIVE_MS,
            "deeper_than_ctrl": None,
            "anchor_delta_depth": depth - ANCHOR_DEPTH_D,
            "anchor_delta_time_ms": t_peak - ANCHOR_TIME_MS,
            "budget": budget,
        }
    ctrl_depth = arms["ctrl"]["peak_depth_D"]
    for arm, entry in arms.items():
        entry["main_race"] = arm in ARMS
        entry["deeper_than_ctrl"] = entry["peak_depth_D"] > ctrl_depth
        entry["hit"] = bool(
            entry["abs_rel_err_vs_experiment"] < HIT_ERR and entry["alive_past_experiment"]
        )
        entry["falsified"] = bool(
            entry["abs_rel_err_vs_experiment"] > FALSIFY_ERR
            or not entry["alive_past_experiment"]
            or entry["peak_depth_D"] <= ctrl_depth
        )
    ctrl = arms["ctrl"]
    ctrl_anchor_ok = (
        abs(ctrl["anchor_delta_depth"]) < 1e-9 and abs(ctrl["anchor_delta_time_ms"]) < 1e-6
    )
    hits = [arm for arm in ARMS if arm != "ctrl" and arms[arm]["hit"]]
    falsified = [arm for arm in ARMS if arm != "ctrl" and arms[arm]["falsified"]]
    repair_hits = [arm for arm in REPAIR_ARMS if arm in arms and arms[arm]["hit"]]
    if hits:
        verdict = "HIT"
    elif falsified and len(falsified) == 2:
        verdict = "FALSIFIED"
    elif falsified:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "repair_hits": repair_hits,
        "experiment": {"depth_D": EXP_DEPTH_D, "time_ms": EXP_TIME_MS},
        "anchor": {"depth_D": ANCHOR_DEPTH_D, "time_ms": ANCHOR_TIME_MS},
        "gates": {
            "hit_abs_rel_err": HIT_ERR,
            "falsify_abs_rel_err": FALSIFY_ERR,
            "survive_ms": SURVIVE_MS,
        },
        "ctrl_anchor_bit_exact": ctrl_anchor_ok,
        "verdict": verdict,
        "arms": arms,
    }


def to_markdown(verdict: dict) -> str:
    lines = []
    lines.append("| 臂 | momconv | 峰深 (D) | 峰时刻 (ms) | 存活 (ms) | err vs 实验 | 判定 |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in ARMS + tuple(a for a in REPAIR_ARMS if a in verdict["arms"]):
        e = verdict["arms"][arm]
        lines.append(
            f"| {arm} | {e['momconv'] or 'off'} | {e['peak_depth_D']:.6f} | "
            f"{e['peak_time_ms']:.4f} | {e['alive_ms']:.4f} | "
            f"{e['rel_err_vs_experiment'] * 100:+.4f}% | "
            f"{('基线（锚）' if arm == 'ctrl' else ('HIT' if e['hit'] else ('证伪' if e['falsified'] else '—')))} |"
        )
    lines.append("")
    lines.append("| 臂 | conv_loss/E_scale | conv_gain/E_scale | 净转换损耗/E_scale | 峰时轴向动能 KE_axial |")
    lines.append("|---|---|---|---|---|")
    for arm in ARMS + tuple(a for a in REPAIR_ARMS if a in verdict["arms"]):
        b = verdict["arms"][arm]["budget"]
        gain = (
            "质量爆炸（单步 conv_gain 尖峰）"
            if b.get("mass_blowup")
            else f"{b['conv_gain_share'] * 100:+.4f}%"
        )
        net = (
            "质量爆炸"
            if b.get("mass_blowup")
            else f"{b['conv_net_share'] * 100:+.4f}%"
        )
        lines.append(
            f"| {arm} | {b['conv_loss_share'] * 100:+.4f}% | {gain} | "
            f"{net} | {b['ke_axial_at_peak']:.6e} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default=".")
    args = parser.parse_args()
    tree = Path(args.tree).resolve()
    verdict = summarise(tree)
    out_dir = tree / "runs" / "k3_momconv_verdict"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    md = to_markdown(verdict)
    (out_dir / "verdict.md").write_text(
        "# K3-MOMCONV verdict\n\n"
        f"verdict = **{verdict['verdict']}**\n\n"
        f"ctrl anchor bit-exact = {verdict['ctrl_anchor_bit_exact']}\n\n" + md + "\n"
    )
    print(f"verdict = {verdict['verdict']}  ctrl_anchor_bit_exact={verdict['ctrl_anchor_bit_exact']}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble the ELBM channel/DNS verification report from a raw run artefact.

Everything in the output is computed from the run fields and from the
SHA-256-verified Moser-Kim-Mansour (1999) reference profiles -- no number is
typed by hand:

* wall units (``y+``, ``U+``, ``Re_tau`` from the measured viscous sublayer),
* the campaign gates (a)-(d) through :func:`tensorlbm.elbm_channel.channel_gates`,
* figures (mean profile vs DNS, Reynolds stresses vs DNS, stability trace),
* a markdown results section and a machine-readable ``results.json``.

Usage::

    python tools/elbm_dns_report.py --raw artifacts/elbm_dns/raw/elbm_re180.npz \\
        --out artifacts/elbm_dns/re180 --label re180 --markdown docs/reports/elbm_dns_re180.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tensorlbm.validation import mkm1999  # noqa: E402


def _load_channel_report_module():
    spec = importlib.util.spec_from_file_location("elbm_channel_report", _HERE / "elbm_channel_report.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def build_markdown(res: dict, meta: dict, gate_verdicts: dict, label: str) -> str:
    gates = res["gates"]
    extra = res["extra"]
    cfg = meta.get("config", {})
    a, b, c, d = gates["a_log_slope"], gates["b_reynolds_stress_peak"], gates["c_friction_coefficient"], gates["d_long_time_stability"]
    rows = [
        ("网格 (nx x ny x nz)", f"{cfg.get('nx')} x {cfg.get('ny')} x {cfg.get('nz')}"),
        ("域 (Lx/h x Lz/h)", f"{cfg.get('nx', 0) / max(extra['half_height'], 1e-9):.2f} x {cfg.get('nz', 0) / max(extra['half_height'], 1e-9):.2f}"),
        ("Re_tau (目标)", _fmt(extra["re_tau_target"], 1)),
        ("Re_tau (由测量粘性底层斜率反推)", _fmt(extra.get("re_tau_from_measured_nu"), 1)),
        ("u_tau = sqrt(g h)", _fmt(extra["u_tau"], 6)),
        ("nu = (tau-1/2)/3", _fmt(extra["nu_nominal"], 8)),
        ("Delta y+ (第一层流体)", _fmt(extra["delta_plus"], 2)),
        ("时间步 / 预热 / 采样间隔",
         f"{d['steps']} / {meta.get('warmup')} / {meta.get('sample_every')}"),
        ("统计样本数", str(extra["samples"])),
        ("<alpha> / min / max", f"{_fmt(extra['alpha_mean'], 5)} / {_fmt(extra['alpha_min'], 4)} / {_fmt(extra['alpha_max'], 4)}"),
        ("min population", f"{extra['min_population']:.3e}"),
        ("max |u| (lattice)", _fmt(extra["max_abs_u"], 5)),
        ("max dH (原始, fp32)", f"{extra['entropy_drop_max_collision']:.3e}"),
        ("max dH 超出 fp32 噪声底 (H 定理残差)", f"{extra.get('entropy_excess_max_collision', float('nan')):.3e}"),
        ("吞吐 (Mcell/s)", _fmt(meta.get("throughput_Mcell_s", float('nan')), 2)),
        ("墙钟 (h)", _fmt((meta.get("wall_clock_s") or 0) / 3600.0, 2)),
    ]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)

    gate_rows = [
        ("(a) 对数区斜率", f"拟合斜率 {_fmt(a.get('slope'), 4)} (1/kappa=2.439), 相对误差 {100 * a.get('relative_error', float('nan')):.2f}%",
         "PASS" if gate_verdicts["a"] else "FAIL"),
        ("(b) 雷诺应力峰值位置", f"u_rms+ 峰值 y+={_fmt(b.get('y_plus_peak'), 1)}, DNS y+={_fmt(b.get('dns_y_plus'), 1)}, "
         f"偏差 {_fmt(b.get('error_vs_dns'), 2)} (判据 12±2: {_fmt(b.get('error_vs_campaign_target'), 2)})",
         "PASS" if gate_verdicts["b"] else "FAIL"),
        ("(c) 摩擦系数 C_f", f"C_f(run)={_fmt(c.get('cf_run'), 5)}, C_f(DNS)={_fmt(c.get('cf_dns'), 5)}, "
         f"相对误差 {100 * c.get('cf_rel_error', float('nan')):.2f}%", "PASS" if gate_verdicts["c"] else "FAIL"),
        ("(d) 长时间积分稳定性", f"{d.get('steps')} 步, min f={d.get('min_population'):.3e}, "
         f"H 残差 {d.get('entropy_excess_max_collision') if d.get('entropy_excess_max_collision') is not None else d.get('entropy_drop_max_collision'):.2e}",
         "PASS" if gate_verdicts["d"] else "FAIL"),
    ]
    gate_table = "\n".join(f"| {k} | {v} | {s} |" for k, v, s in gate_rows)

    u_b_err = c.get("u_bulk_plus_rel_error", float("nan"))
    dnsref = gates.get("dns_reference", {})

    return f"""# ELBM 槽道湍流 DNS 验证结果（{label}）

> 本文件的全部数字由 `tools/elbm_dns_report.py` 从原始算例文件
> `{res['extra']['run']}` 程序化提取（零手抄），DNS 参照为
> Moser–Kim–Mansour (1999) `Re_tau={mkm1999.load_profile(180 if extra['re_tau_target'] < 280 else 395).re_tau:.1f}`
> 谱方法数据（SHA-256 已登记于 `tensorlbm/data/mkm1999/provenance.json`）。

## 1. 算例参数

| 项目 | 值 |
|---|---|
{table}

## 2. 判据对照

| 判据 | 数值 | 判定 |
|---|---|---|
{gate_table}

上列 (c) 判据的等价形式是体积平均速度 `U_b+`：`C_f = 2/U_b+^2`。
本次 `U_b+` = {_fmt(c.get('u_bulk_plus_run'), 3)}（DNS {_fmt(c.get('u_bulk_plus_dns'), 3)}，
相对误差 {100 * u_b_err:.2f}%）。

## 3. 剖面一致性

| 量 | run | DNS | 相对误差 |
|---|---|---|---|
| U_b+ | {_fmt(c.get('u_bulk_plus_run'), 3)} | {_fmt(c.get('u_bulk_plus_dns'), 3)} | {100 * u_b_err:.2f}% |
| u_rms+ 峰值 | {_fmt(b.get('peak_value'), 3)} | {_fmt(dnsref.get('u_rms_peak_value'), 3)} | n/a（峰值位置见判据 b） |
| -<u'v'>+ 最大 | {_fmt(extra.get('max_minus_uv_plus'), 3)} | {_fmt(dnsref.get('minus_uv_peak_value'), 3)} | n/a |

> `-<u'v'>+` 在槽道中由应力平衡给出 `≈ 1 - y/h`，壁面处最大；它与
> `u_rms+` 的峰值位置是两个不同的量，判据 (b) 用的是后者（`u_rms+` 的
> 内层峰才在 y+ ≈ 12–15），两者都在报告中单列，避免把"应力峰值在壁面"
> 误读为判据失败。

## 4. 声明

* 判据 (e) pytest 见 `tests/test_elbm.py` + `tests/test_elbm_channel_dns.py`。
* 本报告的所有判定（含 FAIL）均为机器计算，未做任何人工修饰。
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    label = args.label or raw.stem
    out = Path(args.out)
    meta_path = raw.parent / f"{raw.stem}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cfg = meta.get("config", {})
    re_tau = float(meta.get("re_tau", cfg.get("re_tau", 180.0)))
    u_tau = float(meta.get("u_tau", cfg.get("u_tau", 0.01)))

    cr = _load_channel_report_module()
    res = cr.report(raw, re_tau=re_tau, u_tau=u_tau, out_dir=out)
    figs = cr.make_figures(res, out, label=label)

    # criterion (e): the pytest verdict is only "green" if the count is passed in
    gates = res["gates"]
    verdicts = {
        "a": bool(gates["a_log_slope"].get("pass")),
        "b": bool(gates["b_reynolds_stress_peak"].get("pass")),
        "c": bool(gates["c_friction_coefficient"].get("pass")),
        "d": bool(gates["d_long_time_stability"].get("pass")),
    }
    all_pass = all(verdicts.values())

    # honest headline: the DNS-relative error of the key quantities
    summary = {
        "label": label,
        "raw": str(raw),
        "grid": meta.get("grid"),
        "domain_over_h": [cfg.get("nx", 0) / max(res["wall_units"]["half_height"], 1e-9),
                          cfg.get("nz", 0) / max(res["wall_units"]["half_height"], 1e-9)],
        "re_tau_target": re_tau,
        "re_tau_from_measured_nu": res["extra"].get("re_tau_from_measured_nu"),
        "u_tau": u_tau,
        "nu": res["extra"]["nu_nominal"],
        "delta_plus": res["extra"]["delta_plus"],
        "steps": res["gates"]["d_long_time_stability"].get("steps", meta.get("steps")),
        "samples": res["extra"]["samples"],
        "alpha_mean": res["extra"]["alpha_mean"],
        "alpha_min": res["extra"]["alpha_min"],
        "min_population": res["extra"]["min_population"],
        "entropy_drop_max_collision": gates["d_long_time_stability"]["entropy_drop_max_collision"],
        "entropy_excess_max_collision": gates["d_long_time_stability"].get("entropy_excess_max_collision"),
        "throughput_Mcell_s": meta.get("throughput_Mcell_s"),
        "wall_clock_s": meta.get("wall_clock_s"),
        "gates": {
            "a_log_slope": {k: v for k, v in gates["a_log_slope"].items() if not isinstance(v, list)},
            "b_reynolds_stress_peak": gates["b_reynolds_stress_peak"],
            "c_friction_coefficient": gates["c_friction_coefficient"],
            "d_long_time_stability": gates["d_long_time_stability"],
        },
        "verdict": {**verdicts, "all_pass_except_pytest": all_pass},
        "dns_reference": gates["dns_reference"],
        "figures": [str(f) for f in figs],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=1, default=float))

    if args.markdown:
        md = build_markdown(res, meta, verdicts, label)
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(md)

    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
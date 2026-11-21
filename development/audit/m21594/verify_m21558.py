#!/usr/bin/env python3
"""m21594 — 交差审计（非重做）：m21558 XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM

只读取证 + 程序化复算，零手抄：
  E1 provenance      : 交付提交 / 快照文件 sha256（快照 = audit/m21594/snapshot，只读副本）
  E2 三件套          : run_artifacts / benchmark（tensor_benchmark_cases 裁判行）/ repro_cmd
  E3 数值复算        : 从交付 json 的 profile 数组复算 u_peak_err / eps_inf / eps_l2 并判 15% 门
  E4 D1 层序口径     : 解析解 nus=[nu_a,nu_b] 的层序 vs 数值 init 的实际层序（镜像口径偏差）
  E4 D2/D3/D4        : 读 audit/m21594/sim_diagnostics.json（sim_diag.py，同一个交付求解器；
                       稳态检验 / 单组分对照 / 质量与壁格账）
  E5 报告结构        : docs 六段核验
  E6 verdict         : 生成 tensor_maturity_verdicts INSERT（verdict_m21594.sql）+ tables.md + 图

求解器与解析解均按文件路径 hermetic 载入（不经 site-packages 的 editable 安装，
避免 import 到别的工作树），见 audit/m21594/snapshot。

用法: python3 audit/m21594/verify_m21558.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ORIGIN = Path("/srv/tensor/agent/work/m21558/TensorLBM")   # 被核单原始工作树（只读，记录出处）
SNAP = HERE / "snapshot"                                # 本次审计的只读快照
ART = SNAP / "artifacts" / "xval_two_component_sc_layered_poiseuille_edm"
OWNER_TOL = 0.15
ANCHOR = "XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM"
DELIVERED_COMMIT = "1b685d33c1016089ccaaba225c39215e692c2360"
MARGIN = 3
NY, NX, H, G_FORCE = 32, 4, 30.0, 1e-4
NU_A = (1.0 / 3.0) * (0.8 - 0.5)
NU_B = (1.0 / 3.0) * (1.2 - 0.5)

EV = {}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ana = load("_audit_analytic", SNAP / "src/tensorlbm/verification/layered_poiseuille/analytic.py")
tcs = load("_audit_two_component_sc",
           SNAP / "src/tensorlbm/verification/layered_poiseuille/two_component_sc.py")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def psql(sql: str) -> str:
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-F", "|", "-c", sql],
                       capture_output=True, text=True)
    return r.stdout.strip()


# --------------------------------------------------------------------------- #
# E1 provenance
# --------------------------------------------------------------------------- #
FILES = [
    "src/tensorlbm/verification/layered_poiseuille/two_component_sc.py",
    "src/tensorlbm/verification/layered_poiseuille/run_two_component_sc_validation.py",
    "src/tensorlbm/verification/layered_poiseuille/analytic.py",
    "tests/test_two_component_sc_layered_poiseuille.py",
    "run_quick_validation.py",
    "docs/XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM.md",
    "docs/XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM-STATUS.md",
    "artifacts/xval_two_component_sc_layered_poiseuille_edm/quick_validation_results.json",
]
EV["provenance"] = {
    "delivered_commit": DELIVERED_COMMIT,
    "merge_commit_repo_master": "481ac88da (Merge #2475 (m21558) via merge-master)",
    "origin_tree": str(ORIGIN),
    "snapshot_tree": str(SNAP.relative_to(REPO)),
    "files": {f: {"exists": (SNAP / f).exists(),
                  "sha256": sha256(SNAP / f) if (SNAP / f).exists() else None,
                  "sha256_origin": sha256(ORIGIN / f) if (ORIGIN / f).exists() else None,
                  "bytes": (SNAP / f).stat().st_size if (SNAP / f).exists() else 0}
              for f in FILES},
    "artifact_dir_listing": sorted(p.name for p in ART.iterdir()) if ART.exists() else [],
    "png_present": (ART / "quick_validation.png").exists(),
    "full_campaign_artifacts_present": sorted(
        p.name for p in ART.iterdir()
        if p.name not in ("quick_validation.png", "quick_validation_results.json")) if ART.exists() else [],
}

# --------------------------------------------------------------------------- #
# E2 benchmark 裁判行 + 报告台账
# --------------------------------------------------------------------------- #
EV["benchmark"] = {
    "anchor_slug": ANCHOR,
    "rows_total": int(psql(f"SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'") or 0),
    "rows": psql(f"SELECT case_name,metric,tolerance,status FROM tensor_benchmark_cases "
                 f"WHERE anchor_slug='{ANCHOR}'"),
    "poiseuille_rows_in_judge_table": psql(
        "SELECT anchor_slug,coalesce(status,'') FROM tensor_benchmark_cases "
        "WHERE anchor_slug ILIKE '%POISEUILLE%' ORDER BY anchor_slug").splitlines(),
}
EV["reports"] = {"m21558_rows": psql(
    "SELECT id,mission_id,title FROM tensor_reports WHERE mission_id=21558"),
    "anchor_title_rows": psql(
        "SELECT id,mission_id,title FROM tensor_reports WHERE title ILIKE '%TWO-COMPONENT-SC-LAYERED%'"
    ).splitlines()}
EV["prior_verdicts"] = psql(
    "SELECT id,repo,scope,level,verdict_by FROM tensor_maturity_verdicts "
    f"WHERE scope ILIKE '%TWO-COMPONENT-SC-LAYERED%'").splitlines()

# --------------------------------------------------------------------------- #
# E2 integration callers
# --------------------------------------------------------------------------- #
grep = subprocess.run(["grep", "-rln", "--include=*.py", "two_component_sc", str(ORIGIN)],
                      capture_output=True, text=True).stdout.splitlines()
callers = [p.replace(str(ORIGIN) + "/", "") for p in grep]
EV["integration_callers"] = {
    "all_hits": callers,
    "non_self_hits": [c for c in callers
                      if "verification/layered_poiseuille/" not in c and "tests/" not in c
                      and not c.startswith("run_")],
    "note": "命中仅为 verification 子包自身 + 测试 + 独立 run 脚本；src/tensorlbm 主求解器/平台入口无 caller",
}

# --------------------------------------------------------------------------- #
# E3 交付 json 数值复算
# --------------------------------------------------------------------------- #
art = json.loads((ART / "quick_validation_results.json").read_text())
prof = art["profile"]
y_num = np.asarray(prof["y"], float)
ux_num = np.asarray(prof["ux_total"], float)
ux_ana = np.asarray(prof["ux_analytic"], float)
rho_a = np.asarray(prof["rho_a"], float)
rho_b = np.asarray(prof["rho_b"], float)
ref_layered = ana.solve_layered(height=H, interfaces=[15.0], nus=[NU_A, NU_B], g=G_FORCE)
mirror = ana.solve_layered(height=H, interfaces=[15.0], nus=[NU_B, NU_A], g=G_FORCE)


def _errs(m: int):
    sc = float(np.max(np.abs(ux_ana[m:-m])))
    return (float(np.max(np.abs(ux_num[m:-m] - ux_ana[m:-m])) / sc),
            float(np.linalg.norm(ux_num[m:-m] - ux_ana[m:-m]) / np.linalg.norm(ux_ana[m:-m])))


# 交付入口 run_quick_validation.py:91 用 margin=2；campaign 脚本用 margin=3
eps_inf, eps_l2 = _errs(2)
eps_inf_m3, eps_l2_m3 = _errs(MARGIN)
u_peak_num = float(np.max(ux_num))
u_peak_ana = float(ref_layered.u_peak())     # 独立解析峰，不取自交付 json
u_peak_err = abs(u_peak_num - u_peak_ana) / u_peak_ana
EV["recompute"] = {
    "config_delivered": art["config"],
    "stored": art["metrics"],
    "recomputed": {"u_peak_num": u_peak_num, "u_peak_ana": u_peak_ana, "u_peak_err": u_peak_err,
                   "eps_inf": eps_inf, "eps_l2": eps_l2,
                   "analytic_u_peak_independent": u_peak_ana,
                   "metric_definition": "eps 定义在交付内不一致：run_quick_validation.py:96 margin=2，"
                                        "run_two_component_sc_validation.py:125 margin=3；"
                                        f"m=2 -> {eps_inf*100:.2f}/{eps_l2*100:.2f}%，"
                                        f"m=3 -> {eps_inf_m3*100:.2f}/{eps_l2_m3*100:.2f}%"
                                        "（无预注册度量定义）"},
    "delta_vs_stored": {k: float(EV_store - art["metrics"][k]) for k, EV_store in
                        (("u_peak_err", u_peak_err), ("eps_inf", eps_inf), ("eps_l2", eps_l2))},
    "pass_15pct": {"u_peak_err": bool(u_peak_err < OWNER_TOL), "eps_inf": bool(eps_inf < OWNER_TOL),
                   "eps_l2": bool(eps_l2 < OWNER_TOL)},
    "all_within_15pct": bool(max(u_peak_err, eps_inf, eps_l2) < OWNER_TOL),
    "wall_cell_density": {"rho_a[0](y=0.5)": float(rho_a[0]), "rho_b[0](y=0.5)": float(rho_b[0]),
                          "rho_a[-1](y=29.5)": float(rho_a[-1]), "rho_b[-1](y=29.5)": float(rho_b[-1]),
                          "nominal": 2.1,
                          "peak_over_nominal": float(max(rho_a[0], rho_b[-1]) / 2.1)},
    "profile_shape": {"interior_mean": float(np.mean(ux_num[2:-2])),
                      "interior_std": float(np.std(ux_num[2:-2])),
                      "g_t_at_2000": float(art["config"]["g_x"] * art["config"]["steps"]),
                      "u_interior_over_g_t": float(np.mean(ux_num[2:-2]) / (1e-4 * 2000))},
}

# --------------------------------------------------------------------------- #
# E4-D1 层序口径
# --------------------------------------------------------------------------- #
s0 = tcs.TwoComponentSCSolver(tcs.TwoComponentSCConfig(ny=NY, nx=NX, tau_a=0.8, tau_b=1.2, G=2.0,
                                                       g_x=G_FORCE, g_y=0.0, steps=1))
ra0, _, _ = tcs.macroscopic(s0.f_a)
rb0, _, _ = tcs.macroscopic(s0.f_b)
u_ref = ref_layered.u(y_num)
u_mir = mirror.u(y_num)
EV["layer_order"] = {
    "analytic_side": "solve_layered(nus=[nu_a=0.1, nu_b=0.2333]) -> layer0 (ν=0.1) 贴 y=0 壁",
    "numeric_side": {"rho_a_y0.5": float(ra0[1].mean()), "rho_b_y0.5": float(rb0[1].mean()),
                     "rho_a_y29.5": float(ra0[-2].mean()), "rho_b_y29.5": float(rb0[-2].mean()),
                     "A_index_range": "16..30 => A(ν=0.1) 位于 y>15 一侧",
                     "code_comment": "two_component_sc.py:131 'Component A in bottom layer'（实现 132 行 rho[16:-1]=2 即 y>15 一侧） 与实现相反"},
    "verdict": "FLIPPED（数值低黏层在上、解析低黏层在下）",
    "mirror_reference_relative_dev": float(np.max(np.abs(u_mir - u_ref)) / np.max(np.abs(u_ref))),
    "u_peak_ref": float(ref_layered.u_peak()), "u_peak_mirror": float(mirror.u_peak()),
}

# --------------------------------------------------------------------------- #
# E4-D2a 解析起动瞬态参考：u(y,t) = g/(2ν) y(H-y) - Σ_{n odd} 4gH²/(ν n³π³) sin(nπy/H) e^{-n²π²νt/H²}
# 2000 步时一个"黏度正确"的求解器应当已经出现明显的壁面制动层（√(νt)=14 lu 与 H/2=15 同量级），
# 用来判断交付 profile（几乎无制动）是否连合理瞬态都不是。
# --------------------------------------------------------------------------- #
def start_up(y, t, nu, gg=G_FORCE, hh=H, n_max=401):
    u = gg / (2.0 * nu) * y * (hh - y)
    for n in range(1, n_max, 2):
        u -= 4.0 * gg * hh**2 / (nu * n**3 * np.pi**3) * np.sin(n * np.pi * y / hh) \
             * np.exp(-n**2 * np.pi**2 * nu * t / hh**2)
    return u


t_del = art["config"]["steps"]
u_t_nua = start_up(y_num, t_del, NU_A)
u_t_nub = start_up(y_num, t_del, NU_B)
EV["transient_reference"] = {
    "formula": "start-up series for single-viscosity plane Poiseuille (nu=0.1 / 0.2333), t=2000, H=30, g=1e-4",
    "boundary_layer_sqrt_nu_t": float(np.sqrt(NU_A * t_del)),
    "half_height": H / 2,
    "y": y_num.tolist(), "u_nu0.1": u_t_nua.tolist(), "u_nu0.2333": u_t_nub.tolist(),
    "delivered_vs_transient_nu0.1_eps_inf": float(
        np.max(np.abs(ux_num[2:-2] - u_t_nua[2:-2])) / np.max(np.abs(u_t_nua[2:-2]))),
    "delivered_braked_depth_lu": float(np.sum(ux_num < 0.9 * np.max(ux_num[2:-2])) * 1.0),
    "expected_braked_depth_lu_nu0.1": float(np.sum(u_t_nua < 0.9 * np.max(u_t_nua))),
    "note": "交付 profile 在 2000 步几乎未出现壁面剪切层（仅贴壁 1 格），"
            "而 nu=0.1 的正确起动解在同期已有 ~14 lu 量级的制动层；"
            "两支都不是稳态，故 177.78% 的成因不是「有效黏度未标定」，而更接近「动量扩散未发生 / 强迫尺度错」",
}

# --------------------------------------------------------------------------- #
# E4-D2/D3/D4 读诊断腿
# --------------------------------------------------------------------------- #
SIM = HERE / "sim_diagnostics.json"
if SIM.exists():
    sim = json.loads(SIM.read_text())
    EV["D2_steady_state"] = {k: {kk: vv for kk, vv in v.items() if kk != "profile"}
                             for k, v in sim["D2_steady_state"].items()}
    EV["D3_single_component_control"] = {kk: vv for kk, vv in sim["D3_single_component_control"].items()
                                        if kk != "profile"}
    EV["D4_mass_ledger"] = {k: {"mass_first": v["trace"][0], "mass_last": v["trace"][-1],
                                "rho_wall_max": max(t["rho_wall_max"] for t in v["trace"])}
                            for k, v in list(sim["D2_steady_state"].items())
                            + [("single_component_G0", sim["D3_single_component_control"])]}
    EV["sim_pass_15pct"] = sim["pass_15pct"]
    EV["sim_meta"] = sim["meta"]
else:
    EV["D2_steady_state"] = {"state": "not_run（sim_diagnostics.json 缺席）"}
    EV["D3_single_component_control"] = {"state": "not_run"}
    EV["D4_mass_ledger"] = {"state": "not_run"}
    EV["sim_pass_15pct"] = {"state": "not_run"}
    EV["sim_meta"] = {"state": "not_run"}

# --------------------------------------------------------------------------- #
# E5 报告结构
# --------------------------------------------------------------------------- #
main_doc = (SNAP / "docs" / "XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM.md").read_text()
status_doc = (SNAP / "docs" / "XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM-STATUS.md").read_text()
EV["report_structure"] = {
    "main_doc_headings": [ln for ln in main_doc.splitlines() if ln.startswith("#")],
    "status_doc_headings": [ln for ln in status_doc.splitlines() if ln.startswith("#")],
    "main_doc_has_15pct_gate": ("<15%" in main_doc or "15 %" in main_doc),
    "main_doc_has_acceptance": "Acceptance" in main_doc,
    "status_doc_declares_fail": "FAILS acceptance criteria" in status_doc,
    "status_doc_says_do_not_mark_verified": "Do not mark as verified" in status_doc,
    "main_doc_length_lines": len(main_doc.splitlines()),
    "status_doc_length_lines": len(status_doc.splitlines()),
}

# --------------------------------------------------------------------------- #
# 判级
# --------------------------------------------------------------------------- #
def _check(d, key, field, pred):
    """三态检查：未跑(not_run) -> None；跑过 -> bool。"""
    if not isinstance(d, dict) or "state" in d:
        return None
    row = d.get(key) if key else d
    if not isinstance(row, dict) or field not in row:
        return None
    return bool(pred(row[field]))


tests_ok = True   # 由 pytest 单独记录（12 passed，见 checks 字段/报告）
L = 2
checks = {
    "impl_real_code": (SNAP / FILES[0]).exists(),
    "entry_runs_reproduced": True,
    "unit_tests_12_passed": tests_ok,
    "unit_tests_assert_accuracy": False,
    "benchmark_row_exists": EV["benchmark"]["rows_total"] > 0,
    "benchmark_within_15pct": EV["recompute"]["all_within_15pct"],
    "steady_state_reached": _check(EV["D2_steady_state"], "20000", "u_max_over_g_t", lambda v: v < 1.15),
    "single_component_control_pass": _check(EV["D3_single_component_control"], None, "eps_inf",
                                            lambda v: v < OWNER_TOL),
    "integrated_in_main_solver": len(EV["integration_callers"]["non_self_hits"]) > 0,
}
EV["level_checks"] = checks
EV["level"] = L
EV["level_reason"] = (
    "L2 = 独立功能实现（真代码：输入→计算→输出；交付 profile 可由 json 逐点复算；12 单测通过）。"
    "不到 L3：模块仅存在于 verification 子包，主求解器/平台入口无 caller（integration_callers 空）。"
    "不到 L4：tensor_benchmark_cases 该 anchor 0 行（无裁判行可判），且唯一可独立复算的对照"
    f"（analytic.solve_layered）实测 u_peak_err={u_peak_err*100:.2f}%、eps_inf={eps_inf*100:.2f}%、"
    f"eps_l2={eps_l2*100:.2f}%，全部 > 15% owner 门；无外部锚（OSS/文献 ref）登账。"
    "诚实性成立：STATUS.md 自认 FAIL 并写明「Do not mark as verified」，故判 honest-FAIL 而非虚标。"
)
EV["verdict"] = "honest-FAIL"

# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
ax[0].plot(ux_num, y_num, "o-", ms=3.2, label="numeric (m21558 delivered, 2000 steps)")
ax[0].plot(u_ref, y_num, "k--", lw=1.5, label="layered analytic, $\\nu_A$ at y=0 (as used)")
ax[0].plot(u_mir, y_num, "r:", lw=1.5, label="layered analytic, $\\nu_A$ at y=H (sim layering)")
ax[0].plot(u_t_nua, y_num, "C2-.", lw=1.2, label="start-up series $\\nu$=0.1, t=2000")
ax[0].plot(u_t_nub, y_num, "C4:", lw=1.2, label="start-up series $\\nu$=0.2333, t=2000")
ax[0].axvline(1e-4 * 2000, color="0.6", lw=1, ls="-.", label="$g\\,t=0.20$ (inviscid plateau)")
ax[0].set_xlabel("$u_x$"), ax[0].set_ylabel("y"), ax[0].legend(fontsize=6.6, loc="lower right")
ax[0].set_title(f"(a) profile: $\\epsilon_\\infty$={eps_inf*100:.1f}%, $u_{{pk}}$ {u_peak_num:.3f} vs {u_peak_ana:.3f}")

if SIM.exists():
    for i, (st, blk) in enumerate(sorted(sim["D2_steady_state"].items())):
        tr = np.array([[t["step"], t["u_max"]] for t in blk["trace"]])
        ax[1].plot(tr[:, 0], tr[:, 1], f"C{i}-", label=f"two-component G=2, {st} steps")
    tr = np.array([[t["step"], t["u_max"]] for t in sim["D3_single_component_control"]["trace"]])
    ax[1].plot(tr[:, 0], tr[:, 1], "C3-", label="single-component G=0 control (20k)")
    _tm = max([t["step"] for t in sim["D3_single_component_control"]["trace"]]
              + [t["step"] for k in sim["D2_steady_state"] for t in sim["D2_steady_state"][k]["trace"]])
    ax[1].plot([0, _tm], [0, 1e-4 * _tm], "k--", lw=1, label="$u = g\\,t$ (no viscosity)")
    ax[1].set_xlabel("step"), ax[1].set_ylabel("$|u_x|_{\\max}$"), ax[1].legend(fontsize=7)
    ax[1].set_title("(b) $u_{max}(t)$ vs free-acceleration line")
else:
    ax[1].text(0.5, 0.5, "sim_diag: not_run", ha="center")

ax[2].semilogy(y_num, np.maximum(rho_a, 1e-3), "o-", ms=3, label="$\\rho_A$ (delivered)")
ax[2].semilogy(y_num, np.maximum(rho_b, 1e-3), "s-", ms=3, label="$\\rho_B$ (delivered)")
ax[2].axhline(2.0, color="k", ls=":", lw=1, label="nominal 2.0")
ax[2].set_xlabel("y"), ax[2].set_ylabel("$\\rho$"), ax[2].legend(fontsize=7)
ax[2].set_title(f"(c) wall-cell density: {rho_a[0]:.0f} at y=0.5 (nominal 2.0) -> x{max(rho_a[0], rho_b[-1])/2.1:.0f}")
fig.tight_layout()
FIG = HERE / "fig_m21558_audit.png"
fig.savefig(FIG, dpi=130)

# --------------------------------------------------------------------------- #
# 表 + verdict
# --------------------------------------------------------------------------- #
d2 = EV["D2_steady_state"]
d3 = EV["D3_single_component_control"]
rows = [
    ("① run_artifacts 存在", "quick_validation_results.json + png 在交付提交内（sha256 见 evidence.json）", "PASS"),
    ("①b 全量 campaign 工件", "run_two_component_sc_validation.py 声明 4 例（20k/30k 步）但 artifacts/ 只有 quick_validation_*", "MISSING"),
    ("② benchmark 裁判行", f"tensor_benchmark_cases anchor_slug='{ANCHOR}' -> {EV['benchmark']['rows_total']} 行", "MISSING"),
    ("②b 数值 vs 解析（15% 门）",
     f"u_peak_err={u_peak_err*100:.2f}% / eps_inf={eps_inf*100:.2f}% / eps_l2={eps_l2*100:.2f}%", "FAIL"),
    ("③ repro_cmd 可执行", "python3 run_quick_validation.py 复现 177.78%/253.30%/264.77%", "PASS(有缺陷)"),
    ("③b repro 可移植性", "脚本 sys.path 硬编码 /srv/tensor/agent/work/m21558/TensorLBM/src；去路径后 import 落到别的工作树（m21384 editable 安装）", "DEFECT"),
    ("③c 比特可复现", "init 用 torch.randn_like 无 seed -> eps_l2 交付 264.42% vs 复跑 264.77%", "DEFECT"),
    ("④ 单测", "12 passed，但无一条断言物理精度（test_velocity_develops 仅断言 ux_max 增长）", "L1 only"),
    ("⑤ 积分 caller", f"{EV['integration_callers']['non_self_hits'] or '无'}", "MISSING"),
    ("⑥ 层序口径", f"解析 A 贴 y=0，数值 A 在 y>15 -> 镜像角 {EV['layer_order']['mirror_reference_relative_dev']*100:.2f}%", "DEFECT"),
    ("⑦ 稳态（报告根因假设）", f"2000/20000 步 u_max/g·t = {d2.get('2000',{}).get('u_max_over_g_t','n/a')}/{d2.get('20000',{}).get('u_max_over_g_t','n/a')}（=1 表示仍在自由加速）", "DEFECT"),
    ("⑧ 单组分对照", f"G=0 单组分 20k 步 eps_inf={d3.get('eps_inf','n/a')}", "DIAG"),
    ("⑨ 壁格密度", f"y=0.5 处 rho_A={rho_a[0]:.0f}（名义 2.0）×{max(rho_a[0], rho_b[-1])/2.1:.0f}", "DEFECT"),
]
lines = ["# m21594 交差审计 · XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM (m21558)", "",
         f"被判交付提交 `{DELIVERED_COMMIT}`（master 合并 `481ac88da` Merge #2475）。级别 **L{L}**，判定 **{EV['verdict']}**。", "",
         "| 检查项 | 证据 | 结论 |", "|---|---|---|"]
for a, b, c in rows:
    lines += [f"| {a} | {b} | {c} |"]
lines += ["", "## 交付数值（程序化复算，源自 quick_validation_results.json 的 profile 数组）", "",
          "| 量 | 交付存储 | 复算 | 判 15% 门 |", "|---|---|---|---|",
          f"| u_peak_err | {art['metrics']['u_peak_err']*100:.2f}% | {u_peak_err*100:.2f}% | "
          f"{'PASS' if u_peak_err < OWNER_TOL else 'FAIL'} |",
          f"| eps_inf | {art['metrics']['eps_inf']*100:.2f}% | {eps_inf*100:.2f}% | "
          f"{'PASS' if eps_inf < OWNER_TOL else 'FAIL'} |",
          f"| eps_l2 | {art['metrics']['eps_l2']*100:.2f}% | {eps_l2*100:.2f}% | "
          f"{'PASS' if eps_l2 < OWNER_TOL else 'FAIL'} |",
          f"| u_peak (num / ana) | {u_peak_num:.6f} / {u_peak_ana:.6f} | 独立 solve_layered: "
          f"{float(ref_layered.u_peak()):.6f} | — |", "",
          "## 图", f"![audit]({FIG.name})", ""]
(HERE / "tables.md").write_text("\n".join(lines))

EV["evidence_keys"] = {
    "impl_files": "snapshot/src/tensorlbm/verification/layered_poiseuille/two_component_sc.py:1-331 "
                  f"(sha256 {sha256(SNAP / FILES[0])[:16]})",
    "entry": "snapshot/run_quick_validation.py:14-80（入口 main；:11 sys.path 硬编码 m21558 绝对路径）",
    "run_artifacts": "snapshot/artifacts/xval_two_component_sc_layered_poiseuille_edm/quick_validation_results.json"
                     f" (sha256 {sha256(ART / 'quick_validation_results.json')[:16]}) + quick_validation.png；"
                     "4-case campaign 工件缺失",
    "tests": "snapshot/tests/test_two_component_sc_layered_poiseuille.py:1-192 — 12 passed（无精度断言）",
    "integration_callers": "无（grep two_component_sc 仅 verification 子包/测试/独立 run 脚本）",
    "benchmark": f"tensor_benchmark_cases anchor_slug='{ANCHOR}' -> 0 行（无裁判行）；"
                 f"独立复算 u_peak_err={u_peak_err*100:.2f}% / eps_inf={eps_inf*100:.2f}% / "
                 f"eps_l2={eps_l2*100:.2f}% > 15%",
    "repro_cmd": "cd TensorLBM && python3 run_quick_validation.py（复算 177.78%/253.30%/264.77%；"
                 "非比特级，init 无 seed）",
    "external_anchor": "无（无 OSS/文献数值锚登账；docs References 仅方法文献）",
}
EV["evidence_keys"]["anchor"] = ANCHOR
EV["defects"] = {
    "D1_layer_order_flip": EV["layer_order"],
    "D2_never_steady": {k: {"u_max_over_g_t": v.get("u_max_over_g_t"), "eps_inf": v.get("eps_inf")}
                        for k, v in d2.items() if isinstance(v, dict)},
    "D3_single_component": {k: d3.get(k) for k in ("eps_inf", "u_peak_err", "u_max_over_g_t")},
    "D4_wall_density": EV["recompute"]["wall_cell_density"],
    "D5_repro_portability": {
        "hardcoded_path": "run_quick_validation.py:10 sys.path.insert(0, '/srv/tensor/agent/work/m21558/TensorLBM/src')",
        "probe_cmd": "cp snapshot/run_quick_validation.py /tmp/m21594_port/ ; "
                     "sed -i 's#/srv/tensor/agent/work/m21558/TensorLBM/src#/nonexistent#' .../run_quick_validation.py ; "
                     "cd /tmp/m21594_port && python3 -c 'import tensorlbm.verification.layered_poiseuille.two_component_sc'",
        "probe_result": "ModuleNotFoundError: No module named "
                        "'tensorlbm.verification.layered_poiseuille.two_component_sc'"
                        "（tensorlbm 解析到 /srv/tensor/agent/work/m21384/TensorLBM/src —— pip editable 装到别的工作树）",
        "conclusion": "repro 只在 m21558 工作树绝对路径存在的机器上可跑，非 hermetic",
    },
    "D7_test_scope": {
        "pytest": "python3 -m pytest tests/test_two_component_sc_layered_poiseuille.py -q -> 12 passed (4.64s)",
        "asserts": "equilibrium 和/动量守恒、psi 单调/有界、step_count 递增、ux_max 增大、"
                   "三个 forcing 方案不崩、analytic 模块自身可算/界面连续；无一条把求解器输出与解析解比对并断言 15%",
    },
    "D6_no_seed": "two_component_sc.py:138 torch.randn_like 无 manual_seed",
}
(HERE / "evidence.json").write_text(json.dumps(EV, indent=2, ensure_ascii=False))

ev_json = json.dumps(EV["evidence_keys"], ensure_ascii=False).replace("'", "''")
sql = f"""INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm21558 {ANCHOR}', 21594, {L}, '{ev_json}'::jsonb, 'board-audit');
"""
(REPO / "verdict_m21594.sql").write_text(sql)
(HERE / "verdict.sql").write_text(sql)

print(json.dumps({"level": L, "verdict": EV["verdict"], "benchmark_rows": EV["benchmark"]["rows_total"],
                  "u_peak_err": u_peak_err, "eps_inf": eps_inf, "eps_l2": eps_l2,
                  "checks": checks, "fig": str(FIG)}, indent=2, ensure_ascii=False))
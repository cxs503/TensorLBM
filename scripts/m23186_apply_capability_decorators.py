"""Insert ``@capability`` decorations into the modules that own the code.

Run once from the repository root:  python scripts/m23186_apply_capability_decorators.py
Idempotent: a target that already carries the decorator is skipped.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tensorlbm"

IMPORT_LINE = "from .registry import capability"

# (relative file, target name, decorator kwargs literal)
TABLE = [
    # ---- D3Q19 collision family: measured against the analytic Poiseuille profile
    ("solver3d.py", "collide_bgk3d", dict(
        id="lbm.collision.bgk.d3q19", kind="collision", maturity="mature",
        wired=True, tested=True, verified=True,
        test_path="tests/test_lbm_verification.py",
        evidence={"grade": "ANALYTIC",
                  "reference": "Poiseuille 解析剖面（Guo 体力, ny=12, tau=1.0, u_max=0.05）",
                  "max_error_pct": 0.338,
                  "measured_by": "scripts/m23186_collision_family_verification.py",
                  "note": "稳态剖面相对误差 0.338%（BGK 基准）"},
        endpoint="api:tensorlbm.solver3d.collide_bgk3d")),
    ("solver3d.py", "collide_trt3d", dict(
        id="lbm.collision.trt.d3q19", kind="collision", maturity="mature",
        wired=True, tested=True, verified=True,
        test_path="tests/test_trt.py",
        evidence={"grade": "ANALYTIC",
                  "reference": "同 BGK 的 Poiseuille 解析剖面（TRT, magic=3/16 等效 tau）",
                  "max_error_pct": 0.0012,
                  "measured_by": "scripts/m23186_collision_family_verification.py",
                  "note": "稳态剖面相对误差 0.0012%，比 BGK 低两个量级（半路反弹壁面误差被 TRT 抵消）"},
        endpoint="api:tensorlbm.solver3d.collide_trt3d")),
    ("solver3d.py", "collide_rlbm3d", dict(
        id="lbm.collision.rlbm.d3q19", kind="collision", maturity="mature",
        wired=True, tested=True, verified=True,
        test_path="tests/test_rlbm.py",
        evidence={"grade": "ANALYTIC",
                  "reference": "同 BGK 的 Poiseuille 解析剖面（正则化）",
                  "max_error_pct": 0.338,
                  "measured_by": "scripts/m23186_collision_family_verification.py",
                  "note": "稳态剖面误差与 BGK 重合（0.338%），正则化只改高阶非平衡项"},
        endpoint="api:tensorlbm.solver3d.collide_rlbm3d")),
    ("solver3d.py", "collide_mrt3d", dict(
        id="lbm.collision.mrt.d3q19", kind="collision", maturity="mature",
        wired=True, tested=True, verified=True,
        test_path="tests/test_lbm_verification.py",
        evidence={"grade": "ANALYTIC",
                  "reference": "同 BGK 的 Poiseuille 解析剖面（D2Q9/D3Q19 显式 MRT 矩阵）",
                  "max_error_pct": 0.343,
                  "measured_by": "scripts/m23186_collision_family_verification.py",
                  "note": "稳态剖面 0.343%；本算例 s_e/s_eps/s_q 系数不激活，故与 BGK 同量级——"
                          "非别名由 scripts/m23186_collision_family_verification.py 的 "
                          "s_e 灵敏度探针单独证明（见该脚本 anti_mock 段）"},
        endpoint="api:tensorlbm.solver3d.collide_mrt3d")),
    # ---- D3Q19/D3Q27 advanced families: mechanics-tested, no error band recorded
    ("d3q27.py", "collide_bgk27", dict(
        id="lbm.collision.bgk.d3q27", kind="collision", maturity="developing",
        wired=True, tested=True, verified=False,
        test_path="tests/test_d3q27.py",
        evidence={"grade": "MECHANICS",
                  "note": "D3Q27 玻尔兹曼 BGK；test_d3q27 断言平衡态/守恒/一阶 Chapman-Enskog，"
                          "本仓无该格子的解析剖面误差带"})),
    ("d3q27.py", "collide_trt27", dict(
        id="lbm.collision.trt.d3q27", kind="collision", maturity="developing",
        wired=True, tested=True, verified=False,
        test_path="tests/test_d3q27_trt_rlbm.py",
        evidence={"grade": "MECHANICS", "note": "TRT magics 与 BGK 极限一致性测试；无解析误差带"})),
    ("d3q27.py", "collide_rlbm27", dict(
        id="lbm.collision.rlbm.d3q27", kind="collision", maturity="developing",
        wired=True, tested=True, verified=False,
        test_path="tests/test_d3q27_trt_rlbm.py",
        evidence={"grade": "MECHANICS", "note": "正则化 D3Q27；与 BGK 极限一致性测试；无解析误差带"})),
    ("d3q27.py", "collide_mrt27", dict(
        id="lbm.collision.mrt.d3q27", kind="collision", maturity="developing",
        wired=True, tested=True, verified=False,
        test_path="tests/test_d3q19_d3q27_mrt_consistency.py",
        evidence={"grade": "MECHANICS",
                  "note": "D3Q27 MRT 与 D3Q19 MRT 的粘性/极限一致性测试；无解析误差带"})),
    # ---- D2Q9 collision family
    ("solver.py", "collide_bgk", dict(
        id="lbm.collision.bgk.d2q9", kind="collision", maturity="developing",
        wired=True, tested=True, verified=False,
        test_path="tests/test_lbm_verification.py",
        evidence={"grade": "MECHANICS",
                  "note": "D2Q9 主力碰撞算子；Poiseuille 解析带仅在 D3Q19 侧登记，"
                          "D2Q9 侧仅有机理/收敛测试"})),
    ("solver.py", "collide_mrt", dict(
        id="lbm.collision.mrt.d2q9", kind="collision", maturity="developing",
        wired=True, tested=True, verified=False,
        test_path="tests/test_autograd.py",
        evidence={"grade": "MECHANICS",
                  "note": "D2Q9 显式 MRT 矩阵 + tau_field 吸收层；机理/可微测试，无解析误差带"})),
    # ---- entropic KBC / cascaded: real kernels, main-path wiring honest
    ("entropic_kbc.py", "collide_kbc_d3q19", dict(
        id="lbm.collision.kbc.d3q19", kind="collision", maturity="developing",
        wired=False, tested=True, verified=False,
        test_path="tests/test_entropic_kbc.py",
        evidence={"grade": "MECHANICS",
                  "note": "真 KBC（熵条件二分 + KBC 分解，非调用方给混合系数）；"
                          "主求解路径未接线：仅契约/审计模块引用，无 stepper dispatch"})),
    ("entropic_kbc.py", "collide_kbc_d3q27", dict(
        id="lbm.collision.kbc.d3q27", kind="collision", maturity="developing",
        wired=False, tested=True, verified=False,
        test_path="tests/test_entropic_kbc_viscosity.py",
        evidence={"grade": "MECHANICS",
                  "note": "D3Q27 KBC + 粘性标定测试；主路径未接线（同上）"})),
    ("cascaded_collision.py", "collide_cascaded_d3q19", dict(
        id="lbm.collision.cascaded.d3q19", kind="collision", maturity="developing",
        wired=False, tested=True, verified=False,
        test_path="tests/test_cascaded_collision.py",
        evidence={"grade": "MECHANICS",
                  "note": "全中心矩层级 CM 算子；引用方为 free_surface_lbm 与契约模块，"
                          "主 stepper 无 'cascaded' 分派 ⇒ wired=False"})),
    ("cascaded_collision.py", "collide_cascaded_d3q27", dict(
        id="lbm.collision.cascaded.d3q27", kind="collision", maturity="developing",
        wired=False, tested=True, verified=False,
        test_path="tests/test_cascaded_collision.py",
        evidence={"grade": "MECHANICS", "note": "D3Q27 CM 算子；main path 未接线"})),
    ("advanced_collision.py", "collide_cascaded_d3q27", dict(
        id="lbm.collision.cascaded.d3q27.second_order", kind="collision",
        maturity="developing", wired=False, tested=True, verified=False,
        test_path="tests/test_cascaded_collision.py",
        evidence={"grade": "MECHANICS",
                  "note": "二阶正则化重建（非 CM/KBC）：契约 advanced_collision_contract 已声明"
                          "其高阶中心矩未实现，不得当作 CM 使用"})),
    # ---- multiphase / phase field
    ("multiphase3d.py", "sc_two_component_force_3d", dict(
        id="lbm.multiphase.shan_chen_force.d3q19", kind="multiphase",
        maturity="developing", wired=True, tested=True, verified=False,
        test_path="tests/test_multiphase.py",
        evidence={"grade": "MECHANICS",
                  "note": "Shan-Chen 两相作用力（3D）；静滴/混合物测试通过，"
                          "无 Laplace 压差 vs 解析的误差带"})),
    ("cac_lbm.py", None, dict(
        id="lbm.phasefield.cac_hch", kind="multiphase", maturity="developing",
        wired=False, tested=True, verified=False,
        test_path="tests/test_ac_lattice_generic.py",
        evidence={"grade": "MECHANICS",
                  "feature_ledger": "GAP-PP3-CAC-HCH",
                  "note": "保守 Allen-Cahn + 化学势相场框架；高密度比发散在案，"
                          "登记为框架(mode=module)而非可用求解器"})),
    ("allen_cahn_lbm.py", "measure_interface_width", dict(
        id="lbm.phasefield.interface_width_probe", kind="multiphase",
        maturity="developing", wired=False, tested=True, verified=False,
        test_path="tests/test_ac_lattice_generic.py",
        evidence={"grade": "MECHANICS", "note": "界面宽度反演探针（tanh 剖面），用于相场标定"})),
    # ---- free surface / acoustics / icing
    ("free_surface.py", "vof_sweep_x", dict(
        id="lbm.free_surface.vof_sweep", kind="free_surface", maturity="developing",
        wired=True, tested=True, verified=False,
        test_path="tests/test_dam_break_lbm_face.py",
        evidence={"grade": "MECHANICS",
                  "note": "VOF 方向扫描 + MUSCL 界面重构；质量账本测试，无解析解误差带"})),
    ("acoustics.py", "compute_fwh_far_field", dict(
        id="lbm.acoustics.fwh_farfield", kind="acoustics", maturity="developing",
        wired=True, tested=True, verified=False,
        test_path="tests/test_acoustics.py",
        evidence={"grade": "MECHANICS",
                  "note": "FWH 时域积分远场声压；验证脚本 acoustics_validation 存在，"
                          "未登记 vs 解析单极/偶极源的误差带"},
        endpoint="api:tensorlbm.acoustics.compute_fwh_far_field")),
    ("aircraft_icing.py", "rime_density_macklin", dict(
        id="lbm.icing.rime_density", kind="icing", maturity="developing",
        wired=False, tested=True, verified=False,
        test_path="tests/test_aircraft_icing.py",
        evidence={"grade": "MECHANICS",
                  "note": "Macklin 霜冰密度经验式；结冰整套流程(aircraft_icing)仍为 developing，"
                          "主求解路径未接线"})),
    # ---- AMR / backends / force / differentiable
    ("amr_reflux.py", "restrict_residual", dict(
        id="lbm.amr.reflux_restrict", kind="amr", maturity="developing",
        wired=True, tested=True, verified=False,
        test_path="tests/test_amr_flux_conservation.py",
        evidence={"grade": "MECHANICS", "note": "AMR 残差限制/回灌守恒；通量守恒测试"})),
    ("backends/triton_backend.py", "triton_suboff_step", dict(
        id="lbm.backend.triton.suboff_step", kind="backend", maturity="developing",
        wired=False, tested=False, verified=False,
        evidence={"grade": "NONE",
                  "note": "Triton 融合步：仅在装有 triton 的 GPU 节点可用，"
                          "本环境无法执行 ⇒ 未验证、未接线（诚实标 NONE）"})),
    ("force_methods.py", "force_momentum_exchange", dict(
        id="lbm.force.momentum_exchange", kind="force", maturity="developing",
        wired=True, tested=True, verified=False,
        test_path="tests/test_magnus_boundary.py",
        evidence={"grade": "MECHANICS",
                  "note": "动量交换法壁面力；与其它方法对比测试存在，无实验阻力误差带"},
        endpoint="api:tensorlbm.force_methods.force_momentum_exchange")),
    ("autograd_calib.py", "calibrate", dict(
        id="lbm.differentiable.closure_calibration", kind="differentiable",
        maturity="developing", wired=False, tested=True, verified=False,
        test_path="tests/test_autograd_calib.py",
        evidence={"grade": "MECHANICS",
                  "note": "可微闭包标定（伴随/有限差分对照）；无实测阻力标定闭环 ⇒ developing"})),
]


def render(kwargs: dict) -> str:
    parts = [f'id="{kwargs["id"]}"']
    for key, value in kwargs.items():
        if key == "id":
            continue
        parts.append(f"{key}={value!r}")
    body = ",\n    ".join(parts)
    return f"@capability(\n    {body},\n)"


def insert(path: Path, target: str | None, kwargs: dict) -> str:
    text = path.read_text(encoding="utf-8")
    deco = render(kwargs)
    if kwargs["id"] in text:
        return f"SKIP (already registered): {kwargs['id']}"
    if target is None:
        # module-level claim: place after the module docstring, before first import
        tree = ast.parse(text)
        doc = tree.body[0] if tree.body and isinstance(tree.body[0], ast.Expr) else None
        lines = text.splitlines(keepends=True)
        insert_at = (doc.end_lineno if doc is not None else 0)
        block = (
            "\n\n# module-level capability claim: the framework itself is the unit\n"
            f"{IMPORT_LINE}\n\n\n"
            "@capability(\n    " + ",\n    ".join(
                [f'id="{kwargs["id"]}"'] +
                [f"{k}={v!r}" for k, v in kwargs.items() if k != "id"]
            ) + ",\n)\n"
            "def _capability_claim() -> str:\n"
            f'    """Registry anchor for {kwargs["id"]}."""\n'
            f'    return "{kwargs["id"]}"\n'
        )
        new = "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])
        path.write_text(new, encoding="utf-8")
        ast.parse(new)
        return f"INSERTED module claim: {kwargs['id']}"

    lines = text.splitlines(keepends=True)
    pattern_def = f"def {target}("
    pattern_class = f"class {target}"
    hits = [i for i, ln in enumerate(lines)
            if ln.startswith(pattern_def) or ln.startswith(pattern_class)]
    if len(hits) != 1:
        raise SystemExit(f"{path}: expected exactly one top-level {target}, found {len(hits)}")
    idx = hits[0]
    # keep any existing decorators attached to the target below the new one
    while idx > 0 and lines[idx - 1].lstrip().startswith("@"):
        idx -= 1
    block = f"{IMPORT_LINE}\n\n\n{deco}\n"
    new = "".join(lines[:idx]) + block + "".join(lines[idx:])
    ast.parse(new)
    path.write_text(new, encoding="utf-8")
    return f"INSERTED: {kwargs['id']} -> {path.relative_to(ROOT)}::{target}"


def main() -> int:
    status = []
    for rel, target, kwargs in TABLE:
        path = SRC / rel
        if not path.is_file():
            status.append(f"MISSING FILE: {rel}")
            continue
        status.append(insert(path, target, kwargs))
    print(json.dumps(status, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""m15617 · TensorLBM 国际对标专项审计生成器 (GAP-LBM-INTL-BENCH).

机器化做四件事，避免人工宣布成熟度：
  1) 四参照（OpenLB/waLBerla/Palabos/Lettuce）逐算法引用 + 活体校验；
  2) TensorLBM 模块四级分档 verified/partial/implemented/failed，
     判据=证据层级 1 模块存在 <2 API测试 <3 守恒测试 <4 标准算例 <5 网格收敛 <6 实验对比；
  3) 生成 tensorlbm_module_maturity / tensorlbm_compete_score 的 DDL+INSERT；
  4) 保留/重构/删除清单 + 差距→开发任务(挂 T1-T5)。
数字一律从 result.json / STATUS.md 抽取写盘（零手抄）；审计不宣布 verified：
verified 需 (a) benchmarks/verified/<case>/ 在盘 (b) 其 result.json 有真值裁决字段
(c) 有收敛类字段，缺一降档。
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, platform, re, subprocess, sys
from datetime import datetime, timezone

AUDIT_ID, MISSION = "m15617-intl-bench-v1", 15617

# ── artifact set + provenance manifest ───────────────────────────────────────
# The audit is only as trustworthy as the files it leaves behind: a merge that
# drops them, or a hand edit that nudges a number, must be *detectable* rather
# than discoverable weeks later.  So every run hashes its own outputs into
# ``manifest.json`` and ``verify_manifest()`` is a pure file-system check (no
# re-run, no network) that the artifact set on disk is bit-identical to what the
# generator wrote.  ``expected_artifacts`` is recorded too, so a *missing* file
# is a mismatch and not a silent no-op.
MANIFEST_NAME = "manifest.json"
ARTIFACT_NAMES = ("module_maturity.json", "module_maturity.csv", "compete_score.csv",
                  "maturity.sql", "level_distribution.png", "compete_radar.png")
# SQL deliverable is kept in docs/ under the mission's own name; m15912 is the
# later alias of the same snapshot and is regenerated in lock-step (they must
# never diverge — a stale copy is exactly the drift this guard exists for).
DOC_SQL_FILES = ("tensorlbm_maturity_m15617.sql", "tensorlbm_maturity_m15912.sql")
FAMILIES = ["基础LBM内核", "高阶碰撞", "边界复杂几何", "多相流", "自由表面",
            "湍流", "GPU单卡", "多GPU-HPC", "工程验证", "AI融合"]

UPSTREAM = {
 "OpenLB": {"root": "/root/openlb", "snapshot": "master@6c3934d (2019-12-02)", "tier":
   {"collision": 4, "multiphase": 3, "boundary": 4, "amr": 2, "gpu": 1},
   "notes": {"amr": "仅 superLattice/super-BC 局部加密 src/core/superLattice3D.h，无块结构 AMR",
             "gpu": "git ls-tree 中 .cu 计数=0；MPI 并行走 src/communication",
             "multiphase": "src/dynamics/freeEnergyDynamics.cpp + ForcedShanChenBGKdynamics"},
   "cites": {
     "collision": [("src/dynamics/dynamics.h", "class BGKdynamics"),
                   ("src/dynamics/dynamics.h", "class TRTdynamics"),
                   ("src/dynamics/dynamics.h", "class ForcedShanChenBGKdynamics"),
                   ("src/dynamics/mrtDynamics.h", "class MRTdynamics"),
                   ("src/dynamics/entropicDynamics.cpp", None),
                   ("src/dynamics/freeEnergyDynamics.cpp", None)],
     "multiphase": [("src/dynamics/freeEnergyDynamics.h", None),
                    ("src/dynamics/dynamics.h", "ForcedShanChenBGKdynamics")],
     "boundary": [("src/boundary/offBoundaryCondition3D.h", "createBouzidiBoundaryCondition3D"),
                  ("src/boundary/boundaryCondition3D.h", None)],
     "amr": [("src/core/superLattice3D.h", None)],
     "gpu": [("src/communication/communicator3D.cpp", None)]}},
 "waLBerla": {"root": "/tmp/walberla", "snapshot": "git HEAD (CMake>=3.26 主干)", "tier":
   {"collision": 4, "multiphase": 4, "boundary": 4, "amr": 4, "gpu": 4},
   "notes": {"amr": "src/lbm/refinement/TimeStep.h + blockforest 块结构 AMR",
             "multiphase": "src/lbm/free_surface (FlagInfo/dynamics/bubble_model)",
             "gpu": "src/gpu/*.cu + pystencils 生成 GPU kernel",
             "collision": "srt/trt/mrt/cumulant 四族 cell_operations"},
   "cites": {
     "collision": [("src/lbm/srt/cell_operations/DefaultCellOperation.impl.h", "omega_"),
                   ("src/lbm/trt/cell_operations/DefaultCellOperation.impl.h", None),
                   ("src/lbm/mrt/cell_operations/DefaultCellOperation.impl.h", None),
                   ("src/lbm/cumulant/CellwiseSweep.impl.h", "Cumulant")],
     "multiphase": [("src/lbm/free_surface/FlagInfo.h", "class FlagInfo"),
                    ("src/lbm/free_surface/dynamics", None),
                    ("src/lbm/free_surface/bubble_model/BubbleModel.h", None)],
     "boundary": [("src/lbm/boundary/NoSlip.h", None), ("src/lbm/boundary/DynamicUBB.h", None),
                  ("src/lbm/boundary/FreeSlip.h", None), ("src/lbm/boundary/Curved.h", None)],
     "amr": [("src/lbm/refinement/TimeStep.h", "class TimeStep"),
             ("src/lbm/refinement/PdfFieldPackInfo.h", None),
             ("src/lbm/refinement/CurlBasedLevelDetermination.h", None)],
     "gpu": [("src/gpu", None), ("tests/gpu/Kernels.cu", None)]}},
 "Palabos": {"root": "/tmp/palabos_probe", "snapshot": "2.3.0 + unreleased", "tier":
   {"collision": 4, "multiphase": 4, "boundary": 4, "amr": 3, "gpu": 2},
   "notes": {"multiphase": "src/multiPhysics/shanChenProcessor3D + freeSurfaceAnalysis3D",
             "amr": "src/gridRefinement + src/multiGrid (2.3 开发中)",
             "gpu": "仅 coupledSimulators/npFEM/src_GPU，主求解器无 GPU 后端"},
   "cites": {
     "collision": [("src/basicDynamics/isoThermalDynamics.h", "class BGKdynamics"),
                   ("src/complexDynamics/trtDynamics.h", "class TRTdynamics"),
                   ("src/core/dynamics.h", "class NoDynamics"),
                   ("src/multiPhysics/shanChenProcessor3D.h", "ShanChenMultiComponentProcessor3D")],
     "multiphase": [("src/multiPhysics/shanChenProcessor3D.h", "class ShanChenMultiComponentProcessor3D"),
                    ("src/multiPhysics/shanChenLattices3D.h", None),
                    ("src/multiPhysics/freeSurfaceAnalysis3D.h", None)],
     "boundary": [("src/boundaryCondition/bounceBackModels3D.h", None),
                  ("src/boundaryCondition/boundaryCondition3D.h", None),
                  ("src/boundaryCondition/NLD_boundaryDynamics3D.h", None)],
     "amr": [("src/gridRefinement", None), ("src/multiGrid/defaultMultiGridPolicy3D.h", None)],
     "gpu": [("coupledSimulators/npFEM/src_GPU/device_utilities.cu", None)]}},
 "Lettuce": {"root": "/tmp/up/lettuce", "snapshot": "master@88a6559", "tier":
   {"collision": 4, "multiphase": 1, "boundary": 3, "amr": 1, "gpu": 4},
   "notes": {"multiphase": "git grep -il 'multiphase|shan_chen|free_surface' 仅命中 _force/__init__.py 注释→无多相",
             "amr": "无 AMR（均匀网格）",
             "gpu": "cuda_native 代码生成后端；多卡靠分布式分支",
             "boundary": "bounce_back / anti_bounce_back_outlet / partially_saturated，无两点插值壁面"},
   "cites": {
     "collision": [("lettuce/ext/_collision/kbc_collision.py", "class KBCCollision"),
                   ("lettuce/ext/_collision/mrt_collision.py", None),
                   ("lettuce/ext/_collision/regularized_collision.py", None),
                   ("lettuce/ext/_collision/bgk_collision.py", None)],
     "multiphase": [("lettuce/ext/_force/__init__.py", None)],
     "boundary": [("lettuce/ext/_boundary/bounce_back_boundary.py", None),
                  ("lettuce/ext/_boundary/anti_bounce_back_outlet.py", None),
                  ("lettuce/ext/_boundary/partially_saturated_boundary.py", None)],
     "amr": [("lettuce/base.py", None)],
     "gpu": [("lettuce/cuda_native/_generator.py", None)]}},
}

# key, family, impl glob, test glob, 算例候选, doc, note
MODULES = [
 ("bgk_stream_collide_2d", "基础LBM内核", ["src/tensorlbm/*collision*.py"], ["tests/*collision*"], ["poiseuille_2d", "couette_2d", "taylor_green_2d"], [], ""),
 ("d3q27_kernel", "基础LBM内核", ["src/tensorlbm/*d3q27*"], ["tests/*d3q27*"], ["taylor_green_3d", "sphere_re100_d3q27"], ["docs/D3Q27_ACTIVATION.md"], ""),
 ("solver3d_stream_collide", "基础LBM内核", ["src/tensorlbm/solver3d.py", "src/tensorlbm/solver.py"], ["tests/*solver*"], ["poiseuille_convergence_order", "kovasznay_2d"], [], ""),
 ("mms_order_verification", "基础LBM内核", ["src/tensorlbm/spatial_convergence.py"], ["tests/*convergence*", "tests/*mms*"], ["mms_convergence_2d"], [], ""),
 ("perf_solver_torch", "GPU单卡", ["src/tensorlbm/perf_solver.py"], ["tests/*perf*", "tests/test_triton_fused.py"], [], [], ""),
 ("trt_collision", "高阶碰撞", ["src/tensorlbm/*trt*"], ["tests/*trt*"], [], [], "TRT 无独立算例"),
 ("mrt_collision", "高阶碰撞", ["src/tensorlbm/*mrt*"], ["tests/*mrt*"], [], [], ""),
 ("kbc_entropic_collision", "高阶碰撞", ["src/tensorlbm/*kbc*"], ["tests/*kbc*"], [], [], ""),
 ("cumulant_d3q27", "高阶碰撞", ["src/tensorlbm/*cumulant*", "src/tensorlbm/*cmkbc*"], ["tests/*cumulant*"], ["collision_kernels_3d_tg"], ["docs/CUMULANT_D3Q27_VERIFICATION.md"], ""),
 ("regularized_collision", "高阶碰撞", ["src/tensorlbm/*regularized*"], ["tests/*regular*"], [], [], ""),
 ("halfway_bounce_back", "边界复杂几何", ["src/tensorlbm/*boundar*.py", "src/tensorlbm/*bounce*"], ["tests/*bounce*", "tests/*boundar*"], ["poiseuille_3d_duct", "couette_3d", "staircase_wall_offset"], [], ""),
 ("interpolated_bfl_wall", "边界复杂几何", ["src/tensorlbm/*bouzidi*", "src/tensorlbm/wall_surface_bfl.py"], ["tests/*bfl*", "tests/*bouzidi*"], ["sphere_drag_bfl", "sphere_boundary_sensitivity"], ["docs/IN12_S6_WALL_LADDER.md"], ""),
 ("immersed_boundary_ibm", "边界复杂几何", ["src/tensorlbm/*ibm*", "src/tensorlbm/*immersed*"], ["tests/*ibm*"], ["naca_0012", "propeller_openwater"], ["docs/IN54W1-N02_IBM_ice_propeller_validation.md"], ""),
 ("wall_function_les", "边界复杂几何", ["src/tensorlbm/wall_function*.py", "src/tensorlbm/spalding_wall_model.py"], ["tests/*wall_function*", "tests/*wallmodel*"], ["suboff_re1000", "channel_turb_3d"], ["docs/IN12_S6_WALL_LADDER.md"], ""),
 ("stl_cad_voxelization", "边界复杂几何", ["src/tensorlbm/stl_geometry.py", "src/tensorlbm/voxelize.py", "src/tensorlbm/ship_cad3d.py"], ["tests/*voxel*", "tests/*stl*"], ["dtmb5415_resistance", "wigley_hydrostatics"], ["benchmarks/STATUS.md"], ""),
 ("shan_chen_multiphase", "多相流", ["src/tensorlbm/multiphase.py", "src/tensorlbm/pseudopotential_edm3d.py", "src/tensorlbm/sc_eos_coexistence.py"], ["tests/*multiphase*"], ["rayleigh_taylor", "two_phase_poiseuille", "mcmp_3d_stability"], ["benchmarks/pending/rayleigh_taylor/README.md"], ""),
 ("color_gradient_multiphase", "多相流", ["src/tensorlbm/*color_gradient*"], ["tests/*color*"], ["two_phase_poiseuille_viscous", "poiseuille_two_phase_cg"], ["benchmarks/pending/hadamard_rybczynski_droplet/NOTES.md"], ""),
 ("free_energy_multiphase", "多相流", ["src/tensorlbm/phasefield", "src/tensorlbm/*free_energy*"], ["tests/*free_energy*", "tests/*phasefield*"], ["laplace_cac_phasefield", "laplace_pr_eos"], ["docs/FE_VERIFICATION_STATUS_M14723.md"], ""),
 ("allen_cahn_multiphase", "多相流", ["src/tensorlbm/allen_cahn*.py"], [], [], ["benchmarks/pending/hadamard_rybczynski_droplet/NOTES.md"], "orphan: 无测试无算例"),
 ("laplace_law_surface_tension", "多相流", ["src/tensorlbm/sc_surface_tension.py"], ["tests/*surface_tension*", "tests/*laplace*"], ["laplace_droplet", "laplace_pr_eos"], ["docs/FREE_SURFACE_SURFACE_TENSION_CSF.md"], ""),
 ("vof_free_surface", "自由表面", ["src/tensorlbm/vof_free_surface", "src/tensorlbm/vof_interface_capture.py"], ["tests/*vof*", "tests/*free_surface*"], ["dam_break_2d_multimodel", "rayleigh_taylor"], ["docs/FREE_SURFACE_VERIFICATION.md"], ""),
 ("fs_mass_conservation", "自由表面", ["src/tensorlbm/paired_link_mass_accounting.py"], ["tests/*mass*"], ["dam_break_sc"], ["docs/FS_WALBERLA_ALIGNMENT_REPORT.md"], ""),
 ("wave_tank_2d", "自由表面", ["src/tensorlbm/wave_tank*.py", "src/tensorlbm/wave_bc*.py"], ["tests/*wave_tank*"], ["wave_tank", "sloshing_resonance"], ["docs/IN144_N07_sloshing_report.md"], ""),
 ("smagorinsky_les", "湍流", ["src/tensorlbm/*smagorinsky*", "src/tensorlbm/turbulence"], ["tests/*smag*", "tests/*les*"], ["cavity", "channel_turb_3d"], [], ""),
 ("wale_les", "湍流", ["src/tensorlbm/*wale*"], ["tests/*wale*"], [], [], ""),
 ("rans_ke", "湍流", ["src/tensorlbm/rans_ke.py", "src/tensorlbm/rans_common.py"], ["tests/*rans*"], ["suboff_re1000"], [], ""),
 ("turbulence_stats", "湍流", ["src/tensorlbm/turbulence_stats.py"], ["tests/*turbulence*"], [], [], ""),
 ("triton_fused_single_gpu", "GPU单卡", ["src/tensorlbm/triton_fused.py"], ["tests/test_triton_fused.py"], ["collision_kernels_3d_tg"], ["docs/triton_audit.md", "docs/triton_fused_bwd.md"], "8.6 GLUPS 为项目自记(docstring)，非第三方复现"),
 ("triton_obstacle_gpu", "GPU单卡", ["src/tensorlbm/triton_fused_obstacle.py", "src/tensorlbm/triton_backend.py"], ["tests/test_triton_obstacle_wall_parity.py", "tests/*triton*"], [], [], ""),
 ("torch_fallback_backend", "GPU单卡", ["src/tensorlbm/torch_fallback.py"], ["tests/*fallback*"], [], [], ""),
 ("triton_distributed_multigpu", "多GPU-HPC", ["src/tensorlbm/triton_fused_distributed.py"], ["tests/*distributed*", "tests/*triton*"], [], ["docs/triton_distributed_notes.md"], "69 GLUPS/8GPU 为自记"),
 ("torch_distributed_multigpu", "多GPU-HPC", ["src/tensorlbm/suboff_torch_distributed.py"], ["tests/*distributed*", "tests/*triton*"], [], [], ""),
 ("static_block_amr", "工程验证", ["src/tensorlbm/static_block_amr.py", "src/tensorlbm/suboff_static_amr.py"], ["tests/*amr*"], ["suboff_amr_convergence"], ["docs/AMR_FREE_SURFACE_REPORT.md"], "AMR 归入工程验证面（尤见 STATUS 中 amr 相关门）"),
 ("sixdof_motion", "工程验证", ["src/tensorlbm/rigid_body_6dof.py", "src/tensorlbm/sixdof.py"], ["tests/*sixdof*", "tests/*6dof*"], ["floating_body_free_decay", "maneuvering_steady_turn"], ["docs/IN59_N02_KCS_MANEUVERING_CFD.md"], ""),
 ("seakeeping_rao", "工程验证", ["src/tensorlbm/seakeeping", "src/tensorlbm/rao_analysis.py"], ["tests/*rao*", "tests/*seakeep*"], ["wave_body_rao"], ["docs/IRREGULAR_WAVE_6DOF_SUMMARY.md"], ""),
 ("ship_resistance_hull", "工程验证", ["src/tensorlbm/ship_resistance_closure.py", "src/tensorlbm/ship_cad.py", "src/tensorlbm/ship_flow.py"], ["tests/*resistance*", "tests/*hull*"], ["suboff_re1000", "suboff_sail_resistance", "kcs_resistance", "dtmb5415_resistance"], ["benchmarks/STATUS.md"], ""),
 ("fwh_acoustics", "工程验证", ["src/tensorlbm/thermal_radiation.py"], ["tests/*fwh*", "tests/*acoustic*"], ["fwh_acoustics", "acoustics"], [], ""),
 ("non_newtonian_yield", "工程验证", ["src/tensorlbm/non_newtonian", "src/tensorlbm/yield_stress_lbm.py", "src/tensorlbm/powerlaw.py"], ["tests/*yield*", "tests/*rheology*"], ["rheology_yield"], ["docs/AUDIT_m14882_H2_FRAZIL_RHEOLOGY.md"], ""),
 ("thermal_convection", "工程验证", ["src/tensorlbm/thermal3d.py", "src/tensorlbm/thermal.py", "src/tensorlbm/rb_convection_validation.py"], ["tests/*thermal*", "tests/*rb_*"], ["thermal_cavity", "thermal_cavity_3d", "cavity_natural_convection", "rayleigh_benard"], [], ""),
 ("compressible_shock", "工程验证", ["src/tensorlbm/weak_compressibility.py"], ["tests/*compressible*", "tests/*shock*"], ["compressible_shock_family", "sod_shock_tube"], ["docs/IN50_W4_N04_COMPRESSIBLE_LBM.md"], ""),
 ("rotating_frame", "工程验证", ["src/tensorlbm/rotating_frame.py", "src/tensorlbm/rotating_interior.py", "src/tensorlbm/rotating_duct.py"], ["tests/*rotating*"], ["rotating_ekman", "rotating_duct_frame_w2"], [], ""),
 ("dem_coupling_prototype", "工程验证", ["src/tensorlbm/particle_suspension.py", "src/tensorlbm/settling_coupling.py"], ["tests/*settling*", "tests/*coupling*"], ["sphere_settling_gcs", "permeability"], ["docs/tensordem_coupling_p1.md"], ""),
 ("regression_gates_contract", "AI融合", ["src/tensorlbm/regression_gate.py", "src/tensorlbm/physics_capability_contract.py"], ["tests/*regression_gate*", "tests/*contract*"], ["mms_convergence_2d"], ["docs/INPUT_CONTRACT_GUARDS_20261111.md"], ""),
 ("neural_wall_law", "AI融合", ["src/tensorlbm/neural_wall_law.py"], ["tests/*neural*"], [], ["docs/AI4S_APPS.md"], ""),
 ("pod_reduced_order", "AI融合", ["src/tensorlbm/pod.py", "src/tensorlbm/topology_opt.py"], ["tests/*pod*"], [], [], ""),
]

SOFT_NOTES = {
 "immersed_boundary_ibm": "STATUS：naca0012 表面 MEM +82%（摩擦主导翼型），propeller_openwater CAD 面积 -32%/-24%（算例级缺陷，非模型结构性失效）",
}

KNOWN_DEFECTS = {
 "allen_cahn_multiphase": "NOTES.md #5：相场无法维持分散液滴（<200 步溶解），sigma 力缺曲率、相方程不守恒、M/tau_phi 硬编码",
 "color_gradient_multiphase": "NOTES.md #2：lambda=rho 锁死；密度差声学塌陷",
 "shan_chen_multiphase": "NOTES.md #3/#6+G1：压力挤压反转/NaN；G_12 符号与 docstring 相反",
 "vof_free_surface": "rayleigh_taylor README：VOF 无自由压力场（结构性）",
 "dem_coupling_prototype": "docs/tensordem_coupling_p1.md：耦合=可执行原型非验证",
 "ship_resistance_hull": "STATUS：DTMB5415 阶梯湿面积 16x 光滑→Cd_tot 1.18；KCS 收敛跑未完成",
}
NUM_KEY = re.compile(r"(err|rel|dev|pct|delta|gci|drift|closure|residual|ratio)", re.I)


def walk_nums(o, pre="", out=None, d=0):
    out = out if out is not None else []
    if d > 4:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            walk_nums(v, f"{pre}.{k}" if pre else str(k), out, d + 1)
    elif isinstance(o, list):
        for i, v in enumerate(o[:6]):
            walk_nums(v, f"{pre}[{i}]", out, d + 1)
    elif isinstance(o, (int, float)) and not isinstance(o, bool) and NUM_KEY.search(pre):
        out.append((pre, o))
    return out


def truthy_verdict(o):
    hits = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("verified", "all_gates_pass", "pass", "passed", "ok", "gates_pass") and v is True:
                hits.append(k)
            hits += truthy_verdict(v)
    elif isinstance(o, list):
        for v in o:
            hits += truthy_verdict(v)
    return hits


def has_convergence(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if re.search(r"converg|grids|refinement|order", str(k), re.I):
                return True
            if has_convergence(v):
                return True
    elif isinstance(o, list):
        return any(has_convergence(v) for v in o)
    return False


def load_status(repo):
    rows, p = {}, os.path.join(repo, "benchmarks", "STATUS.md")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            m = re.match(r"\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|\s*$", line)
            if m:
                rows[m.group(1).strip()] = m.group(2).strip()
    return rows


def gfirst(repo, pats):
    for pat in pats:
        h = sorted(x for x in glob.glob(os.path.join(repo, pat), recursive=True) if os.path.exists(x))
        if h:
            return h
    return []


def grade(repo, status, mod):
    key, fam, impl, tests, cases, docs, note = mod
    impl_h, test_h = gfirst(repo, impl), gfirst(repo, tests)
    doc_h = [d for d in docs if os.path.exists(os.path.join(repo, d))]
    vcase = pcase = None
    for c in cases:
        if not vcase and os.path.isdir(os.path.join(repo, "benchmarks", "verified", c)):
            vcase = c
        if not pcase and os.path.isdir(os.path.join(repo, "benchmarks", "pending", c)):
            pcase = c
    tier, why = 1, "模块存在（实现文件在盘）"
    errs, verdict, conv = [], [], False
    if test_h:
        tier, why = 2, "API/单元测试在盘"
    if pcase:
        tier, why = max(tier, 3), "pending 算例在盘（守恒/gate 或结构性 FAIL）"
    if vcase:
        rj = os.path.join(repo, "benchmarks", "verified", vcase, "result.json")
        obj = {}
        if os.path.exists(rj):
            try:
                obj = json.load(open(rj))
            except Exception:
                obj = {}
        errs, verdict, conv = walk_nums(obj)[:8], truthy_verdict(obj)[:4], has_convergence(obj)
        tier, why = 4, "benchmarks/verified 算例在盘"
        if conv:
            tier, why = 5, "verified 算例含网格/收敛证据字段"
    r = {"key": key, "family": fam, "impl_paths": [os.path.relpath(x, repo) for x in impl_h][:4],
         "test_paths": [os.path.relpath(x, repo) for x in test_h][:4], "doc_paths": doc_h, "note": note,
         "case_verified": vcase, "case_pending": pcase, "evidence_tier": tier, "tier_why": why,
         "verdict_fields": verdict, "has_convergence_field": conv,
         "error_samples": [{"path": p, "value": v} for p, v in errs], "defect": KNOWN_DEFECTS.get(key, "") or SOFT_NOTES.get(key, "")}
    if vcase and status.get(vcase):
        r["status_row"] = status[vcase][:600]
    if pcase and status.get(pcase):
        r["pending_status_row"] = status[pcase][:600]
    if KNOWN_DEFECTS.get(key) and not vcase and tier < 4:
        r["level"] = "failed"
    elif tier >= 5 and verdict:
        r["level"] = "verified"
    elif tier == 5:
        r["level"] = "partial"
        r["downgrade_reason"] = "verified/ 在盘但 result.json 无真值裁决字段"
    elif tier >= 3:
        r["level"] = "partial"
    else:
        r["level"] = "implemented"
    if r["level"] == "verified" and r.get("defect"):
        # A registered structural defect and a verification claim cannot coexist.
        r["level"] = "partial"
        r["downgrade_reason"] = "已登记结构性缺陷未闭合，verified 不予保留：" + str(r["defect"])[:80]
    return r



def apply_evidence_isolation(rows):
    """Honesty pass: one benchmark case may certify one module only.

    A ``benchmarks/verified/<case>`` result is evidence for the *capability it
    exercised*.  When the same case is quoted by several modules (e.g. suboff_re1000
    backing both the wall-function LES and the k-epsilon RANS row) the audit cannot
    tell which of them the numbers actually cover, so the shared credit is withheld
    and the modules are demoted to ``partial`` with the sharing recorded.
    Known structural defects (``KNOWN_DEFECTS``) also block ``verified``: an open
    defect cannot coexist with a verification claim.
    """
    seen = {}
    for r in rows:
        c = r.get("case_verified")
        if c:
            seen.setdefault(c, []).append(r["key"])
    for r in rows:
        c = r.get("case_verified")
        if c and len(seen[c]) > 1:
            r["evidence_shared_with"] = [k for k in seen[c] if k != r["key"]]
            if r["level"] == "verified":
                r["level"] = "partial"
                r["downgrade_reason"] = (f"证据复用：verified 算例 {c} 同时支撑 {len(seen[c])} "
                                         f"个模块（{','.join(seen[c])}），非模块专属证据")
        if r["level"] == "verified" and r.get("defect"):
            r["level"] = "partial"
            r["downgrade_reason"] = "已登记结构性缺陷未闭合，verified 不予保留：" + str(r["defect"])[:80]
    return rows


def check_citations(rep):
    bad = []
    for name, spec in UPSTREAM.items():
        root = spec["root"]
        n_decl = sum(len(v) for v in spec.get("cites", {}).values())
        if not os.path.isdir(root):
            # Honest degradation: an absent checkout is reported, never silently graded.
            bad.append(f"{name}: 上游根目录缺失 {root}")
            rep.setdefault("citation_stats", {})[name] = {"ok": 0, "total": n_decl}
            continue
        ok = tot = 0
        for topic, pairs in spec["cites"].items():
            for path, needle in pairs:
                tot += 1
                full = os.path.join(root, path)
                if not os.path.exists(full):
                    bad.append(f"{name}/{topic}: 缺 {path}")
                    continue
                if needle:
                    txt = "" if os.path.isdir(full) else open(full, encoding="utf-8", errors="replace").read()
                    if needle not in txt:
                        bad.append(f"{name}/{topic}: {path} 不含 {needle!r}")
                        continue
                ok += 1
        rep.setdefault("citation_stats", {})[name] = {"ok": ok, "total": tot}
    return bad


ANCHORS = {"T1": "碰撞与内核（高阶碰撞/守恒/精度阶）", "T2": "边界与复杂几何（亚格点壁面/IBM/CAD 保真）",
           "T3": "多相与自由表面（密度比/黏度比解耦、Laplace、界面守恒）", "T4": "GPU/HPC（单卡吞吐、多卡扩展、口径统一）",
           "T5": "工程验证与 AI 融合（阻力量级、实验对标、神经闭合）"}
DIM_TOPIC = {"基础LBM内核": "collision", "高阶碰撞": "collision", "边界复杂几何": "boundary", "多相流": "multiphase",
             "自由表面": "multiphase", "湍流": "collision", "GPU单卡": "gpu", "多GPU-HPC": "gpu",
             "工程验证": "boundary", "AI融合": "collision"}
ANCHOR_OF = {"基础LBM内核": "T1", "高阶碰撞": "T1", "边界复杂几何": "T2", "多相流": "T3", "自由表面": "T3",
             "湍流": "T5", "GPU单卡": "T4", "多GPU-HPC": "T4", "工程验证": "T5", "AI融合": "T5"}



DIM_EN = {"基础LBM内核": "core LBM", "高阶碰撞": "high-order collision", "边界复杂几何": "boundary/geometry",
          "多相流": "multiphase", "自由表面": "free surface", "湍流": "turbulence", "GPU单卡": "GPU single-card",
          "多GPU-HPC": "multi-GPU/HPC", "工程验证": "engineering validation", "AI融合": "AI coupling"}

def compete(rows):
    out = []
    for dim in FAMILIES:
        mods = [r for r in rows if r["family"] == dim]
        if not mods:
            continue
        lvl = sum(r["evidence_tier"] for r in mods) / len(mods)
        up = {}
        for name, spec in UPSTREAM.items():
            t = spec["tier"].get(DIM_TOPIC[dim], 1)
            if dim == "AI融合":
                t = 1
            if dim in ("多相流", "自由表面") and not spec["cites"].get("multiphase"):
                t = 1
            up[name] = t
        upmax = max(up.values())
        delta = round(lvl - upmax, 2)
        gap = "A(并跑/领先)" if delta >= 0 else ("B(差1级)" if delta >= -1 else ("C(差2级)" if delta >= -2 else "D(差≥3级)"))
        out.append({"dimension": dim, "n_modules": len(mods), "mean_evidence_tier": round(lvl, 2),
                    "maturity_index_0_10": round((lvl - 1) / 5 * 10, 1), "upstream_tier": up,
                    "upstream_max": upmax, "tier_delta": delta, "gap_grade": gap,
                    "counts": {k: sum(1 for r in mods if r["level"] == k) for k in ("verified", "partial", "implemented", "failed")}})
    return out



def _sha256(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError:
        return None


def _git_head(root):
    """HEAD commit of a checkout, read without mutating it (no git config touched)."""
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:  # pragma: no cover - defensive
        return None


def provenance(repo, rows):
    """Auditability block: every claim in the report is bound to a file digest.

    The gate rule is "零手抄": if a number cannot be traced to a hashed input it
    must not appear.  So we hash the audit's own inputs and record the identity of
    the four upstream checkouts (present/absent + HEAD).
    """
    inputs = [os.path.join(repo, "benchmarks", "STATUS.md")]
    for r in rows:
        for c in (r.get("case_verified"), r.get("case_pending")):
            if c:
                inputs.append(os.path.join(repo, "benchmarks", "verified" if c == r.get("case_verified")
                                           else "pending", c, "result.json"))
    digests = {}
    for f in inputs:
        if os.path.exists(f):
            digests[os.path.relpath(f, repo)] = _sha256(f)
    up = {}
    for name, spec in UPSTREAM.items():
        root = spec["root"]
        up[name] = {"root": root, "present": os.path.isdir(root),
                    "head": _git_head(root) if os.path.isdir(root) else None}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_head": _git_head(repo),
        "input_sha256": digests,
        "module_manifest_sha256": hashlib.sha256(
            json.dumps([[m[0], m[1]] for m in MODULES], ensure_ascii=False).encode()).hexdigest(),
        "n_modules": len(MODULES),
        "upstream_checkouts": up,
        "external_anchor_ok": all(v["present"] for v in up.values()),
    }


def artifact_digests(out_dir, names=ARTIFACT_NAMES):
    """sha256 + byte size of every artifact that exists in ``out_dir``."""
    digests = {}
    for name in names:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            digests[name] = {"sha256": _sha256(path), "bytes": os.path.getsize(path)}
    return digests


def sql_copy_digest(repo, sql_text):
    """sha256 of the docs/ SQL delivery copy, so a one-copy artifact dir stays bound."""
    h = hashlib.sha256(sql_text.encode()).hexdigest()
    for name in DOC_SQL_FILES:
        path = os.path.join(repo, "docs", name)
        if os.path.isfile(path):
            return {"path": os.path.relpath(path, repo), "sha256": h}
    return None


def artifact_names(sql_copy=True):
    """Artifacts written this run: the SQL copy is skipped when docs/ already carries it."""
    return tuple(ARTIFACT_NAMES) if sql_copy else tuple(n for n in ARTIFACT_NAMES if n != "maturity.sql")


def write_manifest(out_dir, rep, names=None):
    """Write ``manifest.json``: bind every emitted artifact to its sha256.

    Returns the manifest dict (also on disk).  The manifest deliberately does not
    hash itself, so re-verification needs no re-run: a downstream reader (gate,
    reviewer, platform ingest) only has to recompute the listed digests.
    ``expected_artifacts`` is the set that *this run actually wrote* (plus any
    excluded-and-relocated deliverable recorded under ``sql_deliverable``), so an
    absent file is a mismatch rather than a silent no-op.
    """
    names = tuple(names or ARTIFACT_NAMES)
    man = {
        "audit_id": AUDIT_ID,
        "mission": MISSION,
        "generated_utc": rep.get("generated_utc"),
        "generator": {"module": "tensorlbm.audit.intl_bench",
                      "file": os.path.basename(__file__),
                      "sha256": _sha256(os.path.abspath(__file__))},
        "repo_head": (rep.get("provenance") or {}).get("repo_head"),
        "input_sha256": (rep.get("provenance") or {}).get("input_sha256") or {},
        "module_manifest_sha256": (rep.get("provenance") or {}).get("module_manifest_sha256"),
        "schema": {"module_maturity_rows": len(rep.get("modules") or []),
                   "compete_score_rows": len(rep.get("scores") or [])},
        "expected_artifacts": list(names),
        "artifacts": artifact_digests(out_dir, names),
        "sql_deliverable": rep.get("sql_deliverable"),
        "figure_error": rep.get("figure_error"),
        "note": "机器生成，勿手改：artifacts 内每个文件的 sha256 由生成器写入本清单；"
                "verify_manifest(out_dir) 返回 [] 才算工件集完整未改。",
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return man


def verify_manifest(out_dir):
    """Check an artifact directory against its manifest — pure file-system probe.

    Returns a list of ``{"file", "problem", ...}`` mismatches; ``[]`` means the
    artifact set is complete and bit-identical to what the generator wrote.
    ``problem`` ∈ {manifest_missing, manifest_unreadable, missing, sha256, bytes,
    unlisted} — a missing artifact, a tampered artifact and an unexplained extra
    file are all failures, never warnings.
    """
    mpath = os.path.join(out_dir, MANIFEST_NAME)
    if not os.path.isfile(mpath):
        return [{"file": MANIFEST_NAME, "problem": "manifest_missing"}]
    try:
        man = json.load(open(mpath))
    except Exception as e:  # pragma: no cover - defensive
        return [{"file": MANIFEST_NAME, "problem": "manifest_unreadable", "detail": repr(e)}]
    listed = man.get("artifacts") or {}
    expected = man.get("expected_artifacts") or list(ARTIFACT_NAMES)
    bad = []
    for name in expected:
        rec = listed.get(name)
        path = os.path.join(out_dir, name)
        if rec is None or not os.path.isfile(path):
            bad.append({"file": name, "problem": "missing"})
            continue
        got = _sha256(path)
        if got != rec.get("sha256"):
            bad.append({"file": name, "problem": "sha256", "expected": rec.get("sha256"), "got": got})
        elif os.path.getsize(path) != rec.get("bytes"):
            bad.append({"file": name, "problem": "bytes",
                        "expected": rec.get("bytes"), "got": os.path.getsize(path)})
    for name in sorted(os.listdir(out_dir)):
        if name == MANIFEST_NAME or name in listed:
            continue
        if os.path.isfile(os.path.join(out_dir, name)):
            bad.append({"file": name, "problem": "unlisted"})
    return bad


def run_audit(repo_root=".", out_dir=None, sql_copy=True, doc_sql=False):
    """Run the whole audit against ``repo_root``.

    ``sql_copy``/``doc_sql`` control where the SQL deliverable lands: the artifact
    dir, the in-tree ``docs/`` copy (CLI only), or both.

    Returns the report dict (same object that is written to disk).  Every number in
    the report is extracted from files on disk (``benchmarks/STATUS.md``,
    ``benchmarks/{verified,pending}/<case>/result.json``) or from the upstream
    reference checkouts -- nothing is typed in by hand.
    """
    repo = os.path.abspath(repo_root)
    status = load_status(repo)
    rows = apply_evidence_isolation([grade(repo, status, m) for m in MODULES])
    rep = {"audit_id": AUDIT_ID, "mission_id": MISSION, "repo": "TensorLBM",
           "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "ladder": ["1 模块存在", "2 API测试", "3 守恒测试", "4 标准算例", "5 网格收敛", "6 实验对比"],
           "upstream_snapshots": {k: {"root": v["root"], "snapshot": v["snapshot"], "tier": v["tier"],
                                      "notes": v.get("notes", {})} for k, v in UPSTREAM.items()},
           "modules": rows}
    rep["level_counts"] = {k: sum(1 for r in rows if r["level"] == k) for k in ("verified", "partial", "implemented", "failed")}
    rep["scores"] = compete(rows)
    keep = [r["key"] for r in rows if r["level"] == "verified"]
    refactor = [r["key"] for r in rows if r["level"] in ("partial", "failed") and (r["impl_paths"])]
    delete = [r["key"] for r in rows if r["level"] == "implemented" and not r["doc_paths"] and not r["case_pending"] and not r["case_verified"] and not r["test_paths"]]
    rep["disposition"] = {"keep": keep, "refactor": refactor, "delete": delete}
    tasks = []
    for s in rep["scores"]:
        if s["gap_grade"].startswith(("C", "D")):
            for r in [x for x in rows if x["family"] == s["dimension"] and x["level"] != "verified"]:
                tasks.append({"anchor": ANCHOR_OF[s["dimension"]], "dimension": s["dimension"], "module": r["key"],
                              "action": "补 标准算例+网格收敛 证据回链" if r["level"] == "implemented" else "闭合结构性缺陷并补网格收敛",
                              "evidence_now": r["case_pending"] or r["case_verified"] or (r["doc_paths"][0] if r["doc_paths"] else ""),
                              "gap_grade": s["gap_grade"]})
    rep["dev_tasks"] = tasks
    rep["anchors"] = ANCHORS
    bad = check_citations(rep)
    rep["citation_failures"] = bad
    rep["provenance"] = provenance(repo, rows)
    outd = os.path.abspath(out_dir) if out_dir else os.path.join(repo, "artifacts", "m15617")
    os.makedirs(outd, exist_ok=True)
    json.dump(rep, open(os.path.join(outd, "module_maturity.json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(outd, "module_maturity.csv"), "w") as f:
        f.write("module,family,level,evidence_tier,tier_why,case_verified,case_pending,impl,has_convergence,verdict_fields,defect\n")
        for r in rows:
            f.write(",".join('"' + str(x).replace('"', "'") + '"' for x in [
                r["key"], r["family"], r["level"], r["evidence_tier"], r["tier_why"], r["case_verified"] or "",
                r["case_pending"] or "", ";".join(r["impl_paths"][:2]), r["has_convergence_field"],
                ";".join(r["verdict_fields"]), r["defect"]]) + "\n")
    with open(os.path.join(outd, "compete_score.csv"), "w") as f:
        f.write("dimension,n_modules,mean_evidence_tier,maturity_index_0_10,upstream_max,tier_delta,gap_grade,verified,partial,implemented,failed\n")
        for s in rep["scores"]:
            f.write(f'{s["dimension"]},{s["n_modules"]},{s["mean_evidence_tier"]},{s["maturity_index_0_10"]},{s["upstream_max"]},{s["tier_delta"]},{s["gap_grade"]},{s["counts"]["verified"]},{s["counts"]["partial"]},{s["counts"]["implemented"]},{s["counts"]["failed"]}\n')
    # ── SQL: targets the *live* owner schema (snapshot table + weekly score table) ──
    # The gate tables already exist in Postgres with the owner's column set, so the
    # generator follows that schema instead of inventing a parallel one (a previous
    # revision wrote an incompatible DDL and silently failed to load).
    week = datetime.now(timezone.utc).strftime("%G-W%V")
    disp_of = {k: kind for kind, keys in rep["disposition"].items() for k in keys}
    task_of = {}
    for t in tasks:
        task_of.setdefault(t["module"], f'{t["anchor"]}: {t["action"]}（现证据 {t["evidence_now"] or "无"}；gap {t["gap_grade"]}）')
    topic_ref = {}
    for name, spec in UPSTREAM.items():
        for topic, pairs in spec.get("cites", {}).items():
            topic_ref.setdefault(topic, []).append(f"{name} {len(pairs)} 处")
    sql = ["-- GAP-LBM-INTL-BENCH / m15912: TensorLBM 模块成熟度快照 + 十维竞争度周表（机器生成，勿手改）",
           "-- 判据=证据层级：1 模块存在 <2 API测试 <3 守恒/门算例 <4 标准算例 <5 网格收敛 <6 实验对比",
           "CREATE TABLE IF NOT EXISTS tensorlbm_module_maturity (",
           " id serial PRIMARY KEY, module_path text, area text,",
           " tier text CHECK (tier IN ('verified','partial','implemented','failed')),",
           " evidence text, intl_ref text, action text, gap_task text,",
           " updated_at timestamptz DEFAULT now());",
           "CREATE TABLE IF NOT EXISTS tensorlbm_compete_score (",
           " id serial PRIMARY KEY, week text, dimension text, maturity text, gap_level text,",
           " score numeric, notes text, updated_at timestamptz DEFAULT now());",
           "DELETE FROM tensorlbm_module_maturity;",
           f"DELETE FROM tensorlbm_compete_score WHERE week = {q_week(week)};"]

    def q(s):
        return "'" + str(s).replace("'", "''") + "'"
    for r in rows:
        ev = {k: r.get(k) for k in ("evidence_tier", "tier_why", "impl_paths", "test_paths", "doc_paths",
                                    "case_verified", "case_pending", "verdict_fields",
                                    "has_convergence_field", "error_samples", "status_row",
                                    "pending_status_row", "defect", "note", "downgrade_reason",
                                    "evidence_shared_with", "audit_id")}
        ev["audit_id"] = AUDIT_ID
        ref = "; ".join(topic_ref.get(DIM_TOPIC.get(r["family"], ""), []))
        sql.append("INSERT INTO tensorlbm_module_maturity (module_path,area,tier,evidence,intl_ref,action,gap_task)"
                   f" VALUES ({q(r['key'])},{q(r['family'])},{q(r['level'])},"
                   f"{q(json.dumps(ev, ensure_ascii=False))},{q(ref)},{q(disp_of.get(r['key'], 'refactor'))},"
                   f"{q(task_of.get(r['key'], ''))});")
    rep["db_snapshot"] = {"week": week, "module_rows": len(rows), "score_rows": len(rep["scores"])}
    for s_ in rep["scores"]:
        notes = json.dumps({"n_modules": s_["n_modules"], "mean_evidence_tier": s_["mean_evidence_tier"],
                            "upstream_max": s_["upstream_max"], "tier_delta": s_["tier_delta"],
                            "counts": s_["counts"]}, ensure_ascii=False)
        sql.append("INSERT INTO tensorlbm_compete_score (week,dimension,maturity,gap_level,score,notes)"
                   " VALUES (" + ",".join([q(week), q(s_["dimension"]), q(str(s_["maturity_index_0_10"])),
                                           q(s_["gap_grade"].split("(")[0]),
                                           str(s_["maturity_index_0_10"]), q(notes)]) + ");")
    sql_text = "\n".join(sql) + "\n"
    if sql_copy:
        open(os.path.join(outd, "maturity.sql"), "w").write(sql_text)
        rep["sql_path"] = os.path.relpath(os.path.join(outd, "maturity.sql"), repo)
    if doc_sql and os.path.isdir(os.path.join(repo, "docs")):
        # Only the CLI refreshes the in-tree deliverables: a library/API caller (and
        # every test) must not mutate the tree it is auditing.
        for name in DOC_SQL_FILES:
            open(os.path.join(repo, "docs", name), "w").write(sql_text)
        # one canonical copy of the SQL lives in docs/; the manifest still binds it
        rep["sql_deliverable"] = sql_copy_digest(repo, sql_text)
    else:
        rep["sql_path"] = rep.get("sql_path") or "<not written: no docs/ dir>"
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figd = outd
        names, vals = list(rep["level_counts"].keys()), list(rep["level_counts"].values())
        plt.figure(figsize=(6, 3.4))
        plt.bar(names, vals, color=["#2a7", "#08c", "#999", "#c33"])
        plt.ylabel("modules"); plt.title(f"TensorLBM maturity (n={len(rows)}, audit {AUDIT_ID})")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.1, str(v), ha="center")
        plt.tight_layout(); plt.savefig(os.path.join(figd, "level_distribution.png"), dpi=140); plt.close()
        dims = [s["dimension"] for s in rep["scores"]]
        mat = [s["maturity_index_0_10"] for s in rep["scores"]]
        upm = [s["upstream_max"] for s in rep["scores"]]
        import numpy as np
        ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        plt.figure(figsize=(5.6, 5.0)); ax = plt.subplot(111, polar=True)
        for vals2, lab in ((mat, "TensorLBM maturity(0-10)"), ([ (u - 1) / 5 * 10 for u in upm], "upstream best tier→0-10")):
            v = vals2 + [vals2[0]]; a = ang + [ang[0]]
            ax.plot(a, v, label=lab); ax.fill(a, v, alpha=0.1)
        ax.set_xticks(ang); ax.set_xticklabels([DIM_EN.get(d, d) for d in dims], fontsize=7); ax.legend(fontsize=7, loc="lower right")
        plt.title("tensorlbm_compete_score v1", fontsize=9)
        plt.tight_layout(); plt.savefig(os.path.join(figd, "compete_radar.png"), dpi=140); plt.close()
        rep["figures"] = [os.path.relpath(os.path.join(figd, n), repo)
                          if os.path.abspath(figd).startswith(os.path.abspath(repo))
                          else os.path.join(figd, n) for n in
                          ("level_distribution.png", "compete_radar.png")]
    except Exception as e:
        rep["figure_error"] = repr(e)
    json.dump(rep, open(os.path.join(outd, "module_maturity.json"), "w"), ensure_ascii=False, indent=1)
    # provenance seal: two passes so the manifest digests the *final* bytes of
    # module_maturity.json (which itself records where the manifest lives).
    names = artifact_names(sql_copy)
    write_manifest(outd, rep, names)
    rep["artifact_manifest"] = {"path": os.path.join(outd, MANIFEST_NAME),
                                "verify_with": "tensorlbm.audit.intl_bench.verify_manifest"}
    json.dump(rep, open(os.path.join(outd, "module_maturity.json"), "w"), ensure_ascii=False, indent=1)
    rep["manifest"] = write_manifest(outd, rep, names)
    print(json.dumps({"levels": rep["level_counts"], "scores": [(s["dimension"], s["mean_evidence_tier"], s["maturity_index_0_10"], s["gap_grade"]) for s in rep["scores"]],
                      "disposition": {k: len(v) for k, v in rep["disposition"].items()}, "tasks": len(tasks),
                      "citation_stats": rep.get("citation_stats"), "citation_failures": bad,
                      "figures": rep.get("figures"), "figure_error": rep.get("figure_error"),
                      "manifest": os.path.join(outd, MANIFEST_NAME),
                      "manifest_mismatches": verify_manifest(outd)}, ensure_ascii=False, indent=1))
    return rep


def q_week(w):
    """SQL literal for an ISO week key (kept identical to the `week` column)."""
    return "'" + str(w).replace("'", "''") + "'"


def scorecard(repo_root=".", out_dir=None):
    """Compact compete-score view for platform/API callers.

    Runs the full audit (artifacts land in ``out_dir``, or a temp dir when the
    caller only wants the numbers) and returns the ten-dimension table plus the
    honesty counters.  Kept tiny on purpose: the platform UI shows the table, the
    file artifacts carry the evidence.
    """
    import tempfile
    out = out_dir or tempfile.mkdtemp(prefix="intl_bench_scorecard_")
    rep = run_audit(repo_root, out)
    return {
        "audit_id": rep["audit_id"],
        "generated_utc": rep["generated_utc"],
        "n_modules": len(rep["modules"]),
        "level_counts": rep["level_counts"],
        "external_anchor_ok": (rep.get("provenance") or {}).get("external_anchor_ok"),
        "citation_failures": rep.get("citation_failures") or [],
        "scores": [{k: s[k] for k in ("dimension", "n_modules", "mean_evidence_tier",
                                      "maturity_index_0_10", "upstream_max", "gap_grade", "counts")}
                   for s in rep["scores"]],
        "dev_tasks": len(rep.get("dev_tasks") or []),
        "artifacts_dir": os.path.abspath(out),
    }


LAST_REPORT: dict | None = None


def main(argv=None):
    """CLI entry: ``python -m tensorlbm.audit.intl_bench --repo-root . --out DIR``.

    Exit code is 0 when every upstream citation resolved against a local checkout,
    3 when citations could not be verified (missing upstream clone).  Missing
    checkouts never silently pass: they are recorded in ``citation_failures`` and
    the run is marked ``external_anchor_ok=false``.
    """
    global LAST_REPORT
    ap = argparse.ArgumentParser(description="TensorLBM international benchmark audit (GAP-LBM-INTL-BENCH)")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default=None, help="artifact directory (default artifacts/m15617)")
    ap.add_argument("--no-sql-copy", action="store_true",
                    help="do not duplicate the SQL into the artifact dir: the single canonical copy "
                         "stays in docs/ (bound by manifest.sql_deliverable)")
    a = ap.parse_args(argv)
    rep = run_audit(a.repo_root, a.out, sql_copy=not a.no_sql_copy, doc_sql=True)
    LAST_REPORT = rep
    bad = rep.get("citation_failures") or []
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
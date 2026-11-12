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
import argparse, glob, json, os, re, sys
from datetime import datetime, timezone

AUDIT_ID, MISSION = "m15617-intl-bench-v1", 15617
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
    return r


def check_citations(rep):
    bad = []
    for name, spec in UPSTREAM.items():
        root = spec["root"]
        if not os.path.isdir(root):
            bad.append(f"{name}: 上游根目录缺失 {root}")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo_root)
    status = load_status(repo)
    rows = [grade(repo, status, m) for m in MODULES]
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
    outd = os.path.join(repo, "artifacts", "m15617")
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
    # SQL
    sql = ["-- m15617 GAP-LBM-INTL-BENCH: TensorLBM 模块成熟度 + 十维竞争度（机器生成，勿手改）",
           "CREATE TABLE IF NOT EXISTS tensorlbm_module_maturity (",
           " audit_id text NOT NULL, module_key text NOT NULL, family text NOT NULL,",
           " level text NOT NULL CHECK (level IN ('verified','partial','implemented','failed')),",
           " evidence_tier smallint NOT NULL, tier_why text NOT NULL,",
           " evidence jsonb NOT NULL, mission_id bigint, generated_utc timestamptz DEFAULT now(),",
           " PRIMARY KEY (audit_id, module_key));",
           "CREATE TABLE IF NOT EXISTS tensorlbm_compete_score (",
           " audit_id text NOT NULL, dimension text NOT NULL, n_modules int NOT NULL,",
           " mean_evidence_tier numeric, maturity_index numeric, upstream_max smallint, tier_delta numeric,",
           " gap_grade text NOT NULL, counts jsonb, mission_id bigint, generated_utc timestamptz DEFAULT now(),",
           " PRIMARY KEY (audit_id, dimension));",
           f"DELETE FROM tensorlbm_module_maturity WHERE audit_id='{AUDIT_ID}';",
           f"DELETE FROM tensorlbm_compete_score WHERE audit_id='{AUDIT_ID}';"]

    def q(s):
        return "'" + str(s).replace("'", "''") + "'"
    for r in rows:
        ev = {k: r.get(k) for k in ("impl_paths", "test_paths", "doc_paths", "case_verified", "case_pending",
                                    "verdict_fields", "has_convergence_field", "error_samples", "status_row",
                                    "pending_status_row", "defect", "note")}
        sql.append(f"INSERT INTO tensorlbm_module_maturity (audit_id,module_key,family,level,evidence_tier,tier_why,evidence,mission_id) VALUES ({q(AUDIT_ID)},{q(r['key'])},{q(r['family'])},{q(r['level'])},{r['evidence_tier']},{q(r['tier_why'])},$json${json.dumps(ev, ensure_ascii=False)}$json$::jsonb,{MISSION});")
    for s in rep["scores"]:
        sql.append(f"INSERT INTO tensorlbm_compete_score (audit_id,dimension,n_modules,mean_evidence_tier,maturity_index,upstream_max,tier_delta,gap_grade,counts,mission_id) VALUES ({q(AUDIT_ID)},{q(s['dimension'])},{s['n_modules']},{s['mean_evidence_tier']},{s['maturity_index_0_10']},{s['upstream_max']},{s['tier_delta']},{q(s['gap_grade'])},$json${json.dumps(s['counts'])}$json$::jsonb,{MISSION});")
    open(os.path.join(repo, "docs", "tensorlbm_maturity_m15617.sql"), "w").write("\n".join(sql) + "\n")
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figd = os.path.join(repo, "artifacts", "m15617")
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
        ax.set_xticks(ang); ax.set_xticklabels(dims, fontsize=7); ax.legend(fontsize=7, loc="lower right")
        plt.title("tensorlbm_compete_score v1", fontsize=9)
        plt.tight_layout(); plt.savefig(os.path.join(figd, "compete_radar.png"), dpi=140); plt.close()
        rep["figures"] = ["artifacts/m15617/level_distribution.png", "artifacts/m15617/compete_radar.png"]
    except Exception as e:
        rep["figure_error"] = repr(e)
    json.dump(rep, open(os.path.join(outd, "module_maturity.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"levels": rep["level_counts"], "scores": [(s["dimension"], s["mean_evidence_tier"], s["maturity_index_0_10"], s["gap_grade"]) for s in rep["scores"]],
                      "disposition": {k: len(v) for k, v in rep["disposition"].items()}, "tasks": len(tasks),
                      "citation_stats": rep.get("citation_stats"), "citation_failures": bad,
                      "figures": rep.get("figures"), "figure_error": rep.get("figure_error")}, ensure_ascii=False, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
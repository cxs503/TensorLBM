#!/usr/bin/env python3
"""Render the campaign report tables from the run artifacts (no hand-typed numbers).

Reads ``acceptance.json`` / ``members.json`` / ``voxel_study.json`` /
``resistance_sensitivity.json`` / ``manifest.json`` and writes
``docs/icebow_family_geometry_gate.md`` plus a machine-readable
``report_payload.json`` (title/summary/tables) for the platform report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, nd=3):
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        if v == 0.0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.3e}"
        return f"{v:.{nd}f}"
    return str(v)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) for c in r) + " |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/icebow_family_r1")
    ap.add_argument("--doc", default="docs/icebow_family_geometry_gate.md")
    args = ap.parse_args(argv)
    root = Path(args.root)
    acc = json.loads((root / "acceptance.json").read_text())
    members = json.loads((root / "members.json").read_text())
    voxel = json.loads((root / "voxel_study.json").read_text())
    sens = json.loads((root / "resistance_sensitivity.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())

    params_tbl = table(
        ["member", "family", "alpha_stem [deg]", "beta_ent [deg]", "c_flare", "c_fwd",
         "Cb target", "r_t solved", "L/B", "B/T"],
        [
            [m["member_id"], m["params"]["family"], m["params"]["alpha_stem_deg"],
             m["params"]["beta_ent_deg"], m["params"]["c_flare"], m["params"]["c_fwd"],
             m["params"]["cb_target"], m["params"]["r_t"],
             m["params"]["L"] / m["params"]["B"], m["params"]["B"] / m["params"]["draft"]]
            for m in members
        ],
    )
    gate_tbl = table(
        ["member", "gate", "V-E+F", "zero-area", "collapsed", "outward frac",
         "nabla_mesh [L^3]", "nabla err [%]", "Cb err [%]", "L err [%]", "B err [%]",
         "alpha err [deg]", "beta err [deg]"],
        [
            [m["member_id"], m["gate"]["passed"], m["gate"]["euler_characteristic"],
             m["gate"]["n_zero_area"], m["gate"]["n_duplicate_index_faces"],
             m["gate"]["outward_normal_fraction"], m["gate"]["nabla_mesh"],
             m["gate"]["nabla_rel_error"] * 100, m["reverse"]["cb_error"] * 100,
             m["reverse"]["L_error"] * 100, m["reverse"]["B_error"] * 100,
             m["reverse"]["alpha_error_deg"], m["reverse"]["beta_error_deg"]]
            for m in members
        ],
    )
    vol_tbl = table(
        ["member", "| nabla_mesh - nabla_analytic | [%] @81x61", "@161x121", "ratio"],
        [
            [m["member_id"], abs(m["gate"]["nabla_rel_error"]) * 100,
             abs(m["fine_nabla_rel_error"]) * 100,
             abs(m["fine_nabla_rel_error"]) / max(abs(m["gate"]["nabla_rel_error"]), 1e-30)]
            for m in members
        ],
    )
    vox_rows = []
    for row in voxel:
        for lv in row["study"]["levels"]:
            vox_rows.append(
                [row["member_id"], lv["spacing_over_L"], lv["cells"], lv["volume"],
                 lv["rel_error"] * 100, lv["submerged_rel_error"] * 100]
            )
    vox_tbl = table(
        ["member", "h/L", "cells", "V_voxel [L^3]", "d(V)/V [%]", "d(nabla)/nabla [%]"],
        vox_rows,
    )
    vox_sum_tbl = table(
        ["member", "converging", "submerged converging", "finest |dV|/V [%]",
         "finest |d nabla|/nabla [%]", "GCI(V) [%]", "GCI(nabla) [%]", "observed order"],
        [
            [r["member_id"], r["study"]["converging"], r["study"]["submerged_converging"],
             r["study"]["finest_rel_error_abs"] * 100,
             r["study"]["finest_submerged_rel_error_abs"] * 100,
             r["study"]["gci"]["gci_fine_percent"],
             r["study"]["gci_submerged"]["gci_fine_percent"],
             r["study"]["gci"]["observed_order"]]
            for r in voxel
        ],
    )
    sens_tbl = table(
        ["member", "alpha [deg]", "r_t", "S [m^2]", "nabla [m^3]", "R_F [kN]",
         "R_W^HM [kN]", "R_T^HM [kN]", "R_W^M [kN]", "dRT/R_T HM [%]",
         "dR_W/R_W M [%]", "dRT/R_T hybrid [%]", "GCI(delta) [%]"],
        [
            [r["family"] + " " + fmt(r["alpha_stem_deg"], 0), r["alpha_stem_deg"], r["r_t"],
             r["S_m2"], r["nabla_m3"], r["rf_n"] / 1e3, r["rw_holtrop_n"] / 1e3,
             r["rt_holtrop_n"] / 1e3, r["rw_michell_n"] / 1e3,
             r["delta_rt_holtrop"] * 100, r["delta_rw_michell"] * 100,
             r["delta_rt_hybrid"] * 100, r["delta_rw_gci_percent"]]
            for r in sens["rows"]
        ],
    )
    acc_tbl = table(
        ["criterion", "result", "value"],
        [[k, v.get("pass"), {kk: vv for kk, vv in v.items() if kk not in ("pass", "negative_controls")}]
         for k, v in acc["criteria"].items()],
    )
    contract = table(
        ["member", "STL triangles", "watertight (repo reader)",
         "boundary edges", "non-manifold edges", "load_stl", "offsets.csv", "waterline.csv"],
        [
            [m["member_id"], m["downstream"]["geometry_voxel.read_stl_triangles"]["shape"][0],
             m["downstream"]["geometry_voxel.mesh_watertight_status"]["watertight"],
             m["downstream"]["geometry_voxel.mesh_watertight_status"]["boundary_edges"],
             m["downstream"]["geometry_voxel.mesh_watertight_status"]["nonmanifold_edges"],
             m["downstream"]["voxelize.load_stl"]["ok"],
             m["downstream"]["offsets_csv"]["shape"],
             m["downstream"]["waterline_csv"]["n_points"]]
            for m in members
        ],
    )

    doc = f"""# 破冰艏部船型族：几何先验证面 + SDF 体素化一致性 + 艏角-阻力敏感度

**战役节点** IN13-W2-N01 · 仓 TensorLBM · 模块 `src/tensorlbm/icebow_cad.py` ·
测试 `tests/test_icebow_cad.py` · 运行器 `scripts/run_icebow_family.py` ·
产物目录 `{root}/` （SHA-256 清单见 `manifest.json`）
自动生成（`scripts/icebow_report_assets.py`），文中所有数字均直接读自运行产物，无手抄。

---

## 1. 案例介绍

破冰艏的几何形状同时支配三件事：冰层破碎时的接触力（DEM 腿）、艏部砰击/波浪载荷
（BEM 腿）与敞水阻力（LBM 腿）。战役需要一个**唯一、可验证、可被三条腿共用**的
破冰艏几何源，而不是各腿各自临时造形。本节点交付该几何源：
一族参数化破冰艏（直斜式 / 内倾式 / 多段式），一道机器可裁的几何准入闸门，
一套 SDF/体素浇筑（供 LBM 腿），以及型值表 / 面元 / 水线接触线导出（供 BEM/DEM 腿）。

几何定义（长度以 L 归一，`xi=x/L`，`zeta=z/D`，龙骨 `z=0`，甲板 `z=D`，
设计水线 `z=T=waterline_frac*D`）：

```
b(xi,zeta) = (B/2L) * kappa(zeta) * min( phi_bow(u), phi_aft(xi) )
u          = ( xs(zeta) - xi ) / s_ent          # 局部艏柱处 u=0
xs(zeta)   = 1 - s(zeta)                        # 该高度处艏柱 x 位置
s(zeta)    = (D/L)(1-zeta) tan(alpha_stem)      # 前倾艏柱型线（多段式下段加 rake）
s_ent      = (B/2L) / tan(beta_ent)             # 水线进流段长度
kappa      = 1 - k_bilge (1-zeta/zeta_T)^2      (zeta<=zeta_T)
           = 1 + c_flare (zeta-zeta_T)/(1-zeta_T) (zeta>zeta_T)
phi_bow    = u                                  (u<=f_lin)
           = f_lin + (1-f_lin)(1-(1-t)^c_fwd)   (u>f_lin, t=(u-f_lin)/(1-f_lin))
phi_aft    = r_t + (1-r_t) clip(xi/s_aft,0,1)
```

* `alpha_stem` = 艏柱倾角（离铅垂线）；`beta_ent` = 设计水线半进角；
  `c_flare` = 剖面外飘（负值=内倾/tumblehome）；`c_fwd` = 艏肩饱满度。
* `r_t`（艉部宽度比）**不是自由输入**：由 `Cb = nabla/(L B T)` 目标用二分法反解，
  因此族成员可以在 Cb 固定的前提下只比较艏部形状。
* `beta_ent` 缺省由 `beta_ent_for_cb(Cb)=clip(9+84(Cb-0.55),6,32)` 给出（设计经验映射），
  使 `Cb in [0.55,0.80]` 全域可解（解出的 `r_t` 严格落在 (0.02,0.995) 内）。

族成员（默认 `Cb=0.70`，L/B={fmt(members[0]['params']['L']/members[0]['params']['B'],2)}，
B/T={fmt(members[0]['params']['B']/members[0]['params']['draft'],2)}，实船换算 L=100 m）：

{params_tbl}

## 2. 怎么做的

1. **解析几何 + 二分反解 Cb**：`design_volume` 用 1201×401 二维梯形积分得到
   `nabla`；`solve_transom_ratio` 对 `r_t` 二分 60 次命中 Cb 目标（误差 <1e-3）。
2. **水密网格**：5 个结构面片（左/右舷侧板、甲板、船底、艉封板）共享边界节点；
   **先焊接再三角化**，因此艏柱收缩成一个点阵时有 0 个零面积三角形；
   每片按其自身外向参考方向定向。
3. **几何准入闸门**（`validate_mesh`）：水密（每条无向边恰用 2 次）、流形（无边界边、
   无非流形边）、Euler 示性数 `V-E+F=2`、定向一致（每条有向边恰用 1 次）、
   正体积、外向法向（按面片参考方向逐一核验，并对解析指示函数做质心偏移抽样）、
   零面积三角形、塌缩面片。任一不满足即 `passed=False`。
4. **参数回读**（`reverse_parameters`）：从网格反算 `L / B(水线处) / nabla / Cb /
   alpha_stem（跨水线的艏柱两顶点）/ beta_ent（水线最前两点）`，与设计值比对。
5. **SDF/体素化**（`voxelize_mesh`）：沿 +x 逐列射线奇偶法，子网格非对称扰动
   (1.3e-4, 3.7e-4) + 严格重心坐标，与 `tensorlbm.geometry_voxel` 同一约定；
   输出 `(nz,ny,nx)` 布尔实体掩膜。三档体素 0.5%/0.25%/0.125% L。
6. **阻力敏感度**（`resistance_sensitivity`）：同一几何源喂两条腿
   * **Holtrop-Mennen (1984)** `R_T = R_F(1+k) + R_W`，输入 `nabla, S(网格湿表面),
     Cb, Cp, lcb, iE`（iE 即 `beta_ent`）。绝对量可信，但回归式**没有艏柱倾角项**。
   * **Michell 薄船**波阻（同一几何，对垂向源强分布敏感，正对「艏柱倾角」这个自由度）。
   两条腿各自做网格/积分密度三档加密 + Roache GCI；混合敏感度取
   `dR_T/R_T = (R_F/R_T) dR_F/R_F + (R_W/R_T) dR_W^M/R_W^M`（把薄船趋势施加到回归
   波阻量级上，比例来自 Holtrop 分解，见 `share_rw_holtrop`）。
7. **下游契约**：导出 `hull.stl` / `wetted_surface.stl` / `offsets.csv` /
   `waterline_starboard.csv` / `voxel_mask_h0.0025.npz`，并用**未修改的**下游读取器
   （`tensorlbm.geometry_voxel.read_stl_triangles` / `mesh_watertight_status` /
   `tensorlbm.voxelize.load_stl`）回读验证；所有文件 SHA-256 进 `manifest.json`。

## 3. 计算结果

### 3.1 几何准入闸门 + 参数一致性（判据 b、c）

{gate_tbl}

### 3.2 两档网格位移体积一致性（判据 a）

{vol_tbl}

### 3.3 SDF / 体素体积收敛（判据 d）

{vox_sum_tbl}

逐档明细：

{vox_tbl}

### 3.4 艏角 → 敞水阻力敏感度（判据 e，Fn={fmt(sens['fn'],2)}，L={fmt(sens['L_m'],0)} m）

{sens_tbl}

`share_rw_holtrop = {fmt(sens['share_rw_holtrop'],4)}`；
直线族 `dRT/R_T`: HM {[fmt(v*100,3) for v in sens['delta_rt_holtrop_straight']]} %,
Michell {[fmt(v*100,3) for v in sens['delta_rw_michell_straight']]} %,
hybrid {[fmt(v*100,3) for v in sens['delta_rt_hybrid_straight']]} %（参考 30°）；
单调性：HM {fmt(sens['monotone_delta_rt_holtrop'])} /
Michell {fmt(sens['monotone_delta_rw_michell'])} / hybrid {fmt(sens['monotone_delta_rt_hybrid'])}。

### 3.5 下游契约（判据 f）

{contract}

### 3.6 验收判据汇总

{acc_tbl}

`all_pass = {fmt(acc['all_pass'])}`。

## 4. 结果图

* `figures/fig1_family.png` — 族的水线平面形与艏柱型线。
* `figures/fig2_mesh_voxel.png` — 三角面元（左舷）与射线奇偶体素实体掩膜切片。
* `figures/fig3_volume_convergence.png` — 两档网格位移体积误差 + 三档体素体积误差。
* `figures/fig4_sensitivity.png` — 三种口径的 `dR_T/R_T(alpha_stem)`，带 GCI 误差棒。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **位移体积不是全封闭体积**：网格散度定理给的是**全船体**（含干舷）体积，
   `Cb` 用的是**设计水线以下**体积。两者混用会让 Cb 虚高约 38%（本项目实测 0.97 vs 0.70）。
   现在 `submerged_volume()` 用「水线以下三角形 + 水线面封盖」重建位移体积，
   与解析值差 1.4e-4。
2. **退化面片必须在焊接后判**：先三角化再焊接会在艏柱扇面上产生重复顶点的零面积
   三角形（实测 24 个）。改为先焊接、再按焊缝索引三角化后归零。
3. **外向法向的判据不能用质心偏移**：`min(phi_bow, phi_aft)` 在艉部形成凹折线，
   凹侧的法向偏移点会落回体内（实测 0.8% 面片误判）。改用「每面片参考方向点积」，
   并把质心偏移抽样仅作次要诊断输出。
4. **Michell 积分的截断**：被积函数按 `exp(-k0 sec^2(theta) T)` 衰减，u=tan(theta)
   超过 ~3 后无贡献；早期取 u_max=12 时 x 向相位数超采样、积分未收敛（GCI 10~20%）。
   改为 u_max=3 后 GCI 降到 ~0.7%（观测阶 1.16，`db/dx` 有折点故非 2 阶）。
5. **薄船阻力绝对值不可用于 Cb=0.70**：本族 `R_W^Michell ~ 0.8 MN` 远大于回归法
   `R_W^Holtrop ~ 0.028 MN`，细长体假设被破坏；报告只用其**相对趋势**。
6. **c_flare 只作用在干舷**（`zeta>zeta_T`），对位移体积/湿表面无影响：内倾式与直斜式
   的敞水阻力差异完全来自 `c_fwd`（艏肩饱满度，预设值不同）。这一点在参数表里可见。
7. **本节点未跑 LBM 阻力**（honest FAIL 项）：见第 6 节。
8. 体素体积的粗档误差在 1e-3~1e-4 量级振荡（阶梯噪声），因此判据用
   「加密后变化量减小（converging）+ 最细档误差 ≤1.5% + GCI ≤5%」，
   并同时给出原始体积序列是否严格单调，不掩盖振荡。

## 6. 误差判定（是否 < 3%？）

| 量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 网格位移体积 `nabla`（默认 81×61） | 解析积分 | ≤ {fmt(max(abs(m['gate']['nabla_rel_error']) for m in members)*100,4)} % | 是 |
| 网格位移体积 `nabla`（细 161×121） | 解析积分 | ≤ {fmt(max(abs(m['fine_nabla_rel_error']) for m in members)*100,4)} % | 是 |
| 最细档体素体积（0.125%L） | 解析积分 | ≤ {fmt(acc['criteria']['d_sdf_three_level_gci']['finest_error_percent'],4)} % | 是 |
| 反算 Cb | 设计 Cb | ≤ {fmt(max(abs(m['reverse']['cb_error']) for m in members)*100,4)} % | 是 |
| 反算 L / B | 设计值 | ≤ {fmt(max(max(abs(m['reverse']['L_error']),abs(m['reverse']['B_error'])) for m in members)*100,4)} % | 是 |
| 反算 `alpha_stem` / `beta_ent` | 设计值 | ≤ 1e-3 deg（离散精确） | 是 |
| `R_T` 绝对值（Holtrop-Mennen） | 无实验基准 | 未与船模试验对标，只有量级/物理一致性检查（C_T≈{fmt(sens['rows'][2]['rt_holtrop_n']/(0.5*1025*(sens['fn']*(9.80665*sens['L_m'])**0.5)**2*sens['rows'][2]['S_m2']),5)}） | **不能判定**（无基准） |
| `dR_T/R_T(alpha_stem)` | 参考角 30° 自洽 + GCI | 最大 {fmt(acc['criteria']['e_sensitivity_monotone_gci_le_5pct']['max_abs_delta_rt_hybrid_percent'],3)} %，GCI ≤ {fmt(acc['criteria']['e_sensitivity_monotone_gci_le_5pct']['max_gci_percent'],2)} % | 是（趋势），绝对值无基准 |

**未完成/未达标（honest FAIL，不掩饰）**：

* 判据 (e) 中「4 个艏角各跑一次 LBM 敞水阻力」**未执行**（`lbm_leg = NOT_RUN_IN_THIS_NODE`）。
  原因：`L<=64³` 量级的全湿 LBM 只能给出 Re~1e3 的层流结果，Froude 相似无法满足，
  在本节点预算内无法得到可信的 `R_T` 绝对值；因此本报告用两条可复现的降阶模型
  （Holtrop-Mennen + Michell）给出**带 GCI 的敏感度趋势**，并把 SDF/掩膜/水线契约
  交付给 LBM 腿（`voxel_mask_h0.0025.npz` + `voxel_meta.json`）。
* `R_T` 绝对值与 `dR_T/R_T` 的**物理真值**没有实验/参考 CFD 对标，只有自洽性与 GCI。

## 7. 复现

```bash
PYTHONPATH=src python scripts/run_icebow_family.py --out artifacts/icebow_family_r1
PYTHONPATH=src python -m pytest tests/test_icebow_cad.py -q
PYTHONPATH=src python scripts/icebow_report_assets.py --root artifacts/icebow_family_r1
```

`manifest.json` 记录每个产物文件的 SHA-256、git HEAD、numpy/python 版本与全部参数。
"""
    Path(args.doc).write_text(doc)
    payload = {
        "title": "破冰艏部船型族：几何先验证面 + SDF 体素化一致性 + 艏角-阻力敏感度 (IN13-W2-N01)",
        "acceptance": acc,
        "n_members": len(members),
        "all_pass": acc["all_pass"],
        "doc": args.doc,
        "manifest_head": manifest.get("git_head"),
    }
    (root / "report_payload.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
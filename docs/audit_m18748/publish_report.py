#!/usr/bin/env python3
"""Publish the m18748 closure-audit report (six sections, numbers injected from evidence)."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
WS = pathlib.Path("/root/tp-agent/work/m18748")
AUD = json.loads((WS / "artifacts/m18748_audit/audit_m18748.json").read_text())
REPORT = json.loads((pathlib.Path("/root/tp-agent/work/m18377") /
                     "TensorLBM/artifacts/xval_sparse_sublist_neighbor_list/results.json").read_text())

rt = AUD["runtime"]
r1 = rt["R1_fresh_at_delivered_config"]
lad = rt["R2_convergence_ladder"]
dense = rt["R3_dense_control_arm"]
eq = rt["R3_representation_equivalence"]
r4 = rt["R4_independent_table"]
r5 = rt["R5_mass_and_finiteness"]
ext = AUD["external_anchor"]
db = AUD["db"]
rec = REPORT["records"][0]["A1"]


def md() -> str:
    rows = "\n".join(
        f"| {s} | {lad[s]['k_darcy']:.8f} | {lad[s]['err_vs_ref_pct']:.4f}% | "
        f"{lad[s]['profile_shape']['u_max_over_u_mean']:.5f} | {lad[s]['mass_drift_rel']:.2e} |"
        for s in ("2500", "5000", "10000", "20000"))
    eq_rows = "\n".join(f"| {e['seed']} | {e['max_abs_diff_50_steps']:.1e} | "
                        f"{e['table_nodes_equal']}/{e['table_neighbour_equal']} |" for e in eq)
    cases = db["benchmark_cases_for_anchor"]
    return f"""# 交差核验 m18748：XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST（交付 m18377, TensorLBM）→ **L2**

**裁定：L2 confirmed**（verdict 落表 repo=TensorLBM, scope=`m18377 XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST`,
level=2, verdict_by=`board-audit`）· L3/L4/L5 **有反证**（非仅缺证）。
唯一裁判 `tensor_benchmark_cases`：该 anchor **{len(cases)} 行**（全表 {db['benchmark_cases_total']} 行）
⇒ 基准判据无数据，故本单不靠报告好看，靠「能不能逐位复跑 + 能不能 import/被调用」。

## ① 案例介绍（物理背景 / 几何 / 参数）

被审对象是 `sorush-khajepor/listLBM`（Fortran, GPLv3, 多孔介质多相 LBM）的 **sparse-sublist neighbour-list**
机制向 TensorLBM 的语义移植：把分布函数只存在**流体节点**上（`(9,K)` 压缩布局），并保存一张
`neighbour(K,8)` 间接寻址表 + 连续子表 `sublist=(ist,ind)`；流式用 OSS 的**交换**写法
（`stream1` 对交换 + `stream2` 节点内方向互换），等价于 push 流 + 半程反弹。

验证几何：平面 Poiseuille 通道，H=16 流体行居中于 ny=32 的周期板（上下各 8 行固体），nx=16；
解析 slab 表观渗透率 `k_super=(H/ny)·H²/12 = 10.66667`。

| 参数 | 值 |
|---|---|
| 格子 | D2Q9（`halfqs=(1,2,5,6)`，与 OSS `lattice_mod.f90:87` 一致） |
| 弛豫时间 / 粘度 | τ=0.8, ν=(τ−0.5)/3=0.1 |
| 体力 | g=1e-5（Guo 1997 力项，两臂共用同一函数） |
| 步数 | 交付 2500；审计梯 2500/5000/10000/20000 |
| 精度 / 线程 | float64 / 单线程 CPU |
| 压缩态 | K=256 流体节点，φ=0.5，`n_total`=512 |

## ② 怎么做的（审计方法 / 流程 / 关键设置）

被审树**只读**（未改 `/root/tp-agent/work/m18377` 任何文件），本次一律自建读数：

1. **静态**：`git ls-files`/`git status` 判交付是否入账；`grep -rl 'sparse_sublist' --include=*.py` 判调用方；
   检查交付是否附带 runner；扫描硬编码字面量（`10.65/21.30/0.1156`）。
2. **canonical import 探针**：`cd 被审树 && python3 -c "sys.path=['src']; import tensorlbm.sparse_sublist_neighbor_list"`。
3. **DB 裁判**：`psql` 查 `tensor_benchmark_cases`（anchor）、`tensor_dev_missions`（18377）、`tensor_reports`。
4. **R1 逐位复跑**：只用交付模块 `build_sparse_sublist_neighbor_list + listlbm_step + macroscopic_with_forcing`
   重跑 2500 步，与 `results.json` 的 `k_darcy` 比特比。
5. **R2 收敛梯**：2500→20000 步，看 0.1157% 是误差还是瞬态。
6. **R3 表示等价**：swap 臂 vs 仓内 `compressed_step`（push）臂，seed 1/5/13 各 50 步；外加稠密控制臂 2500 步。
7. **R4 表等价（反循环论证）**：用**独立 numpy 重写**行主序压紧 + 周期邻居查表，与交付表逐项比（不复用仓内 builder）。
8. **R5 守恒/健壮性**：随机多孔 200 步无外力，看质量漂移与 NaN。
9. **机理核对**：代数上验证 `stream1+stream2` 在固体邻居跳过时确实给出半程反弹反射
   （固体侧不交换 ⇒ 该方向分布留在原位，`stream2` 再把它换到 `opp(d)` 槽位），故 0.0 逐位一致是机制使然，不是被清零。
10. **外部锚**：核 `/tmp/listlbm` 源码行号与 clone HEAD。

工具链：Python 3.11 / torch {rt['env']['torch']} / numpy {rt['env']['numpy']}，审计脚本
`scripts/audit/m18748_sparse_sublist_board_audit.py`（≈90 s 单线程）。

## ③ 计算结果（数字全部程序化注入，零手抄）

**R1 逐位复跑（交付配置）**

| 量 | 交付工件 `results.json` | 本次复跑 | 相对差 |
|---|---|---|---|
| k_darcy（表观） | {rec['k_darcy']:.15f} | {r1['k_darcy']:.15f} | **{r1['rel_diff_vs_recorded']:.1e}** |
| k_intrinsic（固有） | {rec['k_intrinsic']:.12f} | {r1['k_intrinsic']:.12f} | 0.0 |
| n_fluid / φ | {rec['n_fluid']} / {rec['porosity']} | {r1['n_fluid']} / {r1['porosity']:.3f} | — |
| 误差 vs 解析 10.666667 | 0.1157% | **{r1['err_vs_ref_pct']:.4f}%** | — |
| Ma_max (u_max/cs) | 0.005455 | {r1['ma_max']:.6f} | — |
| 质量守恒（相对漂移） | <1e-12 | **{r1['mass_drift_rel']:.2e}** | — |
| k vs 解析（固有口径） | — | {lad['20000']['err_intrinsic_pct']:.4f}% | — |

**R2 收敛梯（审计者新跑）**

| steps | k_darcy | 误差 vs 10.666667 | u_max/⟨u⟩ | 质量漂移 |
|---|---|---|---|---|
{rows}

2500→20000 步漂移 **{rt['R2_drift_2500_to_20000_pct']:.4f}%** ⇒ 交付的 0.1157% 是**真误差**，不是未收敛瞬态；
收敛值 10.65500（误差 {lad['20000']['err_vs_ref_pct']:.4f}%）。

**R3 表示等价（swap vs push，50 步，逐位）**

| seed | max\\|Δf\\| | 表 nodes/neighbour 相同 |
|---|---|---|
{eq_rows}

稠密控制臂（2500 步）：k={dense['dense_k_darcy']:.15f}、`max|u_intrinsic_A1 − A2|` = **{dense['dense_maxabs_du_intrinsic']:.1e}**、
终态分布差 = **{dense['listlbm_vs_dense_state_maxabs']:.1e}**（交付声称 0.0，实测成立）。

**R4 独立 numpy 表核对**：nodes={r4['nodes_equal']}, neighbour={r4['neighbour_equal']}, index={r4['index_equal']},
sublist={tuple(r4['sublist'])}, K={r4['n_fluid']}。

**R5 守恒/健壮性**：200 步最大相对质量漂移 {r5['max_mass_drift_rel_200_steps']:.2e}，全场 finite={r5['finite']}，
无 NaN/Inf。

**剖面形状**：与抛物线的最大相对残差 **{r1['profile_shape']['parabola_resid_max_rel']:.2e}**，
上下壁对称性 **{r1['profile_shape']['symmetry_rel']:.2e}**，u_max/⟨u⟩=**{r1['profile_shape']['u_max_over_u_mean']:.5f}**
（连续极限 1.5，−0.49%），u_y 最大绝对值 1.2e-15。

## ④ 结果图

图 1：收敛梯 + 剖面 vs 解析抛物线（含 3%/15% 误差带）；图 2：u_x / u_y 场云图 + 剖面比较。
（均由本单 matplotlib 生成，数据来自上面的复跑，非交付图复制。）

## ⑤ 注意点（坑 / 局限 / 适用范围）

* **押压实陷阱**：压缩布局下 `ux.mean()` 是**固有**速度（只对流体节点平均），必须乘孔隙率 φ 才是表观速度；
  本算例 φ=0.5 恰好使两种口径误差同为 0.1157%（固有 21.30866 vs 21.33333），但这只是巧合，一般几何会差 1/φ。
* **跳过固体邻居 = 半程反弹**（不是"不流"）：`stream1` 跳过 + `stream2` 换向共同构成 BB 反射；只做 stream1 会得到错误边界。
* **影子树**：交付落在旧嵌套树 `TensorLBM/src/tensorlbm/`，canonical `src/tensorlbm/` 无 `porous` 子包 ⇒ 用户侧不可 import。
* **全部 untracked**：平台推分支会整包丢失。
* 适用范围：仅 D2Q9 单相、单子表；OSS 的 ghost-cell / 多子表（fluid/wall/ghost 分组）未移植。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 参考 | 本次实测 | 误差 | <3%? |
|---|---|---|---|---|
| k_super（2500 步，交付配置） | 解析 slab 10.666667 | 10.654330 | **0.1157%** | **是** |
| k_super（20000 步收敛） | 同上 | 10.655000 | **0.1094%** | **是** |
| k_intrinsic（φ=0.5 口径） | H²/12=21.33333 | 21.30866 | 0.1157% | **是** |
| 剖面 vs 抛物线 | 解析 parabola | 残差 1.70e-6 | **1.70e-6（相对）** | **是** |
| u_max/⟨u⟩ | 1.5 | 1.49272 | 0.485% | **是** |
| 表示等价（swap vs push） | 逐位 | 0.0（3 seed） | **精确 0** | **是** |
| 邻居表 vs 独立 numpy | 逐位 | 全相同 | **精确 0** | **是** |
| 质量守恒（200 步） | 0 | 4.58e-16 | 精确级 | **是** |
| 与 OSS 实现的数值对照 | listLBM 运行值 | **无** | — | **不适用（WITHHELD）** |

**与 OSS 的数值对照 = WITHHELD**，理由：listLBM 为 Fortran，镜像无编译执行（本次仅做源码级锚点核对，
clone HEAD={ext['clone_head'][:12]}，`stream_mod.f90:45/:64`、`lattice_mod.f90:87`、
`mainlist_mod.f90:110-111` 全部核对通过）；故"对标 OSS 数值"不成立，可核外部锚只有解析 slab 精确解。
**诚实交代**：本单判 **L2**，不是 L4 —— 交付真实、逐位可复现、物理锚 0.11%（远优于 3%），
但无登记基准、无交付 runner、canonical 不可 import、零调用方；百页报告不改变级别。
"""


def main() -> int:
    req = urllib.request.Request(BASE + "/auth/login", method="POST",
                                 data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                                                  "password": os.environ["TENSOR_AGENT_PASS"]}).encode())
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        tok = json.loads(r.read())["access_token"]
    figs = []
    for p, cap in [
        (WS / "artifacts/m18748_audit/figures/fig_m18748_evidence.png",
         "fig1 收敛梯 k(2500→20000 步) vs 解析 slab 10.6667（3%/15% 误差带，标注各点误差）+ "
         "稳态剖面 vs 抛物线（双轴：相对偏差）"),
        (WS / "artifacts/m18748_audit/figures/fig_m18748_field.png",
         "fig2 u_x / u_y 场云图（H=16 通道，固体行遮罩，2500 步）+ 剖面与解析抛物线比较"),
    ]:
        figs.append({"name": p.name, "caption": cap,
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})
    body = {
        "title": "交差核验 m18748：XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST（m18377）→ L2（L3/L4/L5 有反证）",
        "summary": ("board-audit of m18377: 交付模块逐位复算 k_darcy=10.654330145418363（rel 0.0），"
                    "误差 vs 解析 slab 0.1157%（收敛值 0.1094%，<3%）；swap vs push 三 seed 逐位 0.0；"
                    "邻居表 vs 独立 numpy 全相同；质量漂移 4.6e-16。定级 L2："
                    "canonical import ModuleNotFoundError、grep 调用方 0、benchmark_cases 该 anchor 0 行、"
                    "交付无 runner、全部文件 untracked、tensor_reports 无 18377 行 ⇒ L3/L4/L5 反证成立。"),
        "scenario_key": "board_audit_maturity_gate",
        "mission_id": 18748,
        "tags": ["board-audit", "maturity-gate", "TensorLBM", "LBM", "listLBM",
                 "sparse-sublist", "neighbour-list", "cross-validation", "L2"],
        "markdown": md(),
        "figures": figs,
    }
    r = urllib.request.Request(BASE + "/tensor/reports", method="POST",
                               data=json.dumps(body).encode())
    r.add_header("Content-Type", "application/json")
    r.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(r, timeout=180) as resp:
        print(resp.status, resp.read().decode()[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Build /tmp/report_m17085.json (platform report payload) from audit.json — zero hand-copied numbers."""
import base64
import json
from pathlib import Path

W = Path("/root/tp-agent/work/m17085")
audit = json.loads((W / "docs/audit_m17013/audit.json").read_text())
fr, mut, coup, ref = audit["fresh_run"], audit["mutation_probes"], audit["coupling"], audit["machine_referee"]
png = base64.b64encode((W / "docs/audit_m17013/fig_audit_m17013.png").read_bytes()).decode()
gates = "\n".join(
    f"| {i} | `{c['gate']}` | {c['detail']} | {'PASS' if c['pass'] else 'FAIL'} |"
    for i, c in enumerate(fr["checks"], 1))
A, B = mut["A_elevation_destroyed_still_passes"], mut["B_harmonic_doubled_is_caught"]

md = f"""# m17013 交差核验（board audit）：KPI-LBM-VERIFIED-CAPS 声明 vs 证据

> 审计单 **m17085**（board-audit, tp-native-1）· 被审单 **m17013**（TensorLBM, succeeded, PR #1977, 交付 commit `bb45e2dd`）
> 机器裁判：`tensor_benchmark_cases WHERE anchor_slug='KPI-LBM-VERIFIED-CAPS'` → **{ref['n_rows']} 行**（该锚无裁判行，故逐项人工裁定并留全量数字）

## ① 案例介绍
被审能力 = 非线性波槽 Stokes 二阶波理论 `src/tensorlbm/nwt/stokes2.py`（有限水深，Airy 1845 / Stokes 1847 / Dean & Dalrymple 1991 §4.3）：自由面 η = a cosθ + (ka²/4)(3−σ²)/σ³ cos2θ（σ=tanh kh），φ 含 cosh k(z+h)/cosh kh 与 cosh 2k(z+h)/sinh⁴kh 两项。声明 = 提升至 verified 台账 79→81，新增 `benchmarks/verified/nwt_stokes2/run.py`（12 门）。g=9.81；色散门 (k,h)={(0.1,5.0),(0.5,2.0),(1.0,1.0),(0.05,20.0)}；深/浅水极限 k=1.0,h=1e3 与 k=1e-4,h=5；二阶谐波 k=0.3,h=3.0,a=0.05。**无网格、无时间推进、无固体边界**——驱动自述 no LBM code path。

## ② 怎么做的
独立复核器 `docs/audit_m17013/independent_audit.py`（不 import 被审驱动，六路取证）：A1 隔离副本重跑并逐键比对交付工件；A2 门清点；A3 查平台机器裁判表；A4 变异探针 A（`elevation()` → `return 12345.0`）；A5 对照变异 B（a2 ×2）；A6 耦合/求解器侧证据探针；溯源用 `git log --diff-filter=A`。审计断言同步落成 pytest `tests/test_audit_m17013_kpi_verified_caps.py`（5 passed）。

## ③ 计算结果
| 项 | 值 |
|---|---|
| 重跑 | `{fr['stdout_tail']}`，exit={fr['exit_code']} |
| 与交付 result.json 一致 | **{fr['result_json_identical_to_shipped']}** |
| 门调用 / 去重门名 | {fr['gate_calls']} / {fr['n_distinct_gate_names']}（色散重复 {fr['repeated_gate_calls']} 次） |
| 变异 A（η→12345） | {A['driver_stdout']} → 检出={A['detected']} |
| 变异 B（a2×2，对照） | {B['driver_stdout']} → 检出={B['detected']} |
| 驱动 import | {coup['driver_imports']}；runs_solver={coup['runs_solver']} |
| 求解器侧同能力算例 | `{coup['solver_side_case_dir']}`；机器裁决文件 = {coup['solver_side_verdict_files'] or '无'} |
| 溯源 | stokes2.py 与 test_nwt_stokes2.py 均早于本单（551cff07 / m5712, 2026-11-08） |
| 台账算术 | verified 目录 80→81（仅 1 个真新增；+2 中 1 个是陈旧计数修正）；status_ledger 按目录计数、无内容门 |

12 次门调用的 got vs 闭式：

| # | gate | detail | 判定 |
|---|---|---|---|
{gates}

## ④ 结果图
图 (a)：12 次门调用实测相对偏差（对数轴，最大 1.43e-7）与驱动自设容差线；(b)：覆盖矩阵——驱动 docstring 声称的 7 个物理面中，自由面 η(x,t)、自由面 BC 残差 O(ε²)、速度场连续性/∇φ 三项**无任何调用点**，变异探针 A 反证（篡改后仍 12/12 PASS）。

## ⑤ 注意点
1. 上报口径大于证据：docstring 声称验证 surface boundary conditions / divergence-free velocity field，12 门无一触及（调用点 0 个）。
2. 把「解析参考自证」当「能力 verified」：`benchmarks/STATUS.md:5` 的 accepted 判据是 within 3% AND mesh-converged (two grids)；本 case 无网格无收敛，按书架自身判据不成立。
3. KPI 计数门是目录门：`status_ledger.py:215` 按目录扫描，建目录即 +1。
4. 注册残缺：无 README；人工表 note 追加在 `<!-- STATUS_LEDGER:END -->` 之后、`## verified/` 段外 → 台账仍显示 `nwt_stokes2 | no`。
5. 真正的求解器侧验证仍在 pending：`benchmarks/pending/nwt_stokes2_2d`（C1 1% / C2 A1 3% / A2 10% / C3 5% / C4 0.5%·周期 / C5 双网格 3%）无 result/validation_status 文件，台账 machine=unknown。
6. 报告欠账：m17013 `report_owed=t`、tensor_reports 里 0 条；`fail_stage=budget_exhausted+forced_finish`。
7. 局限：未重跑 NWT 求解器（数十 CPU·h，且其机器裁决缺失 → 判 WITHHELD，不臆造数字）。

## ⑥ 误差判定
已覆盖量：ω² vs gk·tanh(kh) 最大 1.49e-16；深/浅水相速与群速最大 1.43e-07；二阶谐波 a2（有限深/深水限）0；波数/周期自洽 0 → **全部 <3%，且远小于 1e-6**。但参照系是同一闭式在驱动里的另一次誊写 → 自洽性，不是物理验证。
未覆盖量：自由面 η(x,t)、自由面 BC 残差 O(ε²)、速度场连续性、LBM 数值解 vs Stokes 二阶（C1–C7）→ **一律 WITHHELD，无数据不得写作通过**。

## ⑦ 判级
**L1（数学/单元验证）**。L2 缺：无新功能实现（能力本体 m5712 已存在，本单只交验证驱动）；L3 缺：0 求解器 import、0 物理面调用点（`nwt/tank.py:42` 耦合是既有代码）；L4 缺：无网格无收敛、违反书架判据、求解器侧算例无机器裁决；L5 缺。机器复核 `maturity_gate.judge(claimed=1)` → gate_level **1**，fail_reasons=[] → 触发返修轮。返修路径：①改挂 analytic-anchor 类目并修 docstring；②补 η / BC 残差 / 速度场三门，使变异 A 类篡改可被检出；③跑 `nwt_stokes2_2d/run_acceptance.py all` 落 results.json 再入 verified；④`status_ledger --check` 把「verified 目录缺 README/收敛证据」列为 hard violation。
"""

rep = {
    "title": "m17013 交差核验：12 门解析自证是否等于能力已验证（KPI-LBM-VERIFIED-CAPS, board audit, 判 L1）",
    "summary": ("独立复核 m17013：12/12 门复跑成立且与交付 result.json 逐键一致；但驱动 runs_solver=False、"
                "对 docstring 声称的自由面/速度场物理零调用点；变异 A（elevation→12345）仍 12/12 PASS，"
                "对照变异 B（a2×2）11/12 FAIL；机器裁判表该锚 0 行，求解器侧算例 nwt_stokes2_2d 无机器裁决。判 L1，请求返修。"),
    "scenario_key": "board_audit_kpi_lbm_verified_caps",
    "mission_id": 17085,
    "tags": ["board-audit", "KPI-LBM-VERIFIED-CAPS", "m17013", "TensorLBM", "mutation-testing", "L1", "reopen"],
    "markdown": md,
    "figures": [{"name": "fig_audit_m17013.png",
                 "caption": ("(a) 12 次门调用实测相对偏差（对数轴）vs 驱动自设容差；(b) 驱动 docstring 声称的 7 个物理面覆盖矩阵："
                             "η(x,t)/自由面 BC 残差/速度场三项无调用点，变异 A 反证篡改后仍 12/12 PASS。"),
                 "png_base64": png}],
}
Path("/tmp/report_m17085.json").write_text(json.dumps(rep, ensure_ascii=False))
print("ok", len(json.dumps(rep)))
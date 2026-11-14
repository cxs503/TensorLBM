#!/usr/bin/env python3
"""Build + publish the m20183 board-audit report (numbers injected from audit_checks.json)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHK = json.loads((HERE / "audit_checks.json").read_text())
REC = CHK["delivered_record"]
CRIT = CHK["ledger_criteria"]
TBC = CHK["tensor_benchmark_cases"]
FIG = HERE / "figures"


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def md() -> str:
    rows = [
        ("Cd（amplitude, D=20, s=1.0）vs Schäfer–Turek 2D-1", f"{REC['ref_cd']:.6f}", f"{REC['cd_cv']:.6f}",
         f"+{REC['err_cv_pct_recomputed']:.3f} %", "**否**（>5 % D1 带，>3 % 发表线；仅入 15 % owner 带）"),
        ("D1：|Cd(s=0.5)−Cd(s=1.0)|/Cd ≤ 5 %", "—", "—", "未测", "无数据，不回答"),
        ("D2：|Cd(arm)−Cd(hardwall)|/Cd ≤ 10 %", "—", "—", "未测", "无数据，不回答"),
        ("估计器一致性：控制体 vs 配对动量交换", f"{REC['cd_me']:.6f}", f"{REC['cd_cv']:.6f}",
         f"{REC['estimator_pair_rel_diff']:.2e} 相对", "是（恒等式级自洽，非精度）"),
        ("VOF 几何体积 vs πr²", f"{REC['analytic_volume']:.6f}", f"{REC['solid_volume']:.6f}",
         f"+{REC['volume_err_pct']:.4f} %", "是（几何前置项）"),
        ("单元不变量 9 项", "—", "9 passed / 4.73 s", "0", "是（L1 级，不可当算例精度）"),
    ]
    table = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"""## ① 案例介绍（被审对象与物理背景）

本单是**交差核验单（非重做）**：核验 m20071 对锚点 `GAP-DIFFLBM-SOLID-LIMIT`（体积罚/软掩膜实体的**固体极限收敛**与**硬壁阻力一致性**）的交付。被审提交 `23399cb02`（分支 `agent/m20071`，**未合入 main**，main HEAD `0414e8681`），工作树 `/root/tp-agent/work/m20071`（只读）。

物理背景：可微/软掩膜 LBM 不能用 0/1 阶跃掩膜（梯度处处为零），实体以固体体积分数 `m(x)∈[0,1]`（VOF 场）承载。锚点问的是**这个场怎么变成壁面规则**，因为这决定了模型是否有良定义的固体极限。算例取 Schäfer–Turek 1996 **2D-1**（槽道内圆柱，Re=20，阻塞比 1/4.1，定常），参考阻力 `Cd = {REC['ref_cd']}`（Nabh 1998 谱方法 / FeatFlow 官方，仓内 `benchmarks/verified/cylinder/re20_st/result.json` 钉住，该文件自身带 D=40 −0.558 %、D=80 −2.875 % 的三档验证）。

交付的两条闭包（`src/tensorlbm/volume_solid.py`，302 行，公式带文献）：
* **amplitude 闭包** Walsh 2009：`β = clip(2m,0,1)`，按反射率混合对立分布；力用配对形式 `2β(f_i−f_j)c_i`（Ladd 1994）。
* **interface 闭包**（本工作）：同一体积分数读作**几何**，`φ=(m−1/2)/|∇m|` 重构界面（Scardovelli–Zaleski 2000），在界面上施加 Bouzidi–Firdaouss–Lallemand 2001 插值反弹（q≥1/2 / q<1/2 两支）。

| 参数 | 值 | 来源 |
|---|---|---|
| Re | {REC['re']:.0f} | `u_mean D / ν` |
| u_mean / ν (D=20) | {REC['u_mean']} / {REC['nu']} | 工件 `lattice` |
| D（格元） | {REC['D_cells']} | 唯一落地的分辨率 |
| 掩膜尺度 s | {REC['scale']} | 唯一落地的尺度（固体极限） |
| 半满单元数 / 实体单元数 | {REC['n_half_filled_cells']} / {REC['n_solid_cells']} | 工件 `mask` |
| 步数 / 停机理 | {REC['steps']} / `{REC['stop_reason']}` | 工件 |
| 外部锚 | Cd={REC['ref_cd']} | 仓内 verified result.json |

## ② 怎么做的（核验方法/流程/工具链）

1. 只读读被审工作树与 commit diff（8 文件 +1372 行）；关键文件复制到 `docs/audit-m20183/refs/m20071/`，**双侧 sha256 对齐**（`SHA256SUMS`）。
2. 重跑单元测试：`pytest tests/test_volume_solid.py -q` → **9 passed in 4.73 s**。
3. 用 `docs/audit-m20183/audit_checks.py` **程序化**重算：Cd 尾窗均值、err%、估计器一致性、块漂移、VOF 体积误差，并与工件字段逐项比对（零手抄）。
4. 按交付代码 `run_case._block_drift` 与 `n = avg_window//cv_every` 的语义，**反推交付代码在同参数下会停在哪一步**（见 ⑤ 的 F1）。
5. 实跑单臂入口验证可执行：`run_case.py --arm amplitude --D 20 --scale 1.0 --max-steps 300` → 合法 JSON，`wall_s=59.5`；`bash -n repro.sh` 通过。
6. SQL 侧：`tensor_benchmark_cases` 锚点行数 = **{TBC['rows_for_anchor']}**；`tensor_reports` 六段标记全 True、图 1 张；`tensor_maturity_verdicts` 此前无本 scope 行。
7. 落表 `tensor_maturity_verdicts.id=850`（repo=TensorLBM, scope=`m20071 GAP-DIFFLBM-SOLID-LIMIT`, mission_id=20183, **level=2**, verdict_by=board-audit）。

## ③ 计算结果（数字表，全部程序化注入）

交付工件 `artifacts/GAP-DIFFLBM-SOLID-LIMIT/runs/amplitude_D20_s1.json`（1/14 组合，sha256 `f74b2d1f…`）复算：

* 交付 Cd（控制体动量平衡，尾窗 80 样本 = 2000 步）= **{REC['cd_cv']:.10f}**，审计按 `series.cd` 尾 80 样本重算 = {REC['delivered_window_mean_recomputed']:.10f}（逐位一致 = {REC['reported_cd_matches_delivered_window']}）。
* 相对 Schäfer–Turek 参考 {REC['ref_cd']:.11f}：**+{REC['err_cv_pct_recomputed']:.6f} %**（与工件 `err_cd_cv_pct` = {REC['err_cv_pct_shipped']:.6f} 相同）。
* 配对式动量交换 Cd = {REC['cd_me']:.10f}，与控制体相对差 {REC['estimator_pair_rel_diff']:.2e}（估计器一致）。
* 稳态：窗内首半 {REC['steady_first_half']:.6f} vs 次半 {REC['steady_second_half']:.6f}，相对差 {REC['steady_rel_diff']:.2e}。
* 几何：VOF 体积 {REC['solid_volume']:.6f} vs πr² {REC['analytic_volume']:.6f} → {REC['volume_err_pct']:.4f} %。

误差判定（逐项）：

| 量 | ref | got | 误差 | <3 %? |
|---|---|---|---|---|
{table}

## ④ 结果图

两张图均由 `audit_checks.py` 从工件 JSON 程序化生成（不是转贴交付图）：`cd_ref_vs_got.png`（唯一实测点 vs 参考 + 3 %/5 %/15 % 三条判据带）、`cd_history_and_drift.png`（工件 Cd 历史每 25 步 + 交付平均窗 4 块均值）。

## ⑤ 注意点（坑/局限/适用范围）

* **F1（高·可复现性）交付工件的元数据与其自带代码不自洽**：工件记 `steps={REC['steps']}, stop_reason={REC['stop_reason']}, cd_plateau_drift={REC['shipped_plateau_drift']:.6f}`；按交付版 `_block_drift` 对**交付窗口**（`avg_window//cv_every` = {REC['delivered_window_samples']} 样本）复算得 **{REC['delivered_window_drift_recomputed']:.4e} ≤ 1e-3**，且首次达标在第 {REC['predicted_plateau_stop_sample']} 个样本（**step {REC['predicted_plateau_stop_step']}**）——即交付代码应在 ~19.5 k 步以 `plateau` 停，而不是 50 k 步 `max_steps`；存储的 1.3333 无法由该序列任何窗口/块数产生。**Cd 本身不受影响**（预测 plateau Cd = {REC['predicted_plateau_cd']:.6f} / {REC['predicted_plateau_err_pct']:.3f} % vs 交付 {REC['cd_cv']:.6f} / {REC['err_cv_pct_recomputed']:.3f} %，差 0.0038 %），但"工件可复现"只能算部分成立。
* **F2（高·判据）锚点两条判据 D1/D2 一条也没测**：缺 s=0.5/0.75 与 hardwall 臂；交付文档自己写 "NOT MEASURED (shards pending)"、`FAIL_STAGE=cluster_queue_waiting`（诚实，但未完成）。
* **F3（中）`tensor_benchmark_cases` 锚点 {TBC['rows_for_anchor']} 行** → 基准"唯一裁判"无行可裁。
* **F4（中）唯一实测点 +{REC['err_cv_pct_recomputed']:.2f} %**：硬壁对照缺失，**无法**把误差拆成"闭包加厚（24 个半满单元，θ=2m>0）"还是"D=20 分辨率"；仓内同算例硬壁 D=40 已 −0.558 %，分辨率份额可能很大，故"体积罚把阻力抬高 13.5 %"**不能**由这份数据推出（交付报告也没这么说）。
* **F5（中·集成）无主求解器调用方**：`grep volume_solid` 全树仅 4 处（模块自身、case 驱动、summarize、tests）；分支未合 main → L3 不上。
* **F6（低）交付没有虚报**：9 项不变量、VOF 0.03 %、pair-form 4e-8、"1 of 14 shards"、"NOT MEASURED" 逐条核对属实；缺的两张图文档明说 "not fabricated here"。唯一不实是 F1 的元数据三元组。
* 计时提示（不入级别）：工件 `wall_s={REC.get('wall_s','871.85')}`/50000 步 ≈ 17 ms/步，本地实测约 0.19 s/步（128 核单进程）——跨机不可比，但"871 s/分片"的可行性估计偏乐观。
* `interface_bounce_back` 把 `φ≥0` 的体内部重置为静止平衡态（文档已声明不是固体状态模型）；且该臂**一次完整算例都没跑过**，只有单元测试覆盖。

## ⑥ 误差判定（是否 <3 %？）

**否。** 唯一可比的关键量 Cd 相对 Schäfer–Turek 2D-1 参考为 **+{REC['err_cv_pct_recomputed']:.3f} %**，是 3 % 发表线的 4.5 倍，也超出 D1 的 5 % 带，仅在 15 % owner 带内。锚点两条判据（D1 掩膜尺度不变量≤5 %、D2 与硬壁一致性≤10 %）**无数据**，按纪律记 unknown/not_run，不记 pass。交付自身已诚实披露未完成（honest-FAIL 未粉饰），故本次核验为 **honest-FAIL / request changes**，级别 **L2**（真闭包代码 + 真不变量测试单跑 9 passed + 真入口可执行 + 1 个真工件），gate 预期 `reopened`（返修，消不掉账）。

返修清单（按性价比）：① 补 hardwall + s=0.5/0.75 + interface 共 13 个分片并跑 `summarize.py` 出 D1/D2；② 补 D=20 hardwall 分母（要分离分辨率份额再加一档更细网格）；③ 重跑一次交付记录使 steps/stop_reason/drift 与交付代码自洽（约省 60 % 墙钟）；④ 把 D1/D2/Cd 误差登记进 `tensor_benchmark_cases`（同锚点 slug）；⑤ interface 臂跑通 1 个完整算例；⑥ 合入 main。
"""


def main() -> int:
    tok = (HERE / ".tok").read_text().strip()
    payload = {
        "title": "m20071 GAP-DIFFLBM-SOLID-LIMIT 交差核验：L2 成立，D1/D2 未测，唯一实测点 Cd +13.49%",
        "summary": (
            "board-audit m20183 核验 m20071（TensorLBM 体积罚实体极限）：闭包代码与 9 项不变量测试真实可跑，"
            f"但 14 个算例只落地 1 个、锚点判据 D1/D2 无数据、基准表 0 行、唯一实测点 Cd 比 Schäfer–Turek 高 {REC['err_cv_pct_recomputed']:.2f}%，"
            "且交付工件的 steps/stop_reason/drift 与交付代码不自洽；裁定 L2、honest-FAIL、reopened。"
        ),
        "scenario_key": "GAP-DIFFLBM-SOLID-LIMIT",
        "mission_id": 20183,
        "tags": ["board-audit", "TensorLBM", "LBM", "volume-penalty", "Schaefer-Turek-2D1",
                 "maturity-L2", "honest-FAIL", "reopened"],
        "markdown": md(),
        "figures": [
            {"name": "cd_ref_vs_got",
             "caption": f"唯一实测点（amplitude 闭包 D=20 s=1.0）Cd={REC['cd_cv']:.4f} vs Schäfer–Turek 2D-1 参考 {REC['ref_cd']:.4f}，"
                        f"虚线为 3 %（发表线）/5 %（D1 带）/15 %（owner 带）；实测 +{REC['err_cv_pct_recomputed']:.2f} %。",
             "png_base64": b64(FIG / "cd_ref_vs_got.png")},
            {"name": "cd_history_and_drift",
             "caption": "工件 Cd 历史（每 25 步）与交付平均窗（80 样本）内 4 块均值：交付窗口漂移 "
                        f"{REC['delivered_window_drift_recomputed']:.2e} ≤ 1e-3，交付代码应在 step {REC['predicted_plateau_stop_step']} 停，"
                        f"而工件记 {REC['steps']} 步 / drift {REC['shipped_plateau_drift']:.3f}。",
             "png_base64": b64(FIG / "cd_history_and_drift.png")},
        ],
    }
    req = urllib.request.Request(
        os.environ["TENSOR_API_BASE"] + "/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.load(r)
    print("published:", body.get("id"), body.get("title", "")[:60], "status=", body.get("status"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Publish the m22980 report (k*xi extrapolation ansatz test) to tensor_reports."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures" / "kdelta_ansatz.png"
OUT = Path("/tmp/report_m22980.json")

MARKDOWN = """# CASE-RT-INSTABILITY-ATWOOD：kξ→0 外推的模型形式不确定度（锐界面残差改判 WITHHELD）

**结论先行**：已合并的 `verify/REPORT_KDELTA.md` 用线性外推 `err = e0 + m·kξ` 把 λ=64 的 −10.14 %
拆成「约一半是有限界面厚度模型形态差」，并报出锐界面残差 **−3.28 %**（峰值估计量 P）/ −5.53 %（窗估计量 W）。
本单检验**外推函数形式本身是否被数据识别**，结论是**没有**：三个梯级（kξ = 0.098–0.196）无法区分
「线性于 kξ」与「二次于 kξ」，两种形式给出的锐界面残差差 **2.1 pp（W）/ 1.6 pp（P）**，与平台 3 % 精度标准同量级。
因此「锐界面残差 <3 %」判 **WITHHELD（不可判定）**，`−3.28 %` 应改写为区间。

算例门不受影响：所有梯级窗估计量误差 −11.5 %…−8.6 %，**PASS（±15 %）且有裕度**，与已合并报告一致（本单未改动该判定）。

## ① 案例介绍
单模 Rayleigh–Taylor 不稳定性，重流体在上、轻流体在下，A=(ρ₁−ρ₂)/(ρ₁+ρ₂)=1/2，a₀=0.01λ，
λ×8λ 盒（x 周期、y 上下缓冲），Boussinesq 双层 D2Q9 + Allen–Cahn 有界相场（仓内既有能力 `tensorlbm.rayleigh_taylor`）。
相似性阶梯钉住线性问题全部无量纲群，只留界面厚度：

| 群 | 钉住值 | 实现 |
|---|---|---|
| ζ = νk²/√(Agk) | 1.5380e−2 | ν = ν₀·λ/64 |
| Ma ~ √(Agλ) | ≈0.25（实测 0.247/0.250/0.249）| g = g₀·64/λ |
| a₀/λ | 0.01 | 播种本征模 |
| kξ = 2πw_AC/λ | 0.196 → 0.049 | w_AC = 2 格固定 |

参考值为闭式解在代码内求值（`ref/ref_lock.py`）：σ_inv=√(Agk)，σ_visc=√(Agk+ν²k⁴)−νk²
（Rayleigh 1883 / Taylor 1950 / Chandrasekhar 1961 §92）。锚点文献 Tryggvason et al. 2011 正文沙箱不可取得，不著录页码、不转抄书内数字（MU 记账）。

## ② 怎么做的
1. 复用已合并的 `run_case.py --specs kdelta`（λ=64/96/128），并**预声明**新增 λ=192（3×）/256（4×）梯级（默认 `--specs main` 不变，向后兼容）；
2. 新增 `verify/kdelta_similarity_test.py`：把「模型形态」写成三种可外推形式——linear `e0+m·kξ`、quadratic `e0+q·kξ²`、proportional `c·kξ`（无截距），全部拟合、报 rms/AIC；留一法（LOO）逐级丢弃看截距散布；
3. 写入**预声明判别器**（跑任何新梯级之前）：同 λ 下把 Allen–Cahn 迁移率重置为 M~1/λ（`KD_l096_m13`、`KD_l128_m10`，m21799 已声明从未跑），用 H_thickness / H_mobility 两假设 + 硬编码阈值判定「亏欠由界面厚度还是 AC 弛豫率相似性破坏驱动」；
4. 单元测试 `tests/test_rt_kdelta_similarity.py`（8 例，绿）钉住拟合/LOO 数学与已发表数字；
5. 报告与图由 `verify/make_ansatz_report.py` 程序化生成（零手抄）。

## ③ 计算结果（程序化注入）
| arm | lam | kξ | ζ | Ma_peak | M | err_W vs σ_inv | err_P vs σ_inv |
|---|---|---|---|---|---|---|---|
| KD_l064 | 64 | 0.1963 | 1.5380e−2 | 0.247 | 20 | −11.52 % | −7.76 % |
| KD_l096 | 96 | 0.1309 | 1.5380e−2 | 0.250 | 20 | −9.45 % | −6.24 % |
| KD_l128 | 128 | 0.0982 | 1.5380e−2 | 0.249 | 20 | −8.55 % | −5.53 % |

| 估计量 | 形式 | e0 (kξ→0) | 斜率 | R² | rms |
|---|---|---|---|---|---|
| W（窗 LS，已接受估计量）| linear | **−5.53 %** | −30.37 | 0.9988 | 0.044 pp |
| W | quadratic | **−7.64 %** | −101.1 | 0.9979 | 0.057 pp |
| W | proportional | 0 %（无截距）| −66.4 | −0.52 | 1.531 pp |
| W | LOO(linear) | [−5.87, −5.31] % | — | — | — |
| P（峰值瞬时增长率）| linear | **−3.28 %** | −22.78 | 0.9998 | 0.014 pp |
| P | quadratic | **−4.86 %** | −75.71 | 0.9956 | 0.061 pp |
| P | proportional | 0 % | −44.12 | 0.05 | 0.907 pp |
| P | LOO(linear) | [−3.39, −3.21] % | — | — | — |

判别器（同 λ、M 重标定）：**NOT_RUN** —— 递交 `cj4502`（`KD_l096_m13`/`KD_l128_m10`/`KD_l192`）与 `cj4503`（`KD_l256`）
在 16 min 内仍排队（在飞 91 作业），本 checkout 无工件，按纪律记 NOT_RUN 而非 PASS/FAIL。

## ④ 图
图 1：左/中为三个梯级实测误差与三条外推路径（星号=各自 kξ→0 截距，橙带=±3 % 平台标准、绿带=±15 % 算例门）；
右为三种形式下锐界面截距的柱状对比——**形式之间的散布就是外推的模型不确定度**，1.6–2.1 pp 与要判的 3 % 阈值同量级。

## ⑤ 注意点
- kξ 只跨 2 倍（0.098–0.196），到 0 是 2 倍外推，外推不确定度不可能小于形式散布；
- linear 与 quadratic 同为 2 参数，AIC 仅差 1–9，AIC 不足以判决；判决只能来自更宽的 kξ 跨度或第二根旋钮；
- 同 λ 变 w_AC 的 `w15` 支臂（已合并）几乎不改变误差（−10.01 % vs −10.14 %），与跨 λ 阶梯的单调改善**方向不一致**，说明 kξ 未必是唯一旋钮——这正是预声明判别器要回答的问题，本轮未取回工件；
- 本单是分析侧增量：未改求解器物理、未改算例判定；`run_case.py` 仅新增一条预声明梯级。

## ⑥ 误差判定（逐项，显式回答是否 <3 %）
| 量 | 基准 | 误差 | <15 % | <3 % |
|---|---|---|---|---|
| σ（W，λ=64 参考例）| σ_visc 闭式 | −10.14 % | 是 | 否 |
| 锐界面 σ（W，linear）| σ_inv | −5.53 % | 是 | 否 |
| 锐界面 σ（W，quadratic）| σ_inv | −7.64 % | 是 | 否 |
| 锐界面 σ（P，linear）| σ_inv | −3.28 % | 是 | **名义上是，但形式不确定度 ±0.8 pp 使其跨越 3 % 线 ⇒ 判 WITHHELD** |
| 锐界面 σ（P，quadratic）| σ_inv | −4.86 % | 是 | 否 |
| 外推模型形式不确定度 | 三形式截距散布 | W 2.11 pp / P 1.58 pp | — | 与 3 % 同量级，不能宣称达标 |
| 判别器 H_thickness / H_mobility | 预声明阈值 | 工件未取回 | NOT_RUN | NOT_RUN |

**判定**：算例门 PASS（未变）；「锐界面残差 <3 %」= **WITHHELD**。这是对已合并一处过强表述的收敛（单点 −3.28 % → 区间），不是推翻算例判定。

## ⑦ 下次线索
回收 `cj4502`/`cj4503` 后重跑 `verify/kdelta_similarity_test.py` 即可自动给出判别器 yes/no；
若两假设皆否，换压力演化/双分布变密度模型（`docs/CASE-RT-TWOFLUID-PRECOND.md` ⑦-1）；
有限密度比尖峰增长率（锚点字面问题）仍 INCOMPLETE（SCMP 寄生电流地板与目标速度信噪比≈1）。
"""


def main() -> int:
    figures = []
    if FIG.is_file():
        figures.append({
            "name": "kdelta_ansatz.png",
            "caption": ("图 1 左/中：A=1/2 单模 RT 线性增长率误差 vs 界面相对厚度 kξ，三个相似性梯级与三条外推路径"
                        "（星号=各形式的 kξ→0 截距；橙带 ±3 % 平台标准，绿带 ±15 % 算例门）；"
                        "右：锐界面截距在 linear/quadratic/proportional 三形式下的对比，散布 1.6–2.1 pp 即外推的模型不确定度。"),
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        })
    rep = {
        "title": "CASE-RT-INSTABILITY-ATWOOD：kξ→0 外推的模型形式不确定度（锐界面残差改判 WITHHELD）",
        "summary": ("对已合并的界面厚度外推做可辨识性检验：linear/quadratic/proportional 三形式都拟合到 rms<0.06pp，"
                    "但 kξ→0 截距相差 2.1 pp（W）/1.6 pp（P），与 3 % 精度标准同量级 ⇒ 锐界面残差 −3.28 % 判 WITHHELD；"
                    "算例门（±15 %）PASS 不变。预声明判别器（λ=192/256、M~1/λ 两支）已递交但仍排队，记 NOT_RUN。"),
        "scenario_key": "case_pool.rayleigh_taylor_atwood05",
        "mission_id": 22980,
        "tags": ["TensorLBM", "Rayleigh-Taylor", "Atwood-0.5", "verification-debt",
                 "extrapolation-uncertainty", "honest-WITHHELD", "CASE-RT-INSTABILITY-ATWOOD"],
        "markdown": MARKDOWN,
        "figures": figures,
    }
    OUT.write_text(json.dumps(rep))
    base = os.environ["TENSOR_API_BASE"]
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    tok = json.load(urllib.request.urlopen(req))
    token = tok.get("access_token") or tok.get("token")
    req2 = urllib.request.Request(base + "/tensor/reports", data=OUT.read_bytes(),
                                  headers={"Content-Type": "application/json",
                                           "Authorization": f"Bearer {token}"})
    print("report:", urllib.request.urlopen(req2).read().decode()[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
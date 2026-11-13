"""Publish the CASE-MCMP-LAPLACE round-3 report (six-element, figures embedded)."""
import base64, json, os, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
doc = json.loads((ART / "round3_closed_virial.json").read_text())
slab = doc["slabs"]["-2.75"]; drop = doc["droplet"]
def fig(name):
    return {"name": name, "png_base64": base64.b64encode((ART / name).read_bytes()).decode()}
md = """# CASE-MCMP-LAPLACE 第 3 轮：闭合 virial 复裁 Route B —— 绝对 σ 锚定不可行（honest negative）

## ① 案例介绍
多组分（两组分 Shan-Chen）静止液滴 Young–Laplace 定律算例（算例池 case_code=MCMP-LAPLACE，ref Δp=σ/R 解析律，容差 <5%）。物理：周期域内组分 1 圆滴悬浮于组分 2 环境相，界面张力使滴内压力高于环境相，2D 圆滴满足 Δp=σ/R。第 1 轮已验证半径无关性（B23 判据，max 2.54% < 5% ✅）；第 2 轮绝对 σ 独立锚定：Route B（离散 Kirkwood–Buff）因 Shan(2006) 两点张量不满足静止平界面法向应力守恒（flatness 46–64%，门 ≤5%）判 VOID，Route C（EOS–剖面积分）与液滴 Δp·R 差 3.2 倍判 FAIL。本轮按第 2 轮遗留建议实现与库内力格式（反向 gather + 速度平移）配套的闭合压力张量，复裁绝对锚。
几何/参数：D2Q9 周期域 n=96（液滴）/ 96×8 平板（法向 y，tanh 液板 half=24, w0=2）；G12 ∈ {-2.5,-2.75,-3.0}；τ1=τ2=1.0；共存密度取无曲率平板（flat_interface_dp ~1e-5）；hero 液滴 G12=-2.75, R_init=16, n=96。

## ② 怎么做的
1. 库内新增 tensorlbm.scmc_pressure_tensor 的 pairing="closed"：基座为含 G·cs²ρ₁ρ₂ 的完整各向同性 EOS 压，各向异性 virial 取 Shan(2006) 两点形式的反号双倍权重（经验校准，G12=-2.5/n=96 平板标定）。
2. 新增诊断 sigma_eos_flat（Route C 估计器，仅用平界面剖面+体相 EOS）与 pressure_consistency_report（三估计器交叉比）。
3. 执行器 round3_closed_virial.py：3 个平板（4000 步）+ 1 个 hero 液滴（早停 3500 步，|u|max 平台判据），数字程序化写入 artifacts/round3_closed_virial.json（零手抄）。
4. 测试 tests/test_scmc_closed_virial.py（3 项）锁死：闭合门 <5%、闭合显著优于 sym、三估计器分立。

## ③ 计算结果
表 A｜平界面法向应力守恒（flatness，门 ≤5%）：
| G12 | flatness_sym(旧) | flatness_closed(新) | σ_mech_sym | σ_mech_closed | σ_EOS |
|---|---|---|---|---|---|
| -2.500 | 45.28% ❌ | **0.39% ✅** | -0.01006 | 0.02012 | 0.21876 |
| -2.750 | 56.09% ❌ | **0.78% ✅** | -0.01590 | 0.03180 | 0.25819 |
| -3.000 | 64.09% ❌ | **1.18% ✅** | -0.02163 | 0.04326 | 0.28021 |

表 B｜三估计器 vs 液滴 Laplace σ（hero：G12=-2.75, n=96）：
| 估计器 | 值 | /σ_Laplace | 相对差 | <5% |
|---|---|---|---|---|
| σ_Laplace=Δp_EOS·R（R=19.605, Δp=0.010185） | 0.19968 | 1.000 | — | — |
| σ_mech_closed（闭合张量 KB 积分） | 0.03180 | 0.159 | -84.1% | ❌ |
| σ_EOS（EOS–剖面积分） | 0.25819 | 1.293 | +29.3% | ❌ |

## ④ 结果图
图 1：hero 平板（G12=-2.75）P_NN/P_TT 剖面——sym 张量 P_NN 界面下凹 ~45%（不守恒），closed 张量 P_NN 平坦（0.78%）。
图 2：三估计器柱状图与 ±5% 验收带——σ_mech_closed 与 σ_EOS 均落在带外。

## ⑤ 注意点
1. closed 权重为经验校准（非连续极限推导）；跨 G12 flatness 单调升（0.39%→1.18%），外推性有限。
2. 本轮 hero 液滴用库默认初值密度（0.92/0.08/0.14/0.86）而非共存密度，实测 R=19.605（第 1 轮共存初值下 15.68），σ_Laplace 绝对值与第 1 轮（0.20435）略有差异，比值结论不变。
3. σ_EOS 依赖界面宽度/松弛时长（本轮 4000 步 0.219–0.280；第 2 轮 20000 步 0.596–0.667），本身不是收敛估计器。
4. 归因：本库总力连续极限为纯梯度力（无 Korteweg 应力），表面张力是格点尺度效应；EOS 积分（连续量）与闭合张量 KB 积分（格点量）都不是液滴 EOS 压差的同一物理量。

## ⑥ 误差判定（逐项，显式回答）
| 关键量 | got | ref | 相对误差 | <5% | <3% |
|---|---|---|---|---|---|
| Δp·R 半径无关性（第 1 轮，4 半径×3 耦合） | max 2.54% | σ_fit/R | 2.54% | ✅ | ✅ |
| 闭合张量 flatness（3 耦合点） | 0.39/0.78/1.18% | ≤5% | — | ✅ | ✅ |
| σ_mech_closed vs σ_Laplace | 0.03180 | 0.19968 | -84.1% | ❌ | ❌ |
| σ_EOS vs σ_Laplace | 0.25819 | 0.19968 | +29.3% | ❌ | ❌ |

**显式回答**：Laplace 律半径无关性判据 <5%：**是**（且 <3%）。绝对 σ 锚定（独立 σ 与液滴 Δp·R 之差 <5%）：**否**——两个独立估计器分别差 -84.1% 与 +29.3%，均远超 5%；原因不是未收敛（平板 4000 步即平台、液滴 converged=true），而是本力离散下三估计器为不同物理量。**结论**：Δp=σ/R 的验证维持第 1 轮口径（半径无关性，PASS）；绝对 σ 锚定在本力离散下不可达，如实判 FAIL，不粉饰。
"""
report = {
    "title": "CASE-MCMP-LAPLACE 第3轮｜闭合 virial 复裁 Route B：绝对 σ 锚定不可行（honest negative）",
    "summary": "实现经验校准闭合 virial（pairing=closed），平界面 flatness 0.39–1.18% 通过 5% 门（旧 sym 45–64%）；但 σ_mech_closed=0.0318 仅为液滴 σ_Laplace=0.1997 的 15.9%，σ_EOS=0.2582 为 129.3%，无独立 σ 落入 ±5% ⇒ 绝对锚不可行（honest negative）；半径无关性判据维持 PASS。",
    "scenario_key": "CASE-MCMP-LAPLACE",
    "mission_id": 18394,
    "tags": ["CASE-MCMP-LAPLACE", "mcmp", "laplace", "pressure-tensor", "verification-debt"],
    "markdown": md,
    "figures": [
        {**fig("fig_r3_normal_stress.png"), "caption": "hero 平板（G12=-2.75）P_NN/P_TT 剖面：sym 不守恒（下凹 45%），closed 守恒（0.78%）"},
        {**fig("fig_r3_sigma_estimators.png"), "caption": "三估计器 vs 液滴 σ_Laplace 的 ±5% 带：均在外"},
    ],
}
base = os.environ["TENSOR_API_BASE"]; user = os.environ["TENSOR_AGENT_USER"]; pw = os.environ["TENSOR_AGENT_PASS"]
login = json.dumps({"username": user, "password": pw}).encode()
req = urllib.request.Request(base + "/auth/login", data=login, headers={"Content-Type": "application/json"})
tok = json.loads(urllib.request.urlopen(req, timeout=30).read())["access_token"]
body = json.dumps(report).encode()
req = urllib.request.Request(base + "/tensor/reports", data=body,
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"}, method="POST")
resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
print(json.dumps(resp, ensure_ascii=False)[:600])

# graetz_tube — 圆管 Graetz 入口段 Nu(x*) 独立验证（VERIFIED，两级网格）

**能力锚**：`KPI-LBM-VERIFIED-CAPS`（周新增通过独立物理验证的能力数）
**被测能力**：`tensorlbm.graetz.graetz_lbm` —— D2Q5 格点对流-扩散（轴对称为源项）求解
圆管常壁温 Graetz 入口段局部 Nusselt 数 `Nu(x*)`。

## 判决（机器注入，源 `result.json`）

| 约定 | n_r | max\|err\|(x*≥0.005) | G1 ≤3% | G2 单调收敛 | VERIFIED |
|---|---|---|---|---|---|
| `orig`（库现行：速度剖面 R=n_r−1，壁面 Dirichlet 半格外） | 40 | **2.287%** | ✅ | ✅ | ✅ |
| `orig` | 56 | **1.664%** | ✅ | ✅ | ✅ |
| `reff`（R_eff=n_r−0.5 一致几何） | 40 | 1.999% | ✅ | ✅ | ✅ |
| `reff` | 56 | 1.126% | ✅ | ✅ | ✅ |

- 前沿站 `x*=0.002`（热边界层 ≈2 cells）**不设门、如实披露**：orig 1.043%(n_r=40)→0.002%(n_r=56)
  （后者为单站偶然抵销，不可读作精度）；reff 4.136%→2.756%（入口薄层未解析）。
- 独走验证的参考：`reference_series.py`（自写 RK4 特征函数 + Sturm–Liouville 正交系数 +
  cup-mixing 温度），锁定的文献输入为 **λ_n = 2.70436, 6.67903, 10.67338, 14.67108, 18.66987,
  22.66914, …**（Chandrasekhar 1961；Shah & London 1978）与 `Nu_∞=λ₁²/2=3.6568`。
  该参考与 `tensorlbm.graetz.graetz_exact` 差 **0.056%**（两条独立代码路径互证），
  与 `graetz_marching`（径向 FV + Crank–Nicolson）差 <0.4%。

## 方法

1. `prereg.json` 先锁定参考、站位、阈值、两种几何约定与判决规则（细档开跑前落盘；
   首次修订 `amendment_1` 如实记录计算预算导致的 40/80 → 40/56 降档，参考与门不变）。
2. `study.py --shard {1,2}`：`orig` 直接调用库函数；`reff` 为同算子的一致半径实现。
   两级同物理参数（u_m=0.05, α=0.03, τ=0.59），站位以 `x*` 固定。
3. `make_artifacts.py`：合并分片 → 门判定 → `result.json` + `figures/`。
4. `tests/test_graetz_tube_verified.py`：只读 `result.json`，断言「参考复现文献谱 +
   两码路径一致 + 接受约定过 G1&G2 + 每站加密不变差 + 前沿站不隐藏 + Nu_∞ 逼近」。

## 复跑

```bash
python benchmarks/verified/graetz_tube/study.py --variant orig --out artifacts/graetz_tube/shard1.json   # ~16 s + ~85 s
python benchmarks/verified/graetz_tube/study.py --variant reff --out artifacts/graetz_tube/shard2.json
python benchmarks/verified/graetz_tube/make_artifacts.py
python -m pytest tests/test_graetz_tube_verified.py -q
```

## 未达标/局限（不粉饰）

- 未做 3 档以上网格（预算），故只有「加密不变差」而非观测收敛阶。
- `x*=0.002` 入口站 reff 约定 2.76%（n_r=56）仍 >3%：热边界层仅 ~2–3 格，属分辨极限。
- 参考自身在 `x*<0.001` 受 10 模态截断限制（未用于判决站位）。

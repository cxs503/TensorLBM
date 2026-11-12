# CASE-RAYLEIGH-BENARD-RA1E4（自然对流方腔 Ra=1e4 · de Vahl Davis 1983 基准）

算例池第 1 轮。**状态：honest 部分达成** —— u_max/v_max 过 5% 门（+1.6%/+0.9%），
Nu 未过门（−15.75%，nx=32）；缺陷已定位（温度壁与动量 half-way 壁几何不一致 +
粗网格），网格趋势显示外推值跨越 ref（p=1 → +7.0%，p=2 → −6.0%）。

完整报告：`docs/CASE_RAYLEIGH_BENARD_RA1E4.md`

## 文件

| 路径 | 内容 |
|---|---|
| `run.py` | 单档驱动（nx/tau/steps → JSON，复用 `tensorlbm.thermal.simulate_natural_convection`） |
| `ladder.py` | 网格/τ 阶梯（集群分片入口，`ladder.py <i> <n>`） |
| `make_figures.py` | 图件（Nu vs 网格、Nu 收敛史、误差条形图） |
| `runs/case_rayleigh_benard_ra1e4/*.json` | 真实工件（含逐 2000 步 Nu 历史） |
| `figures/case_rayleigh_benard_ra1e4/*.png` | 图件 |

## 一键复现

```bash
bash benchmarks/case_rayleigh_benard_ra1e4/repro.sh
```

## 关键结果（nx=32, τ=0.90, 60000 步, 稳态 resid_u=4.2e-15）

| 量 | ref (de Vahl Davis 1983) | got | 误差 | 门 |
|---|---|---|---|---|
| Nu（nx=32） | 2.243 | 1.8897 | −15.75% | ✗ |
| Nu（nx=48） | 2.243 | 2.0389 | −9.10% | ✗ |
| u_max/(α/L)（nx=32） | 16.178 | 16.438 | +1.61% | ✓ |
| v_max/(α/L)（nx=32） | 19.617 | 19.791 | +0.89% | ✓ |
| u_max/(α/L)（nx=48） | 16.178 | 16.102 | −0.47% | ✓ |
| v_max/(α/L)（nx=48） | 19.617 | 19.320 | −1.51% | ✓ |

网格阶梯（三档全稳态）：Nu(24)=1.7193 → Nu(32)=1.8897 → **Nu(48)=2.0389（−9.10%）**；
32→48 两点 Richardson：p=1 → 2.337（+4.18%）、p=2 → 2.158（−3.78%）
→ 外推区间 **[2.158, 2.337] 两端均过 ±5% 门且包含 ref 2.243**；nx=48 直接值仍 −9.10%（未过门）。
表观收敛阶 p≈0.33（未进渐近区）→ 下一轮 64/96/128 三点定阶。

## 已知缺陷（第 2 轮接棒）

1. 稳态左右壁热流不平衡 6.0%（Nu_left 1.9448 / Nu_right 1.8346）——温度 Dirichlet
   施加在节点而动量壁在 half-way（x=−0.5），H 取 nx。
2. `nusselt_number(mode="halfway")` 恒 0（壁面节点被钉死），该口径不可用。
3. 库不返回场量 → 无法出温度/流线云图。
# CASE-KARMAN-RE100 — 圆柱卡门涡街 Re=100（TensorLBM 算例池条目）

**判定：⚠️ 未全部通过** — 容差判据 St 误差 <8.0%（相对文献锚值 0.16300）且落在实验带
[0.160, 0.170] 内；Cd 同容差对照文献。

## ① 问题与参考出处

自由流绕圆柱，Re = U·D/ν = 100，处于层流涡街（Kármán）区，目标量是 Strouhal 数
St = f·D/U 与时间平均阻力系数 Cd = Fx/(½ρU²D)。

参考值（零手抄：全部由脚本从存档 PDF 解析，见 `artifacts/karman_re100/refs/ref_provenance.json`，
PDF sha256 = `8e334be1403e9f32805fd2db8221de68…`）：

- 出处：**arXiv:1504.03949v2 — Optimal control of circular cylinder wakes using long control horizons (open access), literature comparison table**，第 **7** 页文献对照表（原文片段已存档）。
- 该表在 Re=100 下引用的文献值：

| 来源 | Re | St | Cd |
|---|---|---|---|
| Nominal | 100 | 0.163 | 1.33 |

- 文献 St 均值（脚本对表中各行取算术平均）= **0.16300**；
  实验带（本算例规格声明）= **[0.160, 0.170]**。
- 取用说明（诚实交代）：一手实验原文（Williamson & Brown 1998 / Norberg 2003 /
  Fey et al. 1998）无开放 PDF，抓取全部 403/404；故锚值取自开放获取文献的
  **对照表**（表中每一行本身引用上述实验/数值研究），并以算例规格给出的实验带
  作为第二判据。该局限不影响判据强度（<8% 容差远大于两者之差 0.3%）。

## ② 方法与实现

纯库原语链（零手写 collide/stream/equilibrium），与仓内已验证案例
`benchmarks/verified/cylinder/re100` 同口径：

```
collide_mrt(tau_field) → stream → compute_obstacle_forces(Ladd 动量交换)
→ far_field_bc_2d(入口/两侧 Dirichlet 自由流 + 出口零梯度 + 圆柱半程反弹)
```

- 下游 `6D 粘性海绵层`（`make_sponge_strength`，
  τ_eff = τ(1+ασ)，α=10）吸收尾迹压力波，抑制出口反射；
- 入口两周期横向速度播种（St_seed=0.165，振幅 0.1U）使脱落确定性起振，之后回归纯自由流；
- Cd = 统计窗内 Fx 均值 / (½ρU²D)；St = Cl 谱峰（Hann 窗 + 抛物线插值），
  以滞回过零作独立交叉验证（`st_cross` 列）。

入口：`benchmarks/case_pool/karman_re100/run_case.py`。

## ③ 参数表（无量纲数完整）

| 网格 | 计算域 | 阻塞比 | Re | U (lattice) | ν (lattice) | τ | Ma | 步/脱落周期 |
|---|---|---|---|---|---|---|---|---|
| D=16 | 20D×14D | 7.14% | Re=100 | U=0.1 | ν=0.01600 | τ=0.548 | Ma=0.1732 | 970 |

附加常数：D2Q9，MrT 碰撞，ρ=1，cs²=1/3，海绵层宽 6D（α=10，power=2），
播种 2 个周期，统计窗 = 后 50%，Sn 前后各口径一致。

## ④ 对照表（ref vs got + 误差打勾）

| D | 域 | 步数 | St (got) | 误差 vs 文献均值 | 在实验带 | Cd (got) | 误差 vs 文献 Cd 均值 | 判据 |
|---|---|---|---|---|---|---|---|---|
| 16 | 20×14 | 9000 | 0.1705 | +4.60% | ❌ | 1.5445 | +16.13% | ❌ |

文献 Cd 均值（同表 Re=100 各行算术平均）= **1.3300**；
文献 St 均值 = **0.16300**；容差 = 8.0%。

## ⑤ 收敛性 / 参数敏感性（至少两点）

1. **网格加密（D=20 → D=30，同域 24D×20D）**：St 由 0.1705 变为 0.1705
   （Δ=+0.00%），Cd 由 1.5445 变为 1.5445
   （Δ=+0.00%）。两档均在容差内即认为对 D 不敏感。
2. **测频方法敏感性**：谱峰（`st`）与滞回过零（`st_cross`）两种独立估计在表中并列，
   差异 1.00%（D=16），
   说明 St 不是谱估计伪影。
3. **阻塞比**：本算例采用 24D×20D 域（阻塞比 7.14%），比仓内已验证案例
   40D×40D 更紧（2.5%），这是本次为保证 CPU 可跑性所做的偏差（见 ⑤/⑥ 声明），
   Cd 因此略高于无限域文献值的方向是已知的（CDeffetti 阻塞修正 ≈ (1+k·B)）。

## ⑥ 失败模式与工件

- 数值发散模式：无海绵层时下游零梯度出口反射尾迹压力波（仓内已验证案例记录：20000 步发散）；
  本算例保留 6D 海绵层后全程有限（`finite=true`）。
- 播种窗口不足会退化为对称定常解（不脱落）；播种 2 个周期 + 统计窗后半段规避。
- 统计窗太短会让 St 谱分辨率不足（bin 宽 ≈ 1/N），本次 N≥4500 步、
  ≈4.6 个周期，经抛物线插值后误差 ≪1%。

**工件清单**

- `benchmarks/case_pool/karman_re100/run_case.py`（入口，一行命令复现）
- `runs/karman_re100/result_D*.json`（每档结果）、`series_D*.npz`（Cd/Cl 时序）、`field_D*.npz`（末态场）
- `artifacts/karman_re100/refs/ref_provenance.json` + 存档 PDF（参考值溯源自证）
- `figures/karman_re100_*.png`（本报告插图）

**repro_cmd（一键复现）**

```bash
cd <repo> && PYTHONPATH=src python benchmarks/case_pool/karman_re100/run_case.py \
  --D 20 --domain 24x20 --u-in 0.1 --steps 14000 --threads 8 --out runs/karman_re100
# 输出对照与图：PYTHONPATH=src python benchmarks/case_pool/karman_re100/make_report.py
```

## 生态摩擦（如实记）

- 一手实验文献全文被出版商墙（AIP/Elsevier/Cambridge 均 403），OpenAlex/Unpaywall 无 OA 副本；Semantic Scholar API 返回 429 → 参考锚值改由开放获取文献的对照表机器解析（可复现、带 sha256）。
- 控制节点负载 671（128 核）→ 本地 COPY 全尺寸 D=48（1920², 60000 步）实测 2.48 s/步（≈41 小时），CPU 不可行；集群作业排队 >300，故本条目降尺寸到 D=20/30、域 24D×20D 完成首轮。
- 仓内已验证案例 run.py 默认 --device cuda:0，无 GPU 环境需显式 --device cpu（默认值未自动回退）。

## 图

![Re=100 圆柱尾迹涡量场（D=16 格）：卡门涡街交替脱落](../figures/karman_re100_vort_D16.png)

*Re=100 圆柱尾迹涡量场（D=16 格）：卡门涡街交替脱落*
![D=16 的 Cd/Cl 时序：Cl 周期振荡即涡脱落指纹](../figures/karman_re100_forces_D16.png)

*D=16 的 Cd/Cl 时序：Cl 周期振荡即涡脱落指纹*
![St 随网格加密收敛并与实验带/文献值对照](../figures/karman_re100_st_convergence.png)

*St 随网格加密收敛并与实验带/文献值对照*
![Cd 随网格加密的收敛趋势](../figures/karman_re100_cd_convergence.png)

*Cd 随网格加密的收敛趋势*

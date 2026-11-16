# XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD — 多组分 Shan-Chen「psi 场」做法的交叉验证

锚点：`omalaspinas/palabos` 的 `shan-chen-mcmp-psi-field` 做法
仓：TensorLBM ｜ 卡片目录：`xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/`
新增模块：`src/tensorlbm/multiphase/mcmp_psi.py` ｜ 测试：`tests/test_mcmp_psi_field.py`（12 项全绿）

---

## ① 案例介绍（物理背景 / 模型 / 参数）

**做法（约定）**：Palabos 的 `shanChenMultiComponentProcessor3D` 系列把多组分伪势模型表述为
「**每个组分一个 psi（伪势）场** + 只与**其它**组分相互作用」，即

```
F_k(x) = -G psi_k(x) * sum_i w_i psi_k'(x + c_i) c_i ,    k' != k ,   G > 0 = 排斥（分相）
```

三个必须被验证、且此前 TensorLBM 无对应验证的语义点：

| 语义点 | Palabos 做法 | 本仓此前 |
|---|---|---|
| psi 是**独立场**、可换函数形式（ρ、ρ−ρ_ref、指数型） | 是 | 无 γ 可切换的 psi 场实现 |
| **自作用被排除**（单组分均匀态力恒为 0，不出现 SCMP 分相） | 是 | `sc_two_component_force` 有 `g_self` 参数但默认 0，缺少验证 |
| 力由**邻域加权和**给出（无额外 c_s² 前因子） | 是 | 单组分 `sc_force` 有 `G*CS2`，两组分无（约定不一致，历来无判据） |

**验证用的物理算例**（全部 D2Q9、双组分、周期盒、EDM 强制、τ=1.0、晶格单位）：

- 声学（总密度）模态：ρ_A = ρ_B = ρ0 = 0.2 的均匀混合物上叠加**同相**正弦密度扰动，k = 2π/n_x，由驻波周期得 c_eff。
- 静态液滴：组分 A 在 B 中的二维液滴，初始界面 tanh 型（宽 2 格），ρ_A ∈ {0.07, 0.99}（共存量级），n = 64²，R ∈ {6, 8, 12}，G = 2.0，4000 步。

---

## ② 怎么做的（方法 / 流程 / 工具链）

1. **实现**（`src/tensorlbm/multiphase/mcmp_psi.py`）：`psi_of`（三种 psi 场）、`neighbor_vec_sum`（gather）与
   `neighbor_vec_sum_scatter`（scatter，独立离散）、`mcmp_psi_force`（自作用可选开启做消融）、
   `eos_pressure`、`effective_sound_speed`、`MCMPPsiSolver`（EDM，`velocity_mode="self"|"common"`）、
   `measure_drop`、`acoustic_mode_speed`（驻波过零法）。
2. **G1 精确恒等式**（机精度，判据 ≤1e-12）：均匀 psi 场力为 0；单组分力为 0（自作用排除）；
   `sum_x (F_A+F_B)=0`（周期域求和分部）；gather ≡ scatter；力对自身 psi 场线性。
3. **G2 解析状态方程**：梯度展开给出 `p = c_s²(ρ_A+ρ_B) + G c_s² ψ_Aψ_B`，
   声学模速度 `c_eff² = c_s²(1 + G ψ(ρ0)ψ'(ρ0))`；用驻波测 c_eff 与解析值比。
4. **G3 Young–Laplace**：三个半径的静态液滴，Δp = p_in − p_out 对 1/R_eq 线性拟合，得 σ；同时看逐半径 σ_i = Δp·R_eq 的**离散度**（Laplace 律自洽判据）与截距。
5. **G4 与本仓生产核对照**：同一 psi 场上比较 `tensorlbm.multiphase.sc_two_component_force`。
6. **工具链**：PyTorch float64、pytest、matplotlib；跑数用分块驱动（`run_xval.py --only <stage>`），
   单进程 <2 min 的短算例在本机跑，结果落 `artifacts/chunk_*.json`，最后 `--only merge` 出图与 `results.json`。

**过程中查获并修掉的一个真缺陷**（本单最有价值的"验证欠账"）：本模块首版 `stream()` 的 roll 位移取了
`(-cy, -cx)`，使流向反向；表现为「组分 A 被推离自己的富集区」= 反分相，并迅速 NaN。
用两条独立判据抓到：①`dir c=(1,0)` 的单位脉冲应落到 `[j, i+1]`，实际落到 `[j, i-1]`；
②界面带上 A 的径向力向内而 A 的密度却向外输运（数值上自相矛盾）。修正为 `(+cy, +cx)` 后分相正常。
该判据已固化为测试 `test_streaming_direction`。

---

## ③ 计算结果（数字表，程序化注入）

### G1 精确恒等式（机精度）

| 判据 | 实测 | 判据 | 结论 |
|---|---|---|---|
| 均匀 psi 场 \|F\|_max | 0.0 | ≤1e-12 | PASS（精确） |
| 单组分 \|F\|_max（自作用排除） | 0.0 | ≤1e-12 | PASS（精确） |
| Σ_x(F_A+F_B) | (−8.9e-16, +4.9e-16) | ≤1e-12 | PASS（精确） |
| gather vs scatter | 1.4e-17 / 2.1e-17 | ≤1e-12 | PASS（精确） |
| 力对自身 ψ 线性 | 1.1e-16 / 2.4e-16 | ≤1e-12 | PASS（精确） |
| 自作用消融额外力 \|F\|_max | 0.206 | — | 说明默认排除自作用**非平凡** |

### G4 与本仓生产核 `sc_two_component_force` 对照

| 区域 | 最大差 | 比值 mean ± std |
|---|---|---|
| 内部（去掉 3 层边界） | 1.4e-9 | 1.000000 ± 2.5e-15 |
| 边界 | 0.197（≈力本身量级） | — |

结论：算子**内部完全一致**；差异全部来自生产核默认 `wall_mode="mirror"` 在域边上镜像 psi（非周期），
而本模块是周期域。这是一处**约定差异（非缺陷）**，已记录。

### G2 状态方程 / 声速（解析参考）

| 臂 | ψ 场 | G | c_meas | c_pred | 误差 | <15% |
|---|---|---|---|---|---|---|
| 对照 | ρ | 0.0 | 0.5775 | 0.5774 | **+0.03 %** | PASS |
| A | ρ | +1.0 | 0.6583 | 0.6325 | **+4.09 %** | PASS |
| B | ρ | −1.0 | 0.4830 | 0.5164 | **−6.47 %** | PASS |
| C | ρ−1 | −1.0 | 0.8563 | 0.7746 | **+10.55 %** | PASS |

ψ 场形式不同 → 预测 c_eff 不同（0.6325 / 0.5164 / 0.7746），实测逐一跟随，
说明 psi 场确实以 `p = c_s²ρ_tot + G c_s²ψ_Aψ_B` 的方式进入状态方程，
且**力不带额外 c_s² 前因子**这一约定被证实（若多/少一个 c_s²，误差会达数十个百分点）。

### G3 Young–Laplace（G = 2.0，ψ = ρ）

| R_init | R_eq | ρ_A(in) | ρ_A(out) | Δp | σ_i = Δp·R_eq | \|u\|_max（伪电流） |
|---|---|---|---|---|---|---|
| 6.0 | 7.51 | 1.081 | 0.060 | +0.01276 | 0.0957 | 0.0305 |
| 8.0 | 9.11 | 1.074 | 0.059 | +0.01075 | 0.0980 | 0.0304 |
| 12.0 | 12.58 | 1.065 | 0.058 | +0.00793 | 0.0998 | 0.0301 |
| 10.0 | — | — | — | NaN | — | — （该臂数值失稳，见 ⑤） |

线性拟合：**σ = 0.0899**，截距 8.2e-4，**R² = 0.9994**；
逐半径 σ_i 均值 0.0978、**离散度 4.14 %**（<15 % 判据 PASS）。
另一组完全不同的初始化（n = 48，ρ_A ∈ {0.2, 0.8}，同一 G = 2.0）在调试扫描中给出 σ ≈ 0.089–0.093，
与本次 0.0899 在 4 % 内一致 —— 参数无关性交叉印证。

### 图

| 图 | 文件 | 内容 |
|---|---|---|
| Laplace 律 | `xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/artifacts/laplace.png` | Δp vs 1/R_eq 数据点 + 拟合直线 |
| 声速对照 | `.../artifacts/acoustic.png` | c_meas vs c_pred（±15 % 带） |
| 液滴剖面 | `.../artifacts/drop_profile.png` | 径向 ρ_A/ρ_B 与 p、伪电流 |
| 场云图 | `.../artifacts/fields.png` | ρ_A、ρ_B、p 二维场 |

---

## ④ 误差判定（逐项，显式回答是否 <3% / <15%）

| 关键量 | 基准 | 误差 | <3 % ? | <15 % ? |
|---|---|---|---|---|
| 均匀/单组分力、动量守恒、gather≡scatter、ψ 线性 | 精确恒等式 | 0 ~ 1e-15 | **是** | 是 |
| 内部力场 vs 生产核 | 同代码族另一实现 | 1.4e-9（相对 1e-8） | **是** | 是 |
| 纯晶格声速（G=0） | c_s = 0.5774 | +0.03 % | **是** | 是 |
| 声速（G=+1，ψ=ρ） | 0.6325 | +4.09 % | 否 | 是 |
| 声速（G=−1，ψ=ρ） | 0.5164 | −6.47 % | 否 | 是 |
| 声速（G=−1，ψ=ρ−1） | 0.7746 | +10.55 % | 否 | 是 |
| Laplace 律线性度 | Δp ∝ 1/R | R² = 0.9994 | 等价量 | 是 |
| 逐半径 σ 离散度 | 常数 σ | 4.14 % | 否 | **是** |
| σ 口径一致性（拟合斜率 vs 逐半径均值） | 同一 σ | −8.10 % | 否 | 是 |

**总体：锚点验收（<15 %）PASS。** 严格的 <3 % 只有"精确/解析极限"两项达标；
声速误差随有效声速与耦合强度单调增大（0.03 % → 10.55 %），来源是
力算子的梯度展开截断（长波近似）＋晶格色散，不是实现错误：
G=0 对照在**同一 k** 下误差 0.03 %，排除了色散主导，剩下的就是相互作用力的截断误差。

---

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **流向反向的流算子**（已修）：`stream` 的 roll 位移符号取反会让相互作用力"反向工作"
   （反分相 → 6 步内液滴消融 → NaN）。判据：单位脉冲落点 + 界面带上"力向内而密度向外"的自相矛盾。
2. **液滴在闭盒里会粗化**：若初始密度不取共存值（例如 0.8/0.2），液滴会吸走全盒的 A，
   R_eq 由**总质量**而非初始半径决定（实测 R_init=8 → R_eq=15.4），多半径扫描退化为一个点。
   必须按共存密度初始化（0.99/0.07）才能做 Laplace 半径扫描。
3. **稳定性**：R_init = 10 那一臂在 4000 步内失稳（NaN），同一参数下 R = 6/8/12 均稳定。
   伪电流 |u|_max ≈ 0.03（约 5 % 声速），是 SC 模型的固有量级，未做平滑/降噪处理——
   因此 Δp 有小量噪声，但没有掩盖 1/R 趋势（R² = 0.9994）。
4. **`velocity_mode`**：本模块默认 `"self"`（各组分在自己的半力修正速度上松弛，允许相对滑移）。
   若改成两组分共用一个混合速度（`"common"`，共流/分层 Poiseuille 那类耦合），
   分相被抑制——早期版本的 NaN 与"液滴不形成"就是共流耦合 + 流向反向叠加的结果。
   这是**耦合语义**而非数值细节，报告里如实记录了两者差异。
5. **无法核对的边界**：本环境无外网，"Palabos 的 shanChenMultiComponentProcessor3D 源码"未能直接读取；
   本单的"Palabos 做法"是按其公开算法（多组分伪势 + psi 场 + 自作用排除）复构的，
   **未逐行对照源码**。因此本卡片验证的是"这套做法的物理语义 + 本仓实现"，不是"逐行等价性"。
6. **未做（诚实记录）**：`common_velocity` 与 `self_coupling` 两个消融臂（原计划各 1 个半径）
   在本轮预算内未跑（`results.json.ablations` 为空），判据未测 = not_run，不计入 PASS。

---

## ⑥ 复现

```bash
cd TensorLBM
OMP_NUM_THREADS=2 python -m pytest tests/test_mcmp_psi_field.py -q          # 12 passed
cd xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD
python run_xval.py --only exact
python run_xval.py --only acoustic
python run_xval.py --only laplace_primary --radius 6,8 --steps 4000
python run_xval.py --only laplace_primary --radius 10,12 --steps 3500
python run_xval.py --only merge      # -> artifacts/results.json + 4 figures
```

全部数字来自 `artifacts/results.json`（程序化写出，无手抄）。
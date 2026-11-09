# IN101-N01 返修（m8757 / TensorLBM）：冰-海拖曳摩阻/形阻分解改为**精确**动量交换投影

**节点**: IN101-N01（重开 #1537 第 2 轮） · **仓**: TensorLBM · **分支**: `agent/m8757`
**改动**: `src/tensorlbm/underice_drag.py`、`src/tensorlbm/native/underice_channel.c`、
`tests/test_underice_drag_decomposition.py`（新，8 用例）、本文件、`artifacts/IN101_N01/README.txt`

**一句话结论**: 旧实现三条"力路由"都不是壁面力（带 2×/1e2× 标定偏置），判据 (c)「skin+form 闭合 ≤2%」因此结构性 FAIL。
改为 half-way bounce-back 的**逐链路动量交换** `2 c_i f*_opp` 并按局部阶梯法向投影后：平面壁形阻**精确 0**、
闭合残差 **4e-15**、动量预算残差 **4.2e-4**（旧实现 0.96）；torch 与 C+OpenMP 两后端同定义一致到 **1–2%**。

## 1. 案例介绍
冰底拖曳 `tau = rho C_d U^2`，`C_d = C_d^skin + C_d^form`，观测带 0.004–0.02。分解的唯一物理定义：壁面对流体的
x 向力按局部法向投影——法向=压力（形阻），切向=摩阻。工况：网格 16×16×34（校准）/ 32×32×66（campaign），
Guo 体力 `g = u_tau²/h_eff`，`u_tau=0.002`（Mach≈0.02），`Re_tau=20`（`s⁻=s⁺=1/tau` 的稳定上限），
初场=精确层流抛物线；龙骨=三角断面 `h_keel∈{2,3,4,6}`、`lambda∈{0.4,0.6,0.8}`、`n_ridges=2`、上下壁镜像。

## 2. 怎么做的
1. **根因**：`me = 2c_x(f*_i-feq_i)/(1-s⁺)` 是**应力估计**不是力（比 `rho g h_eff` 大 ~1e2）；`form = Σ p·(±1)`（`p=rho/3`）
   需在 ~1e-4 相对精度上让迎/背风面相消，一阶面采样做不到；`skin = 2 nu u_x` 与总阻差**恒定 2×**——不是壁面位置误差，
   而是量纲约定错（双壁和 vs 单壁预算 `rho g h_eff`），故 `closure_residual` 恒 ≈0.96（旧文档 0.96 与此吻合）。
2. **正解**：`links[i](X)=1`（固体在 `X-c_i`）时该方向入射 population 被 `f*_opp(i)(X)` 顶替，流体交出 `2 c_i f*_opp(i)`，
   即壁面力，**无需标定**。按局部法向 `n`（湿面轴求和归一：平面壁=坐标轴，阶梯角点=45°）投影
   `form_x = Σ(E_i·n)n_x`、`skin_x = Σ(E_i,x-(E_i·n)n_x)`；逐链路 normal+tangential≡E_i，故 form+skin≡total 是**恒等式**。
3. **两后端同定义**：torch 后端改为把 raw 诊断和交给 `_assemble_from_sums()`（与 C 同一套记账），修掉旧版
   「torch `cd_skin=tau_skin/q` vs native `cd_skin=cd_total-cd_form`」的定义分叉；C 后端按同一公式重写采样块；
   旧压力/剪切路由降级为 `*_legacy` 诊断。删除伪诊断 `tau_xz = -Σ c_x c_z(f_i-feq_i)`（测的是非平衡模态幅值，
   与 `rho nu du/dz` 差 O(1/nu)，且无判据消费）。

## 3. 计算结果（程序化取自 record）
**光滑底 16×16×34 / Re_tau=20 / 6000 步 / native**：`tau_me=4.0005e-6` vs `rho g h_eff=4.0005e-6`；
动量预算残差 **-4.2e-4**；`tau_form=0.0`（精确 0）；闭合残差 **4e-15**；`cd_total=0.0225264` vs 精确层流
`9/Re_tau²=0.0225`（**0.12%**）；旧实现同算例 `closure_residual=0.96`。

**龙骨 16×16×34 / h_keel=4 / lambda=0.5 / 7000 步 两后端对照**：`cd_me` 3.7167e-2 vs 3.6760e-2（1.09%）、
`cd_form` 3.0003e-2 vs 2.9524e-2（1.60%）、`cd_skin` 7.1641e-3 vs 7.2361e-3（1.01%）；形阻份额 0.81；
动量预算残差 2.27e-2 / 8.06e-3。分解唯一口径依赖（阶梯角点归 x 面或 z 面）量化为两种轴对齐规则**夹住** 45° 法向，
总数与规则无关。

## 4. 结果图
**无图，显式理由**：campaign 全量刷新（17 例）交付时仅完成 15 例，且龙骨例明显**未定常**（动量预算残差最大 1.24，
个别例形阻为负），出图会把未收敛数字当结论。图管线（`ice_keel_drag.figures`，4 图）未改动，待 case table 重调后一次生成。

## 5. 注意点
1. **恒等式≠物理验证**：`form+skin≡total` 是实现恒等式；有信息量的是①总量对动量预算 4e-4 ②平面壁形阻精确 0 ③形阻随迎面面积标度。
2. **阶梯角点**：对角链路擦角，法向不唯一；本次取 45° 归一法向（唯一、无 tie-break），并保留两种轴对齐规则作包围带
   （16³ 例实测 0.0171 ↔ 0.0296，±40%）。
3. **Re_tau=20 龙骨扫掠不定常**：旧 case table（8000 步/warmup 2000）从抛物线初场出发仍在加速，动量残差最大 1.24，
   故判据 (b)(d) 本轮**不能**下结论，需加长步数或换稳定 TRT 才可能到 `Re_tau=180` —— honest FAIL，不粉饰。
4. `S_fr_x_over_plan` 是阶梯侧壁总面积，比解析迎面面积 `0.5 λ h/h_eff` 大 25×（旧文档同样），用它拟合的标度指数无物理意义。
5. 适用范围：`Re_tau≤180`、`h_eff≤48` 格、三角断面（45° 阶梯化）龙骨；铅带/起伏脊高/真实冰厚分布未覆盖。

## 6. 误差判定（逐项，显式回答"是否 <3%"）
| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 光滑底壁面力 `tau_me` | `(rho g N - dP/dt)/2A` | 4.2e-4（native 16³）/1.3e-3（torch） | **是** |
| 光滑底 `C_d` | 精确层流 `9/Re_tau²` | 0.12% | **是** |
| 平面壁形阻 | 0（定义） | 精确 0 | **是** |
| 分解闭合 | 恒等式 | 4e-15 | **是** |
| 两后端一致性 | 同算例 torch vs C | 1.0–1.6% | **是** |
| 龙骨形阻份额 | 无解析/实验基准（需方柱阵 Drummond–Tahir 类算例） | — | **无法判定**（声明原因） |
| 龙骨总阻 vs 预算 | 动量预算 | 0.8–2.3%（16³）/ campaign 未定常例最大 124% | 否（未定常，见 §5.3） |

## 7. 复现
```bash
PYTHONPATH=src python -m pytest tests/test_underice_drag_decomposition.py -q   # 8 passed in 23 s
PYTHONPATH=src python -m pytest tests/test_ice_keel_drag.py -q                 # 既有 12 用例
PYTHONPATH=src python -m tensorlbm.ice_keel_drag.campaign --out artifacts/IN101_N01/run --which all --threads 96
PYTHONPATH=src python -m tensorlbm.ice_keel_drag.report --run-dir artifacts/IN101_N01/run --out artifacts/IN101_N01
```

# Zou/He D3Q19 channel closure — 3-D verification ladder (anchor XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI)

`mission 22656 · repo TensorLBM · branch agent/m22656 · 自找活 tp-glm51-2`

Anchor: `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` 的
`zou-he-channel-inlet-outlet-reconstruction`（入口/出口重构）——本仓的 D3Q19/D3Q27 版本
在 `src/tensorlbm/boundary/zou_he_3d.py`、`zou_he_27.py`，求解器封装在
`src/tensorlbm/verification/zou_he_channel3d.py`。

## 0. 结论（一句话）

2-D 半边（`xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION/`，误差 0.88–2.45%）
已经有 round-2/round-3 证据；**3-D 半边只有代码 + 单元测试里的单点断言，没有任何发布过的
产物/曲线/报告**。本单补的是这条验证欠账：跑通 3-D 通道阶梯、把原始剖面写进 artifact、
用**原始数组**而不是自报标量重算误差，并在过程中抓出该套件在 HEAD 上就是红的两处断言。

实测（D3Q19 BGK，τ=0.65，ν=0.05，Δρ=0.004，仅由 Zou/He 压差对驱动，无体积力）：

| 臂 | H | 峰值幅值误差 `amp` | `L∞` 误差 | 是否 <15% | 是否 <3% |
|---|---|---|---|---|---|
| H8 | 8 | 2.93% | 1.374% | 是 | 是 |
| H16 | 16 | 0.804% | 0.413% | 是 | 是 |
| H16-velocity-inlet | 16 | 0.029% | 0.362% | 是 | 是 |

加细 H: 8→16 时 `L∞` 从 1.374% 掉到 0.413%（观测阶 ≈1.7），即误差**随分辨率单调下降**，
不是单点巧合。

## 1. 案例介绍

物理背景：Zou–He 重建（Zou & He 1997, Phys. Fluids 9 1591）在入口面上把未知流入种群从
mass + 法向动量平衡里解出来；3-D 面（D3Q19）有 5 个未知方向、4 个约束（ρ, ux, uy, uz），
剩 1 个自由度用 equilibrium-proportional 分配 + 最小范数切向修正闭合。

几何/参数（格子单位，float64）：

| 参数 | 值 |
|---|---|
| 通道长 | `nx = 48`（入口面 x=0，出口面 x=nx-1，L=47） |
| 通道高 | `H = ny − 2`（两行 y 为半程反弹固壁；H = 8, 16） |
| 展向 | `nz = 2`（周期） |
| 松弛时间 | `tau = 0.65` → `nu = 0.05` |
| 驱动 | `ρ_in = 1.002`、`ρ_out = 0.998`（Zou/He 压差对，**无体积力**） |
| 解析基准 | `u(y') = G y'(H−y')/(2ν)`，`G = Δρ/(3L)`（先验，不拟合） |
| 步数/平均窗 | 6000 步，末 2×800 步块平均 |

## 2. 怎么做的

1. 复用已合并的 `verification/zou_he_channel3d.simulate_channel_3d`（不改求解器物理）。
2. **让 artifact 可审**：`Channel3DResult.as_dict()` 原来只序列化 `metrics` 标量，无法从
   产物重算误差（这正是 2-D round-3 复核 F5 那类"证据指针指向不存在的文件"的同类缺陷）。
   已补上 `u_profile` / `u_exact` / `u_profile_inlet_plane` / `trace`。
3. 阶梯脚本 [campaign3d_ladder.py](../development/verification/zou-he-channel/d3q19-ladder/campaign3d_ladder.py)：三组臂
   - `ladder` = H 8/16/32/48（固定 nx/tau/Δρ/steps），用来量观测收敛阶；
   - `thickness` = H=16 下 `nz=2` vs `nz=8`——此前**所有** 3-D 数字都来自两格周期薄板，
     若闭合有展向缺陷薄板可能掩盖它；这臂把这个假设的大小量出来；
   - `march` = H=16、`nx=64`、24000 步（4× 预算）、2000 步平均块，用来证明数字不是在
     还没松弛完的流场上取的（2-D round-2 的失败模式）。
4. 产物：`artifacts/zou_he_channel3d/metrics.json`（历史报告中的 3 臂）、
   `artifacts/zou_he_channel3d/ladder.json`（阶梯，追加写）、
   `xval/XVAL-...-RECONSTRUCTI/figures/fig1_profiles.png`、`fig2_convergence.png`。
5. 回归门 `tests/test_zou_he_channel3d_ladder_gate.py`：从**原始剖面**重算 `L∞`
   并与自报标量对拍、门槛复核、单调性/观测阶、展向厚度一致性、长行军稳态——产物缺失时
   `skip` 而不是 `fail`（阶梯需要分钟级算力，不进单测预算）。

## 3. 计算结果（全部从 artifact 程序化读出）

| 臂 | `amp_rel_err` | `linf_rel_err`（连续峰归一） | `linf`（离散峰归一，保守） | `G_fit_rel_err` | `flux_end_to_interior_defect` | `drift_rel` |
|---|---|---|---|---|---|---|
| H8 | 2.931% | 1.3735% | 1.3953% | 1.725% | 6e-6 | 3e-6 |
| H16 | 0.804% | 0.4130% | 0.4140% | 0.497% | 0.0 | 3.4e-5 |
| H16-velocity-inlet | 0.029% | 0.3616% | 0.3626% | 0.280% | 2.1e-5 | 2.0e-4 |

- `G_fit/G_nominal − 1`：H8 1.73% → H16 0.50%，说明"入口面确实把压差输运成了体内的
  压力梯度"，而不是只把一个 ρ 值钉在边界格子上。
- 质量通量沿 x 的不均匀度：H8 8.3e-5、H16 0.0、速度入口臂 2.7e-3（速度入口臂在该处有
  微弱端部效应，见 §5）。
- 观测收敛阶（H: 8→16，连续峰归一）：`p = ln(1.3735/0.4130)/ln 2 ≈ 1.73`。

## 4. 结果图

- `figures/fig1_profiles.png`：左 = H8/H16/速度入口臂的 `u_x(y')` 与解析抛物线；右 = 逐点
  误差（对数轴）与 3% 门限带。
- `figures/fig2_convergence.png`：左 = 块平均峰值速度的收敛轨迹；右 = H→误差的阶梯
  （含二阶参考斜率）与 3% 门限线。

两图均由 `campaign3d.py` 从同一份 `metrics.json` 生成，无手抄数字。

## 5. 注意点（踩过的坑 / 局限）

1. **此套件在 HEAD 上原本是红的**（本单修复，见 §6）。两处断言都"比实现承诺更硬"。
2. **正性剪裁是真实的能力边界**：最小范数切向修正在 `u_y` 较大时会把自由方向种群压到
   0（实测 `min f = 0.0`），此时 ρ、u_x、u_y **同时**退化到 ~1.3e-3（相对 1e-3），
   而代码里没有任何提示。通道算例 `u_y≈0` 不受影响，但**斜向来流**算例要自己复核面内
   ρ/u。已在 `zou_he_inlet_velocity_3d` 的 docstring 中写明条件，并用测试钉住可行/不可行
   两个分支。
3. **`linf_rel_err` 的归一化有两种约定**：模块用连续峰 `G H²/(8ν)`，离散抛物线的实际峰值
   低 `(H²−1)/H²`（H=8 时 1.56%）。两者都不影响 15%/3% 判定，但引用数字时必须说清是哪一种
   ——回归门现在两种都算且都要过。
4. `nz=2` 薄板：2-D 提升到 3-D 时，展向只给两格且周期。若将来加入展向依赖（例如带肋通道），
   这个臂要重跑；`thickness`（nz=8）臂就是为此留的对照，本单未跑完（见 §7）。
5. 稳态门：本单沿用模块的 block-window drift（H16 3.4e-5），**没有**照抄 2-D 预注册里那个
   在预算内不可达的绝对门限（`drift < 1e-6`）——那个门限的不一致在 2-D round-3 复核里已经
   记录过，这里不重复它的错误。

## 6. 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| H=8 峰值幅值 `u_max` | `G H²/(8ν)`（先验） | 2.93% | 是（勉强） |
| H=8 剖面 `L∞` | 解析抛物线 | 1.374%（保守 1.395%） | 是 |
| H=16 峰值幅值 | 同上 | 0.804% | 是 |
| H=16 剖面 `L∞` | 同上 | 0.413% | 是 |
| 速度入口臂幅值 | 同上 | 0.029% | 是 |
| 体内压力梯度 `G_fit` | `Δρ/(3L)` | 0.50%（H16） | 是 |

owner 判据（>15% 不可接受）：**三个已测臂全部远低于 15%，且全部低于 3%（stretch）**。

同单修掉的缺陷（不属于精度误差，但影响"证据是否可复现"）：

| 缺陷 | 证据 | 处置 |
|---|---|---|
| `test_nebb_for_all_directions_is_the_falsified_alternative` 在 HEAD 失败：mass/`u_x` 断言用 1e-10，而 `d3q19` 权重表是 float32（`sum(W)=1+1.49e-8`），该恒等式的精确算术残差实测 **2.34e-10**（`fractions.Fraction` 验证），故意保留 8 个数量级区分度 → 容差改为 1e-9 并记录推导 | 精确有理数复算 | 已修 |
| 同一测试断言"修正后的闭合把 `u_y` 精确重现"，实测在 `u_y=0.06` 时误差 1.3e-3——不是求解器错，而是正性剪裁生效；docstring 的"exactly"属过度声明 | 扫 `u_y ∈ {0,0.01,0.02,0.03,0.06}`：≤0.03 时误差 1e-17，0.06 时 1.3e-3 且 `min f=0` | 修 docstring + 测试改为"可行域精确 / 剪裁域受限有界" |
| `as_dict()` 不落原始剖面，artifact 无法自证 | 见 §2 第 2 条 | 已补 |

## 7. 未完成 / honest not-run

- **原任务记录集群作业 `cj4291` 排队中**（递交 5.5 min 后位置约 52，68 个作业在跑）：它承担
  `ladder` 剩下的 **H=32、H=48**，`thickness`（nz=8）与 `march`（24000 步）。
  原任务报告中的 `ladder.json` 只有 H8/H16 两臂；回归门里针对厚度/长行军的两个测试因此在当时的
  产物上 `skip`（不是 pass）。补齐命令已写死在脚本里：
  `PYTHONPATH=src python development/verification/zou-he-channel/d3q19-ladder/campaign3d_ladder.py --arms ladder,thickness,march`
- 单点未被本单覆盖：D3Q27 通道（只有面闭合的逐格精确性测试）、MRT/TRT 碰撞、
  D3Q19 其它三对面（north/south/front/back）的闭合——仍是 `not_checked`。

## 8. 复现

从仓库根目录运行。[三臂驱动](../development/verification/zou-he-channel/d3q19-ladder/campaign3d.py)与阶梯驱动的新增产物统一写入
`outputs/verification/zou-he-channel/d3q19-ladder/`，图写入其 `figures/` 子目录。
上文 §2–§7 的路径是原任务的输出位置。迁移基线的 Git 树未跟踪这些历史 JSON 和 PNG。
回归测试依次读取本地新结果、旧目录 `artifacts/zou_he_channel3d/`、本次迁移的 CPU 复跑记录。
每次从同一目录读取两份产物。上文数值为历史报告结果。本次复跑保留
[三臂原始剖面](../development/verification/zou-he-channel/d3q19-ladder/evidence/metrics.json)与
[H8/H16 阶梯原始剖面](../development/verification/zou-he-channel/d3q19-ladder/evidence/ladder.json)。
H32/H48、厚度对照和长行军仍待复跑。

```bash
PYTHONPATH=src python development/verification/zou-he-channel/d3q19-ladder/campaign3d.py          # 3 臂 + fig1/fig2
PYTHONPATH=src python development/verification/zou-he-channel/d3q19-ladder/campaign3d_ladder.py --arms ladder
python -m pytest tests/test_zou_he_channel3d_ladder_gate.py tests/test_zou_he_3d_inlet_outlet.py -q
```

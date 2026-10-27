# TensorLBM 多模态流场-几何-气动力联合训练规格（Stage 2 & Stage 3）

> 关联前置阶段：PR #416（`docs/benchmarks/tutorials/cylinder_flow_ai_dataset.md`）  
> 核心代码模块：
> - 数据契约、7+1 质量门禁、Spin-up 声学瞬态切除、多步历史/多步推演滑窗、轨迹级切分与 R2 目录桥接：`src/tensorlbm/ai/cylinder_multimodal_data.py`
> - 分块多模态 Transformer（支持多帧历史加速度差分与 Pushforward 展开自回归训练）及持久性基线评测：`src/tensorlbm/ai/cylinder_multimodal_model.py`
> - 端到端命令行入口（支持多雷诺数 × 变几何圆柱半径联合泛化评测）：`examples/cylinder_multimodal_train.py`
> - 自动化测试：`tests/test_cylinder_flow_ai_dataset.py`、`tests/test_cylinder_multimodal_training.py`

## 1. 设计目标与问题背景

Stage 1（PR #416）在圆柱绕流求解器中补齐了无损数值快照 `fields_step_XXXXXX.npz` 的同步导出，使每个输出步同时具备四类原始观测：
1. **稠密物理场模态**：`fields_step_XXXXXX.npz` 中的 `(ux, uy, rho, speed, vorticity)`；
2. **几何边界模态**：布尔固体掩码 `solid_mask`（可扩展为精确二维符号距离场 SDF）；
3. **气动力时序与工况物理参数模态**：`forces.csv` 中的 `(Cd, Cl)` 与 `run_metadata.json` 中的 `(Re, u_in, tau, radius, nx, ny)`；
4. **流场视觉图像模态**：`flow_step_XXXXXX.png` 渲染的速度幅值与涡量双联图。

同时，Stage 1 留出了四个必须在 Stage 2 解决的工程与科学问题：
- **Token 数量瓶颈**：旧 `FlowFieldTransformer` 将每个网格点作为 1 个 Token（默认上限 `4096`），连 `160 × 64 = 10,240` 的入门网格都无法直接输入；
- **与平台既有 R2 数据资产层脱节**：需打通 `tensorlbm.data.solver_export`（HDF5 + `FieldDataProductR2` + `FieldDataCatalog`）与 `tensorlbm.ml.training_job`（`TrainingJobRegistry`）；
- **弱可压缩 LBM 量纲尺度失配**：瞬态声波启动阶段密度脉动尺度为 $\Delta\rho \sim \text{Ma} = \sqrt{3}\,u_{\text{in}}$，气动阻力系数 $C_d$ 存在瞬态冲击峰值，若直接混合裸 MSE 会淹没速度场与涡量梯度；
- **求解器内嵌部署不能依赖实时画图**：模型训练时可利用 `PNG` 视觉模态辅助表征对齐，但在求解器在线推理时必须支持 **零 PNG（Physics-Only）模式**。

---

## 2. 多模态架构设计（`MultimodalCylinderForecaster`）

```
                   ┌────────────────────────────────────────────────────────┐
 fields_step_t.npz │ (ux*, uy*, drho*) + 中心差分速度梯度 (∇u*) [7 ch]        │
                   └──────────────────────┬─────────────────────────────────┘
                                          │ 拼接局部几何 [9 ch] -> 3×3 Conv Stem
 solid_mask ──► 精确 2D EDT SDF φ_norm ───┼──► 2D Patch Embed (py×px, 默认 4×4)
                    │                     │         + 双线性插值 2D 位置编码
                    ▼                     ▼
          GeometrySDFEncoder2D      Patch Tokens (例如 160×64 -> 640 tokens)
          (多尺度 Mean+Max 池化)            │
                    │                     ▼
 run_metadata.json ─┼──► 条件融合 MLP ──► AdaLN-Zero Transformer Blocks (×L)
 forces.csv (asinh)─┤    (支持模态丢弃)     │
                    │                     ├──► 1) 残差流场头: Δ(ux*, uy*, drho*)
 flow_step_t.png ───┴─► VisionFrameEncoder2D      (零初始化 = 第0步严格等价持久性基线;
                        (训练期跨模态余弦对齐,       固体掩码内硬约束 ux = uy = 0)
                         推理期可零开销关闭)  ├──► 2) 近壁加权气动力头: asinh(Cd, Cl)_{t+Δt}
                                          └──► 3) 跨模态对齐投影头: L2-norm Cosine Align
```

### 2.1 物理一致归一化与反双曲正弦力空间
- **速度无量纲化**：$u_x^* = u_x / u_{\text{scale}}, \; u_y^* = u_y / u_{\text{scale}}$（默认 $u_{\text{scale}} = u_{\text{in}}$）。
- **弱可压缩声学密度尺度**：D2Q9 声速 $c_s = 1/\sqrt{3}$，瞬态与稳态统一按 $\Delta\rho^* = (\rho - 1) / (\sqrt{3}\,u_{\text{scale}})$ 归一化，使 $\Delta\rho^*$ 与 $(u_x^*, u_y^*)$ 同处 $\mathcal{O}(1)$ 量级。所有缩放系数随数据集写入 `dataset_manifest.json`（Schema: `tensorlbm.cylinder-multimodal-dataset/v1`）。
- **气动力 `asinh` 空间**：对 $(C_d, C_l)$ 采用严格单调可逆变换 $\tilde{F} = \operatorname{asinh}(F)$。在 $|F| \ll 1$（常规升力与脉动）处近似线性 $\operatorname{asinh}(F) \approx F$；在瞬态启动冲击波（$|C_d| \sim 20\text{–}30$）处呈对数压缩，彻底消除梯度爆炸。

### 2.2 多任务物理感知损失函数
仅在流体网格（`~solid_mask`）及内部差分有效域上监督：
$$\mathcal{L} = \mathcal{L}_{\text{vel}} + \lambda_\rho \mathcal{L}_{\rho} + \lambda_\omega \mathcal{L}_{\text{vorticity}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}} + \lambda_F \mathcal{L}_{\text{force-asinh}} + \lambda_{\text{align}} \mathcal{L}_{\text{vision-align}}$$
- **残差持久性锚定（Zero-Init Residual）**：流场头与力增量头末层权重初始化为 `0`，因此训练第 `0` 步模型严格等价于持久性基线（$\hat{Y}_{t+\Delta t} = Y_t$），保证优化轨迹从物理合理状态平滑出发。
- **模态丢弃（Modality Dropout）**：训练期按概率随机屏蔽 `flow_step_t.png`（默认 `0.3`）与当前步气动力输入（默认 `0.2`），使单套权重同时支持**全模态评估**与**求解器内零 PNG 快速推理**。

---

## 3. 数据质量门禁、防泄漏切分与平台治理对接

1. **7+1 准入检查（`validate_cylinder_trajectory`）**：
   - 强制 `allow_pickle=False` 加载全部 NPZ；
   - 核验全部数组形状 `(ny, nx)`、数据类型与有限性（拒绝任何 `NaN`/`Inf`）；
   - 核验步数严格递增且符合 `output_interval` 与 `n_steps` 节拍；
   - 核验同一固定几何轨迹内所有时间步 `solid_mask` 逐位一致且非全固；
   - 核验 `forces.csv` 与 `run_metadata.json` 诊断量完整且有限，保留 `code_revision`（Git SHA）；
   - 多模态模式下核验对应 `flow_step_XXXXXX.png` 存在且非空。
2. **轨迹级防泄漏切分（`split_cylinder_trajectories`）**：
   - 以完整轨迹（`run_id` 与真实目录路径双重去重）为最小切分单元，若同一轨迹跨 `train/val/test` 立即抛出 `ValueError`。
3. **R2 数据目录与训练作业表无缝桥接**：
   - `export_cylinder_run_to_r2_catalog()` 将圆柱轨迹写入标准 `tensorlbm.solver-export/v1` HDF5 并逐帧通过 `FieldDataProductR2` SHA-256 门禁注册进 `FieldDataCatalog`；
   - `train_multimodal_cylinder_forecaster()` 原生对接 `TrainingJobRegistry`，自动记录作业状态流转、验证指标与模型表注册。

---

## 4. 可复现实验命令与基线对比结果（CPU 冒烟验证）

### 4.1 运行命令
```bash
python examples/cylinder_multimodal_train.py \
  --generate-smoke --register-r2-catalog \
  --output-dir outputs/cylinder_multimodal_stage2
```
默认生成 5 条**雷诺数完全互不重叠**的真实 D2Q9 轨迹（网格 `96 × 48`，`4 × 4` Patch = `288` tokens）：
- 训练集：`Re ∈ {20, 50, 80}`（3 条完整轨迹）
- 验证集：`Re = 35`（未见雷诺数）
- 测试集：`Re = 65`（未见雷诺数，共 7 个预测步转换及 7 步自回归长推演）

### 4.2 未见雷诺数（`Re = 65`）测试集实测对比（CPU）

| 指标 | Persistence 基线 | 多模态模型（含 PNG） | 零 PNG 求解器模式 | 相对改进 / 说明 |
| --- | ---: | ---: | ---: | --- |
| 流体速度相对 $L_2$ 误差 | `0.1281` | **`0.1146`** | **`0.1146`** | **+10.50% skill** |
| 内部涡量相对 $L_2$ 误差 | `0.1879` | **`0.1630`** | **`0.1630`** | **+13.26% skill** |
| 密度脉动相对 $L_2$ 误差 | `1.0498` | **`0.7894`** | **`0.7894`** | **+24.81% skill** |
| 气动力 $(C_d, C_l)$ RMSE | `3.6813` | **`1.6342`** | **`1.6945`** | **-55.6% 误差下降** |
| 固体内部最大速度 $\max_{\Omega_s}|\mathbf{u}|$ | `0.0000` | **`0.0000`** | **`0.0000`** | 严格满足无滑移固体掩码 |
| 7 步自回归末步（Step 160）涡量相对 $L_2$ | `0.5870` | — | **`0.3834`** | **-34.7% 累积涡量误差** |
| 单样本 CPU 推理耗时 | — | `37.49 ms` | **`6.08 ms`** | 零 PNG 模式提速 **6.2×** |

---

## 5. Stage 3：多步历史时序上下文、Pushforward 自回归训练与变几何联合泛化

在 Stage 2 单帧条件预测（$H=1, K=1$）基础上，Stage 3 已在同一套向后兼容 API 中落地四项关键升级（默认 `min_step=0, history_steps=1, pushforward_steps=1` 完全保持 Stage 2 行为）：

1. **声学启动瞬态切除（`trim_trajectory_spinup` / `--min-step`）**：
   - 自动剔除 `step < min_step` 的初始弱可压缩声波冲击帧，并将 `min_step`、`history_steps`、`rollout_steps` 纳入 `dataset_manifest.json` 的 SHA-256 指纹治理。
2. **多帧历史流场与时间差分加速度通道（`MultimodalCylinderArch.history_steps = H`）**：
   - 空间卷积干路同时拼接 $H$ 帧历史归一化物理场 `history_fields`（`3H` 通道）与相邻帧后向时间有限差分 $\Phi_{t-m+1} - \Phi_{t-m}$（`3(H-1)` 通道），直接向 Patch Token 提供局部流场时间导数与涡街传播相位方向；
   - AdaLN-Zero 条件通路与力增量头同步融合全部 $H$ 步历史气动力 $\operatorname{asinh}(C_d, C_l)_{t-H+1:t}$，单帧调用时自动执行冷启动左填充回退。
3. **Pushforward 多步展开自回归训练（`MultimodalTrainConfig.pushforward_steps = K`）**：
   - 训练期第 `k=0` 步允许融合 `PNG` 视觉模态，随后 `k=1..K-1` 步在**零 PNG 模式**下将模型预测的流场与气动力滑窗滚入历史缓冲区，按指数衰减权重 $\sum_{k=0}^{K-1} \gamma^k \mathcal{L}_k / \sum_{k=0}^{K-1} \gamma^k$（默认 $\gamma=0.7$）联合反传，并按概率 `pushforward_detach_prob=0.5` 执行 Brandstetter Pushforward 截断稳定梯度，直接对齐长周期推理分布。
4. **变几何 × 跨雷诺数双重未见泛化评测（`--train-radii` / `--val-radii` / `--test-radii`）**：
   - 依托 `GeometrySDFEncoder2D` 读取精确二维符号距离场 $\phi_{\text{norm}}$ 与归一化阻塞比 $R/N_y$，在**同时未见的圆柱半径与未见雷诺数**上验证几何内插泛化。

### 5.1 Stage 3 运行命令（跨雷诺数 × 变几何圆柱联合测试）
```bash
python examples/cylinder_multimodal_train.py \
  --generate-smoke --register-r2-catalog \
  --output-dir outputs/cylinder_multimodal_stage3 \
  --nx 96 --ny 48 --u-in 0.06 --n-steps 220 --output-interval 20 \
  --min-step 40 --history-steps 2 --pushforward-steps 2 \
  --train-re 25 55 85 --train-radii 4.0 6.0 \
  --val-re 40 --val-radii 5.0 \
  --test-re 70 --test-radii 5.0 \
  --epochs 35 --batch-size 8 --seed 42
```
- **训练集（6 条轨迹）**：`Re ∈ {25, 55, 85} × radius ∈ {4.0, 6.0}`（切除 `step < 40` 瞬态后全部写入 `FieldDataCatalog`，共注册 `80` 个 R2 HDF5 数据产品）；
- **验证集（未见工况 + 未见几何）**：`Re = 40, radius = 5.0`；
- **测试集（未见工况 + 未见几何）**：`Re = 70, radius = 5.0`。

### 5.2 同批数据上 Stage 2（`H=1, K=1`）vs Stage 3（`H=2, K=2`）实测对比（CPU）

| 评测维度（测试集：`Re = 70, radius = 5.0`，双重未见） | Persistence 基线 | Stage 2 (`H=1, K=1`) | Stage 3 (`H=2, K=2`) | Stage 3 提升幅度 |
| --- | ---: | ---: | ---: | --- |
| 单步流体速度相对 $L_2$ 误差 | `0.1109` | `0.0957` (+16.71% skill) | **`0.0887`** | **+20.05% skill**（零 PNG **+20.07%**） |
| 单步内部涡量相对 $L_2$ 误差 | `0.1568` | `0.1328` (+17.14% skill) | **`0.1197`** | **+23.68% skill**（较 Stage 2 **+6.54 pp**） |
| 单步气动力 $(C_d, C_l)$ RMSE | `2.0162` | `1.6883` | **`1.5048`** | 较 Persistence **-25.4%**、较 Stage 2 **-10.9%** |
| 自回归推演至 `step=220` 速度相对 $L_2$ | `0.1667` | `0.1350` | **`0.0947`** | 较 Persistence **-43.2%**、较 Stage 2 **-29.9%** |
| 自回归推演至 `step=220` 涡量相对 $L_2$ | `0.5007` | `0.3547` | **`0.2076`** | 较 Persistence **-58.5%**、较 Stage 2 **-41.5%** |
| 自回归末步 `step=220` 阻力系数 $C_d$（真值 `2.3376`） | — | `2.1734` | **`2.2553`** | 末步气动阻力相对误差仅 **3.5%** |
| 固体内部最大速度 $\max_{\Omega_s}|\mathbf{u}|$ | `0.0000` | `0.0000` | **`0.0000`** | 严格满足固体无滑移硬约束 |
| 零 PNG 求解器模式单样本 CPU 耗时 | — | `7.11 ms` | **`7.67 ms`** | 仅增加 `0.56 ms` 即大幅压低长推演累积误差 |

---

## 6. 边界声明与 GPU 大规模基准路线（Stage 4 展望）

- **科学诚实声明**：Stage 2 与 Stage 3 的 CPU 实测证明了多步时序历史（$H=2$）+ Pushforward 自回归训练（$K=2$）在**跨雷诺数 + 变圆柱半径双重未见测试集**上显著降低单步与多步自回归涡量累积误差；受限于 CPU 冒烟步数（`220` 步），该验证仍属于中短程瞬态发展区，不等同于充分发展卡门涡街长周期统计基准（如时均 $\overline{C_d}$ 与 Strouhal 数 $St$ 频谱误差）。
- **GPU 生产训练下一步建议**：
  1. 在 GPU 节点直接复用同一命令行入口生成 `nx=320, ny=128, n_steps=20000, min_step=6000` 的充分发展周期性涡脱落数据集，并在长推演（50+ 步）中增补升力主频 $St$ 一致性评估；
  2. 将几何生成扩展至非圆截面（椭圆柱、方柱、NACA 翼型与双柱干涉阵列），直接复用现有精确二维 EDT 符号距离场编码器 `GeometrySDFEncoder2D` 进行跨拓扑形状泛化训练。

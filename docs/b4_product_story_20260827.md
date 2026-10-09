# B4 产品线总览：船舶设计实时阻力评估（2026-08-27）

产品方向（owner 定向 2026-08-24）：面向水下航行器的**任意几何实时性能评估**，
与设计软件融合。三阶段路线：L1 参数族代理（已达成）、L2 任意几何编码
（SDF v2 失败，两阶段重做为待决方向）、L3 服务化/UQ/闭环（本波基本交付）。
本文串联 #239–#254 的架构与已验证结论；每条数字都有独立复核记录。

## 1. 分层架构

```
设计参数 (CAD/参数)
   │  cad_stl.py (#246)           CAD STL 门：掩码 IoU 0.990–0.9985
   ▼
几何语料  scan 链 (#194/#204)     v4 语料 274 点 · fam 语料 350 点
   │  b4_serve (#243)             5 种子 CondFNODrag 集成 · z-score fit-split
   ▼
代理层    v3 条件化 0.66% (L1)    SDF v2 (#247, 负结果) · diff_vox (#248, 伴随梯度)
   │  inference_service           回显服务 + 诚实契约 (ok/review/reject)
   ▼
服务层    TRT 后端 (#249/#252)    B=1 端到端 1.99 ms · verdict 逐字一致
   │  uq_calibration (#251)       σ 语义 · 温度缩放 · 护栏 ROC
   ▼
闭环      active_learning (#243 模拟 → 战役真标注 → #253/#254)
```

## 2. 几何与数据层

- **CAD STL 门（#246）**：`roundtrip_mask` 将设计参数→STL→体素掩码与生产
  掩码对拍，IoU 0.990–0.9985，任何几何路径改动有了位级回归网。
- **语料**：v4 274 点（部署训练语料，fit 186/val 33/test 55）；fam 350 点
  （4 个几何族 ×28 + 基础 238，跨族泛化考卷）。
- **标注真值化（战役实测）**：3 控制点重扫与缓存 rel_diff = 0.0（逐位）；
  24 个缓存外新点与族内 log-log 插值差 median 6.6e-5 / max 2.5e-4 ——
  **族窗内插即真值（≤0.03%）**，插值 oracle 构造被实测背书；窗外外推未检验。

## 3. 代理层

- **L1 参数族（达成）**：5 种子 CondFNODrag 集成，v3 条件化在随机测试上
  MAPE 0.35%（fit 0.27%/val 0.42%）；母语料守卫 0.3–0.5%。
- **SDF v2（#247，诚实负结果）**：联合训练目标是瓶颈而非编码器——latent
  塌缩（participation ratio 0.000），但几何信息存在（v2_reg2 fin probe
  0.991）；LOFO 灾难（10,335%）。下一步两阶段（监督 probe→冻结→代理头），
  等 owner 决策。
- **可微体素化（#248）**：参数→解析 SDF→平滑布尔并→STE 占用→条件向量→
  冻结集成伴随。硬掩码 IoU = 1.000000（与 build_suboff_mask 逐位）；
  tau=0.02 标定（0.05 翻转附体 STE 符号）；批量梯度 5.4→74 designs/s。

## 4. 服务层

- **部署阶梯（#249）**：TRT FP16 B=1 0.607 ms / FP32-strict 0.672 ms
  （parity 1.9e-7）；TF32 默认破坏 parity ~8e-5；INT8 动态量化损失
  12–40×；torch fp16 被 fno.py 谱路径 ComplexHalf 阻塞。plans 不可移植
  （sm_120/TRT11.2），ONNX 是可移植工件。
- **后端选择（#252）**：`from_checkpoints(backend_kind=...)`，优先级
  arg > env `TENSORLBM_DRAG_BACKEND` > torch；默认路径构造上逐位不变；
  请求的 plan 缺失时带原因回退 torch。端到端 B=1 8.19→1.99 ms（4.1×）；
  verdict 24/24 一致，fp32_strict Δ≤2.6e-7。滑杆 miss ~20 ms 瓶颈在
  CPU 几何预处理（14–48 ms）——下一个优化目标。
- **演示前端（#245/#250）**：echo_slider + `GET /demo` 挂载（#250 CI 绿）。

## 5. 诚实契约与 UQ（#251）

- **σ 语义是条件性的**：包络内（v4 随机留出）cov95 87.9–93.0%，近校准
  但重尾；点级留出低估 σ ~2×（cov95 66.7–88.6%）；外推彻底失真（LOHO
  bare_hull cov95=0，geoM 折 rms z 27，fam_slender 成员偏 3 个数量级）。
- **verdict 语义**：包络内 ok ⇒ P(err≤1%) = 90–96%；训练缺失的 hull 型
  （LOHO）仍在包络内，ok 只保证 P(err≤5%)；**reject 可靠地对应灾难**
  （112/112 出族点全 reject 且 err≥15%）。
- **护栏 = 包络边界探测器，非误差排序器**：出族捕获/精度 100%，包络内
  AUC 0.43–0.72（近随机）。手册条件空间无法排序包络内误差。
- **温度缩放**：包络内不需要（T=1.15/1.35）；新设计语义下建议 **×2.3**
  （半折交叉 T=2.21/2.50，cov95 70→94/84→96%）。标量修不了外推——
  那是护栏的职责。

## 6. 主动学习闭环

- **模拟循环（#243）**：coverage 池 17 点，k=17 符号翻转刀刃。
- **真标注战役（al_campaign_20260827，已验收）**：24 真扫 + 3 控制
  （27/27，0 失败）；缓存臂逐位复现 #243（k=17 翻转）；**真实臂 k=13
  翻转（提前 4 标签）且 k=13–24 无回翻**——前 12 点放齐 trend 锚点；
  holdout MAPE 16.73%→k=13 骤降 7.56%，终点 8.42%（最优 6.22%@k20），
  优于模拟终点 10.53%。
- **梯度策略（#253）**：gradient/mixed 选取在 **16 标签**恢复 l_over_d
  正确符号（+0.27~0.31）且 MAPE 16.1→11.0%，coverage 仍错；刀刃定位到
  1 个标签（coverage 买 sail_x@Re430，gradient 买 slender@Re591 锚点）。
  honest：gradient@8 略差于 coverage@8；sail_x 任何预算学不会。
  裕度校准条件成立（≥16 标签拟合在 held-out 验证 in-band 1.00）。
- **fresh-Re（#254）**：`propose_acquisition(fresh_re=True)` + exact-key
  去重——修复战役暴露的 G1（Re 候选仅取缓存行 + floor 过拒新鲜角点，
  10/24 个低于 floor 的点被证明全新且最有信息量）。

## 7. 路线图状态

| 阶段 | 状态 | 依据 |
|---|---|---|
| L1 参数族 | **达成** | v3 0.66% L1 / 0.35% 随机测试 |
| L2 任意几何 | **v2 失败，方向已明** | #247；两阶段重做待 owner |
| L3 服务化 | **基本交付** | 1.99 ms 端到端、verdict 一致、回退安全 |
| L3 UQ | **交付** | σ 语义量化、T=2.3 建议、护栏边界语义 |
| L3 闭环 | **交付** | 真标注战役兑现模拟预测 + G1 修复解锁下一轮 |

**已知短板**（按影响排序）：滑杆 miss 的 CPU 几何预处理 14–48 ms；
sail_x 轴不可学习（语料事实）；SDF 两阶段重做未启动；窗外插值未检验；
INT8 不可用（精度损失）；plans 不可移植（ONNX 兜底）。

## 8. 复现索引

- 运行工件：`/nfs/wangxi/runs/{b4_serve_20260824, b4_sdf2_20260825,
  deploy_latency_20260825, main_health_20260827, trt_echo_20260827,
  uq_calibration_20260827, al_grad_20260827, al_campaign_20260827}/`
- 数据集：`scan_suboff_al_campaign_20260827`（27 点，3.4G）及既有扫描集
- 各 PR 文档：`docs/{active_learning_20260825, trt_echo_20260827,
  uq_calibration_20260827, al_grad_20260827, active_learning_fresh_re_20260827}.md`

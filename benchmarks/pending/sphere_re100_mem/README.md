# 球 Re=100 直接力法基准 — 力法修复 + 解侧误差预算与地板定源

pending/sphere_re100/（2026-08-19 G15 时代记录：压力+摩擦积分 −16.6%、
MEM standard +264.6%）的后续：力法修复（库变更 = PR #309）与残余误差的
完整定源。源钉定只读 worktree main @ cf5709db3c；运行 GPU cuda:6 独占。

## 设置

- 判定路线 BFL（插值边界，精确球面）：域 (y,z) 200×200/300×300、流向 180/270
  （lat 2.0），D=40/60，τ=0.56/0.59，u_lb=0.05，硬 far field，12000 步。
- 对照路线 BB（体素楼梯 + 引擎 + 修复估计器）：lat 1.25，域 (y,z) 140×140/
  210×210，其余同。
- 修复估计器（PR #309）：`momentum_exchange_wet_node`（F = 2·Σ_links
  c_q·f_q(x_solid)，遍历**全部**流体→固体穿越链含 13.8% 角链）+
  重写 `momentum_exchange_pair`（同一全链集）；`SolverConfig.mem_variant`
  接线，默认 "standard" 位同。

## 参考

    Cd_ref(Re=100) = 24/100·(1 + 0.15·100^0.687) = 1.0917311  （Schiller–Naumann）
    交叉：经典 S-N (0.681) = 1.06852、Clift–Gauvin = 1.10923
    门：两档 |Cd−Cd_ref|/Cd_ref ≤ 3% 且 |err| 单调下降

## 判定表

| 运行 | 路线/估计器 | D | 堵塞比 | Cd | err | steady* | CV 对账 |
|---|---|---|---|---|---|---|---|
| formal_bfl_D40 | BFL 链账本（判定） | 40 | 3.14% | 1.2149 | +11.28% | yes | 0.044% |
| formal_bfl_D60 | BFL 链账本（判定） | 60 | 3.14% | 1.2142 | +11.21% | no（−0.316%） | 0.154% |
| formal_bb_D40 | 引擎 + wet-node MEM | 40 | 6.41% | 1.2923 | +18.37% | yes | — |
| formal_bb_D60 | 引擎 + wet-node MEM | 60 | 6.41% | 1.2831 | +17.53% | yes | — |

*steady = 末 20% 窗 vs 前一 20% 窗漂移 <0.3%。bfl D60 落在 −0.3158%——超判据
0.016pp，仍向稳态衰减；该量比 11.2% 误差低两个量级，不改任何结论。

**判定：FAIL（3% 门）。** 两判定档 +11.2%（|err| D40→D60 单调降但 ≤3% 腿两档
皆败）。直接测力已不是瓶颈——对独立控制体动量预算闭合 0.04–0.15%——残余为
解侧 +11% 平坦偏移（误差预算见下），定源记录（definitive-source）。

## MEM 失效根因与修复验证

标准 Ladd 配对在近壁链子集上求和 `f_q(x_f)`（穿越前一步的流体胞分布）与
`f_opp(x_s)`（上一步残留于固相节点的分布）——两项来自不同时间层，压力驱动、
体静止工况下 termA 携带不抵消的自由流背景（D40 t=0 态 +43.64 Cd），即历史
+264.6% 失败模式；termB≈0，termA+termB ≡ `momentum_exchange_standard`。

修复后测量验证：

1. 空域：全变体恒 0。
2. t=0 闭合：wet-node 恰 0 vs 标准配对 +43.64 Cd（同一初态）。
3. 纯 python 双循环暴力参考：一致至 6e-7（fp32 求和顺序级）。
4. 稳态恒等式 pair == wet-node（每案 4 位小数）。
5. 独立控制体动量预算（体侧 6 格盒、精确 In−Out 链账本、pre-streaming 态）：
   D40 lat1.25 Cd 1.2768 vs 1.2753（0.12%）；lat2.0 1.2155 vs 1.2154（0.01%）；
   t=0 基线恰 [0,0,0]。
6. 判别测试 `tests/test_momentum_exchange_wet_node.py` 9/9（PR #309）；
   全库回归 5975 passed / 0 failed；默认路径位同（存量力路径族 40 项不翻）。

## 残差误差预算（D40，BFL，CV 对账）

| 探针 | Cd | err |
|---|---|---|
| lat1.25 down2.25（基线，6.41% 堵塞） | 1.2753 | +16.83% |
| lat1.25 down6.0 | 1.2864 | +17.85% |
| lat1.25 down10.0 | 1.2875 | +17.95% |
| lat2.0 down2.25（3.14% 堵塞） | 1.2154 | +11.34% |
| lat1.25 sponge + 非平衡 far field | 1.4103 | +29.20% |

- 横向：堵塞项 ≈3% 以上生效（≈−1.8pp/1% 堵塞），饱和到堵塞无关地板
  ≈+11%（BB 路线：+11.65% @1.86% vs +11.36% @0.97%）。
- 流向：平坦（2.25D→10D 仅 +1.1pp 且方向相反）——出流截断排除。
- 壁处理：BFL（插值、精确球）比楼梯路线低 1.5pp——楼梯不是残余。
- D 伸缩（固定横向倍数）：平坦（堵塞比 D 不变）；D40→D60 −0.17pp（BFL）/
  −0.84pp（BB）。
- u_lb（Ma）：0.05→0.03 升至 +19.51%——残余的 τ/ν 依赖而非可压缩性。
- `wet_near`（仅面邻链）：+1.83% @lat1.25 → −4.03% @lat4.0 恰好穿越参考——
  楼梯/堵塞抵消假象，仅作诊断列，不作判定路线。
- PF 积分列 −48%…−77%：遗留控制面积分法在这些网格仍不可用。
- 库生产管线探针：`sphere_bfl_control_volume` HEAD 默认（R=12, cumulant,
  非平衡+sponge）GPU/CPU 双端 step 18 发散——该案例的 5%-门先例在 HEAD 不可运行。

## 数值注记

- `far_field_bc_3d` docstring 自记自由流 far-field 族 ~9% 球 Cd 误差——与
  本记录 ≈+11% 地板一致。
- `correct_mass3d`（每 200 步）后质量漂移 ≤0.3 ppm；运行确定性（无 RNG）；
  |Cl| ≤ 1.2e-3。

## 复现

```bash
cd /nfs/wangxi/runs/bm_widen_w5_20260921/sphere_force
PYTHONPATH=src_patched /nfs/wangxi/venvs/tensorlbm/bin/python run.py 40 12000 bfl --lat 2.0
PYTHONPATH=src_patched /nfs/wangxi/venvs/tensorlbm/bin/python run.py 60 12000 bfl --lat 2.0
PYTHONPATH=src_patched /nfs/wangxi/venvs/tensorlbm/bin/python run.py 40 12000 bb  --lat 1.25
PYTHONPATH=src_patched /nfs/wangxi/venvs/tensorlbm/bin/python run.py 60 12000 bb  --lat 1.25
/nfs/wangxi/venvs/tensorlbm/bin/python verify.py            # 检查模式
cd tests && PYTHONPATH=../src_patched pytest test_momentum_exchange_wet_node.py -q
```

工件：result.json、formal_{bfl,bb}_{D40,D60}.json（240 采样力史 + CV 账本逐
采样）、run.py / verify.py。库变更本体在 PR #309；完整过程工件（协议 NOTES、
诊断通道 diag/、全库回归）留服务器暂存目录。

## W7 域形修复复判 — 入口钳制定源（w7/）

W5-B 记录的 +11% 堵塞无关地板定源为**入口平面 Dirichlet 钳制**：
`far_field_bc_3d` 在 x=0 强制自由流平衡态，压制球前轴向上游的势流减速。
修复=纯域形（入口距离 up 从 1.25D 扩到 2.75D 正式对、3-4D 与联合大域
签名），无任何模型修正。

### 设置（相对 W5-B 的增量）

- 几何/物理全同（D=40/60、Re=100、u_lb=0.05、τ=0.56/0.59、D3Q19、
  MRT、BFL 判定路线 + bb 副观测、堵塞 3.14%）。
- 正式对域形 lat2.0 / up2.75 / down2.25；bfl 24000 步（up2.75 域
  入口→球对渡 ≈3900 步，12k 步仅 3.1 渡时残漂移：12k↔24k 判定数差
  D40 0.03pp / D60 0.21pp，24k 双档 drift ≤0.02%）；bb 副观测与签名
  族 12000 步。12k 旧正式对保全于 diag/。
- D60 显存工程：库低内存流入口 `stream3d_roll` + expandable_segments
  （与 gather 版 300 步 A/B 逐位一致：cd/cl/cs/mass 最大差 0.0）。

### 判定表（lat2.0 / up2.75 / down2.25）

| 运行 | 路线/估计器 | D | Cd | err | steady | CV 窗闭合 |
|---|---|---|---|---|---|---|
| formal_bfl_D40 | BFL 链账本（判定） | 40 | 1.13761 | +4.20% | yes（−0.005%） | 0.102% |
| formal_bfl_D60 | BFL 链账本（判定） | 60 | 1.13552 | +4.01% | yes（−0.018%） | 0.010% |
| formal_bb_D40 | 楼梯 + wet-node MEM（12k） | 40 | 1.15097 | +5.43% | yes | — |
| formal_bb_D60 | 楼梯 + wet-node MEM（12k） | 60 | 1.14630 | +5.00% | 触线（−0.357%） | — |

**判定：FAIL 维持。** 双档均出 3% 门（+4.20/+4.01%），|err| 单调下降
成立；对 W5-B 同库 +11.28/+11.21% 压掉 ~7.2pp。楼梯差 +1.20/+1.00pp
（bb 副观测，与 W5-B up1.25 的 1.5pp 同量级）。

### 地板定源（诊断矩阵，D40/lat2.0/12k/稳态 <0.1%）

| 通道 | 探针 | 结果 | 判决 |
|---|---|---|---|
| 入口距离 | up 1.25→2.0→2.75→3.0→4.0 | +11.28→+5.83→+4.23→+3.94→+3.31% | **主因定源：入口钳制** |
| 碰撞算子 | MRT / TRT Λ=3/16 / MRT 魔术 s_q / BGK | +11.28/+11.16/+11.24/+11.30%（0.15pp 带） | 排除（墙位 τ 耦合假设不成立） |
| 远场反射 | noneq 远场单独 @up1.25 | +11.17% ≈ hard +11.28% | 排除（非 populations 反射问题） |
| τ 依赖 | τ 0.53/0.59（Re 200/66.7 浮动参考） | +6.21/+13.69% | 黏性型入口伪影（低 Re 上游更厚），非墙位 |
| 出口 | down 2.25→4.0 @up2.75 | 12k +5.24% / 24k +5.20%（formal +4.20%） | 稳定 +1.0pp 反向偏移（复现 W5-B down→10 +1.1pp），尾长真实弱敏感性 |
| 联合大域 | lat3/up3/down4 | **+2.77%（D40 达标形）** | 横向钳置另贡献 ~1pp |
| 壁离散 | BFL vs BB 同域 | 差 1.5pp | 楼梯非主因 |

五点最小二乘签名（12k 协议、逐点稳态窗）：

    err(up) ≈ E0 + C·(a/x)³，  x=(up+0.5)D，a=0.5D
    E0 ≈ 3.0%，C ≈ 355（lat2.0）

独立复算（不同窗尾约定）：E0=2.9%、C=361、R²=0.9998，对 up2.75/up3.0/
up4.0 三点预测差 ≤0.07pp。W5-B 横向扫描（堵塞饱和）全程固定 up=1.25D
（nx 恒 180），与入口假设完全相容——"堵塞饱和"实为入口项主导下的饱和。

### 3% 双档门可达性（本硬件）

- 达标域形（lat3/up3/down4）的 D60 孪生 = 480×420×420 = 84.7M 胞 =
  **2.6× 实测显存天花板**（D60 ledger 天花板 = 32.4M 胞 FITS @up2.75/
  lat2；33.75M 胞 OOM ×3 于 `bouzidi_bounce_back_d3q19` 内部表达式树，
  empty_cache 无效 = 真活中间量）。
- `stream3d`（gather）缓存 4×[19,N] int64 索引（D60/up3 = 19.1GB）；
  换库入口 `stream3d_roll` 后墙移至 BFL 核内部中间量（33.75M 胞 ~21GB
  同时活）。
- D40 大域 +2.77% 单点过门但无 D60 阶梯（铁律 ≥2 档）→ 不晋级，仅作
  达标形存在证据。
- 破门路径：多卡分域或 BFL 核显存优化（另立项）。

### 数值注记

- 基线 bitwise：diag/t1_mrt_base 与 W5-B formal_bfl_D40 **逐位一致**
  （1.214927，+11.284%）。
- 时间收敛：12000 步即收敛值，叠加 ±0.45% 极限环振荡（周期 ~1000 步
  ≈1.25 t_c）；窗均值不受影响（CV 窗闭合 0.102%/0.010%）。
- 意外保全点：D60/up2.75/τ0.56 = Re150 对自身参考 ≈+2%（窗尾约定差
  内）——入口伪影随 Re 增大而减，与 Re 扫描（Re200 +6.21% < Re100
  +11.28% < Re66.7 +13.69%）相容。判定只在 Re=100。
- roll/gather A/B（diag/bit_*）：6 个共同采样 cd/cl/cs/mass 最大差 0.0。
- 参考族位置：锁定 S-N(0.687)=1.0917 位于交叉族（1.0685-1.1092）低端；
  大域 D40 点 vs Clift-Gauvin 仅 +1.15%。锁定参考不重开，门按锁定值判。

### 复现

```bash
cd <repo>/benchmarks/pending/sphere_re100_mem/w7
CUDA_VISIBLE_DEVICES=6 W7B_DEV=cuda:0 TMPDIR=<tmp> \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True W7B_STREAM=roll \
  python run_diag.py formal_bfl_D40.json --route bfl --collision mrt \
    --D 40 --steps 24000 --lat 2.0 --up 2.75 --down 2.25 --cv_tail 2500
# D60 同参数 --D 60 --tau 0.59（必须 roll + expandable_segments）
python verify.py    # 从原始力史重算判定表/门/单调/CV/域收敛签名
```

### w7/ 文件清单

- `run_diag.py` — 诊断/正式 runner（全库公共入口）。
- `cv_instrument.py` — CV 动量预算仪器（W5-B bfl_worker.cv_box_force_exact
  原样移植；纯测量）。
- `verify.py` / `result.json` — 判据独立重算 + 判定汇总。
- `formal_{bfl,bb}_{D40,D60}.json` — 正式档（采样力史 + CV 尾账本）。
- `diag/` — 19 份诊断档案：t1_*（碰撞四列 + 基线复现）、h1_*（τ 端点）、
  h2_*（up 扫描 / noneq / 大域）、v1_*（down 复核 12k+24k）、p1_up4.0
  （签名点）、formal12k_*（12k 旧正式对）、bit_*（roll/gather 逐位 A/B）、
  accidental_re150_*（Re150 保全点）、mem_D60_up2.75（显存探针）。

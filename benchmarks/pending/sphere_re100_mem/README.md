# 球 Re=100 直接力法基准 — 力法修复 + 解侧误差预算（W5-B）

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

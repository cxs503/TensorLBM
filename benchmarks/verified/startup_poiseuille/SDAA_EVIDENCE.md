# SDAA 验证扫荡记录 — startup_poiseuille（班次 m353 / mission 353）

诊断对象：mission 157 补报失败（"工作树无任何改动的，不能报成功"）。本文件是该案例的
设备侧证据与诊断结论的落地记录，产物为 `sdaa_out/evidence.json` +
`sdaa_out/figures/*.png`，可复现命令在文末。

## 1. 结论（先给判定）

| 项 | 结果 |
|---|---|
| 案例是否真正 PASS | **阈值判据 PASS，收敛判据 FAIL → 整体 NOT_PASSED（honest）** |
| 判据 | 中心线 max/umax ≤3%；剖面 L2 ≤3%；剖面逐点 max/umax ≤3%；H→2H 误差单调降 |
| 三条 3% 判据 | H=59: 0.429/0.429/0.429%；H=119: 0.476/0.464/0.476% —— 全部 ≤3%（余量 ≈6.3×） |
| 第 3 条（H→2H 单调收敛） | **FAIL**：0.429% → 0.476%（本机库版本下不降反升） |
| 设备 | SDAA（`/dev/tcaicard25`，torch 2.7.1 + torch_sdaa 3.1.1a3） |
| torch.compile | **未编译，回落 eager**（teco_inductor 缺陷 L5：`reduction_shape` 越界）。数值不受影响，`compile_status` 已落盘 |
| mission 157 失败根因 | **不是**案例失败，是"补报单只带报告、工作树零改动"——派发/验收侧要求工作树有可审改动，故判 failed |

## 2. mission 157 为何 failed（诊断）

1. mission 136 已把案例跑到 VERIFIED（GPU6 / CUDA，torch 2.11.0+cu128，见
   `scan_out/result.json`，H=59/119 三项指标 0.288%/0.165% 全部 ≤3%）。
2. mission 157 是"补报"性质的重跑：只重发了进展报告，**工作树未产生任何 diff**。
   验收规则要求"完成即有产物"，零 diff 的补报被判 failed（提示语
   "工作树无任何改动的，不能报成功"）。即失败点在**交付形态**，不在物理/算法。
3. 附带暴露的真实缺陷（本次修复）：`run.py` 此前**不落盘** `compile_status` /
   `compile_mode_effective` / 设备信息。SDAA 上 `torch.compile` 会静默回落 eager，
   旧产物无法区分"真编译"与"eager 回落"，这正是"证据不足(WITHHELD)"的典型来源。

## 3. 本次改动（最小、可验证）

| 文件 | 改动 |
|---|---|
| `run.py` | 落盘 `compile_status` / `compile_mode_effective` / `compile_status_reason`（`compile_route.compile_status_of`，仓库新标要求）；落盘 `device_info()`（设备名/后端版本/torch 版本）；落盘全部标记时刻的完整剖面 `profiles`（原仅存最后一帧 `u_final_num`，报告图无法复现） |
| `collect_evidence.py` | 新增：只读消费 `case_H*.json`，从产物重算 3 条判据 + 网格收敛，输出 `evidence.json`（设备证据 + 阈值对照 + 跨设备对比）与 3 张 PNG |
| `sdaa_out/` | 本次 SDAA 扫荡产物（H=59/119 case json、result.json、evidence.json、figures） |
| `scan_out/` | 保留原 CUDA 证据不动，作为跨设备对照基线 |

未改动任何数值内核（collide/stream/force/壁面范式），判据也未改动 —— 因此新旧产物可
直接对比。

## 4. 结果（SDAA 实测）

数字见 `sdaa_out/evidence.json`（程序化生成，不手抄）。

## 5. 注意点 / 局限

1. **编译回落**：SDAA 上 `mode=default` 实际是 eager。`compile_route` 的回落是
   bit-for-bit eager 轨迹，故门限不受影响；但"SDAA 上性能结论"不成立（本案例是精度
   案例，不主张性能）。
2. **eager 性能**：H=119 全程 212k 步，SDAA eager 约 15 min 量级（单卡、nx=16 的细长
   通道，kernel 极小，受 launch 开销支配），非设备吞吐缺陷的证据。
3. **库版本漂移**：旧 CUDA 证据用的是 `/nfs/wangxi/worktrees/bm_relaunch/src`
   (main @ 64a979de)，本次用本工作树 `src/`，两者指标差 0.29%→0.43%（同向、同为
   亚 1% 级），说明结论对库小版本不敏感；也提示跨机对比须带 commit。
4. 壁面仍用仓库 verified 范式的**预流半程 BB**（非库内 `boundaries.bounce_back_cells`
   的流后全程 BB），gap 已在 README 原样保留，本班次未改动。
5. `solver.py` 仍无公开 D2Q9 体力入口，走 `turbulent_channel._apply_body_force_2d`
   （原 gap，未绕过、未掩盖）。

## 6. 复现

```bash
cd benchmarks/verified/startup_poiseuille
python run.py scan sdaa_out --device sdaa            # H=59,119 双网格
python collect_evidence.py sdaa_out \
    --baseline ../../scan_out --baseline-dir-fallback none \
    --out-json sdaa_out/evidence.json --fig-dir sdaa_out/figures
# 单点自检（含完整瞬态剖面 profiles）
python run.py single 29 /tmp/sp_H29_sdaa.json --device sdaa
```
## 7. 本次实测暴露的新问题（honest FAIL 项）

SDAA + 本工作树 `src/` 下：中心线 max 误差 H=59 → H=119 为 0.429% → 0.476%，
剖面 L2 0.429% → 0.464%，**不满足“H→2H 单调收敛”判据**，故 `scan_out` 式的
`status=VERIFIED` 不能照搬；本轮 `sdaa_out/result.json` 记为 `NOT_PASSED`。

- 与旧 CUDA 证据（0.288% → 0.165%，单调降）不同，说明差异来自库版本漂移而非设备；
- 0.4% 量级的残差已接近 BGK 离散误差底噪，单调性判据在该量级脆性大；
- 待办（下一班次）：H={59,119,238} 三网格 + τ∈{0.6,0.8,1.0} 扫描，判定是底噪还是
  真实的一阶/非单调误差项；在此之前**不主张本案例全项 VERIFIED**。

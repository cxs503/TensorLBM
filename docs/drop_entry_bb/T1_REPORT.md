# T1 REPORT · DE-T1-E1 双相(CACSim)路径存活修复 — mach_limit(伪电流)处方

## 1. 补丁 diff 摘要（commit c5a031689，2 行）
`benchmarks/water_entry/single_drop/run.py`（CLI 默认值，L288-289）：
- `--speed` 0.04 → **0.03**（晶格速度↓ → 同一物理速度的晶格 Mach 基线↓）
- `--width` 4.0 → **6.0**（界面宽度↑ → 序参量梯度变缓，表面张力离散伪电流↓）
处方写「run() L101/L103」，但 run() 签名无默认值，实际默认在 CLI（见异议③）。

## 2. 实测（cells=8 最便宜档，处方成本节命令）
命令（与处方等价：`--n`→`--cells`，在本树 m26277 执行）：
```
python -m benchmarks.water_entry.single_drop.run --cells 8 --speed 0.03 --width 6.0 --out runs/t1_cac_w6_s03_n8.json
python -m benchmarks.water_entry.single_drop.score --output runs/t1_score.json runs/t1_cac_w6_s03_n8.json
```
| 量 | 值 | json 字段路径 |
|---|---|---|
| postcontact_healthy_ms | **None** | runs/t1_score.json → `runs.t1_cac_w6_s03_n8.postcontact_healthy_ms` |
| stop_reason | volume_drift | runs/t1_cac_w6_s03_n8.json → `stop_reason` |
| 终止点 | step 26 = 0.158 ms，触水前（contact_step=null） | `history[26]` |
| 终止时 volume_drift | 1.002e-4（门 1e-4） | `history[26].volume_drift` |
| 终止时 mach_max | 0.112（门 0.3） | `history[26].mach_max` |
| numeric_pass / status | False / pending | runs/t1_score.json → `numeric_pass`, `status` |

## 3. 与处方预测区间对照
- 处方预测 postcontact_healthy_ms ∈ **[12, 24] ms**（覆盖 10.394 ms 峰值）。
- n=8 实测 **None**（0.158 ms 死于 volume_drift，从未触水）→ 判据①触发 → **honest-FAIL**。
- n=12 全量（集群 cj5656）：step 41 = 0.166 ms 死于 **mach_limit**（mach=0.3055），contact_step=null → **honest-FAIL**。
- 受控基线（w4/s0.04，集群 cj5659）：n=8 与 n=12 均死于 0.065 ms（mach_limit）→ 处方杠杆把存活从 0.065→0.158/0.166 ms（约 2.5×），但两档都远未达 10.394 ms 峰值。
- **A3#1 参考值不可复现**：处方引「case3_cac12 于 1.362 ms 终止」，本单基线（n=12,w4/s0.04,默认 gap/span/pool）死于 0.065 ms，差 21×（见异议⑥）。
- 关键诊断（n=12）：`phi_at_mach_max`≈0.001（气相）、`mach_liquid_max`≈0.05 → 最大 mach 在**界面附近气相**，非液相；width 杠杆压不住气相伪电流。

## 4. 异议节（处方缺陷，仍按处方实现）
① 成本节命令用 `--n`，实际 CLI flag 是 `--cells`（run.py L287）；照抄 `--n` 直接报错。
② 成本节 `cd /root/tp-agent/work/m26266` 指向他人工作树；本单在本树 m26277 执行。
③ 处方指「run() L101/L103 默认值」，run() 签名无默认值；实际默认在 CLI L288/289（本单改 CLI 默认）。
④ score.py 的 numeric_pass/converged 对双相路径**结构性不可达**：run.py 不输出 side_view_measurement → camera_error=None → numeric_pass 恒 False（score.py L120-128、L212-235）。
⑤ volume_drift 门（1e-4，phi 和）对界面宽度**非不变量**：width↑ 系统性抬高 phi 和漂移（n=8：6.3e-4/ms vs 基线 1.5e-4/ms，约 4×），风险①在 n=8 兑现。
⑥ **A3#1 参考值不可复现**：基线（n=12,w4/s0.04）死于 0.065 ms 而非处方所引 1.362 ms；处方「基线已活到 1.362 ms」的前提在当前代码+默认参数下不成立，需处方方核对 A3#1 的确切参数/代码版本。

## 5. 结论
- n=8：honest-FAIL（postcontact_healthy_ms=None，0.158 ms 死于 volume_drift）。
- n=12：honest-FAIL（0.166 ms 死于 mach_limit）。
- width/speed 杠杆把存活从 0.065 ms 提到 0.158/0.166 ms（约 2.5×），但远未达 10.394 ms 峰值。
- 伪电流在气相；下一步应攻气相力离散（曲率滤波/气相力截断），而非界面宽度。

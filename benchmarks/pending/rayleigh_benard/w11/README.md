# w11/ — W11-C 判决机器档案（rayleigh_benard，verdict FAIL）

本目录是 2026-10 Wave-11 判决工件的入库副本；判决叙述、门禁口径与全部
判据数的 PROVENANCE 溯源见上级 [README.md](../README.md)。本 README 的
全部表格由机器脚本生成（`w11_archive_gen/gen_archive.py`，判决数字零手抄；
核验明细 `w11_archive_gen/verify_manifest.json`，均在战役暂存内）。

## 判决文件（md5）

| 文件 | md5 |
|---|---|
| `formal/verdict.json` | `d12cf82bcd96bf1e9d2271558c7924a2` |
| `formal/verdict_ext.json` | `b8fb8774a63846f0f7118cfd76bdaeb2` |

## 判决读数（机器读自判决 JSON）

| 项 | 值 | 备注 |
|---|---|---|
| G1 rel<=3% >=2 levels | True |  |
| G2 monotone E | False | E_ny 0.012238/0.010061/0.031010 -> ext256 0.046322 |
| G3 sign | True |  |
| G4 health | False |  |
| VERDICT | FAIL | ext FAIL |

## 归档清单（48 文件，157535 B）

| 路径 | B | md5 |
|---|---|---|
| `formal/ext_driver.py` | 612 | `59ebe63a52f09d5382a14357a082d8f9` |
| `formal/formal_run.py` | 6359 | `867e287bb4e9a495cf898e8478db1776` |
| `formal/formal_run_batched.py` | 12790 | `3c6f64f317950d12c8ba9b4c70946ee0` |
| `formal/results/ext_f_ny256_arm0.95.json` | 1167 | `ae5ad4eb836637a2529db316e08df03d` |
| `formal/results/ext_f_ny256_arm1.05.json` | 1123 | `0f5e59490c4b56204b4cb5cab8d26474` |
| `formal/results/f_ny128_arm0.85.json` | 1144 | `84e474db277d793f87292cd8dc3c4cd7` |
| `formal/results/f_ny128_arm0.95.json` | 1139 | `47ee47aa15c3f655f4ec366f325c64a8` |
| `formal/results/f_ny128_arm1.0.json` | 1186 | `0397482d9ba97a34bce7ec66c6fd360f` |
| `formal/results/f_ny128_arm1.05.json` | 1171 | `5c635120d0795f5b889035ccb4a17803` |
| `formal/results/f_ny128_arm1.15.json` | 1137 | `635a26facc7c62b67f147aec56ad93e4` |
| `formal/results/f_ny128_arm1.3.json` | 1131 | `83d98ddd04cc9e25ba57ed0290ef2b20` |
| `formal/results/f_ny256_arm0.85.json` | 1131 | `7bad3cc09fa3a903771a04b2d0cb94df` |
| `formal/results/f_ny256_arm0.95.json` | 1166 | `e5d788954f60e2ae58d130cd2b67fdeb` |
| `formal/results/f_ny256_arm1.0.json` | 1133 | `04f5bdd4d5ca84f02b6484ecfca4b2d2` |
| `formal/results/f_ny256_arm1.05.json` | 1127 | `5337f808926f8f0d09feddc723dc6a5b` |
| `formal/results/f_ny256_arm1.15.json` | 1164 | `73ea7f39ed0e08d068fbe05b048664c5` |
| `formal/results/f_ny256_arm1.3.json` | 1123 | `5539a3069b07cec86f56aaf710d4d7dc` |
| `formal/results/f_ny64_arm0.85.json` | 1130 | `fc164b94a1655f055189ed6595c3aad4` |
| `formal/results/f_ny64_arm0.95.json` | 1134 | `debebb488143eeff862118eb519085d3` |
| `formal/results/f_ny64_arm1.0.json` | 1141 | `a6bde7ba4341ae79b7a28c254be5ca09` |
| `formal/results/f_ny64_arm1.05.json` | 1123 | `b0993bf75dba78f849a68d1cc52d26dc` |
| `formal/results/f_ny64_arm1.15.json` | 1121 | `0aa37dea7b0ecd247b62e9e66abdd3e1` |
| `formal/results/f_ny64_arm1.3.json` | 1121 | `ffec1cf21aa0bfc54fafd6869f415375` |
| `formal/verdict.json` | 9478 | `d12cf82bcd96bf1e9d2271558c7924a2` |
| `formal/verdict_ext.json` | 12159 | `b8fb8774a63846f0f7118cfd76bdaeb2` |
| `formal/verify.py` | 4613 | `9e50640b5943399f76e134bd245a6ca6` |
| `formal/verify_ext.py` | 5180 | `6371c13a4b606bd9e7d1c81080cdb657` |
| `prereg/PREREG_MD5` | 154 | `c5c204f92a46a46c20c80ae6cca5734c` |
| `prereg/SHA256SUMS` | 860 | `295913a1fa583541ce1756802afc5ff1` |
| `prereg/prereg.md` | 11883 | `44ee931819b80bdcb1b635a969c68a50` |
| `probe/rb_lbm.py` | 19230 | `6e9e8f95f91871ad31e5c4f45ecbe177` |
| `probe/results/p2fp64_ny64_arm0.85.json` | 787 | `be622c37d4e9be92ea8d8d46882c49c1` |
| `probe/results/p2fp64_ny64_arm0.95.json` | 791 | `344ef549f35a4eba11f86a0e922371e7` |
| `probe/results/p2fp64_ny64_arm1.05.json` | 785 | `5fdfa96a20e92c6c22aa22795f946876` |
| `probe/results/p2fp64_ny64_arm1.15.json` | 781 | `3f172ce54c6b156d33cf202cb0a16890` |
| `probe/results/p2m_ny128_arm0.95.json` | 818 | `670452676d844f3dc7b13a614e948b8a` |
| `probe/results/p2m_ny128_arm1.3.json` | 793 | `ca780d20c1889e52d585fe1a2f8acd5f` |
| `probe/results/p2m_ny64_arm0.85.json` | 795 | `79e30bc2d4828880326b05b7003ff2e8` |
| `probe/results/p2m_ny64_arm0.95.json` | 801 | `7cc0251093b791da0e1e7823c7ddd7f0` |
| `probe/results/p2m_ny64_arm1.05.json` | 815 | `2561ab39f28ae7b8c29e92861c7562cf` |
| `probe/results/p2m_ny64_arm1.15.json` | 791 | `bca7d20913e61bdf55b848ab09a6f9aa` |
| `probe/results/p2m_ny64_arm1.3.json` | 817 | `efe954cc5b48a8f4236981190a74179b` |
| `ref/anchor_and_convergence.json` | 3465 | `c21e34472ab05d72c4d702105ea4ce1d` |
| `ref/lattice_mode.py` | 12875 | `95a9a6f8c9f32218f1716e79203300f8` |
| `ref/lattice_predictions.json` | 6236 | `e8f80925053df26fc5dc7c9b0fac34a6` |
| `ref/make_tables.py` | 3801 | `d9ba91d45ea68c34365a0795c3f4ce6e` |
| `ref/rb_eig.py` | 12761 | `942a3826afba7924ff430d12c59dcc7f` |
| `ref/ref_table.json` | 3493 | `78775aa2da759d9fb3e0672b675c7636` |

## 复现（已在临时副本逐位复现，暂存未动）

- `formal/verify.py` / `verify_ext.py`（BASE 重定向至副本）分别重生成 `formal/verdict.json` / `formal/verdict_ext.json`：均逐位一致（d12cf82bcd96bf1e9d2271558c7924a2 / b8fb8774a63846f0f7118cfd76bdaeb2）。

## 暂存留档（不入库）

- formal/results/sc_*.json、vb_*.json（敏感性复跑，不在判决链：verify*.py 只读 f_*/ext_f_*）
- probe/results/ 除 p2fp64_*/p2m_* 外的 Phase-0 开发探针（p1/p05/p07/p2_growth/ab_long 迹线）
- ref/eig_ny*.npz（特征向量种子二进制）
- old_archive/（2026-08-19 旧代档；其 run.py/result.json 已在库 pending 根目录）
- 各 *.log / watch 脚本 / __pycache__

staging 根：`/nfs/wangxi/runs/bm_widen_w11_20261006/`

## 披露

- 本目录 9 个 .py 经 ruff format/check --fix 规范化（import 重排/拆分、空白、未用 import 删除：见 verify_manifest.json:lint_normalization 的逐文件staging/repo md5 对照）；AST 等价已验证，判决复现不受影响（上文逐位一致）。
- run/verify 脚本内 `BASE` 仍指向战役暂存绝对路径（冻结记录原样入库，未改写）。
- README PROVENANCE 复核：48/48 条数字对归档 JSON 逐键机器复核一致（48/48 all_match=True）。

# sphere_re100_d3q27（pending）

**状态：🟡 达标待晋级——预注册门 G1–G4 机器判定全过（verdict=PASS，两档 |err|≤3% 且严格单调），卡在 owner 晋级决策与归档 PR。**

## 物理问题

三维定常绕球流（Re=100）阻力系数基准，是 verified/sphere_re100（D3Q19 + MRT-low-memory，远端 main 20cd23472 晋级）的孪生重制：格子换 D3Q27、碰撞换 BGK27、流迁移走 roll27、边界走 BFL 稀疏核链（prereg.md 冻结 + md5 回执的披露偏差）。参考阻力 cd_ref=1.0917310910948732，由远端 verified 存档按 sha256 四件套（result.json / t1 / t2 / manifest）钉定，非手抄。目的：把 27 速格子 + bgk27 + BFL 稀疏核组合纳入与 19 速已验证档同门的定量验证。

## 计算结果现状

机器判定档（staging gate/verdict27.json，2026-10-07 判定）：

| 档 | 域（格） | τ | 步数 | cd_window | cd_ref | 偏差 | G3 健康 |
|---|---|---|---|---|---|---|---|
| T1 D40 | 280×280×320 | 0.56 | 12000 | 1.1222091369579943 | 1.0917310910948732 | +2.791717311316592% | drift 0.14403977443611804%、cv 0.0028590423223743343、质量 0.0 ppm |
| T2 D60 | 420×420×480 | 0.59 | 24000 | 1.1197800349898392 | 1.0917310910948732 | +2.569217284710319% | drift 0.04177535342103482%、cv 0.0003830838162871174、质量 −0.18896452824013463 ppm |

两档 re_eff=99.99999999999991 / 100.00000000000004（u_lb=0.05；阻塞比 0.0160285339468867）。cd_window 取末 20% 采样窗均值（prereg 冻结口径）。

G4 披露行（19↔27 一致性与碰撞核敏感性）：

| 披露项 | D40 | D60 |
|---|---|---|
| 27−19 差（同域孪生） | 0.02038675950992681% | 0.02840952573897856% |
| 19 速存档误差 | 2.7707656824703957% | 2.5400861325477218% |
| 采样展宽 27 vs 19 存档 | 0.216457154745655 | 0.231（19 存档） |

同域碰撞核分解（小域 200×200×180）：bgk19 vs mrt19 差 0.017581948244776234%——27 速孪生落在与已验证 19 速档同一误差带内。

共性模块入口：equilibrium27/feq27、collide27、stream27_roll、BFL 稀疏核（库 PR #316）全部走库函数；实现走镜像 P1–P5（feq27_perq_into / collide27_inplace / bb27_solid_scatter / far_field27_inplace / correct_mass27_inplace），A/B 证据 diag/ab27.json（torch.equal）。资源：峰值显存 8.528244018554688 / 28.782794952392578 GiB，墙钟 1446.9706346988678 s / 9440.398661136627 s（BFL 链接 46106 / 104018）。

## 不达标清单

无。预注册门逐条（prereg.md 冻结文本）：

| 门 | 冻结条款 | 实测 | 判 |
|---|---|---|---|
| G1 精度 | 两档 \|err\|≤3% | 2.791717311316592% / 2.569217284710319% | ✓ |
| G2 单调 | 细档严格更优 | 2.7917 → 2.5692 下降 | ✓ |
| G3 健康 | drift<0.3%、cv≤1e-2、质量≤100 ppm、finite | 全过（两档） | ✓ |
| G4 披露 | 披露行齐 | 上表 | ✓ |

## 根因/诊断

无卡点。两道防线均位级核验：对远端 19 速存档 archive_twin 六项检查（domain_lu/tau/steps/u_lb/re_eff/lat）全真；镜像实现 P1–P5 A/B torch.equal。

## 晋级路径

晋级 = 归档 PR 将 pending/sphere_re100_d3q27 移至 verified/（连同 gate 工件 verdict27.json、t1_D40_27.json、t2_D60_27.json 与 prereg md5 链）。owner 决策项，无新计算，成本 0 GPU·h（原门消耗 = 两档墙钟合计约 3.0 GPU·h，见上）。

<!-- PROVENANCE [{"v":1.0917310910948732,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"cd_ref"},{"v":1.1222091369579943,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.cd_window"},{"v":2.791717311316592,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.err_pct"},{"v":12000,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.steps_used"},{"v":0.14403977443611804,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.g3.drift_pct"},{"v":0.0028590423223743343,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.g3.cv_closure"},{"v":0.0,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.g3.mass_drift_ppm"},{"v":1.1197800349898392,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.cd_window"},{"v":2.569217284710319,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.err_pct"},{"v":24000,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.steps_used"},{"v":0.04177535342103482,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.g3.drift_pct"},{"v":0.0003830838162871174,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.g3.cv_closure"},{"v":-0.18896452824013463,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.g3.mass_drift_ppm"},{"v":40,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.meta.D"},{"v":0.56,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.meta.tau"},{"v":280,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.meta.domain_lu[0]"},{"v":320,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.meta.domain_lu[2]"},{"v":60,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.meta.D"},{"v":0.59,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.meta.tau"},{"v":420,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.meta.domain_lu[0]"},{"v":480,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.meta.domain_lu[2]"},{"v":99.99999999999991,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.meta.re_eff"},{"v":100.00000000000004,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.meta.re_eff"},{"v":0.0160285339468867,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/t1_D40_27.json","k":"meta.blockage"},{"v":46106,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.meta.bfl_links"},{"v":104018,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.meta.bfl_links"},{"v":1446.9706346988678,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t1.meta.wall_s"},{"v":9440.398661136627,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"t2.meta.wall_s"},{"v":8.528244018554688,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/t1_D40_27.json","k":"meta.mem_peak_loop_gib"},{"v":28.782794952392578,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/t2_D60_27.json","k":"meta.mem_peak_loop_gib"},{"v":0.02038675950992681,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"G4_disclosure.delta27_19_D40_pct"},{"v":0.02840952573897856,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"G4_disclosure.delta27_19_D60_pct"},{"v":0.017581948244776234,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"G4_disclosure.delta_collision_pct"},{"v":0.216457154745655,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"G4_disclosure.span27_pct"},{"v":0.231,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"G4_disclosure.span19_pct_archived"},{"v":2.7707656824703957,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"G4_disclosure.err19_archived_D40"},{"v":2.5400861325477218,"f":"/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27/gate/verdict27.json","k":"G4_disclosure.err19_archived_D60"}] -->

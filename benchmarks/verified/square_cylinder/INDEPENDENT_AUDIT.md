# 方柱 Re=100 算例 · 独立审计（非导入 refspec 的第二意见）

常量（本审计自带，不从算例目录继承）：Cd_ref=1.48, St_ref=0.147, 门=3.0%

| D | 列 | Cd | vs 1.48 | St | vs 0.147 | 3% 门 | 块 std | St 估计子跨度 |
|---|---|---|---|---|---|---|---|---|
| 32 | summary | 1.4816 | +0.11% | 0.1482 | +0.82% | PASS | n/a | 0.61% |
| 48 | summary | 1.4844 | +0.30% | 0.1480 | +0.68% | PASS | n/a | 0.82% |
| 64 | pair | 1.4870 | +0.47% | 0.1479 | +0.61% | PASS | 0.0031 | 0.96% |

## 参考簇独立重导（从 result.json 存储的来源列表解析）

- 解析到 10 条 Cd/St 对：Cd ∈ [1.44, 1.55] 中位 1.4915，St ∈ [0.145, 0.149] 中位 0.1455
- 审计常量 1.48 落在 Cd 区间内：True（vs 中位 -0.77%）；0.147 落在 St 区间内：True（vs 中位 +1.03%）

## 发现（按严重度）

- **[advisory] stored-error-legacy** — D=64 cd: stored 0.5700% vs locked-convention 0.4730%; exactly reproduces the superseded 1.4785 convention, so it is legacy (declared), not drift
- **[advisory] stored-error-legacy** — D=64 st: stored 1.5800% vs locked-convention 0.6122%; exactly reproduces the superseded 0.1456 convention, so it is legacy (declared), not drift
- **[blocking] anchor-scope-mismatch** — ledger acceptance registers Re=200 / blockage 1/20 / <8% St,<10% Cd; the delivered case is Re=100 / blockage 1/40 / 3% bars. The delivered number cannot close the registered anchor as written.
- **[advisory] verdict-conditional-on-centre** — cd: the 3% PASS is conditional on the chosen centre. Over the cited band the same measurement spans -4.41% .. +3.26%, so |err| can reach 4.41% > 3% at the unfavourable end. Declaring 'Cd within 3%' without naming the centre oversells the result.
- **[advisory] centre-not-the-median** — the locked centre St=0.147 is +1.03% off the median (0.1455) of the 10 entries the case itself enumerates; Cd centre is -0.77% off its median (1.4915). The pass survives (worst |St err| over the band stays 2.21% <= 3%), but the centre is a round adopted value, not a statistic of the sources.
- **[advisory] st-estimator-ambiguity** — D=32: spectral St 0.1482 vs zero-crossing 0.1473 (0.61% apart); the gate scores the spectral value, i.e. the one nearer the reference. Both are inside the 3% bar, so the verdict does not flip -- but the definition must be declared.
- **[advisory] st-estimator-ambiguity** — D=48: spectral St 0.1480 vs zero-crossing 0.1468 (0.82% apart); the gate scores the spectral value, i.e. the one nearer the reference. Both are inside the 3% bar, so the verdict does not flip -- but the definition must be declared.
- **[advisory] st-estimator-ambiguity** — D=64: spectral St 0.1479 vs zero-crossing 0.1465 (0.96% apart); the gate scores the spectral value, i.e. the one nearer the reference. Both are inside the 3% bar, so the verdict does not flip -- but the definition must be declared.
- **[not_run] archived-uncertainty** — D=32: archive stores no block averages, so the statistical error of the reported Cd/St cannot be reconstructed from the repository.
- **[not_run] archived-uncertainty** — D=48: archive stores no block averages, so the statistical error of the reported Cd/St cannot be reconstructed from the repository.
- **[not_run] independent-reproduction** — fresh full re-run (audit item 4): not_run -- no fresh re-run archive found

计数：blocking 1 / advisory 7 / not_run 3  → 处置：**request_changes**

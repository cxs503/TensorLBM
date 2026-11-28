# dam_break_2d_multimodel（基线审计包，非 verified）

2D 溃坝四模型（SC/SCMP/CG/FE）对 Martin & Moyce (1952) 14 点实验表的基线审计：
流场演化 + 定量比较，判决 4/4 FAIL。详见 [REPORT.md](REPORT.md)。
修复主线=MP-DAMBREAK 常设线（S1 m211 在飞）；锚点冻结见 agent_work/mp_dambreak/prereg/anchors.md。
判据数字一律机器读自 metrics_vs_MM.json / result.json，零手抄。

## Historical post-processing

The [case-specific summary tool](dam_break_summary.py) retains its original
repository-relative input/output contract. To repeat the archived **front-position**
comparison without running a solver, copy the original four tracked CSVs and
reference JSON to the expected layout in an isolated scratch directory:

```bash
repo="$(pwd)"  # from the checkout root
scratch="$(mktemp -d)"
mkdir -p "$scratch/benchmarks/pending/dam_break_sc"
cp "$repo/benchmarks/pending/dam_break_sc/result.json" "$scratch/benchmarks/pending/dam_break_sc/"
for m in sc scmp cg fe; do
    dest="$scratch/outputs/dam_break_benchmark/$m/dam_break/bench_dam_${m}_fast"
    mkdir -p "$dest"
    cp "$repo/benchmarks/pending/dam_break_2d_multimodel/front_position_$m.csv" "$dest/front_position.csv"
done
(cd "$scratch" && python "$repo/benchmarks/pending/dam_break_2d_multimodel/dam_break_summary.py")
```

The tracked CSVs reproduce the four historical rounded RMSE/MAE values and
generate a front-comparison PNG. Original per-step snapshots are not tracked;
the optional GIF cannot be regenerated from these inputs. This is offline
post-processing of historical failed results, not a new scientific validation.

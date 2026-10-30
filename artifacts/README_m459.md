# Mission m459 — Gartling 1990 Backward-Facing Step Re=800 Checkpoint/Resume

**Status**: 🟡 IN PROGRESS — Checkpoint/resume validated, 25k/150k steps completed  
**Device**: SDAA (sdaa:0)  
**Report**: POSTed as report #382

---

## Summary

This mission validates the checkpoint/resume mechanism for overcoming SDAA wall-clock limits in long-running simulations. The Gartling 1990 backward-facing step benchmark (Re=800, ER=2) requires 150k+ steps to reach steady state, but SDAA limits single runs to ~2 minutes.

**Solution**: Segment the simulation into 5k-step chunks, save/load distribution function `f` between segments.

**Result**: 
- ✅ Checkpoint/resume mechanism works correctly
- ✅ 25,000 steps accumulated across 5 segments
- ⚠️ Flow not yet converged (X₁/H = 2.05 vs reference 6.10, drift 70%)
- ⚠️ Need 150k+ steps total (currently at 25k)

**Verdict**: HONEST-FAIL (insufficient steps, not algorithm error)

---

## Artifacts

### Run Data
- `m459_run/seg[1-5].json` — Per-segment results (X₁/H series, timing, diagnostics)
- `m459_ckpt/seg[1-5].pt` — Checkpoint files (distribution function f + step counter)

### Analysis
- `m459_analysis.json` — Convergence analysis summary
- `m459_convergence_curve.json` — Full X₁/H time series (for plotting)
- `m459_convergence_plot.png` — Convergence curve (X₁/H vs steps)
- `m459_segment_summary.png` — Segment-by-segment evolution

### Scripts
- `benchmarks/pending/backward_step/run_segment.py` — Single-segment runner
- `benchmarks/pending/backward_step/analyze_m459.py` — Analysis script
- `benchmarks/pending/backward_step/plot_m459.py` — Plotting script
- `benchmarks/pending/backward_step/generate_report_m459.py` — Report generator

---

## How to Continue the Run

To complete the full 150k steps:

```bash
# Continue from checkpoint seg5 (step 25000)
cd /root/tp-agent/work/m459

# Run segments 6-30 (each 5000 steps, total 125k more steps)
for i in {6..30}; do
  prev=$((i-1))
  timeout 90 python3 benchmarks/pending/backward_step/run_segment.py \
    --m 32 --steps 5000 --device sdaa:0 \
    --resume artifacts/m459_ckpt/seg${prev}.pt \
    --out artifacts/m459_run/seg${i}.json \
    --save-ckpt artifacts/m459_ckpt/seg${i}.pt \
    --seg-id $i --compile-mode eager
  
  if [ $? -ne 0 ]; then
    echo "Segment $i failed!"
    break
  fi
  
  echo "Segment $i complete. Checkpoint: artifacts/m459_ckpt/seg${i}.pt"
done

# Re-run analysis
python3 benchmarks/pending/backward_step/analyze_m459.py

# Re-generate plots
python3 benchmarks/pending/backward_step/plot_m459.py

# Re-generate and POST report
python3 benchmarks/pending/backward_step/generate_report_m459.py
# Then POST /tmp/report_m459.json to the API
```

**Estimated time**: ~31 minutes for remaining 125k steps (at 66 steps/s)

---

## Convergence Criteria

1. **Steady state**: X₁/H drift <1% over 3 consecutive 10k-step windows
2. **Accuracy**: |X₁/H - 6.10| / 6.10 ≤ 3%

Current status (25k steps):
- X₁/H = 2.05 (error -66.44%)
- Drift = 70% (not converged)
- **Need**: Continue to 150k+ steps

---

## Technical Details

### Configuration
- Grid: m=32 (H=64 cells, ny=65, nx=968)
- Re = 800 (based on channel height H)
- Collision: MRT
- Wall treatment: Mode K (pre-stream bounce-back + sponge layer)
- Segment size: 5,000 steps
- Checkpoint: Full distribution function f (9 × ny × nx × 4 bytes ≈ 2.2 MB)

### Performance
- Rate: ~66 steps/s (SDAA, m=32)
- Memory: ~50 MB per segment
- Checkpoint I/O: ~1s per save/load

### Reference
- Gartling, D. K. (1990). "A test problem for outflow boundary conditions..."
- Int. J. Numer. Meth. Fluids 11(7):953–967
- X₁/H = 6.10 (lower-wall reattachment length / channel height)

---

## Report

Full report POSTed to Tensor API as **report #382**.

Six elements:
1. ① Case introduction (physics, geometry, parameters)
2. ② Methodology (checkpoint/resume strategy, convergence criteria)
3. ③ Results (25k steps, X₁/H = 2.05, error -66.44%)
4. ④ Figures (convergence curve, segment summary)
5. ⑤ Notes (limitations, pitfalls, applicability)
6. ⑥ Error assessment (HONEST-FAIL, insufficient steps)

---

**Generated**: 2026-10-30  
**Mission**: m459 (SDAA 验证案例扫荡 · backward_step · 第2轮)

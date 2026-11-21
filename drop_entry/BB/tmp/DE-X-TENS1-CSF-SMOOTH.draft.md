# DE-X-TENS1-CSF-SMOOTH Draft

## Status
- [x] Read TRACK_TENS_W1.md
- [x] Read csf.py
- [ ] Implement curvature smoothing
- [ ] Run experiment
- [ ] Analyze results
- [ ] Write final card

## Plan
1. Modify `src/tensorlbm/free_surface_tension/csf.py`:
   - Add a `smooth_kappa` parameter to `csf_force`.
   - If `smooth_kappa` is True, apply 3-point moving average to `kappa` along each axis.
2. Run with `--arm TENS1-CSF-SMOOTH`.
3. Compare `mach_max` and `peak_depth` with baseline.

## Baseline
- Peak depth: 1.4352D @ 11.091ms
- Mach max: (to be checked)

## Prediction
- Mach max drops > 20%.
- Peak depth error < 15%.

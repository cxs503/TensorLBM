## Summary of Cross-Validation Results

| Case | Kind | $u_{peak}$ err | $\epsilon_\infty$ | $Q$ err | $y_{iface}$ err | Gate |
|------|------|----------------|---------------------|---------|-----------------|------|
| `control_single` | control | 0.37% | 0.37% | 0.38% | — | ✓ PASS |
| `control_mixture` | control | 0.37% | 0.37% | 0.38% | 0.00% | ✓ PASS |
| `layered_r167` | layered | 7.44% | 8.84% | 8.16% | 20.05% | ✗ FAIL |
| `layered_static` | static | 0.00% | 0.00% | 0.00% | 13.06% | ✓ PASS |

**Layered cases summary:**
- Total: 1
- Gate pass: 0/1
- Average $u_{peak}$ error: 7.44%
- Average $\epsilon_\infty$: 8.84%
- Max interface position error: 20.05% of H
"""Quantitative benchmark suite for all TensorLBM models.

Tests every model configuration with consistent metrics:
  - Front position X* at t*=0.1, 0.2, 0.3
  - Mass conservation ratio
  - NaN status
  - Runtime (steps/sec)

Run: .venv/bin/python3 benchmarks/bench_all_models.py
"""

import json, math, time, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm.dam_break_3d import DamBreak3DConfig, run_dam_break_3d
from tensorlbm.free_surface_lbm_2d import (
    init_fill_rectangular_2d, init_flags_from_fill_2d, free_surface_step_2d,
)
from tensorlbm.d2q9 import equilibrium


# ===================================================================
# Test case definitions
# ===================================================================

def make_3d_cases():
    """Generate 3D dam-break test cases."""
    base = dict(
        nx=150, ny=60, nz=60, dam_width=60, fill_height=59,
        gravity=8e-5, rho_heavy=1.0, rho_light=0.5,
        n_steps=200, output_interval=40, free_slip_y=True,
        device='cpu', overwrite=True,
    )
    cases = []

    # Free-surface at various tau
    for tau_val in [0.53, 0.55, 0.6, 0.8]:
        cases.append(dict(base, model='fs', tau=tau_val, C_s=0.1,
                         collision='mrt_smag', A=0.0,
                         run_name=f'bench_fs_tau{int(tau_val*100)}',
                         label=f'FS τ={tau_val} SGS'))

    # Free-surface without SGS
    cases.append(dict(base, model='fs', tau=0.8, C_s=0.0,
                     collision='bgk', A=0.0,
                     run_name='bench_fs_bgk',
                     label='FS τ=0.8 BGK'))

    # CG-BGK baseline
    cases.append(dict(base, model='cg', tau=1.0, A=0.005,
                     collision='bgk', C_s=0.0,
                     run_name='bench_cg_bgk',
                     label='CG τ=1.0 BGK'))

    # CG-MRT
    cases.append(dict(base, model='cg', tau=0.8, A=0.002,
                     collision='mrt', C_s=0.0,
                     run_name='bench_cg_mrt',
                     label='CG τ=0.8 MRT'))

    # CG-MRT+Smagorinsky
    cases.append(dict(base, model='cg', tau=0.6, A=0.002,
                     collision='mrt_smag', C_s=0.1,
                     run_name='bench_cg_mrt_smag',
                     label='CG τ=0.6 MRT+SGS'))

    # SC-BGK
    cases.append(dict(base, model='sc', tau=1.0, G_sc=0.9,
                     collision='bgk', C_s=0.0,
                     run_name='bench_sc_bgk',
                     label='SC τ=1.0 BGK'))

    # SC-MRT
    cases.append(dict(base, model='sc', tau=0.8, G_sc=0.9,
                     collision='mrt', C_s=0.0,
                     run_name='bench_sc_mrt',
                     label='SC τ=0.8 MRT'))

    # FE
    cases.append(dict(base, model='fe', tau=0.8,
                     collision='bgk', C_s=0.0,
                     run_name='bench_fe',
                     label='FE τ=0.8'))

    return cases


def make_2d_cases():
    """Generate 2D free-surface test cases."""
    return [
        dict(ny=150, nx=400, column_width=80, column_height=120,
             tau=0.8, gravity=2e-3, n_steps=300, label='FS-2D τ=0.8'),
        dict(ny=150, nx=400, column_width=80, column_height=120,
             tau=0.6, gravity=2e-3, n_steps=300, label='FS-2D τ=0.6'),
        dict(ny=150, nx=400, column_width=80, column_height=120,
             tau=0.55, gravity=2e-3, n_steps=300, label='FS-2D τ=0.55'),
    ]


# ===================================================================
# 3D benchmark runner
# ===================================================================

def run_3d_case(cfg: dict) -> dict:
    """Run one 3D case and return metrics."""
    label = cfg.pop('label')
    config = DamBreak3DConfig(**{k: v for k, v in cfg.items()
                                 if k in DamBreak3DConfig.__dataclass_fields__})
    t0 = time.time()
    try:
        run_dir = run_dam_break_3d(config)
        elapsed = time.time() - t0
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        diags = meta['diagnostics']

        # Extract metrics
        x_star_values = [d['x_star'] for d in diags]
        mass_values = [d['mean_rho'] for d in diags]
        nan_free = all(
            (not isinstance(d.get('x_star'), float) or d['x_star'] == d['x_star'])
            and (not isinstance(d.get('mean_rho'), float) or d['mean_rho'] == d['mean_rho'])
            for d in diags
        )

        steps = cfg['n_steps']
        return {
            'label': label,
            'model': cfg.get('model', '?'),
            'tau': cfg.get('tau', 0),
            'X*_final': x_star_values[-1] if x_star_values else None,
            'X*_t0.1': _interp_xstar(diags, 0.1),
            'X*_t0.2': _interp_xstar(diags, 0.2),
            'X*_t0.3': _interp_xstar(diags, 0.3),
            'mass_ratio': mass_values[-1] / mass_values[0] if len(mass_values) > 1 else 1.0,
            'mass_stable': abs(mass_values[-1] - mass_values[-2]) < 0.01 if len(mass_values) > 2 else True,
            'nan_free': nan_free,
            'time_s': elapsed,
            'steps_per_s': steps / elapsed if elapsed > 0 else 0,
        }
    except Exception as e:
        return {'label': label, 'error': str(e), 'nan_free': False}


def _interp_xstar(diags, t_target):
    """Interpolate X* at target t* from diagnostics."""
    if not diags:
        return None
    ts = [d['t_star'] for d in diags]
    xs = [d['x_star'] for d in diags]
    if t_target <= ts[0]:
        return xs[0]
    if t_target >= ts[-1]:
        return xs[-1]
    for i in range(len(ts) - 1):
        if ts[i] <= t_target <= ts[i + 1]:
            frac = (t_target - ts[i]) / (ts[i + 1] - ts[i])
            return xs[i] + frac * (xs[i + 1] - xs[i])
    return xs[-1]


# ===================================================================
# 2D benchmark runner
# ===================================================================

def run_2d_case(cfg: dict) -> dict:
    """Run one 2D free-surface case and return metrics."""
    label = cfg.pop('label')
    ny, nx = cfg['ny'], cfg['nx']
    cw, ch = cfg['column_width'], cfg['column_height']
    tau, g, n_steps = cfg['tau'], cfg['gravity'], cfg['n_steps']

    device = 'cpu'
    fill, solid = init_fill_rectangular_2d(ny, nx, cw, ch, device)
    flags = init_flags_from_fill_2d(fill, solid)
    feq = equilibrium(torch.ones((ny, nx)), torch.zeros((ny, nx)), torch.zeros((ny, nx)))
    active = (flags == 1) | (flags == 2)  # LIQUID | INTERFACE
    f = torch.where(active.unsqueeze(0), feq, torch.zeros_like(feq))

    mass0 = f.sum().item()
    front_series = []
    t0 = time.time()
    nan_free = True

    for step in range(1, n_steps + 1):
        f, fill, flags = free_surface_step_2d(f, fill, flags, solid, tau=tau, gy=-g)
        if torch.isnan(f).any():
            nan_free = False
            break
        if step % 30 == 0:
            front = ((flags == 1) | (flags == 2)).any(dim=0).int().nonzero()
            fx = front.max().item() if front.numel() > 0 else 0
            T = step * math.sqrt(g / cw)
            Z = fx / cw
            front_series.append((T, Z))

    elapsed = time.time() - t0
    mass_final = f.sum().item()
    return {
        'label': label,
        'model': 'FS-2D',
        'tau': tau,
        'X*_final': front_series[-1][1] if front_series else None,
        'X*_t0.1': _interp_2d(front_series, 0.1),
        'X*_t0.5': _interp_2d(front_series, 0.5),
        'X*_t1.0': _interp_2d(front_series, 1.0),
        'X*_t1.5': _interp_2d(front_series, 1.5),
        'mass_ratio': mass_final / mass0 if mass0 > 0 else 0,
        'nan_free': nan_free,
        'time_s': elapsed,
        'steps_per_s': n_steps / elapsed if elapsed > 0 else 0,
    }


def _interp_2d(series, t_target):
    if not series: return None
    for i in range(len(series) - 1):
        if series[i][0] <= t_target <= series[i + 1][0]:
            frac = (t_target - series[i][0]) / (series[i + 1][0] - series[i][0])
            return series[i][1] + frac * (series[i + 1][1] - series[i][1])
    return series[-1][1] if t_target >= series[-1][0] else series[0][1]


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("TensorLBM Quantitative Benchmark Suite")
    print("=" * 70)

    results = []

    # 3D cases
    print("\n--- 3D Dam-Break Cases ---")
    for cfg in make_3d_cases():
        label = cfg.get('label', '?')
        print(f"  Running: {label} ...", end=' ', flush=True)
        r = run_3d_case(cfg.copy())
        status = "✓" if r.get('nan_free') else "✗ FAIL"
        xf = r.get('X*_final', '?')
        print(f"{status} X*={xf}")
        results.append(r)

    # 2D cases
    print("\n--- 2D Free-Surface Cases ---")
    for cfg in make_2d_cases():
        label = cfg.get('label', '?')
        print(f"  Running: {label} ...", end=' ', flush=True)
        r = run_2d_case(cfg.copy())
        status = "✓" if r.get('nan_free') else "✗ FAIL"
        xf = r.get('X*_final', '?')
        print(f"{status} X*={xf}")
        results.append(r)

    # Print table
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    header = f"{'Case':<28} {'τ':>5} {'X*@0.1':>8} {'X*@0.2':>8} {'X*@0.3':>8} {'Mass':>7} {'NaN':>5} {'sp/s':>8}"
    print(header)
    print("-" * 70)

    for r in results:
        if 'error' in r:
            print(f"{r['label']:<28} {'ERR':>5} {r['error'][:40]}")
            continue
        x01 = f"{r.get('X*_t0.1', 0):.2f}" if r.get('X*_t0.1') else 'N/A'
        x02 = f"{r.get('X*_t0.2', 0):.2f}" if r.get('X*_t0.2') else 'N/A'
        x03 = f"{r.get('X*_t0.3', 0):.2f}" if r.get('X*_t0.3') else 'N/A'
        mass = f"{r.get('mass_ratio', 1):.2f}x" if r.get('mass_ratio') else '?'
        nan = "OK" if r.get('nan_free') else "FAIL"
        sps = f"{r.get('steps_per_s', 0):.0f}"
        print(f"{r['label']:<28} {r.get('tau',0):.2f} {x01:>8} {x02:>8} {x03:>8} {mass:>7} {nan:>5} {sps:>8}")

    # Summary
    passed = sum(1 for r in results if r.get('nan_free', False))
    total = len(results)
    print(f"\nPASSED: {passed}/{total} ({100*passed//total if total else 0}%)")

    # Save
    out = Path(__file__).parent.parent / "outputs" / "bench_all_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, 'w'), indent=2, default=str)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

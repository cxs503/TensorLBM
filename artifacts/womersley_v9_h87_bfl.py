#!/usr/bin/env python
"""Run H87 convergence + BFL wall correction at H=29."""
from __future__ import annotations
import json, math, time, sys, os
from pathlib import Path
import numpy as np

C = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=np.float64)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
OPP = np.array([0,3,4,1,2,7,8,5,6])

def equilibrium_np(rho, ux, uy):
    u2 = ux*ux + uy*uy
    feq = np.zeros((9, rho.shape[0], rho.shape[1]), dtype=np.float64)
    for i in range(9):
        cu = C[i,0]*ux + C[i,1]*uy
        feq[i] = W[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2)
    return feq

def macroscopic_np(f):
    rho = f.sum(axis=0)
    ux = (f * C[:,0,None,None]).sum(axis=0) / rho
    uy = (f * C[:,1,None,None]).sum(axis=0) / rho
    return rho, ux, uy

def collide_bgk_np(f, tau):
    rho, ux, uy = macroscopic_np(f)
    feq = equilibrium_np(rho, ux, uy)
    return f - (f - feq) / tau

def stream_np(f):
    f_new = np.empty_like(f)
    for i in range(9):
        f_new[i] = np.roll(f[i], shift=(C[i,0], C[i,1]), axis=(1,0))
    return f_new

def apply_force_np(f, a_x):
    rho = f.sum(axis=0)
    for i in range(9):
        f[i] += W[i] * 3.0 * rho * C[i,0] * a_x
    return f

def womersley_phasor(y_phys, H, alpha, a0, omega):
    lam = np.sqrt(1j) * alpha
    yp = (y_phys - H / 2.0) / (H / 2.0)
    return (a0 / (1j * omega)) * (1.0 - np.cosh(lam * yp) / np.cosh(lam))

MEASURE_PERIODS = 4
N_PHASES = 8
TOL_PCT = 3.0

def run_case(H, alpha, tau, u_peak_target, nx, wall_mode="bb", bfl_q=0.5, seed=0):
    np.random.seed(seed)
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    H_eff = float(ny - 2)
    omega = 4.0 * alpha * alpha * nu / (H_eff * H_eff)
    T = 2.0 * math.pi / omega
    T_int = int(round(T))
    rows = np.arange(1, ny - 1, dtype=np.float64)
    y_phys = rows - 0.5
    lam = np.sqrt(1j) * alpha
    yp = (y_phys - H_eff / 2.0) / (H_eff / 2.0)
    shape_max = float(np.max(np.abs(1.0 - np.cosh(lam * yp) / np.cosh(lam))))
    a0 = u_peak_target * omega / shape_max
    u_peak_ana = a0 * shape_max / omega
    tau_diff = H_eff * H_eff / (math.pi**2 * nu)
    skip_periods = max(5, math.ceil(6.5 * tau_diff / T))
    skip_steps = skip_periods * T_int
    n_meas = MEASURE_PERIODS * T_int
    total_steps = skip_steps + n_meas
    Ma = u_peak_target / math.sqrt(1.0 / 3.0)
    stokes_delta = H_eff / (alpha * math.sqrt(2.0))

    rho0 = np.ones((ny, nx))
    f = equilibrium_np(rho0, np.zeros((ny,nx)), np.zeros((ny,nx)))
    initial_mass = float(f.sum())

    # Unknown directions at bottom wall (row 0): populations pointing up
    # f2 (0,1), f5 (1,1), f6 (-1,1) -> opposite: f4, f7, f8
    # Unknown at top wall (row ny-1): f4, f7, f8 -> opposite: f2, f5, f6
    bot_unknown = [2, 5, 6]  # these stream from row 0 to row 1
    top_unknown = [4, 7, 8]  # these stream from row ny-1 to row ny-2

    if wall_mode == "bb":
        def _step(f, a_t):
            f_pre = f.copy()
            f = collide_bgk_np(f, tau)
            f[:, 0, :] = f_pre[OPP, 0, :]
            f[:, ny-1, :] = f_pre[OPP, ny-1, :]
            f = apply_force_np(f, a_t)
            return stream_np(f)
    elif wall_mode == "bfl":
        # BFL interpolated bounce-back with wall fraction q
        # For q >= 0.5: f_i(row0) = (1/(2q)) * f_opp_pre(row0) + ((2q-1)/(2q)) * f_opp_post(row2)
        # For q < 0.5:  f_i(row0) = q * f_opp_pre(row0) + (1-q) * f_opp_post(row1)
        q = bfl_q
        def _step(f, a_t):
            f_pre = f.copy()
            f = collide_bgk_np(f, tau)  # post-collision
            if q >= 0.5:
                w1 = 1.0 / (2.0 * q)
                w2 = (2.0*q - 1.0) / (2.0 * q)
                for i in bot_unknown:
                    oi = OPP[i]
                    f[i, 0, :] = w1 * f_pre[oi, 0, :] + w2 * f[oi, 2, :]
                for i in top_unknown:
                    oi = OPP[i]
                    f[i, ny-1, :] = w1 * f_pre[oi, ny-1, :] + w2 * f[oi, ny-3, :]
            else:
                w1 = q
                w2 = 1.0 - q
                for i in bot_unknown:
                    oi = OPP[i]
                    f[i, 0, :] = w1 * f_pre[oi, 0, :] + w2 * f[oi, 1, :]
                for i in top_unknown:
                    oi = OPP[i]
                    f[i, ny-1, :] = w1 * f_pre[oi, ny-1, :] + w2 * f[oi, ny-2, :]
            f = apply_force_np(f, a_t)
            return stream_np(f)
    else:
        raise ValueError(wall_mode)

    Sc = np.zeros(ny); Ss = np.zeros(ny)
    snap_steps = [skip_steps + 1 + (m * T_int) // N_PHASES for m in range(N_PHASES)]
    snaps = {}
    t0 = time.time()
    for step in range(1, total_steps + 1):
        a_t = a0 * math.cos(omega * step)
        f = _step(f, a_t)
        if step > skip_steps:
            _, ux_step, _ = macroscopic_np(f)
            col_mean = ux_step.mean(axis=1)
            Sc += col_mean * math.cos(omega * step)
            Ss += col_mean * math.sin(omega * step)
            if step in snap_steps:
                snaps[step] = col_mean.copy()
    elapsed = time.time() - t0

    ur = (2.0 / n_meas) * Sc
    ui = (-(2.0 / n_meas)) * Ss
    z_num = (ur + 1j * ui)[1 : ny - 1]
    z_ana = womersley_phasor(y_phys, H_eff, float(alpha), a0, omega)
    row_mask = np.abs(z_ana) >= 0.10 * u_peak_ana
    complex_rel = np.abs(z_num / z_ana - 1.0)[row_mask] * 100.0

    _a, _b = np.real(z_ana), np.imag(z_ana)
    _na, _nb, _ab = float(_a @ _a), float(_b @ _b), float(_a @ _b)
    _tr, _det = _na + _nb, _na * _nb - _ab * _ab
    ref_norm_peak = math.sqrt(0.5 * (_tr + math.sqrt(max(_tr * _tr - 4.0 * _det, 0.0))))

    phase_rows = []; l2p_list, ptp_list = [], []
    for sstep in snap_steps:
        u_snap = snaps[sstep][1 : ny - 1]
        u_ref = np.real(z_ana * np.exp(1j * omega * sstep))
        du = u_snap - u_ref
        l2p = float(np.linalg.norm(du) / ref_norm_peak)
        ptp = float(np.max(np.abs(du)) / u_peak_ana)
        l2p_list.append(l2p); ptp_list.append(ptp)
        phase_rows.append({"step": sstep, "phase_deg": round(math.degrees((omega*sstep)%(2*math.pi)),2),
            "l2_rel_peaknorm_pct": round(l2p*100,4), "max_abs_over_upeak_pct": round(ptp*100,4)})

    per_row_rel = np.abs(z_num / z_ana - 1.0) * 100.0
    center_row = ny // 2
    zc_num = (ur + 1j * ui)[center_row]
    zc_ana = womersley_phasor(np.array([H_eff / 2.0]), H_eff, float(alpha), a0, omega)[0]

    result = {
        "H": H, "alpha": alpha, "tau": tau, "nu_lb": nu, "omega": omega,
        "T_int": T_int, "a0": a0, "u_peak_ana": u_peak_ana, "Ma": Ma,
        "stokes_delta_lu": round(stokes_delta, 3),
        "skip_steps": skip_steps, "total_steps": total_steps,
        "wall_mode": wall_mode + (f"_q{bfl_q}" if wall_mode=="bfl" else ""),
        "profile_l2_max_pct": round(float(np.max(l2p_list))*100, 4),
        "profile_pointwise_max_over_upeak_pct": round(float(np.max(ptp_list))*100, 4),
        "complex_rel_max_pct": round(float(np.max(complex_rel)), 4),
        "center_complex_rel_pct": round(float(abs(zc_num/zc_ana-1.0))*100, 4),
        "mass_drift_pct": round((float(f.sum())-initial_mass)/initial_mass*100, 6),
        "finite": bool(np.isfinite(f).all()),
        "elapsed_s": round(elapsed, 1),
        "phase_table": phase_rows,
        "first_row_complex_rel_pct": round(float(per_row_rel[0]), 4),
        "second_row_complex_rel_pct": round(float(per_row_rel[1]), 4) if len(per_row_rel)>1 else None,
        "third_row_complex_rel_pct": round(float(per_row_rel[2]), 4) if len(per_row_rel)>2 else None,
    }
    extra = {"z_num": z_num, "z_ana": z_ana, "y_phys": y_phys,
             "per_row_rel": per_row_rel, "row_mask": row_mask, "u_peak_ana": u_peak_ana}
    return result, extra


def main():
    tau = 0.80; alpha = 8.0; u_peak = 0.03; nx = 16
    results = {}; extras = {}

    # H87 convergence point
    print("[H87_bb] running...", flush=True)
    r, e = run_case(87, alpha, tau, u_peak, nx, wall_mode="bb")
    results["H87_bb"] = r; extras["H87_bb"] = e
    print(f"  complex={r['complex_rel_max_pct']}% first_row={r['first_row_complex_rel_pct']}% "
          f"steps={r['total_steps']} t={r['elapsed_s']}s", flush=True)

    # BFL wall correction at H=29 with various q
    for q in [0.45, 0.55, 0.60, 0.65]:
        label = f"H29_bfl_q{q}"
        print(f"[{label}] running...", flush=True)
        try:
            r, e = run_case(29, alpha, tau, u_peak, nx, wall_mode="bfl", bfl_q=q)
            results[label] = r; extras[label] = e
            print(f"  complex={r['complex_rel_max_pct']}% first_row={r['first_row_complex_rel_pct']}% "
                  f"finite={r['finite']} t={r['elapsed_s']}s", flush=True)
        except Exception as ex:
            results[label] = {"error": str(ex)}
            print(f"  ERROR: {ex}", flush=True)

    # Save results for merging
    out = Path(__file__).resolve().parent / "womersley_v9_extra.json"
    # Convert numpy arrays to lists for JSON
    serializable = {}
    for k, v in results.items():
        if "error" in v:
            serializable[k] = v
        else:
            serializable[k] = {kk: vv for kk, vv in v.items()}
    out.write_text(json.dumps(serializable, indent=2))
    print(f"\nExtra results saved to {out}", flush=True)

    # Also save extras (phasor data) as npz for plotting
    npz_path = Path(__file__).resolve().parent / "womersley_v9_extras.npz"
    save_data = {}
    for k, e in extras.items():
        save_data[f"{k}_z_num"] = e["z_num"]
        save_data[f"{k}_z_ana"] = e["z_ana"]
        save_data[f"{k}_y_phys"] = e["y_phys"]
        save_data[f"{k}_per_row_rel"] = e["per_row_rel"]
        save_data[f"{k}_row_mask"] = e["row_mask"]
    np.savez(npz_path, **save_data)
    print(f"Extras saved to {npz_path}", flush=True)


if __name__ == "__main__":
    main()

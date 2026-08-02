"""Domain sensitivity worker — logs to file, enables flush via unbuffered stdout."""
import json, math, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch

KAPPA, B_CONST, REF_CT = 0.41, 5.0, 0.00405


def equilibrium3d_efficient(rho, ux, uy, uz, device):
    from tensorlbm.d3q19 import C as C19, W as W19
    c = C19.to(device).float()
    w = W19.to(device).float().view(19, 1, 1, 1)
    nz, ny, nx = rho.shape
    u_sq = ux * ux + uy * uy + uz * uz
    f = torch.empty(19, nz, ny, nx, dtype=torch.float32, device=device)
    cx, cy, cz = c[:, 0].view(19, 1, 1, 1), c[:, 1].view(19, 1, 1, 1), c[:, 2].view(19, 1, 1, 1)
    cu = cx * ux + cy * uy + cz * uz
    f.copy_(cu); f.mul_(3.0)
    cu_sq = cu * cu; f.add_(4.5 * cu_sq); del cu_sq, cu
    f.add_(1.0); f.add_(-1.5 * u_sq.unsqueeze(0))
    f.mul_(rho.unsqueeze(0)); f.mul_(w)
    return f


def collide_mrt_smag_chunked(f, tau, C_s, s_e=1.19, s_eps=1.4, s_q=1.2, s_pi=1.4, chunk=500_000):
    from tensorlbm.d3q19 import macroscopic3d, C as C19, W as W19
    from tensorlbm.turbulence import _get_d3q19_mrt_matrices, _neq_stress_norm_3d, _smagorinsky_tau

    device = f.device; M, M_inv = _get_d3q19_mrt_matrices(device)
    nz, ny, nx = f.shape[1:]; N = nz * ny * nx

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d_efficient(rho, ux, uy, uz, device)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_3d(f_neq)
    tau_eff = _smagorinsky_tau(tau, pi_norm, rho, C_s)
    s_nu_field = 1.0 / tau_eff
    del feq, f_neq, pi_norm, tau_eff
    torch.sdaa.empty_cache()

    s_fixed = torch.tensor([0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, 0.0, s_q,
                             0.0, 0.0, 0.0, 0.0, 0.0, s_pi, s_pi, 1.0, 1.0, 1.0],
                           dtype=f.dtype, device=device)

    f_flat = f.reshape(19, N).contiguous()
    rho_f = rho.reshape(N); ux_f = ux.reshape(N); uy_f = uy.reshape(N); uz_f = uz.reshape(N)
    snu_f = s_nu_field.reshape(N)
    del rho, ux, uy, uz, s_nu_field

    c = C19.to(device).float(); w = W19.to(device).float()
    result = torch.empty(19, N, dtype=f.dtype, device=device)

    for s in range(0, N, chunk):
        e = min(s + chunk, N); ch = slice(s, e)
        fc = f_flat[:, ch]
        rc, uc, vc, wc = rho_f[ch], ux_f[ch], uy_f[ch], uz_f[ch]
        sc = snu_f[ch]

        usq = uc*uc + vc*vc + wc*wc
        cuc = (c[:,0:1]*uc.unsqueeze(0) + c[:,1:2]*vc.unsqueeze(0) + c[:,2:3]*wc.unsqueeze(0))
        feqc = w.unsqueeze(1) * rc.unsqueeze(0) * (1.0 + 3.0*cuc + 4.5*cuc*cuc - 1.5*usq.unsqueeze(0))

        m = M @ fc; meq = M @ feqc; dm = m - meq; del meq
        ms = m - s_fixed.unsqueeze(1) * dm
        for k in (9,10,11,12,13): ms[k] = m[k] - sc * dm[k]
        del m, dm
        result[:, ch] = M_inv @ ms
        del ms, feqc, fc, cuc

    out = result.reshape(19, nz, ny, nx)
    del result, f_flat
    return out


def wallfn(f, solid, nu, y_val=0.5):
    from tensorlbm.d3q19 import macroscopic3d, C as C19
    dev = f.device; c = C19.to(dev).float()
    cx, cy, cz = c[:,0].view(19,1,1,1), c[:,1].view(19,1,1,1), c[:,2].view(19,1,1,1)
    fluid = ~solid; near = torch.zeros_like(solid)
    for ax, sg in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near |= torch.roll(solid, sg, dims=ax) & fluid
    rho, ux, uy, uz = macroscopic3d(f)
    um = torch.sqrt(ux*ux+uy*uy+uz*uz).clamp(min=1e-12)
    ut = torch.sqrt(nu*um/y_val).clamp(min=1e-12)
    yp = y_val*ut/nu; turb = (yp>11.6) & near
    if turb.any():
        uu = ut[turb].clone(); vm = um[turb]
        for _ in range(8):
            ly = torch.log(y_val*uu/nu)
            fv = uu*(ly/KAPPA+B_CONST)-vm; fp = (ly/KAPPA+B_CONST)+1.0/KAPPA
            uu = (uu-fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu
    tw = ut*ut; ium = 1.0/um
    coef = -(tw/y_val)*near.to(f.dtype)
    fx, fy, fz = coef*(ux*ium), coef*(uy*ium), coef*(uz*ium)
    w19 = torch.tensor([1/3]+[1/18]*6+[1/36]*12, dtype=f.dtype, device=dev).view(19,1,1,1)
    cs2 = 1.0/3.0; cur = cx*ux+cy*uy+cz*uz
    f = f + w19*(1.0+cur/cs2)*(cx*fx+cy*fy+cz*fz)/cs2
    df = (tw*(ux*ium)*near.to(f.dtype)).sum().item()
    p = (rho-1.0)/3.0
    sp, sm = torch.roll(solid,1,dims=2), torch.roll(solid,-1,dims=2)
    dp = (p*(sp.to(f.dtype)-sm.to(f.dtype))*fluid.to(f.dtype)).sum().item()
    return f, df, dp


def log(msg):
    """Log to both stdout and logfile."""
    print(msg, flush=True)
    if hasattr(log, 'fp'):
        log.fp.write(msg + '\n')
        log.fp.flush()


def main():
    did = int(sys.argv[1]); nx = int(sys.argv[2]); ny = int(sys.argv[3]); nz = int(sys.argv[4])
    hl = float(sys.argv[5]); n_steps = int(sys.argv[6])

    # Set up log file
    logfile = Path(f"/tmp/domain_sensitivity/worker_{did:02d}.log")
    logfile.parent.mkdir(exist_ok=True)
    log.fp = open(str(logfile), 'w')

    u_in, re, cs = 0.06, 2e6, 0.05
    nu = u_in * hl / re; tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)
    torch.sdaa.empty_cache()

    blockage = (hl / nx) ** 3 * 100
    tag = f"[{did}] {nx}³_h{int(hl)}_b{blockage:.0f}pct"
    log(f"{tag} tau={tau:.6f} steps={n_steps}")

    from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
    from tensorlbm.suboff_resistance import _voxel_wetted_area
    from tensorlbm.boundaries3d import far_field_bc_3d
    from tensorlbm.solver3d import correct_mass3d, stream3d

    cx_g, cy_g, cz_g = nx * 0.35, ny / 2.0, nz / 2.0
    t0 = time.time()
    with torch.no_grad():
        solid, _ = build_suboff_mask(
            hull_type=SuboffHullType.BARE_HULL, nx=nx, ny=ny, nz=nz,
            cx=cx_g, cy=cy_g, cz=cz_g, length=hl, device=device)
    S = _voxel_wetted_area(solid, 1.0)
    dpS = 0.5 * 1.0 * u_in ** 2 * S
    log(f"{tag} mask={time.time()-t0:.1f}s S={S:.1f} dpS={dpS:.6f}")

    with torch.no_grad():
        rho0 = torch.ones(nz, ny, nx, device=device)
        ux0 = torch.full((nz, ny, nx), u_in, device=device)
        ux0[solid] = 0
        f = equilibrium3d_efficient(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device)
        del rho0, ux0
    torch.sdaa.empty_cache()
    log(f"{tag} init done t={time.time()-t0:.1f}s")

    im = float(nx * ny * nz)
    warmup = n_steps // 3
    fric, pres = [], []
    t_run = time.time()
    last_log = t_run

    for step in range(1, n_steps + 1):
        with torch.no_grad():
            f = collide_mrt_smag_chunked(f, tau=tau, C_s=cs, chunk=500_000)
            f = stream3d(f)
            f, df, dp = wallfn(f, solid, nu)
            f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            with torch.no_grad():
                f = correct_mass3d(f, im)
        if step > warmup and math.isfinite(df):
            fric.append(df); pres.append(dp)

        now = time.time()
        if step % 500 == 0 or step == n_steps or (now - last_log > 30):
            n_avg = max(len(fric), 1)
            cf = sum(fric)/n_avg/dpS if fric else 0
            cp = sum(pres)/n_avg/dpS if pres else 0
            ct = cf + cp
            rate = step / (now - t_run) if now > t_run else 0
            log(f"{tag} step={step:4d} Ct={ct:.5f} Cf={cf:.5f} Cp={cp:.5f} "
                f"n={len(fric)} r={rate:.1f}st/s ({now-t_run:.0f}s)")
            last_log = now

        if not torch.isfinite(f).all():
            log(f"{tag} DIV at {step}"); break

    elapsed = time.time() - t_run
    n_avg = max(len(fric), 1)
    cf = sum(fric)/n_avg/dpS if fric else 0
    cp = sum(pres)/n_avg/dpS if pres else 0
    ct = cf + cp
    err = abs(ct - REF_CT) / REF_CT * 100

    result = {
        "label": f"{nx}³_h{int(hl)}", "grid": f"{nx}x{ny}x{nz}",
        "hull_length": hl, "domain_size": nx,
        "hull_pct_of_domain": round(hl/nx*100, 1),
        "blockage_vol_pct": round(blockage, 2),
        "steps_total": n_steps, "n_averaged": n_avg,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct, "error_pct": err,
        "wetted_area": S, "elapsed_s": elapsed,
        "finite": bool(torch.isfinite(f).all().item()),
    }

    out = Path(f"/tmp/domain_sensitivity/result_{did:02d}.json")
    out.write_text(json.dumps(result))
    log(f"{tag} DONE Ct={ct:.5f} err={err:.1f}% elapsed={elapsed:.0f}s")
    log.fp.close()


if __name__ == "__main__":
    main()

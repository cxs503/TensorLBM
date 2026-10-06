"""IC-E Phase 1 shared library (task #233, prereg frozen 2026-10-06T09:55Z).

Archive edition: output to ./output/ and reference data from
./reference/ beside this file; tensorlbm from the installed package
(needs the IC-E-D1 warmup-bins fix, branch exp/icing-warmup-bins;
A/B-verified bitwise-identical to the pristine Phase-1 tree for
mono-disperse runs). The RG-15 geometry patch is applied to the
module object directly.

RG-15 geometry/condition helpers copied verbatim from
/nfs/wangxi/runs/icing_rg15_20260917/rg15_common.py (IC-C, controller-
verified lineage); build_cfg copied verbatim from runlib.py with the module
and seed/device plumbing localised.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
RUNROOT = _HERE / "output"
REF = _HERE / "reference"
DAT = REF / "RG-15_c0.3_tgap2mm.dat"

_SRC = _HERE.parents[2] / "src"  # <repo>/src beside benchmarks/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import tensorlbm  # noqa: E402
import tensorlbm.aircraft_icing as ai  # noqa: E402

assert tensorlbm.__file__.startswith(str(_SRC) + "/"), (
    f"tree misalignment: tensorlbm resolves to {tensorlbm.__file__},"
    f" expected {_SRC} (run from the repo root; the module needs the"
    " IC-E-D1 warmup-bins fix)",
)

# ---------------------------------------------------------------- conditions
CONDITIONS = {
    "3.1_glaze": dict(t_static_c=-2.0, re_phys=5.7e5),
    "3.2_mixed": dict(t_static_c=-4.0, re_phys=5.8e5),
    "3.3_rime": dict(t_static_c=-10.0, re_phys=6.0e5),
}
COMMON_PHYS = dict(
    chord_phys=0.30,
    v_inf=25.0,
    aoa_deg=4.0,
    lwc=0.44e-3,
    mvd=24.0e-6,
    t_exposure=1200.0,
    p_static=101300.0,
    span=0.58,
    rh=0.975,
)
LATTICE = dict(
    nx=320,
    ny=160,
    chord_frac=0.4,
    u_in=0.05,
    collision="cumulant",
    c_s=0.1,
    droplet_phase="eulerian",
    eulerian_scheme="donor2",
    rime_density_mode="macklin",
    beta_window_mode="clean",
    warmup_steps=6000,
    steps=3000,
)


def air_props(t_c: float, p_pa: float = 101300.0) -> dict:
    t_k = t_c + 273.15
    rho = p_pa / (287.05 * t_k)
    mu = 1.716e-5 * (t_k / 273.15) ** 1.5 * (273.15 + 110.4) / (t_k + 110.4)
    return {"rho_air": rho, "mu_air": mu, "t_k": t_k}


def sound_speed(t_c: float) -> float:
    return math.sqrt(1.4 * 287.05 * (t_c + 273.15))


def parse_dat(path=DAT) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            rows.append((x, y))
    pts = np.asarray(rows, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 10:
        raise ValueError(f"bad dat parse: {pts.shape}")
    return pts


def closed_polygon(pts: np.ndarray) -> np.ndarray:
    return np.vstack([pts, pts[:1]])


def rotate_grid_to_airfoil(x_lu, y_lu, cx, cy, chord, aoa_deg):
    """Grid offsets (from the LE at (cx, cy)) -> airfoil-frame chord units.

    Inverse of the mask transform: a grid point p maps to airfoil-frame
    coordinates (xr, yr) with xr along the chord (TE positive)."""
    aoa = math.radians(aoa_deg)
    cos_a, sin_a = math.cos(aoa), math.sin(aoa)
    dx = np.asarray(x_lu) - cx
    dy = np.asarray(y_lu) - cy
    xr = (dx * cos_a - dy * sin_a) / chord
    yr = (dx * sin_a + dy * cos_a) / chord
    return xr, yr


def le_radius_chord(pts: np.ndarray) -> float:
    r = np.hypot(pts[:, 0], pts[:, 1])
    sel = pts[r < 0.012]
    if len(sel) < 5:
        sel = pts[np.argsort(r)[:20]]
    a_mat = np.column_stack([2 * sel[:, 0], 2 * sel[:, 1], np.ones(len(sel))])
    rhs = sel[:, 0] ** 2 + sel[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    xc, yc = sol[0], sol[1]
    return float(math.hypot(xc, yc))


def make_rg15_mask_fn(pts: np.ndarray):
    import torch
    from matplotlib.path import Path as MplPath

    poly = closed_polygon(pts)
    path = MplPath(poly)

    def rg15_mask_2d(nx, ny, chord, aoa_deg=4.0, cx=None, cy=None, t=0.12, device="cpu"):
        if cx is None:
            cx = nx / 3.0
        if cy is None:
            cy = ny / 2.0
        dev = torch.device(device)
        xx = torch.arange(nx, device=dev, dtype=torch.float32)
        yy = torch.arange(ny, device=dev, dtype=torch.float32)
        gx, gy = torch.meshgrid(xx, yy, indexing="xy")  # (ny, nx)
        dx = gx - cx
        dy = gy - cy
        aoa = math.radians(aoa_deg)
        cos_a, sin_a = math.cos(aoa), math.sin(aoa)
        xr = ((dx * cos_a - dy * sin_a) / chord).cpu().numpy().reshape(-1)
        yr = ((dx * sin_a + dy * cos_a) / chord).cpu().numpy().reshape(-1)
        inside = path.contains_points(np.column_stack([xr, yr]))
        mask = torch.from_numpy(inside.reshape(ny, nx))
        return mask.to(dev)

    return rg15_mask_2d


def install_rg15() -> None:
    """Patch the icing_bm module object in place (no extra sys.path entry)."""
    ai.naca0012_mask_2d = make_rg15_mask_fn(parse_dat())


def le_diameter_phys() -> float:
    return 2.0 * le_radius_chord(parse_dat()) * COMMON_PHYS["chord_phys"]


def build_cfg(case, t_exposure, steps, warmup, **overrides):
    """IcingConfig for an RG-15 case (runlib.build_cfg verbatim, module
    localised)."""
    c = CONDITIONS[case]
    ap = air_props(c["t_static_c"], COMMON_PHYS["p_static"])
    kw = dict(
        nx=LATTICE["nx"],
        ny=LATTICE["ny"],
        chord_frac=LATTICE["chord_frac"],
        u_in=LATTICE["u_in"],
        aoa_deg=COMMON_PHYS["aoa_deg"],
        chord_phys=COMMON_PHYS["chord_phys"],
        v_inf=COMMON_PHYS["v_inf"],
        lwc=COMMON_PHYS["lwc"],
        mvd=COMMON_PHYS["mvd"],
        t_static_c=c["t_static_c"],
        t_exposure=t_exposure,
        p_static=COMMON_PHYS["p_static"],
        rh=COMMON_PHYS["rh"],
        rho_air=ap["rho_air"],
        mu_air=ap["mu_air"],
        collision=LATTICE["collision"],
        c_s=LATTICE["c_s"],
        re_lu_target=c["re_phys"],
        droplet_phase=LATTICE["droplet_phase"],
        eulerian_scheme=LATTICE["eulerian_scheme"],
        rime_density_mode=LATTICE["rime_density_mode"],
        beta_window_mode=LATTICE["beta_window_mode"],
        steps=steps,
        warmup_steps=warmup,
        seed=0,
        device="cuda",
        log_every=max(1, steps // 6),
        thermo_model="messinger",
        le_diameter=le_diameter_phys(),
        naca_t=0.091,
    )
    kw.update(overrides)
    return ai.IcingConfig(**kw)


def curve_stats(curve, chord_phys):
    b = curve["beta"]
    if len(b) == 0:
        return {"beta_max": None}
    im = int(np.argmax(b))
    is_ = int(np.argmin(np.abs(curve["s_over_c"])))
    return {
        "beta_max": float(b[im]),
        "argmax_s_over_c": float(curve["s_over_c"][im]),
        "beta_stag": float(b[is_]),
        "stag_s_over_c": float(curve["s_over_c"][is_]),
        "n_bins": int(len(b)),
    }


DET_ARRAYS = (
    "airfoil",
    "solid",
    "m_w",
    "impact_mass",
    "s_grid",
    "beta_grid",
    "stag",
    "hist_step",
    "hist_t",
    "hist_cd",
    "hist_cl",
    "hist_ice",
    "alpha_e",
    "impact_mass_e",
    "beta_e_grid",
)


def npz_array_hashes(npz_path) -> list:
    """IC-A determinism protocol: per-array sha256 (key/shape/dtype/hash)."""
    z = np.load(npz_path)
    out = []
    for k in DET_ARRAYS:
        a = z[k]
        out.append(
            {
                "key": k,
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "sha256_first8": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:8],
            }
        )
    return out


def beta_md5(curve) -> str:
    return hashlib.md5(
        curve["s_over_c"].tobytes() + curve["beta"].tobytes() + curve["n_cells"].tobytes()
    ).hexdigest()


def guards(r, wall_s, name) -> None:
    """Prereg §4 stop-batch guards: NaN / zero-collection / wall cap."""
    assert wall_s < 3600.0, f"{name}: wall {wall_s:.0f}s exceeds 1 GPU.h cap"
    for k in ("m_w", "impact_mass"):
        a = np.asarray(r[k], dtype=np.float64)
        assert np.isfinite(a).all(), f"{name}: NaN in {k}"
    dep = float(r["eulerian"]["audit"]["deposited"])
    assert dep > 0.0, f"{name}: zero collection (deposited={dep})"
    assert int(np.asarray(r["ice_only"]).sum()) > 0, f"{name}: zero ice cells"


def jdump(obj, rel) -> Path:
    p = RUNROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    print(f"[json] wrote {p}")
    return p

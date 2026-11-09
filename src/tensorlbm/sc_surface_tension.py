"""SC-LBM interface tension made *measurable*: isotropic gradient stencils,
momentum-flux pressure readout, wetting (contact-angle) boundary condition and a
three-radius Laplace self-consistency gate.

Node P143W2-N01 (TensorLBM).  Companion driver:
``benchmarks/sc_surface_tension/run.py``; artifacts ``artifacts/sc_surface_tension/``.

Why this module exists
----------------------
The Shan-Chen (pseudopotential) lattice-Boltzmann model used to be verifiable
only through a *closed-form* pressure closure

    p_eos(rho) = rho cs^2 + (G_eff cs^2 / 2) psi^2 ,                    (1)

i.e. the local-equilibrium mechanical pressure.  Reading pressure that way is
circular -- (1) *is* the model assumption -- and it needs recalibration whenever
the force discretisation or the forcing scheme changes.  Reading the pressure
directly from the simulated momentum flux is the model-free alternative, but in
its raw form

    p_kin = [ sum_i |c_i|^2 f_i - rho |u|^2 ] / D                       (2)

it returns rho cs^2 and therefore misses the whole interaction pressure (for the
verified SC94 state below p_kin = 0.65 while the true pressure is 0.037).  That
missing interaction pressure is the mechanism that left the interface tension
unmeasurable (WITHHELD).  The readout implemented here is

    p_flux = (2) + G_eff cs^2 psi^2 / 2 ,                               (3)

i.e. the kinetic stress is taken from the *simulated* distributions (so it
carries every non-equilibrium / spurious-current contribution of the lattice)
while only the interaction part comes from the lattice pseudopotential -- never
from an equilibrium closure of the density.

Surface tension in the pseudopotential model is a lattice truncation effect
--------------------------------------------------------------------------
A shell-weighted sampling of psi,

    G_alpha psi = (1/a1) sum_d w_d d_alpha psi(x+d)
                = d_alpha psi + kappa3 d_alpha laplacian(psi) + O(d^5), (4)

with ``a1 = sum_d w_d d_x^2`` and ``kappa3 = a3/a1``,
``a3 = (1/6) sum_d w_d d_x^4`` (the fourth lattice moment being isotropic,
``sum_d d_x^4 = 3 sum_d d_x^2 d_y^2``).  Since ``F = -G_eff psi G psi``, the
O(d^3) term of (4) is a pure-divergence (Korteweg) stress

    P^K_ab = K [ psi d_a d_b psi - d_a psi d_b psi ]
             + K [ d_a psi d_b psi - (1/2)|grad psi|^2 delta_ab ],  K = G_eff kappa3 ,
                                                                        (5)

whose normal-tangential difference integrates to

    sigma_K = |K| Integral (d_n psi)^2 dn
            = |G_eff| kappa3 Integral (d_n psi)^2 dn .                  (6)

Equation (6) is the *independent* surface-tension anchor of the Laplace gate: it
contains no radius, so comparing ``Delta p * R`` with ``sigma_K`` tests the
curvature relation itself.  It also predicts that ``sigma`` scales with
``kappa3`` -- a stencil property, not a material property -- and that a
spectrally exact gradient (``kappa3 = 0``, ``mode="spectral"``) keeps phase
separation but removes the capillary stress entirely: that is the ``kappa -> 0``
negative control.

Two consequences the caller must know
-------------------------------------
* "Higher-order isotropy" and "surface tension" pull in opposite directions in
  this model: completing the stencil shells changes ``kappa3`` (hence ``sigma``)
  and removes the rank>=6 anisotropic truncation (hence the directional artefact
  of ``sigma``).  Both effects are measured here, not assumed.
* The 2-D convention is a cylindrical drop, ``Delta p * R = sigma``; the 3-D
  sphere convention ``Delta p * R = 2 sigma`` is the same number halved.  Gates
  report both.

Wetting boundary condition
--------------------------
Solid cells carry a pseudopotential ``wall_psi`` and the fluid-solid interaction
is the same Shan-Chen force (library convention, see
:func:`tensorlbm.multiphase._sc_neighbor_weighted_sum`).  Young's relation is
imposed as

    cos(theta) = (psi_w - psi_g) / (psi_l - psi_g) ,                    (7)

inverted by :func:`wall_psi_from_contact_angle`.  The *measured* angle comes
from a least-squares circle fit of the mid-density contour of the sessile drop,
``cos(theta) = z_c / R_fit`` (``z_c`` = fitted centre height above the wall),
which needs no assumption about the drop volume.

References
----------
Shan & Chen (1993) Phys. Rev. E 47, 1815; Shan & Chen (1994) Phys. Rev. E 49,
2941; Shan (2006) Phys. Rev. E 73, 047701 (spurious currents, pressure tensor);
Sbragaglia et al. (2007) Phys. Rev. E 75, 026702 (surface tension of the
pseudopotential model); Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308
(forcing); Kumar (2004) J. Comput. Phys. 201, 109 (isotropic finite
differences); Lee & Fischer (2006) Phys. Rev. E 74, 046709 and Lee (2010)
J. Comput. Phys. 229, 7212 (high-order isotropic gradient); Benzi et al. (2006)
J. Fluid Mech. 548, 257 (mesoscopic drop contact angle in SC-LBM).
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from .d2q9 import C, W, equilibrium, macroscopic
from .multiphase import psi_exp

CS2 = 1.0 / 3.0
G_EFF = -5.0  # physical (standard-convention) SC94 coupling: attractive
TAU = 1.0
#: Discrete coexistence of SC94 (psi = 1 - exp(-rho)) at G_eff = -5, tau = 1
#: (measured; see benchmarks/verified/laplace_droplet).
RHO_L = 1.957
RHO_V = 0.1596
MID_RHO = 0.5 * (RHO_L + RHO_V)

#: complete square-lattice shells up to |d|^2 = 8
_SHELLS: Dict[int, List[tuple[int, int]]] = {}
for _dx in range(-2, 3):
    for _dy in range(-2, 3):
        _k = _dx * _dx + _dy * _dy
        if 0 < _k <= 8:
            _SHELLS.setdefault(_k, []).append((_dx, _dy))


def shell_offsets(shell: int) -> List[tuple[int, int]]:
    """All lattice offsets with ``|d|^2 == shell`` (the complete shell)."""
    return list(_SHELLS[shell])


def shell_moments(shell: int) -> Dict[str, float]:
    """``(n, S2, S4, Sxy)`` of a complete shell; isotropic iff ``S4 == 3 Sxy``."""
    offs = shell_offsets(shell)
    return {
        "n": float(len(offs)),
        "S2": float(sum(d[0] ** 2 for d in offs)),
        "S4": float(sum(d[0] ** 4 for d in offs)),
        "Sxy": float(sum(d[0] ** 2 * d[1] ** 2 for d in offs)),
    }


def build_stencil(name: str = "d2q9", extra: Dict[int, float] | None = None) -> Dict[str, object]:
    """Build an isotropic gradient stencil and report its truncation moments.

    The operator is ``G_alpha psi = (1/a1) sum_d w_d d_alpha psi(x+d)`` with
    ``a1 = sum w d_x^2``, so ``G`` approximates ``d_alpha`` with unit gain at
    leading order; the interaction force is then ``-G_eff psi G psi`` (the
    standard SC94 force).

    ``"d2q9"``
        shells 1 and 2 with the D2Q9 weights ``(1/9, 1/36)``; ``kappa3 = 1/6``,
        the value behind every published ``psi_exp`` SC94 surface tension.
    ``"shell4"``
        shells 1, 2, 4 and 8, the (shell 1, shell 2) weights being solved from
        the two constraints "unit gain" + "fourth moment isotropic"; the outer
        shells cancel the rank-6 lattice anisotropy and shift ``kappa3`` in a
        *predicted* way (default ratio 1.375 vs D2Q9).

    Returns a dict with ``name``, ``offsets``, ``weights``, ``shell_weights``,
    ``a1``, ``a3``, ``kappa3``, ``isotropy_residual`` (relative violation of
    ``S4 = 3 Sxy``; 0 for an isotropic stencil) and ``shells``.
    """
    if name == "d2q9":
        if extra:
            raise ValueError("the 'd2q9' stencil is fixed; use 'shell4' for extra shells")
        w = {1: 1.0 / 9.0, 2: 1.0 / 36.0}
    elif name == "shell4":
        extra = dict(extra) if extra else {4: 1.0 / 288.0, 8: 1.0 / 1152.0}
        if set(extra) - {4, 8}:
            raise ValueError("'shell4' extra weights are restricted to shells 4 and 8")
        m1, m2 = shell_moments(1), shell_moments(2)
        rhs1, rhs2 = 1.0 / 3.0, 0.0
        for k, wk in extra.items():
            m = shell_moments(k)
            rhs1 -= wk * m["S2"]
            rhs2 -= wk * (m["S4"] - 3.0 * m["Sxy"])
        a11, a12 = m1["S2"], m2["S2"]
        a21, a22 = m1["S4"] - 3.0 * m1["Sxy"], m2["S4"] - 3.0 * m2["Sxy"]
        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-14:
            raise ValueError("degenerate shell system")
        w1 = (rhs1 * a22 - a12 * rhs2) / det
        w2 = (a11 * rhs2 - rhs1 * a21) / det
        w = {1: w1, 2: w2, **extra}
    elif name == "iso4":
        # kappa3-free isotropic stencil: the 4th-order axis difference carried to
        # 2-D with the (2,0) shell.  a1 = 1/3, a3 = 0 exactly, so by eq. (6) it
        # carries NO truncation-induced surface tension at leading order -- the
        # local implementation of the "sigma -> 0" control.
        w = {1: 2.0 / 9.0, 4: -1.0 / 72.0}
    else:
        raise ValueError(f"unknown stencil {name!r}; expected 'd2q9', 'shell4' or 'iso4'")

    offsets: List[tuple[int, int]] = []
    weights: List[float] = []
    for k, wk in sorted(w.items()):
        for d in shell_offsets(k):
            offsets.append(d)
            weights.append(float(wk))
    a1 = sum(wk * shell_moments(k)["S2"] for k, wk in w.items())
    a3 = sum(wk * shell_moments(k)["S4"] for k, wk in w.items()) / 6.0
    s4 = sum(wk * shell_moments(k)["S4"] for k, wk in w.items())
    sxy = sum(wk * shell_moments(k)["Sxy"] for k, wk in w.items())
    iso = abs(s4 - 3.0 * sxy) / abs(s4) if abs(s4) > 1e-14 else 0.0
    return {
        "name": name,
        "offsets": offsets,
        "weights": weights,
        "shell_weights": {int(k): float(v) for k, v in w.items()},
        "a1": float(a1),
        "a3": float(a3),
        "kappa3": float(a3 / a1),
        "isotropy_residual": float(iso),
        "shells": sorted(int(k) for k in w),
    }


# ---------------------------------------------------------------------------
# Gradient operators
# ---------------------------------------------------------------------------


def _shifted(field: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """``field(x + d)`` under periodic wrap (axis 0 = y, axis 1 = x)."""
    return torch.roll(field, shifts=(-dy, -dx), dims=(-2, -1))


def gradient_stencil(psi: torch.Tensor, stencil: Dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient of a scalar field with an isotropic lattice stencil (eq. 4)."""
    gx = torch.zeros_like(psi)
    gy = torch.zeros_like(psi)
    a1 = float(stencil["a1"])
    for (dx, dy), wk in zip(stencil["offsets"], stencil["weights"]):
        sh = _shifted(psi, int(dx), int(dy))
        if dx:
            gx = gx + (wk * dx) * sh
        if dy:
            gy = gy + (wk * dy) * sh
    return gx / a1, gy / a1


def gradient_spectral(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fourier (spectrally exact) gradient of a periodic field.

    Rotationally exact -- ``kappa3 = 0`` in eq. (4) -- hence the isotropy limit
    and, by eq. (6), a *zero* truncation-induced surface tension.  Used as the
    ``sigma -> 0`` negative control.
    """
    ny, nx = field.shape[-2], field.shape[-1]
    ky = 2.0 * math.pi * torch.fft.fftfreq(ny, d=1.0, device=field.device)
    kx = 2.0 * math.pi * torch.fft.fftfreq(nx, d=1.0, device=field.device)
    fhat = torch.fft.fft2(field)
    gx = torch.fft.ifft2(1j * kx.view(1, nx) * fhat).real
    gy = torch.fft.ifft2(1j * ky.view(ny, 1) * fhat).real
    return gx, gy


def psi_masked(
    rho: torch.Tensor,
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
) -> torch.Tensor:
    """Pseudopotential field with the wetting potential imposed on solid cells."""
    psi = psi_exp(rho)
    if solid_mask is not None:
        psi = psi.masked_fill(solid_mask, 0.0 if wall_psi is None else float(wall_psi))
    return psi


def sc_force(
    rho: torch.Tensor,
    G_eff: float = G_EFF,
    stencil: Dict[str, object] | None = None,
    mode: str = "stencil",
    solid_mask: torch.Tensor | None = None,
    wall_psi: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shan-Chen interaction force ``F = -G_eff psi grad(psi)``.

    ``G_eff < 0`` is attraction, consistent with
    ``p_mech = rho cs^2 + G_eff cs^2 psi^2 / 2`` (identical to the library's
    ``G = +5`` sign-flipped argument, see ``multiphase`` CONVENTION NOTE).
    ``mode="spectral"`` swaps in the Fourier gradient (kappa3 = 0 control).
    """
    stencil = stencil if stencil is not None else build_stencil("d2q9")
    psi = psi_masked(rho, solid_mask, wall_psi)
    if mode == "spectral":
        gx, gy = gradient_spectral(psi)
    elif mode == "stencil":
        gx, gy = gradient_stencil(psi, stencil)
    else:
        raise ValueError(f"unknown gradient mode {mode!r}; expected 'stencil' or 'spectral'")
    # cs^2 weight: the lattice force is F = -G_eff cs^2 psi grad(psi), which is
    # exactly the library's -G_lib psi sum_i w_i c_i psi(x-c_i) with G_lib = -G_eff.
    coeff = -G_eff * CS2
    return coeff * psi * gx, coeff * psi * gy


def laplacian_stencil(field: torch.Tensor, stencil: Dict[str, object]) -> torch.Tensor:
    """Isotropic Laplacian ``sum_d w_d [psi(x+d) - psi(x)] / a1`` (diagnostics)."""
    out = torch.zeros_like(field)
    a1 = float(stencil["a1"])
    for (dx, dy), wk in zip(stencil["offsets"], stencil["weights"]):
        out = out + wk * (_shifted(field, int(dx), int(dy)) - field)
    return out / a1


# ---------------------------------------------------------------------------
# Collision / streaming with an explicit force field
# ---------------------------------------------------------------------------


def collide_guo(
    f: torch.Tensor,
    Fx: torch.Tensor,
    Fy: torch.Tensor,
    tau: float = TAU,
) -> torch.Tensor:
    """Guo (2002) forcing: BGK collision + mass-conserving source term.

    ``u_phys = u + F/(2 rho)``; ``S_i = w_i (1 - 1/(2 tau)) [(c_i - u).F/cs^2 +
    (c_i.u)(c_i.F)/cs^4]`` with ``sum_i S_i = 0``.  Same formula as
    ``multiphase.collide_sc_single_component(forcing="guo")`` but with an
    externally supplied force field (needed for the alternative stencils).
    """
    rho, ux, uy = macroscopic(f)
    rho_s = torch.clamp(rho, min=1e-12)
    uxp = ux + 0.5 * Fx / rho_s
    uyp = uy + 0.5 * Fy / rho_s
    feq = equilibrium(rho, uxp, uyp)
    c = C.to(f.device).float()
    w = torch.tensor(_D2Q9_W, device=f.device, dtype=f.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    w3 = w.view(9, 1, 1)
    cf = cx * Fx.unsqueeze(0) + cy * Fy.unsqueeze(0)
    cu = cx * uxp.unsqueeze(0) + cy * uyp.unsqueeze(0)
    uf = uxp * Fx + uyp * Fy
    src = w3 * (1.0 - 1.0 / (2.0 * tau)) * ((cf - uf.unsqueeze(0)) / CS2 + cu * cf / CS2 ** 2)
    return f - (f - feq) / tau + src


_D2Q9_W = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
           1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)
_OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)


def stream_bounce_back(f_post: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
    """Streaming with half-way bounce-back at fluid cells facing a solid.

    For each direction ``i``: ``f_i(x) = f_post,i(x - c_i)`` unless ``x - c_i``
    is solid, in which case ``f_i(x) = f_post,opp(x)`` (the wall sits half-way
    between the two nodes).  Solid cells keep their post-collision state so the
    wetting force can still be evaluated there; they are never streamed.
    """
    out = torch.empty_like(f_post)
    for i in range(9):
        c = _C_INT[i]
        src_solid = _shifted(solid, -int(c[0]), -int(c[1]))
        streamed = _shifted(f_post[i], -int(c[0]), -int(c[1]))
        out[i] = torch.where(src_solid, f_post[_OPP[i]], streamed)
    if solid.any():
        out = torch.where(solid.unsqueeze(0), f_post, out)
    return out


_C_INT = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))


def step_sc(
    f: torch.Tensor,
    G_eff: float = G_EFF,
    tau: float = TAU,
    stencil: Dict[str, object] | None = None,
    mode: str = "stencil",
    solid: torch.Tensor | None = None,
    wall_psi: float | None = None,
    bounce_back: bool = False,
) -> torch.Tensor:
    """One full SC-LBM step (Guo forcing) with optional wetting walls."""
    rho = f.sum(dim=0)
    Fx, Fy = sc_force(rho, G_eff, stencil, mode, solid, wall_psi)
    f_post = collide_guo(f, Fx, Fy, tau)
    if solid is not None:
        f_post = torch.where(solid.unsqueeze(0), f, f_post)
    if bounce_back and solid is not None:
        return stream_bounce_back(f_post, solid)
    # periodic streaming (gather convention of solver.stream)
    ny, nx = f.shape[1], f.shape[2]
    idx = _stream_index(ny, nx, f.device, f.dtype)
    return f_post[idx[0], idx[1], idx[2]]


_STREAM_CACHE: Dict[tuple, tuple] = {}


def _stream_index(ny: int, nx: int, device: torch.device, dtype: torch.dtype):
    key = (ny, nx, device.type, device.index)
    if key not in _STREAM_CACHE:
        cy = torch.tensor([c[1] for c in _C_INT], device=device)
        cx = torch.tensor([c[0] for c in _C_INT], device=device)
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - cy.unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - cx.unsqueeze(1)) % nx
        q_idx = torch.arange(9, device=device).view(9, 1, 1).expand(9, ny, nx)
        _STREAM_CACHE[key] = (q_idx, y_src.unsqueeze(2).expand(9, ny, nx),
                              x_src.unsqueeze(1).expand(9, ny, nx))
    return _STREAM_CACHE[key]


# ---------------------------------------------------------------------------
# Pressure readouts
# ---------------------------------------------------------------------------


def pressure_kinetic(f: torch.Tensor) -> torch.Tensor:
    """Raw momentum-flux pressure (eq. 2): kinetic stress only."""
    rho, ux, uy = macroscopic(f)
    c = C.to(f.device).float()
    cc = (c[:, 0] ** 2 + c[:, 1] ** 2).view(9, 1, 1)
    tr = (cc * f).sum(dim=0)
    return 0.5 * (tr - rho * (ux ** 2 + uy ** 2))


def pressure_eos(rho: torch.Tensor, G_eff: float = G_EFF) -> torch.Tensor:
    """Local-equilibrium closure (eq. 1) -- the historical baseline."""
    psi = psi_exp(rho)
    return rho * CS2 + 0.5 * G_eff * CS2 * psi * psi


def pressure_flux(f: torch.Tensor, G_eff: float = G_EFF) -> torch.Tensor:
    """Momentum-flux pressure (eq. 3): kinetic trace from ``f`` + interaction."""
    psi = psi_exp(f.sum(dim=0))
    return pressure_kinetic(f) + 0.5 * G_eff * CS2 * psi * psi


def pressure_tensor(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kinetic part of the pressure tensor ``Lambda_ab = sum_i c_ia c_ib f_i - rho u_a u_b``.

    Returns ``(Lxx, Lyy, Lxy)``.  ``(Lxx + Lyy)/2`` is :func:`pressure_kinetic`;
    ``(Lxx - Lyy)/2`` and ``Lxy`` are the (viscous + non-equilibrium) shear
    stresses, and the spurious-current level is visible in them.
    """
    rho, ux, uy = macroscopic(f)
    c = C.to(f.device).float()
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    lxx = (cx * cx * f).sum(dim=0) - rho * ux * ux
    lyy = (cy * cy * f).sum(dim=0) - rho * uy * uy
    lxy = (cx * cy * f).sum(dim=0) - rho * ux * uy
    return lxx, lyy, lxy


def korteweg_pn_minus_pt(
    rho: torch.Tensor,
    stencil: Dict[str, object] | None = None,
    G_eff: float = G_EFF,
) -> torch.Tensor:
    """``p_N - p_T`` of the Korteweg stress (eq. 5), normal taken along ``grad psi``.

    With ``K = G_eff cs^2 kappa3`` and ``n = grad psi / |grad psi|``,
    ``p_N - p_T = K [ psi (n.nabla)^2 psi - psi (t.nabla)^2 psi ]``.  For a flat
    interface (``(t.nabla)psi = 0``) this reduces to ``K psi psi''``, whose
    integral is ``-|K| int (psi')^2 dn`` -- eq. (6).
    """
    stencil = stencil if stencil is not None else build_stencil("d2q9")
    psi = psi_exp(rho)
    gx, gy = gradient_stencil(psi, stencil)
    mag = torch.clamp(torch.sqrt(gx * gx + gy * gy), min=1e-12)
    nx, ny_ = gx / mag, gy / mag
    tx, ty = -ny_, nx
    hxx, hxy, hyy = _hessian(psi, stencil)
    pnn = nx * nx * hxx + 2.0 * nx * ny_ * hxy + ny_ * ny_ * hyy
    ptt = tx * tx * hxx + 2.0 * tx * ty * hxy + ty * ty * hyy
    # K = G_eff cs^2 kappa3 makes Integral(p_N - p_T) dn = +sigma (eq. 6)
    K = G_eff * CS2 * float(stencil["kappa3"])
    return K * psi * (pnn - ptt)


def _hessian(field: torch.Tensor, stencil: Dict[str, object] | None = None):
    """Spectral second derivatives ``(Hxx, Hxy, Hyy)`` of a periodic field.

    Exact (no lattice anisotropy), used only by the Korteweg diagnostic.  The
    ``stencil`` argument is accepted for call symmetry and ignored on purpose:
    a stencil Hessian needs a traceless projection to be diagonal, which would
    add noise to a diagnostic without changing any gate.
    """
    ny, nx = field.shape[-2], field.shape[-1]
    ky = 2.0 * math.pi * torch.fft.fftfreq(ny, d=1.0, device=field.device)
    kx = 2.0 * math.pi * torch.fft.fftfreq(nx, d=1.0, device=field.device)
    fhat = torch.fft.fft2(field)
    hxx = torch.fft.ifft2(-(kx.view(1, nx) ** 2) * fhat).real
    hyy = torch.fft.ifft2(-(ky.view(ny, 1) ** 2) * fhat).real
    hxy = torch.fft.ifft2(-(ky.view(ny, 1) * kx.view(1, nx)) * fhat).real
    return hxx, hxy, hyy


# ---------------------------------------------------------------------------
# Geometry / measurement helpers
# ---------------------------------------------------------------------------


def droplet_rho(L: int, R: float, rho_l: float = RHO_L, rho_v: float = RHO_V,
                width: float = 2.0, device: torch.device | str = "cpu") -> torch.Tensor:
    """Diffuse circular droplet (liquid inside) on an ``L x L`` periodic box."""
    y = torch.arange(L, dtype=torch.float32, device=device).view(L, 1)
    x = torch.arange(L, dtype=torch.float32, device=device).view(1, L)
    cx = 0.5 * (L - 1)
    r = torch.sqrt((x - cx) ** 2 + (y - cx) ** 2)
    phi = torch.clamp((R - r) / width, -1.0, 1.0)
    w = 0.5 * (1.0 + phi)
    return rho_v + (rho_l - rho_v) * w


def radius_field(L: int, device: torch.device | str = "cpu") -> torch.Tensor:
    y = torch.arange(L, dtype=torch.float32, device=device).view(L, 1)
    x = torch.arange(L, dtype=torch.float32, device=device).view(1, L)
    cx = 0.5 * (L - 1)
    return torch.sqrt((x - cx) ** 2 + (y - cx) ** 2)


def equivalent_radius(rho: torch.Tensor, mid: float = MID_RHO) -> float:
    """``R_eq`` from the liquid area: ``pi R^2 = sum(rho > mid)``."""
    return math.sqrt(float((rho > mid).sum()) / math.pi)


def azimuthal_bins(L: int, n_bins: int = 36, device: torch.device | str = "cpu"):
    """Bin index map ``theta`` (0..n_bins-1) and radial distance on an ``L x L`` box."""
    r = radius_field(L, device)
    y = torch.arange(L, dtype=torch.float32, device=device).view(L, 1)
    x = torch.arange(L, dtype=torch.float32, device=device).view(1, L)
    cx = 0.5 * (L - 1)
    th = torch.atan2(y - cx, x - cx) % (2.0 * math.pi)
    bins = torch.clamp((th / (2.0 * math.pi) * n_bins).long(), 0, n_bins - 1)
    return bins, r


def _band_mean(field: torch.Tensor, bins: torch.Tensor, r: torch.Tensor,
               r_lo: float, r_hi: float, n_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-wedge mean of ``field`` in the radial band ``[r_lo, r_hi]`` + sample counts."""
    mask = (r >= r_lo) & (r <= r_hi)
    out = torch.zeros(n_bins, device=field.device, dtype=field.dtype)
    cnt = torch.zeros(n_bins, device=field.device, dtype=field.dtype)
    out.index_add_(0, bins[mask], field[mask])
    cnt.index_add_(0, bins[mask], torch.ones_like(field[mask]))
    return out / torch.clamp(cnt, min=1.0), cnt


def angular_pressure_jump(
    p: torch.Tensor,
    R: float,
    n_bins: int = 36,
    inner: tuple[float, float] = (0.4, 0.6),
    outer: tuple[float, float] = (1.4, 1.6),
) -> Dict[str, object]:
    """``Delta p(theta)`` in ``n_bins`` azimuthal wedges.

    The inner/outer bands are bulk on purpose: the Korteweg terms of eq. (5) are
    localised on the interface, so bulk-band differencing isolates the Laplace
    jump.  The spread of ``Delta p(theta)`` is the *directional artefact* of the
    interface tension.
    """
    L = p.shape[-1]
    bins, r = azimuthal_bins(L, n_bins, p.device)
    p_in, c_in = _band_mean(p, bins, r, inner[0] * R, inner[1] * R, n_bins)
    p_out, c_out = _band_mean(p, bins, r, outer[0] * R, outer[1] * R, n_bins)
    used = (c_in > 0) & (c_out > 0)
    dp = (p_in - p_out)[used]
    vals = dp.tolist()
    mean = float(dp.mean()) if dp.numel() else float("nan")
    dev = float((dp - mean).abs().max() / abs(mean)) if dp.numel() and mean else float("nan")
    return {"dp_theta": vals, "dp_mean": mean,
            "dp_min": float(dp.min()) if dp.numel() else float("nan"),
            "dp_max": float(dp.max()) if dp.numel() else float("nan"),
            "spread_frac": (float(dp.max() - dp.min()) / abs(mean)) if dp.numel() and mean else float("nan"),
            "max_dev_frac": dev, "n_wedges_used": int(used.sum())}


def radial_profile(field: torch.Tensor, L: int, n_bins: int = 200) -> Dict[str, object]:
    """Azimuthally averaged radial profile; empty bins are filled from neighbours.

    Empty bins occur near ``r = 0`` (the cell centres of a square lattice do not
    hit every radius) and would otherwise inject spurious zeros into the
    derivative that feeds the Korteweg integral.
    """
    r = radius_field(L, field.device)
    rmax = 0.5 * L
    idx = torch.clamp(((r / rmax) * n_bins).long(), 0, n_bins - 1).reshape(-1)
    flat = field.reshape(-1)
    sums = torch.zeros(n_bins, device=field.device, dtype=field.dtype).index_add_(0, idx, flat)
    cnts = torch.zeros(n_bins, device=field.device, dtype=field.dtype).index_add_(
        0, idx, torch.ones_like(flat))
    prof = sums / torch.clamp(cnts, min=1.0)
    have = cnts > 0
    if bool(have.any()):
        first = int(torch.nonzero(have)[0])
        last = int(torch.nonzero(have)[-1])
        prof = prof.clone()
        prof[:first] = prof[first]
        prof[last + 1:] = prof[last]
        bad = ~have.clone()
        bad[:first] = False
        bad[last + 1:] = False
        good_idx = torch.nonzero(have).squeeze(1)
        if bad.any():
            targets = torch.nonzero(bad).squeeze(1)
            pos = torch.searchsorted(good_idx, targets)
            prof[targets] = prof[good_idx[torch.clamp(pos, max=len(good_idx) - 1)]]
    centres = (torch.arange(n_bins, device=field.device, dtype=field.dtype) + 0.5) * (rmax / n_bins)
    return {"r": centres.tolist(), "mean": prof.tolist(), "counts": cnts.tolist(),
            "n_bins": n_bins, "rmax": rmax}


def sigma_korteweg(rho: torch.Tensor, L: int, stencil: Dict[str, object] | None = None,
                   G_eff: float = G_EFF, n_bins: int = 200,
                   r_max_frac: float = 0.45) -> Dict[str, float]:
    """Independent surface tension from the Korteweg integral, eq. (6).

    ``sigma_K = |G_eff kappa3| * Integral (d psi / dr)^2 dr`` on the azimuthal
    mean radial profile.  No radius enters the definition, so comparing it with
    ``Delta p * R`` tests the Laplace curvature relation, not the closure.
    """
    stencil = stencil if stencil is not None else build_stencil("d2q9")
    prof = radial_profile(psi_exp(rho), L, n_bins)
    rr = prof["r"]
    ps = prof["mean"]
    integ = 0.0
    for i in range(1, len(rr)):
        if rr[i] > r_max_frac * L:
            break
        dpsi = ps[i] - ps[i - 1]
        dr = rr[i] - rr[i - 1]
        integ += (dpsi / dr) ** 2 * dr
    K = abs(G_eff * CS2 * float(stencil["kappa3"]))
    return {"sigma_K": K * integ, "integral_dpsi2": integ,
            "korteweg_coeff": K, "kappa3": float(stencil["kappa3"])}


def _circle_fit(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    """Algebraic (Kasa) circle fit; returns ``(xc, yc, R)``."""
    A = torch.stack([x, y, torch.ones_like(x)], dim=1).double()
    b = -(x * x + y * y).double()
    sol = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
    xc = -0.5 * float(sol[0])
    yc = -0.5 * float(sol[1])
    r2 = xc * xc + yc * yc - float(sol[2])
    return xc, yc, math.sqrt(max(r2, 1e-12))


def contact_angle_from_rho(
    rho: torch.Tensor,
    solid: torch.Tensor,
    mid: float = MID_RHO,
    wall_offset: float = 0.5,
    n_angles: int = 180,
    ignore_below: float = 2.0,
    fit_from_height_frac: float = 0.25,
) -> Dict[str, float]:
    """Measure the contact angle of a sessile drop.

    Returns:

    ``theta_deg``
        two-parameter circle fit of the mid-density contour with the centre on
        the symmetry axis: Gauss-Newton on
        ``sqrt((x-xc)^2 + (y-yc)^2) - R``; then
        ``cos(theta) = (yc - y_wall_eff) / R``.  Only contour points above
        ``max(ignore_below, fit_from_height_frac*h)`` are used, so the wetting
        film along the wall (where the mid-level is flat) cannot drag the fit.
    The wall is the topmost solid row below the drop; the no-slip plane lies
    half-way into the first solid row (``wall_offset = 0.5``).
    """
    ny, nx = rho.shape
    fluid = ~solid
    ys = torch.arange(ny, dtype=torch.float32, device=rho.device).view(ny, 1).expand(ny, nx)
    xs = torch.arange(nx, dtype=torch.float32, device=rho.device).view(1, nx).expand(ny, nx)
    liq = (rho > mid) & fluid
    nan = float("nan")
    out = {"theta_deg": nan, "cos_theta": nan, "R_fit": nan, "z_c": nan,
           "n_contour": 0, "y_wall": nan, "apex": nan, "base_half_width": nan,
           "xc": nan, "status": "no-liquid"}
    if not bool(liq.any()):
        return out
    y_c0 = float(ys[liq].mean())
    solid_rows = ys[solid & (ys < y_c0)]
    y_wall = float(solid_rows.max()) if solid_rows.numel() else -0.5
    w = liq.float()
    xc = float((w * xs).sum() / torch.clamp(w.sum(), min=1.0))
    # --- geometry first guess from the liquid mask -------------------------
    top = torch.where(liq, ys, torch.full_like(ys, -1.0)).max(dim=0).values  # per column
    apex = float(top.max())
    h = apex - (y_wall + wall_offset)
    if h <= 1.0:
        out["status"] = "no-cap"
        return out
    yc = y_wall + wall_offset + h - (0.5 * h + 1e6)  # placeholder, replaced below
    row_band = (ys > y_wall + 1.0) & (ys < y_wall + 1.0 + max(3.0, 0.3 * h)) & liq
    if bool(row_band.any()):
        a_half = float((xs[row_band] - xc).abs().max())
    else:
        a_half = float((xs[liq] - xc).abs().max())
    a_half = max(a_half, 1.0)
    R_guess = (a_half ** 2 + h ** 2) / (2.0 * h)
    yc = (y_wall + wall_offset) + R_guess - h

    # --- contour by radial marching, two re-centring passes -----------------
    pts_x, pts_y = [], []
    for _ in range(3):
        yc = min(max(yc, y_wall + wall_offset), y_wall + wall_offset + apex)
        pts_x, pts_y = [], []
        for k in range(n_angles):
            th = 2.0 * math.pi * k / n_angles
            dx, dy = math.cos(th), math.sin(th)
            prev = float(rho[int(yc), int(xc)])
            for rr in range(1, int(0.9 * max(ny, nx))):
                xi = int(round(xc + dx * rr))
                yi = int(round(yc + dy * rr))
                if xi < 0 or yi < 0 or xi >= nx or yi >= ny or bool(solid[yi, xi]):
                    break
                cur = float(rho[yi, xi])
                if (prev - mid) * (cur - mid) <= 0.0:
                    pts_x.append(xc + dx * rr)
                    pts_y.append(yc + dy * rr)
                    break
                prev = cur
        if len(pts_x) < 12:
            break
        px = torch.tensor(pts_x)
        py = torch.tensor(pts_y)
        keep = py > (y_wall + max(ignore_below, fit_from_height_frac * h))
        if int(keep.sum()) < 8:
            keep = py > (y_wall + ignore_below)
        xc_new = float(px[keep].mean())
        yc_new, r_new = _fit_axis_radius(px[keep], py[keep], xc_new, y_floor=y_wall + wall_offset)
        if (not math.isfinite(r_new)) or r_new < 2.0 or r_new > 4.0 * max(ny, nx):
            break
        xc, yc = xc_new, yc_new
    if len(pts_x) < 12:
        out.update({"n_contour": len(pts_x), "status": "insufficient-contour",
                    "y_wall": y_wall, "apex": apex})
        return out

    px = torch.tensor(pts_x)
    py = torch.tensor(pts_y)
    keep = py > (y_wall + max(ignore_below, fit_from_height_frac * h))
    if int(keep.sum()) < 8:
        keep = py > (y_wall + ignore_below)
    xc_fit = float(px[keep].mean())
    yc_fit, r_fit = _fit_axis_radius(px[keep], py[keep], xc_fit, y_floor=y_wall + wall_offset)
    z_c = yc_fit - (y_wall + wall_offset)
    # sessile-drop convention: the cap has height h = R(1-cos theta), so the
    # circle centre sits at z_c = -R cos(theta) (below the wall for theta < 90)
    cos_t = max(-1.0, min(1.0, -z_c / r_fit)) if r_fit > 0 else nan
    theta = math.degrees(math.acos(cos_t)) if math.isfinite(cos_t) else nan

    status = "ok"
    if r_fit > 0.42 * nx:
        status = "cap-too-flat"
    if (float(py.max()) - (y_wall + wall_offset)) < 4.0:
        status = "no-cap-height"
    return {
        "theta_deg": theta if status == "ok" else float("nan"),
        "theta_raw_deg": theta,
        "cos_theta": cos_t,
        "R_fit": float(r_fit),
        "z_c": float(z_c),
        "n_contour": len(pts_x),
        "y_wall": y_wall,
        "apex": float(py.max()),
        "base_half_width": float((px[py < (y_wall + 4.0 * ignore_below)] - xc_fit).abs().max())
        if int((py < (y_wall + 4.0 * ignore_below)).sum()) >= 4 else float(a_half),
        "xc": xc_fit,
        "status": status,
    }


def contact_angle_from_mask(
    rho: torch.Tensor,
    solid: torch.Tensor,
    mid: float = MID_RHO,
    wall_offset: float = 0.5,
    min_rows: int = 6,
    max_half_width_frac: float = 0.40,
) -> Dict[str, float]:
    """Contact angle from a linear least-squares fit of the cap profile.

    For a circular cap the half width ``w(y)`` obeys ``w^2 + y^2 = A + B y`` with
    ``B = 2 y_c`` and ``R^2 = A + y_c^2``, so a *linear* fit of ``w^2 + y^2``
    against ``y`` determines the circle without iteration -- far more robust than
    a nonlinear contour fit on a film-contaminated interface.

    Rows whose half width exceeds ``max_half_width_frac * nx`` are dropped: those
    are the rows where the wetting precursor film (not the cap) sets the width.
    """
    ny, nx = rho.shape
    fluid = ~solid
    ys = torch.arange(ny, dtype=torch.float32, device=rho.device).view(ny, 1).expand(ny, nx)
    xs = torch.arange(nx, dtype=torch.float32, device=rho.device).view(1, nx).expand(ny, nx)
    liq = (rho > mid) & fluid
    nan = float("nan")
    out = {"theta_deg": nan, "cos_theta": nan, "R_fit": nan, "z_c": nan, "apex": nan,
           "y_wall": nan, "n_rows": 0, "status": "no-liquid", "half_widths": []}
    if not bool(liq.any()):
        return out
    y_c0 = float(ys[liq].mean())
    solid_rows = ys[solid & (ys < y_c0)]
    y_wall = float(solid_rows.max()) if solid_rows.numel() else -0.5
    w = liq.float()
    xc = float((w * xs).sum() / torch.clamp(w.sum(), min=1.0))
    apex = float(ys[liq].max())
    h = apex - (y_wall + wall_offset)
    if h < 3.0:
        out.update({"status": "no-cap-height", "y_wall": y_wall, "apex": apex})
        return out
    rows_y, rows_w = [], []
    for y in range(int(y_wall + wall_offset + 1), int(apex)):
        idx = torch.nonzero(liq[y]).squeeze(1)
        if idx.numel() < 3:
            continue
        half = float((xs[y, idx] - xc).abs().max())
        if half > max_half_width_frac * nx:
            continue
        rows_y.append(float(y))
        rows_w.append(half)
    if len(rows_y) < min_rows:
        out.update({"status": "insufficient-rows", "y_wall": y_wall, "apex": apex,
                    "n_rows": len(rows_y)})
        return out
    yy = torch.tensor(rows_y, dtype=torch.float64)
    ww = torch.tensor(rows_w, dtype=torch.float64)
    v = ww * ww + yy * yy
    A = torch.stack([torch.ones_like(yy), yy], dim=1)
    sol = torch.linalg.lstsq(A, v.reshape(-1, 1)).solution.squeeze(1)
    a0, b0 = float(sol[0]), float(sol[1])
    yc = 0.5 * b0
    r2 = a0 + yc * yc
    if r2 <= 0:
        out.update({"status": "degenerate-fit", "y_wall": y_wall, "apex": apex})
        return out
    R = math.sqrt(r2)
    z_c = yc - (y_wall + wall_offset)
    cos_t = max(-1.0, min(1.0, -z_c / R))
    resid = float(((A @ sol - v) ** 2).mean() ** 0.5)
    return {
        "theta_deg": math.degrees(math.acos(cos_t)),
        "cos_theta": cos_t,
        "R_fit": R,
        "z_c": z_c,
        "apex": apex,
        "y_wall": y_wall,
        "n_rows": len(rows_y),
        "fit_rms": resid,
        "half_widths": rows_w,
        "status": "ok",
    }


def _fit_axis_radius(x: torch.Tensor, y: torch.Tensor, xc: float,
                     y_floor: float | None = None,
                     damping: float = 0.5) -> tuple[float, float]:
    """Fit ``(y_c, R)`` of a circle with the centre on the symmetry axis.

    Damped Gauss-Newton from the analytic cap estimate
    ``R = (a^2+h^2)/(2h)``, ``y_c = y_wall + R - h``.  If the iteration does not
    reduce the RMS residual the analytic estimate is returned, so a noisy or
    film-contaminated contour can never produce a wild radius.
    """
    y_wall_guess = float(y.min())
    a = float((x - xc).abs().max())
    h = float(y.max()) - y_wall_guess
    if h <= 1e-9 or a <= 1e-9:
        return float(y.mean()), float(max(a, h))
    R = (a * a + h * h) / (2.0 * h)
    yc = y_wall_guess + R - h

    def rms(yc_, R_):
        r = torch.sqrt((x - xc) ** 2 + (y - yc_) ** 2)
        return float(((r - R_) ** 2).mean() ** 0.5)

    best = (rms(yc, R), yc, R)
    for _ in range(80):
        r = torch.clamp(torch.sqrt((x - xc) ** 2 + (y - yc) ** 2), min=1e-9)
        e = r - R
        jy = -(y - yc) / r
        jr = -torch.ones_like(r)
        a11 = float((jy * jy).sum())
        a12 = float((jy * jr).sum())
        a22 = float((jr * jr).sum())
        b1 = -float((jy * e).sum())
        b2 = -float((jr * e).sum())
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
            break
        dyc = damping * (b1 * a22 - a12 * b2) / det
        dR = damping * (a11 * b2 - b1 * a12) / det
        yc += dyc
        R += dR
        if R < 0:
            R = -R
        if y_floor is not None and yc < y_floor - 20.0 * abs(R):
            yc = y_floor - 20.0 * abs(R)
        cur = rms(yc, R)
        if cur < best[0]:
            best = (cur, yc, R)
        if abs(dyc) < 1e-9 and abs(dR) < 1e-9:
            break
    return float(best[1]), float(abs(best[2]))


def _fit_circle_axis(x: torch.Tensor, y: torch.Tensor, xc: float) -> tuple[float, float, float]:
    yc, R = _fit_axis_radius(x, y, xc)
    return float(xc), float(yc), float(R)


# ---------------------------------------------------------------------------
# Wetting potential <-> contact angle
# ---------------------------------------------------------------------------


def wall_psi_from_contact_angle(
    theta_deg: float,
    psi_l: float | None = None,
    psi_g: float | None = None,
    rho_l: float = RHO_L,
    rho_v: float = RHO_V,
) -> float:
    """Invert Young's relation (eq. 7) for the wall pseudopotential.

    ``psi_w = psi_g + cos(theta) (psi_l - psi_g)``; ``theta = 0`` gives a fully
    wetting wall (``psi_w = psi_l``), ``theta = 90`` a neutral wall
    (``psi_w = psi_g``) and ``theta = 180`` a completely dry one
    (``psi_w = 2 psi_g - psi_l``).
    """
    pl = psi_exp(torch.tensor(float(rho_l))) if psi_l is None else float(psi_l)
    pg = psi_exp(torch.tensor(float(rho_v))) if psi_g is None else float(psi_g)
    return float(pg + math.cos(math.radians(theta_deg)) * (pl - pg))


def contact_angle_from_wall_psi(
    wall_psi: float,
    rho_l: float = RHO_L,
    rho_v: float = RHO_V,
) -> float:
    """Forward Young relation: ``theta`` implied by a ``wall_psi`` (eq. 7)."""
    pl = float(psi_exp(torch.tensor(float(rho_l))))
    pg = float(psi_exp(torch.tensor(float(rho_v))))
    if abs(pl - pg) < 1e-12:
        return float("nan")
    cos_t = (wall_psi - pg) / (pl - pg)
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


def sessile_rho(
    ny: int,
    nx: int,
    y_wall: float,
    theta_deg: float,
    area: float,
    rho_l: float = RHO_L,
    rho_v: float = RHO_V,
    width: float = 2.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Initial sessile drop already consistent with ``theta`` (fixed 2-D area).

    For a circular cap of radius ``R`` and contact angle ``theta`` the area is
    ``A = R^2 (theta - sin theta cos theta)`` and the centre sits at
    ``z_c = R cos theta`` above the wall, so the initial shape is close to
    equilibrium and only a short relaxation is needed.
    """
    th = math.radians(max(1.0, min(179.0, theta_deg)))
    R = math.sqrt(area / (th - math.sin(th) * math.cos(th)))
    # cap height h = R(1-cos theta)  =>  centre at  h - R = -R cos(theta)
    zc = -R * math.cos(th)
    y = torch.arange(ny, dtype=torch.float32, device=device).view(ny, 1)
    x = torch.arange(nx, dtype=torch.float32, device=device).view(1, nx)
    cx = 0.5 * (nx - 1)
    r = torch.sqrt((x - cx) ** 2 + (y - (y_wall + zc)) ** 2)
    phi = torch.clamp((R - r) / width, -1.0, 1.0)
    w = 0.5 * (1.0 + phi)
    return rho_v + (rho_l - rho_v) * w
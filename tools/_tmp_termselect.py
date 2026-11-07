"""Numerically select the consistent O(a^2) truncation of the potential-flow
free-surface system, by requiring the classical Stokes 2nd-order solution to be
an exact steady solution of the truncated model.

Everything is done with complex Fourier coefficients of e^{i theta}, e^{2i theta}.
theta = k x - omega t; eta_1 = Re[A e^{i theta}], psi_1 = Re[B e^{i theta}].
"""

import numpy as np

G = 9.81


def W(k, h):
    """Linear Dirichlet-Neumann symbol at z=0 (bottom z=-h)."""
    return k * np.tanh(k * h)


def solve_case(kh, a=1.0, variants=None):
    """Return dict variant -> (a2_model / a2_stokes)."""
    h = 1.0
    k = kh / h
    sig = np.tanh(kh)
    om2 = G * k * sig
    om = np.sqrt(om2)
    wk = W(k, h)          # k*tanh(kh)
    w2k = W(2 * k, h)     # 2k*tanh(2kh)

    A = a + 0j                       # eta_1 = a cos(theta)
    # linear: -i om A = wk B  =>  B = -i om A / wk
    B = -1j * om * A / wk

    # ---- 2-theta coefficients of the quadratic terms of the eta equation ----
    dtN_G0 = -0.5 * w2k * wk * A * B      # -G0(eta G0 psi)
    dtN_adv = 1.0 * k**2 * A * B          # -d_x(eta d_x psi) = -eta_x psi_x - eta psi_xx
    eta_lhs = -2j * om                   # multiplies A2
    lin_psi2 = w2k                        # multiplies B2 (G0 psi_2 at 2-theta)

    # ---- 2-theta coefficients of the quadratic terms of the psi equation ----
    psix2 = 0.25 * k**2 * B * B           # -1/2 psi_x^2
    wsq = -0.25 * wk**2 * B * B           # -1/2 w^2   (w = G0 psi)
    wetat = 0.5 * wk * B * (-1j * om * A)  # + w * eta_t
    psi_lhs = -2j * om                    # multiplies B2
    psi_lin = G                          # multiplies A2 (-g eta -> -g A2 = ... sign below)

    a2s = k * a**2 * (3 - sig**2) / (4 * sig**3)   # classical Stokes 2nd harmonic

    out = {}
    for name, (dtN, use_w2, use_weta) in variants.items():
        # eta eq:  -2i om A2 = w2k B2 + dtN
        # psi eq:  -2i om B2 = -g A2 + psix2 + wsq*use_w2 + wetat*use_weta
        rhs_eta_nl = dtN * (wk * A * B) if False else dtN
        M = np.array([[eta_lhs, -lin_psi2],
                      [psi_lin, psi_lhs]], dtype=complex)  # [A2, B2]
        rhs = np.array([rhs_eta_nl,
                        psix2 + wsq * use_w2 + wetat * use_weta], dtype=complex)
        A2, B2 = np.linalg.solve(M, rhs)
        out[name] = (A2 / a2s, A2, B2)
    return out


variants = {
    "DtN_G0 + w2 + weta":   (-0.5 * 0 - 0, True, True),
}
# build variants explicitly (dtN term needs kv values, so parametrise by flag)
def run(kh):
    h = 1.0
    k = kh / h
    sig = np.tanh(kh)
    om = np.sqrt(G * k * sig)
    wk, w2k = W(k, h), W(2 * k, h)
    a = 1.0
    A = a + 0j
    B = -1j * om * A / wk
    psix2 = 0.25 * k**2 * B * B
    wsq = -0.25 * wk**2 * B * B
    wetat = 0.5 * wk * B * (-1j * om * A)
    a2s = k * a**2 * (3 - sig**2) / (4 * sig**3)
    res = {}
    for dtN_name in ("G0", "adv"):
        for use_w2 in (False, True):
            for use_weta in (False, True):
                dtN = (-0.5 * w2k * wk * A * B) if dtN_name == "G0" else (k**2 * A * B)
                M = np.array([[-2j * om, -w2k], [G, -2j * om]], dtype=complex)
                rhs = np.array([dtN,
                                psix2 + wsq * use_w2 + wetat * use_weta],
                               dtype=complex)
                A2, B2 = np.linalg.solve(M, rhs)
                res[(dtN_name, use_w2, use_weta)] = A2 / a2s
    return res


for kh in (0.7, 1.5, 3.0, 10.0):
    r = run(kh)
    print(f"kh={kh:5.2f}")
    for key, val in r.items():
        print(f"   {str(key):28s} A2/Stokes = {val.real:+.4f} {val.imag:+.4f}j")
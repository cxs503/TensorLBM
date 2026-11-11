/* Fused D3Q19-BGK + four-SGS kernel for the LBM-LES verification surface.
 *
 * Why this file exists
 * --------------------
 * The production torch collision operators in
 * ``tensorlbm/turbulence/turbulence.py`` are eager-kernel chains over
 * ``(19, nz, ny, nx)`` tensors: measured on this host a 64^3 step costs
 * ~0.5 s and a 128^3 step ~3.5 s (float64).  A four-model verification
 * campaign (decaying isotropic turbulence at 32/64/128^3, Taylor-Green
 * vortex, laminar-degeneracy controls) is O(1e4) steps per case, i.e. tens of
 * CPU-hours per case on the torch path.  This kernel fuses moments / SGS
 * eddy viscosity / equilibrium / collision / streaming into a handful of
 * OpenMP passes, bringing the same case to O(1) minute.
 *
 * Physics / numerics -- a *bit-level replica* of the production operators
 * (``sgs_lbm_kernel.py`` ships the equivalence tests against the torch path):
 *
 *   model 0  Smagorinsky          tau_eff = 1/2 (tau + sqrt(tau^2 + 18 Cs^2 Q/rho))
 *                                 with Q = Frobenius norm of the non-equilibrium
 *                                 stress  (== collide_smagorinsky_bgk3d)
 *   model 1  Dynamic Smagorinsky  Germano-Lilly, global <L:M>/<M:M> over the
 *                                 homogeneous volume, box test filter width 2,
 *                                 Cs^2 clipped to [clip, 0.1]
 *                                 (== collide_dynamic_smagorinsky_bgk3d)
 *   model 2  WALE                 nu_t = Cw^2 |S^d|^3 / (|S|^2.5 + |S^d|^1.25)
 *                                 (== collide_wale_bgk3d)
 *   model 3  Vreman               nu_t = Cv sqrt(B_beta / A_alpha)
 *                                 (== collide_vreman_bgk3d)
 *
 *   * lattice D3Q19, cs^2 = 1/3, BGK collision, periodic box
 *   * tau_eff clamped to [0.5001, max(1.0, tau + 0.5)] exactly as
 *     ``_TAU_EFF_MIN`` / ``_tau_eff_max`` in the production module
 *   * velocity gradients: bc_mode 0 = exact replica of the production
 *     ``_nonperiodic_derivative`` (centred interior, second-order one-sided
 *     edges) + replicate-padded box filter; bc_mode 1 = periodic centred
 *     differences + wrap-padded filter (physically correct for the periodic
 *     DIT / TGV boxes used by the campaign)
 *
 * Layout: i = (z*n + y)*n + x; populations are SoA, f[q*N + i].
 *
 * Build: gcc -O3 -fopenmp -march=native -shared -fPIC -o sgs_lbm.so sgs_lbm.c
 * (done automatically by ``sgs_lbm_kernel.build_kernel``).
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define Q 19
#define TAU_EFF_MIN 0.5001

/* D3Q19 velocity set, ordering identical to tensorlbm.d3q19.C:
 *   0 (0,0,0)
 *   1 ( 1,0,0)   2 (-1,0,0)   3 (0, 1,0)   4 (0,-1,0)   5 (0,0, 1)   6 (0,0,-1)
 *   7 ( 1, 1,0)  8 (-1,-1,0)  9 (1,-1,0)  10 (-1, 1,0)
 *  11 ( 1,0, 1) 12 (-1,0,-1) 13 (1,0,-1)  14 (-1,0, 1)
 *  15 (0, 1, 1) 16 (0,-1,-1) 17 (0, 1,-1) 18 (0,-1, 1)
 */
static const int CX[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
static const int CY[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
static const int CZ[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
static const double WQ[Q] = {
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

static int g_n = 0;
static long g_N = 0;
static int g_bc_mode = 1;

static double *g_f, *g_fn;
static double *g_rho, *g_ux, *g_uy, *g_uz, *g_te, *g_nut;
static double *g_g[9], *g_gf[9];
static double *g_sab[6], *g_smag, *g_lab[6], *g_mab[6], *g_fu[3], *g_st[6];
static double *g_fbuf[2], *g_prod;
static int *g_nb[Q];
static double g_stats[16];

static void *xalloc(size_t bytes) { return calloc(1, bytes); }

void sgs_lbm_free(void);

int sgs_lbm_init(int n) {
    int q, k;
    long i, N, z, y, x;
    if (n < 4) return 1;
    sgs_lbm_free();
    g_n = n;
    N = (long)n * n * n;
    g_N = N;
    g_f = (double *)xalloc(sizeof(double) * Q * N);
    g_fn = (double *)xalloc(sizeof(double) * Q * N);
    g_rho = (double *)xalloc(sizeof(double) * N);
    g_ux = (double *)xalloc(sizeof(double) * N);
    g_uy = (double *)xalloc(sizeof(double) * N);
    g_uz = (double *)xalloc(sizeof(double) * N);
    g_te = (double *)xalloc(sizeof(double) * N);
    g_nut = (double *)xalloc(sizeof(double) * N);
    for (k = 0; k < 9; k++) {
        g_g[k] = (double *)xalloc(sizeof(double) * N);
        g_gf[k] = (double *)xalloc(sizeof(double) * N);
    }
    for (k = 0; k < 6; k++) {
        g_sab[k] = (double *)xalloc(sizeof(double) * N);
        g_lab[k] = (double *)xalloc(sizeof(double) * N);
        g_mab[k] = (double *)xalloc(sizeof(double) * N);
        g_st[k] = (double *)xalloc(sizeof(double) * N);
    }
    g_smag = (double *)xalloc(sizeof(double) * N);
    for (k = 0; k < 3; k++) g_fu[k] = (double *)xalloc(sizeof(double) * N);
    for (k = 0; k < 2; k++) g_fbuf[k] = (double *)xalloc(sizeof(double) * N);
    g_prod = (double *)xalloc(sizeof(double) * N);
    if (!g_f || !g_fn || !g_rho || !g_ux || !g_uy || !g_uz || !g_te || !g_nut ||
        !g_smag || !g_prod)
        return 2;
    for (k = 0; k < 4; k++) {
        if (!g_sab[k] || !g_lab[k] || !g_st[k]) return 2;
    }
    for (k = 0; k < Q; k++) {
        g_nb[k] = (int *)xalloc(sizeof(int) * N);
        if (!g_nb[k]) return 2;
    }
    for (q = 0; q < Q; q++) {
        int cx = CX[q], cy = CY[q], cz = CZ[q];
        for (z = 0; z < n; z++)
            for (y = 0; y < n; y++)
                for (x = 0; x < n; x++) {
                    long sx = (x - cx + n) % n;
                    long sy = (y - cy + n) % n;
                    long sz = (z - cz + n) % n;
                    g_nb[q][(z * n + y) * n + x] = (int)((sz * n + sy) * n + sx);
                }
    }
    for (q = 0; q < Q; q++)
        for (i = 0; i < N; i++) g_f[q * N + i] = WQ[q];
    return 0;
}

void sgs_lbm_free(void) {
    int q, k;
    free(g_f); g_f = NULL;
    free(g_fn); g_fn = NULL;
    free(g_rho); g_rho = NULL;
    free(g_ux); g_ux = NULL;
    free(g_uy); g_uy = NULL;
    free(g_uz); g_uz = NULL;
    free(g_te); g_te = NULL;
    free(g_nut); g_nut = NULL;
    free(g_smag); g_smag = NULL;
    free(g_prod); g_prod = NULL;
    for (k = 0; k < 9; k++) {
        free(g_g[k]); g_g[k] = NULL;
        free(g_gf[k]); g_gf[k] = NULL;
    }
    for (k = 0; k < 6; k++) {
        free(g_sab[k]); g_sab[k] = NULL;
        free(g_lab[k]); g_lab[k] = NULL;
        free(g_mab[k]); g_mab[k] = NULL;
        free(g_st[k]); g_st[k] = NULL;
    }
    for (k = 0; k < 3; k++) { free(g_fu[k]); g_fu[k] = NULL; }
    for (k = 0; k < 2; k++) { free(g_fbuf[k]); g_fbuf[k] = NULL; }
    for (q = 0; q < Q; q++) { free(g_nb[q]); g_nb[q] = NULL; }
    g_n = 0;
    g_N = 0;
}

int sgs_lbm_n(void) { return g_n; }
long sgs_lbm_cells(void) { return g_N; }

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */
static inline double feq_q(int q, double rho, double ux, double uy, double uz) {
    double cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
    double usq = ux * ux + uy * uy + uz * uz;
    return WQ[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
}

static inline double clamp_te(double te, double tau) {
    double hi = (tau + 0.5 > 1.0) ? (tau + 0.5) : 1.0;
    if (te < TAU_EFF_MIN) te = TAU_EFF_MIN;
    if (te > hi) te = hi;
    return te;
}

/* Moments + SGS inputs in one sweep over f.
 *
 * The non-equilibrium stress six-tuple is accumulated in the same loop that
 * builds rho/u (pure ALU, no extra memory traffic), which removes one full
 * 19-array sweep per step compared with a separate closure pass.  Depending
 * on ``model`` the loop also finishes the closure:
 *
 *   model 0  Smagorinsky   -> tau_eff from the neq-stress norm Q
 *   model 1  dynamic       -> S_ab and |S| stored for the Germano ratio
 *   model 2/3 (WALE/Vreman)-> nothing (they need velocity gradients)
 *   model 4  none          -> tau_eff = tau, nu_t = 0 (DNS arm)
 */
static void moments_sgs(const double *f, int model, double tau, double C) {
    long i, N = g_N;
    int q;
    double cs2 = C * C;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < N; i++) {
        double r = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
        for (q = 0; q < Q; q++) {
            double fq = f[(long)q * N + i];
            r += fq;
            mx += CX[q] * fq;
            my += CY[q] * fq;
            mz += CZ[q] * fq;
        }
        if (r < 1e-12) r = 1e-12;
        mx /= r;
        my /= r;
        mz /= r;
        g_rho[i] = r;
        g_ux[i] = mx;
        g_uy[i] = my;
        g_uz[i] = mz;
        if (model == 2 || model == 3) continue; /* gradients only */
        {
            double pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
            for (q = 0; q < Q; q++) {
                double fn = f[(long)q * N + i] - feq_q(q, r, mx, my, mz);
                pxx += CX[q] * CX[q] * fn;
                pyy += CY[q] * CY[q] * fn;
                pzz += CZ[q] * CZ[q] * fn;
                pxy += CX[q] * CY[q] * fn;
                pxz += CX[q] * CZ[q] * fn;
                pyz += CY[q] * CZ[q] * fn;
            }
            if (model == 0) {
                double rad = pxx * pxx + pyy * pyy + pzz * pzz +
                             2.0 * (pxy * pxy + pxz * pxz + pyz * pyz);
                if (rad < 1e-30) rad = 1e-30;
                double disc = tau * tau + 18.0 * cs2 * sqrt(rad) / r;
                if (disc < 0.0) disc = 0.0;
                double te = clamp_te(0.5 * (tau + sqrt(disc)), tau);
                g_te[i] = te;
                g_nut[i] = (te - tau) / 3.0;
            } else if (model == 1) {
                double den = 2.0 * r * tau * (1.0 / 3.0);
                if (den < 1e-12) den = 1e-12;
                double sxx = -pxx / den, sxy = -pxy / den, sxz = -pxz / den;
                double syy = -pyy / den, syz = -pyz / den, szz = -pzz / den;
                g_mab[0][i] = sxx; g_mab[1][i] = sxy; g_mab[2][i] = sxz;
                g_mab[3][i] = syy; g_mab[4][i] = syz; g_mab[5][i] = szz;
                double m2 = 2.0 * (sxx * sxx + syy * syy + szz * szz +
                                   2.0 * (sxy * sxy + sxz * sxz + syz * syz));
                g_smag[i] = sqrt(m2 > 0.0 ? m2 : 0.0);
            } else { /* model 4: SGS off */
                g_te[i] = tau;
                g_nut[i] = 0.0;
            }
        }
    }
}

/* derivative of the three velocity components along `axis` (0=x,1=y,2=z) in
 * one parallel region.
 * mode 0 = exact replica of the production _nonperiodic_derivative
 *          (centred interior, second-order one-sided edges),
 * mode 1 = periodic centred differences.
 * One flat parallel loop over all cells: per-line loops would fork/join once
 * per lattice line, which on 128 threads costs more than the arithmetic. */
static void grad_axis3(const double *a0, const double *a1, const double *a2,
                       double *o0, double *o1, double *o2, int axis, int mode) {
    int n = g_n;
    long stride = (axis == 0) ? 1 : (axis == 1 ? (long)n : (long)n * (long)n);
    long i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < g_N; i++) {
        long t = (i / stride) % n;
        long ip, im, ip2, im2;
        const double *src[3];
        double *dst[3];
        int k;
        if (mode == 1) {
            long tp = (t + 1 == n) ? 0 : (t + 1);
            long tm = (t == 0) ? (n - 1) : (t - 1);
            ip = i + (tp - t) * stride;
            im = i + (tm - t) * stride;
            src[0] = a0; src[1] = a1; src[2] = a2;
            dst[0] = o0; dst[1] = o1; dst[2] = o2;
            for (k = 0; k < 3; k++) dst[k][i] = 0.5 * (src[k][ip] - src[k][im]);
        } else if (t == 0) {
            ip = i + stride;
            ip2 = i + 2 * stride;
            src[0] = a0; src[1] = a1; src[2] = a2;
            dst[0] = o0; dst[1] = o1; dst[2] = o2;
            for (k = 0; k < 3; k++)
                dst[k][i] = 0.5 * (-3.0 * src[k][i] + 4.0 * src[k][ip] - src[k][ip2]);
        } else if (t == n - 1) {
            im = i - stride;
            im2 = i - 2 * stride;
            src[0] = a0; src[1] = a1; src[2] = a2;
            dst[0] = o0; dst[1] = o1; dst[2] = o2;
            for (k = 0; k < 3; k++)
                dst[k][i] = 0.5 * (3.0 * src[k][i] - 4.0 * src[k][im] + src[k][im2]);
        } else {
            ip = i + stride;
            im = i - stride;
            src[0] = a0; src[1] = a1; src[2] = a2;
            dst[0] = o0; dst[1] = o1; dst[2] = o2;
            for (k = 0; k < 3; k++) dst[k][i] = 0.5 * (src[k][ip] - src[k][im]);
        }
    }
}

/* g_ij = du_i/dx_j for i,j in {x,y,z}, stored as dst[3*i + j] */
static void gradients_of(const double *u0, const double *u1, const double *u2,
                         double *dst[9]) {
    grad_axis3(u0, u1, u2, dst[0], dst[3], dst[6], 0, g_bc_mode);
    grad_axis3(u0, u1, u2, dst[1], dst[4], dst[7], 1, g_bc_mode);
    grad_axis3(u0, u1, u2, dst[2], dst[5], dst[8], 2, g_bc_mode);
}

static void gradients_into(double *dst[9]) { gradients_of(g_ux, g_uy, g_uz, dst); }

/* separable box filter, width 2 (replicate pad in mode 0, wrap in mode 1) */
static void filter2(const double *a, double *out, double *tmp) {
    int n = g_n;
    long i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < g_N; i++) {
        long x = i % n;
        long lo = (g_bc_mode == 1) ? (i - x + ((x - 1 + n) % n)) : (i - (x > 0 ? 1 : 0));
        tmp[i] = 0.5 * (a[lo] + a[i]);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < g_N; i++) {
        long y = (i / n) % n;
        long row = i - y * n;
        long lo = (g_bc_mode == 1) ? (row + ((y - 1 + n) % n) * n)
                                   : (row + (y > 0 ? (y - 1) : 0) * n);
        out[i] = 0.5 * (tmp[lo] + tmp[i]);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < g_N; i++) {
        long z = i / ((long)n * n);
        long plane = i - z * (long)n * n;
        long lo = (g_bc_mode == 1) ? (plane + ((z - 1 + n) % n) * (long)n * n)
                                   : (plane + (z > 0 ? (z - 1) : 0) * (long)n * n);
        tmp[i] = 0.5 * (out[lo] + out[i]);
    }
    memcpy(out, tmp, sizeof(double) * g_N);
}

/* neq-stress strain tensor + magnitude: g_sab[0..5] = xx,xy,xz,yy,yz,zz,
 * g_smag = |S|  (production _strain_from_fneq_3d / _strain_magnitude_3d). */
/* ---- SGS closures ------------------------------------------------- */

static void te_wale(double tau, double C) {
    long i;
    double cw2 = C * C;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < g_N; i++) {
        double g11 = g_g[0][i], g12 = g_g[1][i], g13 = g_g[2][i];
        double g21 = g_g[3][i], g22 = g_g[4][i], g23 = g_g[5][i];
        double g31 = g_g[6][i], g32 = g_g[7][i], g33 = g_g[8][i];
        double q11 = g11 * g11 + g12 * g21 + g13 * g31;
        double q12 = g11 * g12 + g12 * g22 + g13 * g32;
        double q13 = g11 * g13 + g12 * g23 + g13 * g33;
        double q21 = g21 * g11 + g22 * g21 + g23 * g31;
        double q22 = g21 * g12 + g22 * g22 + g23 * g32;
        double q23 = g21 * g13 + g22 * g23 + g23 * g33;
        double q31 = g31 * g11 + g32 * g21 + g33 * g31;
        double q32 = g31 * g12 + g32 * g22 + g33 * g32;
        double q33 = g31 * g13 + g32 * g23 + g33 * g33;
        double tr = q11 + q22 + q33;
        double d11 = q11 - tr / 3.0, d22 = q22 - tr / 3.0, d33 = q33 - tr / 3.0;
        double d12 = 0.5 * (q12 + q21), d13 = 0.5 * (q13 + q31), d23 = 0.5 * (q23 + q32);
        double sd2 = d11 * d11 + d22 * d22 + d33 * d33 +
                     2.0 * (d12 * d12 + d13 * d13 + d23 * d23);
        double S12 = 0.5 * (g12 + g21), S13 = 0.5 * (g13 + g31), S23 = 0.5 * (g23 + g32);
        double S2 = g11 * g11 + g22 * g22 + g33 * g33 +
                    2.0 * (S12 * S12 + S13 * S13 + S23 * S23);
        if (sd2 < 0.0) sd2 = 0.0;
        if (S2 < 0.0) S2 = 0.0;
        double nut = cw2 * pow(sd2, 1.5) / (pow(S2, 2.5) + pow(sd2, 1.25) + 1e-30);
        g_nut[i] = nut;
        g_te[i] = clamp_te(tau + 3.0 * nut, tau);
    }
}

static void te_vreman(double tau, double C) {
    long i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < g_N; i++) {
        double g11 = g_g[0][i], g12 = g_g[1][i], g13 = g_g[2][i];
        double g21 = g_g[3][i], g22 = g_g[4][i], g23 = g_g[5][i];
        double g31 = g_g[6][i], g32 = g_g[7][i], g33 = g_g[8][i];
        double b11 = g11 * g11 + g21 * g21 + g31 * g31;
        double b22 = g12 * g12 + g22 * g22 + g32 * g32;
        double b33 = g13 * g13 + g23 * g23 + g33 * g33;
        double b12 = g11 * g12 + g21 * g22 + g31 * g32;
        double b13 = g11 * g13 + g21 * g23 + g31 * g33;
        double b23 = g12 * g13 + g22 * g23 + g32 * g33;
        double A = g11 * g11 + g12 * g12 + g13 * g13 + g21 * g21 + g22 * g22 +
                   g23 * g23 + g31 * g31 + g32 * g32 + g33 * g33;
        double B = b11 * b22 - b12 * b12 + b11 * b33 - b13 * b13 + b22 * b33 - b23 * b23;
        if (B < 0.0) B = 0.0;
        double nut = C * sqrt(B / (A + 1e-30));
        g_nut[i] = nut;
        g_te[i] = clamp_te(tau + 3.0 * nut, tau);
    }
}

/* 2x2x2 box filter == separable width-2 box with replicate padding
 * (bc_mode 0) or periodic wrap (bc_mode 1); one pass, 8 reads, no temporaries
 * -- the separable version needed two scratch arrays and a memcpy per axis and
 * made the dynamic model ~4x slower than the rest. */
static void box2(const double *a, double *out) {
    int n = g_n;
    long nn = (long)n * n;
    long i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < g_N; i++) {
        long x = i % n, y = (i / n) % n, z = i / nn;
        long xs = (g_bc_mode == 1) ? ((x - 1 + n) % n) : (x > 0 ? x - 1 : 0);
        long ys = (g_bc_mode == 1) ? ((y - 1 + n) % n) : (y > 0 ? y - 1 : 0);
        long zs = (g_bc_mode == 1) ? ((z - 1 + n) % n) : (z > 0 ? z - 1 : 0);
        long zy = z * nn + y * n, zs_y = z * nn + ys * n;
        long zs_ = zs * nn + y * n, zs_ys = zs * nn + ys * n;
        out[i] = 0.125 * (a[zy + x] + a[zy + xs] + a[zs_y + x] + a[zs_y + xs] +
                          a[zs_ + x] + a[zs_ + xs] + a[zs_ys + x] + a[zs_ys + xs]);
    }
}

/* Dynamic Smagorinsky (Germano-Lilly) from the strain already stored by
 * moments_sgs(): g_mab[0..5] = S_ab, g_smag = |S|.
 * The test-filtered strain goes to g_st[0..5] / g_fbuf[0], L_ab to g_lab. */
static void te_dynamic_from_strain(double tau, double lambda_clip, double *cs2_out) {
    long i;
    int k;
    static const int pi[6] = {0, 0, 0, 1, 1, 2};
    static const int pj[6] = {0, 1, 2, 1, 2, 2};
    double num = 0.0, den = 0.0, cs2;
    double *smag_f = g_fbuf[0];

    box2(g_ux, g_fu[0]);
    box2(g_uy, g_fu[1]);
    box2(g_uz, g_fu[2]);
    gradients_of(g_fu[0], g_fu[1], g_fu[2], g_gf);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < g_N; i++) {
        double g11 = g_gf[0][i], g12 = g_gf[1][i], g13 = g_gf[2][i];
        double g21 = g_gf[3][i], g22 = g_gf[4][i], g23 = g_gf[5][i];
        double g31 = g_gf[6][i], g32 = g_gf[7][i], g33 = g_gf[8][i];
        double txx = g11, tyy = g22, tzz = g33;
        double txy = 0.5 * (g12 + g21), txz = 0.5 * (g13 + g31), tyz = 0.5 * (g23 + g32);
        g_st[0][i] = txx; g_st[1][i] = txy; g_st[2][i] = txz;
        g_st[3][i] = tyy; g_st[4][i] = tyz; g_st[5][i] = tzz;
        double m2 = 2.0 * (txx * txx + tyy * tyy + tzz * tzz +
                           2.0 * (txy * txy + txz * txz + tyz * tyz));
        smag_f[i] = sqrt(m2 > 0.0 ? m2 : 0.0);
    }
    for (k = 0; k < 6; k++) {
        int a = pi[k], b = pj[k];
        long j;
        const double *ua = (a == 0) ? g_ux : (a == 1 ? g_uy : g_uz);
        const double *ub = (b == 0) ? g_ux : (b == 1 ? g_uy : g_uz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (j = 0; j < g_N; j++) g_prod[j] = ua[j] * ub[j];
        box2(g_prod, g_lab[k]);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (j = 0; j < g_N; j++) g_lab[k][j] -= g_fu[a][j] * g_fu[b][j];
    }
    for (i = 0; i < g_N; i++) {
        double sm = smag_f[i], s0 = g_smag[i];
        double m[6];
        for (k = 0; k < 6; k++)
            m[k] = 2.0 * (4.0 * sm * g_st[k][i] - s0 * g_mab[k][i]);
        num += g_lab[0][i] * m[0] + g_lab[3][i] * m[3] + g_lab[5][i] * m[5] +
               2.0 * (g_lab[1][i] * m[1] + g_lab[2][i] * m[2] + g_lab[4][i] * m[4]);
        den += m[0] * m[0] + m[3] * m[3] + m[5] * m[5] +
               2.0 * (m[1] * m[1] + m[2] * m[2] + m[4] * m[4]);
    }
    num /= (double)g_N;
    den /= (double)g_N;
    cs2 = num / (den > 1e-12 ? den : 1e-12);
    if (cs2 < lambda_clip) cs2 = lambda_clip;
    if (cs2 > 0.1) cs2 = 0.1;
    if (cs2 < 0.0) cs2 = 0.0;
    *cs2_out = cs2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < g_N; i++) {
        double nut = cs2 * g_smag[i];
        g_nut[i] = nut;
        g_te[i] = clamp_te(tau + 3.0 * nut, tau);
    }
}

/* Fill tau_eff (and nu_t) for the closures that need velocity gradients.
 * Models 0/1/4 are already finished by moments_sgs(). */
static void build_tau_eff(int model, double tau, double C, double lambda_clip,
                          double *cs2_out) {
    *cs2_out = 0.0;
    if (model == 2) {
        gradients_into(g_g);
        te_wale(tau, C);
    } else if (model == 3) {
        gradients_into(g_g);
        te_vreman(tau, C);
    } else if (model == 1) {
        te_dynamic_from_strain(tau, lambda_clip, cs2_out);
    }
}


void sgs_lbm_set_threads(int nthreads) {
#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
#endif
}

void sgs_lbm_set_tau_for_diag(double tau) { g_stats[10] = (tau - 0.5) / 3.0; }

void sgs_lbm_set_bc_mode(int mode) { g_bc_mode = (mode == 0) ? 0 : 1; }

/* set f = feq(rho=1, u) where u is (3, nz, ny, nx) C-order */
void sgs_lbm_set_velocity(const double *u) {
    long i, N = g_N;
    int q;
    double rho = 1.0;
    for (i = 0; i < N; i++) {
        double ux = u[i], uy = u[N + i], uz = u[2 * N + i];
        g_rho[i] = rho;
        g_ux[i] = ux;
        g_uy[i] = uy;
        g_uz[i] = uz;
        for (q = 0; q < Q; q++) g_f[(long)q * N + i] = feq_q(q, rho, ux, uy, uz);
    }
}

void sgs_lbm_get_velocity(double *u) {
    long i, N = g_N;
    /* stored moments are those of the pre-step state */
    moments_sgs(g_f, 4, 0.5, 0.0);
    for (i = 0; i < N; i++) {
        u[i] = g_ux[i];
        u[N + i] = g_uy[i];
        u[2 * N + i] = g_uz[i];
    }
}

void sgs_lbm_get_populations(double *f) {
    memcpy(f, g_f, sizeof(double) * Q * g_N);
}

void sgs_lbm_set_populations(const double *f) {
    memcpy(g_f, f, sizeof(double) * Q * g_N);
}

void sgs_lbm_get_nut(double *nut) { memcpy(nut, g_nut, sizeof(double) * g_N); }

int sgs_lbm_stats_n(void) { return 16; }

/* Recompute the diagnostics for the current state without advancing it. */
void sgs_lbm_diagnostics(void) {
    long i, N = g_N;
    double mass = 0.0, tke = 0.0, eps_mol = 0.0, eps_sgs = 0.0, ens = 0.0;
    double nut_sum = 0.0, nut_max = 0.0, umax = 0.0, te_max = 0.0;
    double nu = g_stats[10];
    moments_sgs(g_f, 4, 0.5, 0.0);
    gradients_into(g_g);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : mass, tke, eps_mol, eps_sgs, ens, nut_sum) reduction(max : nut_max, umax, te_max)
#endif
    for (i = 0; i < N; i++) {
        double rho = g_rho[i], ux = g_ux[i], uy = g_uy[i], uz = g_uz[i];
        double g11 = g_g[0][i], g12 = g_g[1][i], g13 = g_g[2][i];
        double g21 = g_g[3][i], g22 = g_g[4][i], g23 = g_g[5][i];
        double g31 = g_g[6][i], g32 = g_g[7][i], g33 = g_g[8][i];
        double S12 = 0.5 * (g12 + g21), S13 = 0.5 * (g13 + g31), S23 = 0.5 * (g23 + g32);
        double SS = g11 * g11 + g22 * g22 + g33 * g33 +
                    2.0 * (S12 * S12 + S13 * S13 + S23 * S23);
        double w1 = g32 - g23, w2 = g13 - g31, w3 = g21 - g12;
        mass += rho;
        tke += 0.5 * (ux * ux + uy * uy + uz * uz);
        eps_mol += 2.0 * nu * SS;
        ens += 0.5 * (w1 * w1 + w2 * w2 + w3 * w3);
        {
            double um = fabs(ux) > fabs(uy) ? fabs(ux) : fabs(uy);
            if (fabs(uz) > um) um = fabs(uz);
            if (um > umax) umax = um;
        }
    }
    g_stats[0] = mass / (double)N;
    g_stats[1] = tke / (double)N;
    g_stats[2] = eps_mol / (double)N;
    g_stats[4] = ens / (double)N;
    g_stats[7] = umax;
    g_stats[13] = g_stats[2] + g_stats[3];
    (void)eps_sgs;
    (void)nut_sum;
    (void)nut_max;
    (void)te_max;
}

void sgs_lbm_get_stats(double *out) {
    int k;
    for (k = 0; k < 16; k++) out[k] = g_stats[k];
}

/* One collision (no streaming) on `fin`, result into `fout` -- the
 * production ``collide_*`` semantics, used by the equivalence tests. */
int sgs_lbm_collide_only(const double *fin, double *fout, int model, double tau,
                         double C, double lambda_clip) {
    long i, N = g_N;
    int q;
    double cs2 = 0.0;
    moments_sgs(fin, model, tau, C);
    build_tau_eff(model, tau, C, lambda_clip, &cs2);
    for (i = 0; i < N; i++) {
        double rho = g_rho[i], ux = g_ux[i], uy = g_uy[i], uz = g_uz[i];
        double te = g_te[i];
        for (q = 0; q < Q; q++) {
            double fq = fin[(long)q * N + i];
            fout[(long)q * N + i] = fq + (feq_q(q, rho, ux, uy, uz) - fq) / te;
        }
    }
    g_stats[12] = cs2;
    return 0;
}

/* Evaluate nu_t (and Cs^2) for an externally supplied velocity field. */
int sgs_lbm_nu_t_field(const double *u, int model, double tau, double C,
                       double lambda_clip, double *nut_out, double *cs2_out) {
    double cs2 = 0.0;
    long i;
    sgs_lbm_set_velocity(u);
    moments_sgs(g_f, model, tau, C);
    build_tau_eff(model, tau, C, lambda_clip, &cs2);
    for (i = 0; i < g_N; i++) nut_out[i] = g_nut[i];
    if (cs2_out) *cs2_out = cs2;
    return 0;
}

static int g_timing = -1;

/* collide + stream (pull scheme), one LBM step.
 *
 * Two passes on purpose: an earlier one-pass version recomputed the
 * equilibrium of the upstream cell for all 19 directions (19 redundant reads
 * of rho/u/tau_eff per cell) and ran ~2.5x slower than this formulation.
 *
 * Pass A writes the post-collision populations of every cell (reading the
 * local moments once), pass B gathers them from the upstream neighbour with
 * precomputed integer offsets.  ``stats`` selects the dissipation /
 * enstrophy / eddy-viscosity diagnostics, which cost a full extra sweep over
 * the gradient fields and are only needed at the sampling steps. */
int sgs_lbm_step(int model, double tau, double C, double lambda_clip, int stats) {
    long i, N = g_N;
    int q;
    double cs2 = 0.0;
    double mass = 0.0, tke = 0.0, eps_mol = 0.0, eps_sgs = 0.0;
    double ens = 0.0, nut_sum = 0.0, nut_max = 0.0, umax = 0.0, te_max = 0.0;
    double nu = (tau - 0.5) / 3.0;
    double t_m0 = 0.0, t_s0 = 0.0;
#ifdef _OPENMP
    if (g_timing < 0) g_timing = getenv("SGS_LBM_TIMING") ? 1 : 0;
    if (g_timing) t_m0 = omp_get_wtime();
#endif
    moments_sgs(g_f, model, tau, C);
    build_tau_eff(model, tau, C, lambda_clip, &cs2);
#ifdef _OPENMP
    if (g_timing) t_s0 = omp_get_wtime();
#endif

    /* --- pass A: local collision ------------------------------------- */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < N; i++) {
        double rho = g_rho[i], ux = g_ux[i], uy = g_uy[i], uz = g_uz[i];
        double te = g_te[i], inv = 1.0 / te;
        double usq = ux * ux + uy * uy + uz * uz;
        double base = 1.0 - 1.5 * usq;
        int qq;
        for (qq = 0; qq < Q; qq++) {
            double cu = CX[qq] * ux + CY[qq] * uy + CZ[qq] * uz;
            double fe = WQ[qq] * rho * (base + 3.0 * cu + 4.5 * cu * cu);
            double fq = g_f[(long)qq * N + i];
            g_fn[(long)qq * N + i] = fq + (fe - fq) * inv;
        }
    }

    /* --- pass B: streaming (pull from the upstream neighbour) --------- */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < N; i++) {
        int qq;
        for (qq = 0; qq < Q; qq++) {
            g_f[(long)qq * N + i] = g_fn[(long)qq * N + g_nb[qq][i]];
        }
    }

    /* --- diagnostics ------------------------------------------------- */
    if (stats) {
        gradients_into(g_g);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : mass, tke, eps_mol, eps_sgs, ens, nut_sum) reduction(max : nut_max, umax, te_max)
#endif
        for (i = 0; i < N; i++) {
            double rho = g_rho[i], ux = g_ux[i], uy = g_uy[i], uz = g_uz[i];
            double g11 = g_g[0][i], g12 = g_g[1][i], g13 = g_g[2][i];
            double g21 = g_g[3][i], g22 = g_g[4][i], g23 = g_g[5][i];
            double g31 = g_g[6][i], g32 = g_g[7][i], g33 = g_g[8][i];
            double S12 = 0.5 * (g12 + g21), S13 = 0.5 * (g13 + g31), S23 = 0.5 * (g23 + g32);
            double SS = g11 * g11 + g22 * g22 + g33 * g33 +
                        2.0 * (S12 * S12 + S13 * S13 + S23 * S23);
            double w1 = g32 - g23, w2 = g13 - g31, w3 = g21 - g12;
            double nut = g_nut[i];
            mass += rho;
            tke += 0.5 * (ux * ux + uy * uy + uz * uz);
            eps_mol += 2.0 * nu * SS;
            eps_sgs += 2.0 * nut * SS;
            ens += 0.5 * (w1 * w1 + w2 * w2 + w3 * w3);
            nut_sum += nut;
            if (nut > nut_max) nut_max = nut;
            if (g_te[i] > te_max) te_max = g_te[i];
            {
                double um = fabs(ux) > fabs(uy) ? fabs(ux) : fabs(uy);
                if (fabs(uz) > um) um = fabs(uz);
                if (um > umax) umax = um;
            }
        }
        g_stats[0] = mass / (double)N;
        g_stats[1] = tke / (double)N;
        g_stats[2] = eps_mol / (double)N;
        g_stats[3] = eps_sgs / (double)N;
        g_stats[4] = ens / (double)N;
        g_stats[5] = nut_sum / (double)N;
        g_stats[6] = nut_max;
        g_stats[7] = umax;
        g_stats[8] = te_max;
        g_stats[11] = (double)N;
        g_stats[13] = g_stats[2] + g_stats[3];
    }
    g_stats[9] = cs2;
    g_stats[10] = nu;
#ifdef _OPENMP
    if (g_timing) {
        double t_end = omp_get_wtime();
        fprintf(stderr, "[sgs_lbm] model=%d closure+grad=%.4f s  collide+stream=%.4f s\n",
                model, t_s0 - t_m0, t_end - t_s0);
    }
#endif
    return 0;
}

/* Three-segment KBC (Karlin-Boesch-Chikatamarla) D3Q27 plane-channel kernel.
 *
 * OpenMP port of tensorlbm.kbc_collision.collide_kbc plus the channel
 * scaffolding of the ELBM kernel (elbm_channel_kernel.c): same array layout,
 * same Guo forcing with the half-force velocity shift, same half-way
 * bounce-back walls, same statistics reduction.  The collision is what changes:
 *
 *   f^neq = s + b + g                                    (Hermite split)
 *   f*    = f^eq + (1 - 1/tau) s + (1 - 1/tau_b) b + (1 - alpha*omega_g) g
 *
 * with tau the exact BGK shear rate (nu = cs2 (tau-1/2)), tau_b the bulk rate
 * and alpha from a per-cell monotone entropy solve (Newton on H'(alpha) with a
 * bisection safeguard; H(alpha) is convex in alpha because f*(alpha) is affine
 * in it).  This is the operator the campaign node asks for: the shear stress
 * can never be over-damped (unlike the ELBM H-limiter that relaminarised the
 * m9102 channel), so the log law survives on coarse grids.
 *
 * Positivity is protected by a per-cell relaxation scale gamma in (0,1]:
 *   f_post = f + gamma (f* - f),  gamma = min_i (f_i - GROUND)/(f_i - f*_i)
 * The entropy-condition solve guarantees H(f*) <= H(f) by construction
 * (alpha = 0 is always admissible and gives Delta H = 0 exactly), and when
 * enforce_entropy is on the kernel additionally bisects gamma until the
 * measured H(f_post) <= H(f).
 *
 * Layout: C-contiguous float arrays shaped (Q, nz, ny, nx),
 * index = q*(nz*ny*nx) + z*(ny*nx) + y*nx + x.  Loops run z -> y -> x so the
 * accesses are contiguous.  Wall rows y = 0 and y = ny-1 carry no populations;
 * the no-slip planes sit at y = 0.5 and y = ny - 1.5, so the channel height is
 * Hc = ny - 2 and the half height h = Hc/2.
 *
 * Build: gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC -lm
 * (kbc_channel_native.py hash-caches the shared object).
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define Q 27
#define FLOOR 1e-30
#define GROUND 1e-12

static const int CX[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1};
static const int CY[Q] = {0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1};
static const int CZ[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1};
static const int OPP[Q] = {0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15, 26, 25, 24, 23, 22, 21, 20, 19};
static const double W[Q] = {
    0.29629629629629628, 0.07407407407407407, 0.07407407407407407, 0.07407407407407407,
    0.07407407407407407, 0.07407407407407407, 0.07407407407407407, 0.018518518518518517,
    0.018518518518518517, 0.018518518518518517, 0.018518518518518517, 0.018518518518518517,
    0.018518518518518517, 0.018518518518518517, 0.018518518518518517, 0.018518518518518517,
    0.018518518518518517, 0.018518518518518517, 0.018518518518518517, 0.0046296296296296294,
    0.0046296296296296294, 0.0046296296296296294, 0.0046296296296296294, 0.0046296296296296294,
    0.0046296296296296294, 0.0046296296296296294, 0.0046296296296296294};

static const double CS2_ = 1.0 / 3.0;
static const double FACTOR = 4.5; /* 9/2 : Hermite projector prefactor 1/(2 cs^4) */

/* Direction contractions.  Because c_a in {-1,0,1} the second-order Hermite
 * projections reduce to a handful of multiplies:
 *     K_q(T) = cx^2 Txx + cy^2 Tyy + cz^2 Tzz + 2(cx cy Txy + cx cz Txz + cy cz Tyz)
 * and (the deviatoric tensor being traceless) the "-|c|^2/3 delta" part of the
 * traceless projector vanishes identically, so the shear projector is the same
 * contraction with T = P_dev.  IXX..CYZ are the integer c_a c_b products. */
static const int IXX[Q] = {0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1};
static const int IYY[Q] = {0,0,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1};
static const int IZZ[Q] = {0,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static const int CXY[Q] = {0,0,0,0,0,0,0,1,-1,-1,1,0,0,0,0,0,0,0,0,1,-1,-1,1,1,-1,-1,1};
static const int CXZ[Q] = {0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1,-1,1,-1,1};
static const int CYZ[Q] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,1,1,-1,-1,-1,-1,1,1};
static void lattice_init(void) { }

/* ------------------------------------------------------------------ */
/* equilibrium, entropy, K/B/C split                                    */
/* ------------------------------------------------------------------ */

static inline void feq_quad(double rho, double ux, double uy, double uz, double *feq) {
    double usq = ux * ux + uy * uy + uz * uz;
    for (int q = 0; q < Q; ++q) {
        double cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
        feq[q] = W[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
    }
}

static inline double cell_entropy(const double *f) {
    double h = 0.0;
    for (int q = 0; q < Q; ++q) {
        double v = f[q] > FLOOR ? f[q] : FLOOR;
        h += v * log(v / W[q]);
    }
    return h;
}

/* H, H' and H'' of f*(a) = base + a*d -- one logarithm per direction. */
static inline void entropy_scalar(const double *base, const double *d, double a,
                                  double *H, double *dH, double *d2H) {
    double h = 0.0, d1 = 0.0, d2 = 0.0;
    for (int q = 0; q < Q; ++q) {
        double v = base[q] + a * d[q];
        if (v < FLOOR) v = FLOOR;
        double lg = log(v / W[q]);
        h += v * lg;
        d1 += d[q] * lg;
        d2 += d[q] * d[q] / v;
    }
    *H = h;
    *dH = d1;
    *d2H = d2;
}

/* Per-cell collision.  diag = {alpha, gamma, dH, iters, skipped, |g|/rho}.
 * The force enters as the Guo source (velocity-shift form) with components
 * (fx, fy, fz); only the wall-normal component is non-zero in this channel. */
static void kbc_cell(const double *f, double tau, double tau_b, double omega_g,
                     double alpha_cap, double alpha_min, int n_iter, double trigger, int enforce_entropy,
                     double fx, double fy, double fz, double *fo, double *diag) {
    double rho = 0.0, jx = 0.0, jy = 0.0, jz = 0.0;
    for (int q = 0; q < Q; ++q) {
        double v = f[q];
        rho += v;
        jx += CX[q] * v;
        jy += CY[q] * v;
        jz += CZ[q] * v;
    }
    if (rho < FLOOR) rho = FLOOR;
    /* Guo splitting: the equilibrium, the source and the *output* velocity all
     * use the half-force-shifted velocity u = (j + F/2)/rho (Guo, Zheng &
     * Shi 2002).  Evaluating the equilibrium at the unshifted j/rho instead
     * double-corrects the half step and was measured to give a 3x too viscous
     * channel (caught by the laminar Poiseuille regression in
     * tests/test_kbc_channel.py). */
    double ux = (jx + 0.5 * fx) / rho;
    double uy = (jy + 0.5 * fy) / rho;
    double uz = (jz + 0.5 * fz) / rho;

    double feq[Q];
    feq_quad(rho, ux, uy, uz, feq);

    double Pxx = 0.0, Pyy = 0.0, Pzz = 0.0, Pxy = 0.0, Pxz = 0.0, Pyz = 0.0;
    for (int q = 0; q < Q; ++q) {
        double v = f[q] - feq[q];
        Pxx += CX[q] * CX[q] * v;
        Pyy += CY[q] * CY[q] * v;
        Pzz += CZ[q] * CZ[q] * v;
        Pxy += CX[q] * CY[q] * v;
        Pxz += CX[q] * CZ[q] * v;
        Pyz += CY[q] * CZ[q] * v;
    }
    const double Ptr = Pxx + Pyy + Pzz;
    const double Dxx = Pxx - Ptr / 3.0, Dyy = Pyy - Ptr / 3.0, Dzz = Pzz - Ptr / 3.0;

    double s[Q], b[Q], g[Q];
    for (int q = 0; q < Q; ++q) {
        double cxy = (double)CXY[q], cxz = (double)CXZ[q], cyz = (double)CYZ[q];
        double shear = (double)IXX[q] * Dxx + (double)IYY[q] * Dyy + (double)IZZ[q] * Dzz +
                       2.0 * (cxy * Pxy + cxz * Pxz + cyz * Pyz);
        double full = (double)IXX[q] * Pxx + (double)IYY[q] * Pyy + (double)IZZ[q] * Pzz +
                      2.0 * (cxy * Pxy + cxz * Pxz + cyz * Pyz);
        double sq = FACTOR * W[q] * shear;
        double f2 = FACTOR * W[q] * (full - Ptr / 3.0);
        s[q] = sq;
        b[q] = f2 - sq;
        g[q] = (f[q] - feq[q]) - f2;
    }

    const double om_s = 1.0 / tau;
    const double om_b = 1.0 / tau_b;
    double base[Q], d[Q];
    double gnorm = 0.0;
    for (int q = 0; q < Q; ++q) {
        base[q] = feq[q] + (1.0 - om_s) * s[q] + (1.0 - om_b) * b[q] + g[q];
        /* ghost rate = alpha * omega_g * (1/tau): alpha = omega_g = 1 is BGK */
        d[q] = -omega_g * om_s * g[q];
        gnorm += fabs(g[q]);
    }

    double alpha = 1.0;
    int iters = 0, skipped = 0, newton_fail = 0;
    double dbg_dH0 = 0.0, dbg_d2H0 = 0.0;
    if (gnorm > trigger * rho) {
        double hi = alpha_cap;
        for (int q = 0; q < Q; ++q) {
            if (d[q] < 0.0) {
                double bound = (base[q] - GROUND) / (-d[q]);
                if (bound < hi) hi = bound;
            }
        }
        if (hi < 0.0) hi = 0.0;
        double H0, dH0, d2H0;
        entropy_scalar(base, d, 0.0, &H0, &dH0, &d2H0);
        dbg_dH0 = dH0;
        dbg_d2H0 = d2H0;
        if (dH0 >= 0.0 || hi <= 0.0) {
            alpha = alpha_min > 0.0 ? alpha_min : 0.0;
        } else {
            double Hhi, dHhi, d2Hhi;
            entropy_scalar(base, d, hi, &Hhi, &dHhi, &d2Hhi);
            if (dHhi <= 0.0) {
                alpha = hi; /* minimum at the positivity bound */
            } else {
                double lo = 0.0;
                double a = d2H0 > 1e-300 ? -dH0 / d2H0 : 0.5 * hi;
                if (!(a > lo && a < hi)) a = 0.5 * hi;
                double Ha, dHa, d2Ha;
                for (int it = 0; it < n_iter; ++it) {
                    iters++;
                    entropy_scalar(base, d, a, &Ha, &dHa, &d2Ha);
                    if (dHa < 0.0)
                        lo = a;
                    else
                        hi = a;
                    double step = d2Ha > 1e-300 ? -dHa / d2Ha : 0.0;
                    double next = a + step;
                    if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
                    int done = fabs(next - a) <= 1e-10 * (1.0 + fabs(a));
                    a = next;
                    if (done) break;
                }
                if (!(a >= 0.0)) { a = 0.0; newton_fail = 1; }
                if (a > hi) a = hi;
                alpha = a;
            }
        }
    } else {
        skipped = 1;
    }
    /* the entropy solve is bounded below by the BGK ghost rate (alpha_min = 1
     * is the default): alpha < 1 would *under*-damp the ghost modes and shift
     * the effective viscosity away from cs2(tau - 1/2) even in laminar flow
     * (measured 3x at alpha -> 0 in the laminar Poiseuille regression). */
    if (alpha < alpha_min) alpha = alpha_min;
    if (alpha > alpha_cap) alpha = alpha_cap;

    double cand[Q];
    for (int q = 0; q < Q; ++q) cand[q] = base[q] + alpha * d[q];

    double gamma = 1.0;
    for (int q = 0; q < Q; ++q) {
        double delta = cand[q] - f[q];
        if (delta < 0.0) {
            double bound = (f[q] - GROUND) / (-delta);
            if (bound < gamma) gamma = bound;
        }
    }
    if (!(gamma > 0.0)) gamma = 0.0;

    double Hpre = 0.0, Hpost = 0.0;
    if (gamma < 1.0) {
        for (int q = 0; q < Q; ++q) cand[q] = f[q] + gamma * (cand[q] - f[q]);
    }
    if (enforce_entropy) {
        Hpre = cell_entropy(f);
        Hpost = cell_entropy(cand);
        if (Hpost > Hpre) {
            double dir[Q];
            for (int q = 0; q < Q; ++q) dir[q] = cand[q] - f[q];
            double lo = 0.0, hi = gamma;
            for (int q = 0; q < Q; ++q) cand[q] = f[q];
            for (int it = 0; it < 24; ++it) {
                double mid = 0.5 * (lo + hi);
                double tmp[Q];
                for (int q = 0; q < Q; ++q) tmp[q] = f[q] + mid * dir[q];
                double hm = cell_entropy(tmp);
                if (hm <= Hpre) {
                    lo = mid;
                    for (int q = 0; q < Q; ++q) cand[q] = tmp[q];
                    Hpost = hm;
                } else {
                    hi = mid;
                }
            }
            gamma = lo;
        }
    }

    const double src = 1.0 - 0.5 * om_s;
    const double uf = ux * fx + uy * fy + uz * fz;
    for (int q = 0; q < Q; ++q) {
        double cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
        double cf = CX[q] * fx + CY[q] * fy + CZ[q] * fz;
        double guo = W[q] * src * (3.0 * (cf - uf) + 9.0 * cu * cf);
        double v = cand[q] + guo;
        fo[q] = v > FLOOR ? v : FLOOR;
    }

    diag[0] = alpha;
    diag[1] = gamma;
    diag[2] = Hpost - Hpre;
    diag[3] = (double)iters;
    diag[4] = (double)skipped;
    diag[5] = gnorm / rho;
    diag[6] = (double)newton_fail;
    diag[7] = dbg_dH0;
    diag[8] = dbg_d2H0;
}

/* Debug hook: expose one cell's whole collision algebra for cross-validation
 * against the Python reference (tensorlbm.kbc_collision).
 * out[0..9] = {rho, ux, uy, uz, alpha, gamma, dH, iters, skipped, |g|/rho}
 * out[10..10+Q) = feq, out[10+Q..10+2Q) = f_post (after the Guo source). */
int kbc_debug_cell(const float *f, int nx, int ny, int nz, int zi, int yi, int xi,
                   double tau, double tau_b, double omega_g, double alpha_cap, double alpha_min,
                   int n_iter, double trigger, int enforce_entropy, double fx, double fy, double fz,
                   double *out) {
    lattice_init();
    const size_t N = (size_t)nx * ny * nz;
    const size_t i = ((size_t)zi * ny + yi) * nx + xi;
    double fv[Q], fo[Q], diag[9];
    for (int q = 0; q < Q; ++q) fv[q] = (double)f[(size_t)q * N + i];
    kbc_cell(fv, tau, tau_b, omega_g, alpha_cap, alpha_min, n_iter, trigger, enforce_entropy,
             fx, fy, fz, fo, diag);
    double rho = 0.0, jx = 0.0, jy = 0.0, jz = 0.0;
    for (int q = 0; q < Q; ++q) {
        rho += fv[q];
        jx += CX[q] * fv[q];
        jy += CY[q] * fv[q];
        jz += CZ[q] * fv[q];
    }
    out[0] = rho;
    out[1] = (jx + 0.5 * fx) / rho;
    out[2] = (jy + 0.5 * fy) / rho;
    out[3] = (jz + 0.5 * fz) / rho;
    for (int k = 0; k < 9; ++k) out[4 + k] = diag[k];
    double feq[Q];
    feq_quad(rho, out[1], out[2], out[3], feq);
    for (int q = 0; q < Q; ++q) {
        out[13 + q] = feq[q];
        out[13 + Q + q] = fo[q];
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* one full step: collision (into fs) -> streaming + bounce back (into f) */
/* ------------------------------------------------------------------ */

/* stats[0]=alpha_mean [1]=alpha_min [2]=alpha_max [3]=gamma_mean [4]=gamma_min
 * [5]=fraction of gamma-limited cells [6]=sum_cells dH [7]=max_cells dH
 * [8]=fraction of cells with the entropy solve skipped [9]=mean |g|/rho
 * [10]=mean Newton iterations */
int kbc_channel_step(float *f, float *fs, int nx, int ny, int nz, double tau, double tau_b,
                     double omega_g, double force, double alpha_cap, double alpha_min, int n_iter,
                     double trigger, int enforce_entropy, double *stats) {
    const size_t N = (size_t)nx * ny * nz;
    double a_sum = 0.0, a_min = 1e300, a_max = -1e300;
    double g_sum = 0.0, g_min = 1e300, n_lim = 0.0, n_skip = 0.0, gnorm_sum = 0.0;
    double dh_sum = 0.0, dh_max = -1e300, it_sum = 0.0;

#pragma omp parallel for schedule(static) collapse(2) \
    reduction(+ : a_sum, g_sum, n_lim, n_skip, gnorm_sum, dh_sum, it_sum) \
    reduction(max : a_max, dh_max) reduction(min : a_min, g_min)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            const int wall = (y == 0 || y == ny - 1);
            const size_t row = ((size_t)z * ny + y) * nx;
            for (int x = 0; x < nx; ++x) {
                const size_t i = row + x;
                if (wall) {
                    for (int q = 0; q < Q; ++q) fs[(size_t)q * N + i] = 0.0f;
                    continue;
                }
                double fv[Q], fo[Q], diag[9];
                for (int q = 0; q < Q; ++q) fv[q] = (double)f[(size_t)q * N + i];
                /* streamwise body force: only fx is non-zero in this channel */
                kbc_cell(fv, tau, tau_b, omega_g, alpha_cap, alpha_min, n_iter, trigger,
                         enforce_entropy, force, 0.0, 0.0, fo, diag);
                for (int q = 0; q < Q; ++q) fs[(size_t)q * N + i] = (float)fo[q];
                a_sum += diag[0];
                if (diag[0] < a_min) a_min = diag[0];
                if (diag[0] > a_max) a_max = diag[0];
                g_sum += diag[1];
                if (diag[1] < g_min) g_min = diag[1];
                if (diag[1] < 1.0 - 1e-12) n_lim += 1.0;
                n_skip += diag[4];
                gnorm_sum += diag[5];
                it_sum += diag[3];
                if (enforce_entropy) {
                    dh_sum += diag[2];
                    if (diag[2] > dh_max) dh_max = diag[2];
                }
            }
        }
    }

    const double ncell = (double)((size_t)(ny - 2) * nz * nx);
    stats[0] = a_sum / ncell;
    stats[1] = a_min;
    stats[2] = a_max;
    stats[3] = g_sum / ncell;
    stats[4] = g_min;
    stats[5] = n_lim / ncell;
    stats[6] = dh_sum;
    stats[7] = dh_max;
    stats[8] = n_skip / ncell;
    stats[9] = gnorm_sum / ncell;
    stats[10] = it_sum / ncell;

    /* Streaming + half-way bounce back (walls at y = 0.5 and y = ny - 1.5).
     * Row-wise: for a given q the x-shift is a cyclic rotation of a row, so it
     * is two memcpys instead of a per-element modulo.  The y/z shifts are row
     * offsets.  Cells that would cross a wall row bounce back from the
     * opposite direction (wall rows themselves carry no populations).       */
#pragma omp parallel for schedule(static)
    for (int q = 0; q < Q; ++q) {
        const int cx = CX[q], cy = CY[q], cz = CZ[q];
        const float *src = fs + (size_t)q * N;
        const float *bsrc = fs + (size_t)OPP[q] * N;
        float *dst = f + (size_t)q * N;
        const size_t rowbytes = (size_t)nx * sizeof(float);
        for (int z = 0; z < nz; ++z) {
            int zs = z - cz;
            if (zs < 0) zs += nz;
            if (zs >= nz) zs -= nz;
            for (int y = 0; y < ny; ++y) {
                const size_t row_d = ((size_t)z * ny + y) * nx;
                if (y == 0 || y == ny - 1) {
                    memset(dst + row_d, 0, rowbytes);
                    continue;
                }
                int ys = y - cy;
                int bounce = 0;
                if (ys < 1) { ys = 1; bounce = 1; }
                if (ys > ny - 2) { ys = ny - 2; bounce = 1; }
                if (bounce) {
                    memcpy(dst + row_d, bsrc + row_d, rowbytes);
                } else {
                    const size_t row_s = ((size_t)zs * ny + ys) * nx;
                    const float *sr = src + row_s;
                    float *dr = dst + row_d;
                    if (cx == 0) {
                        memcpy(dr, sr, rowbytes);
                    } else if (cx > 0) {
                        /* new[x] = old[x - 1]: shift the row right by one */
                        memcpy(dr + 1, sr, (size_t)(nx - 1) * sizeof(float));
                        dr[0] = sr[nx - 1];
                    } else {
                        /* new[x] = old[x + 1]: shift the row left by one */
                        memcpy(dr, sr + 1, (size_t)(nx - 1) * sizeof(float));
                        dr[nx - 1] = sr[0];
                    }
                }
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* driver: initialise, spin up, sample                                  */
/* ------------------------------------------------------------------ */

/* profiles: 8 * ny doubles -> u, uu, vv, ww, uv, rho, Pi_xy(total stress), Pxy_neq
 * scalars : 18 doubles, see kbc_channel_native.KBC_SCALARS
 * history : hist_count * 12 doubles */
int kbc_channel_run(int nx, int ny, int nz, int steps, int warmup, int sample_every, int n_iter,
                    double re_tau, double u_tau, double alpha_cap, double alpha_min, double omega_g,
                    double tau_bulk, double trigger, int enforce_entropy, double noise_amp, int seed,
                    int log_every,
                    double *profiles, double *scalars, double *history, int *hist_count) {
    lattice_init();
    const size_t N = (size_t)nx * ny * nz;
    float *f = (float *)calloc((size_t)Q * N, sizeof(float));
    float *fs = (float *)calloc((size_t)Q * N, sizeof(float));
    double *acc = (double *)calloc((size_t)8 * ny, sizeof(double));
    double *sum_u = (double *)calloc((size_t)ny, sizeof(double));
    double *sum_v = (double *)calloc((size_t)ny, sizeof(double));
    double *sum_w = (double *)calloc((size_t)ny, sizeof(double));
    double *sum_p = (double *)calloc((size_t)ny, sizeof(double));
    if (!f || !fs || !acc || !sum_u || !sum_v || !sum_w || !sum_p) {
        free(f); free(fs); free(acc); free(sum_u); free(sum_v); free(sum_w); free(sum_p);
        return -1;
    }
    const double Hc = (double)(ny - 2);
    const double half = 0.5 * Hc;
    const double force = u_tau * u_tau / half;
    const double nu = u_tau * half / re_tau;
    const double tau = 0.5 + 3.0 * nu;
    const double tau_b = tau_bulk > 0.0 ? tau_bulk : tau;

    /* law-of-the-wall start, mirrored about the centreline */
#pragma omp parallel for schedule(static)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            const size_t row = ((size_t)z * ny + y) * nx;
            double yw = (double)y - 0.5;
            if (yw > half) yw = Hc - yw;
            double yp = yw * re_tau / half;
            if (yp < 1e-6) yp = 1e-6;
            double up = yp < 11.4 ? yp : log(yp) / 0.41 + 5.2;
            double ux = (y > 0 && y < ny - 1) ? u_tau * up : 0.0;
            double feq[Q];
            feq_quad(1.0, ux, 0.0, 0.0, feq);
            for (int x = 0; x < nx; ++x)
                for (int q = 0; q < Q; ++q) f[(size_t)q * N + row + x] = (float)feq[q];
        }
    }
    /* deterministic noise on top of the mean profile (sequential RNG) */
    uint64_t rng = (uint64_t)(seed ? seed : 1) * 2654435761u + 1u;
    for (int z = 0; z < nz; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            const size_t row = ((size_t)z * ny + y) * nx;
            for (int x = 0; x < nx; ++x) {
                double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
                for (int q = 0; q < Q; ++q) {
                    double v = (double)f[(size_t)q * N + row + x];
                    rho += v;
                    ux += CX[q] * v;
                    uy += CY[q] * v;
                    uz += CZ[q] * v;
                }
                double n[3];
                for (int k = 0; k < 3; ++k) {
                    uint64_t s = rng;
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17; rng = s;
                    double u1 = ((double)(s >> 11) + 0.5) / 9007199254740992.0;
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17; rng = s;
                    double u2 = ((double)(s >> 11) + 0.5) / 9007199254740992.0;
                    n[k] = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
                }
                ux = ux / rho + noise_amp * u_tau * n[0];
                uy = uy / rho + noise_amp * u_tau * n[1];
                uz = uz / rho + noise_amp * u_tau * n[2];
                double feq[Q];
                feq_quad(rho, ux, uy, uz, feq);
                for (int q = 0; q < Q; ++q) f[(size_t)q * N + row + x] = (float)feq[q];
            }
        }
    }

    long samples = 0, hist = 0;
    double min_pop = 1e300, max_u = 0.0, mean_rho = 0.0;
    double a_mean_last = 0.0, a_min_last = 0.0, a_max_last = 0.0;
    double g_mean_last = 1.0, g_min_last = 1.0, lim_frac_last = 0.0, skip_frac_last = 0.0;
    double ghost_last = 0.0, dh_sum_last = 0.0, dh_max_last = 0.0, it_last = 0.0;
    double dh_sum_worst = -1e300, dh_max_worst = -1e300, wall_stress_sum = 0.0;
    int finite = 1;

    for (int step = 1; step <= steps; ++step) {
        double stats[11];
        if (kbc_channel_step(f, fs, nx, ny, nz, tau, tau_b, omega_g, force, alpha_cap, alpha_min,
                             n_iter, trigger, enforce_entropy, stats) != 0) {
            finite = 0;
            break;
        }
        a_mean_last = stats[0]; a_min_last = stats[1]; a_max_last = stats[2];
        g_mean_last = stats[3]; g_min_last = stats[4]; lim_frac_last = stats[5];
        ghost_last = stats[9]; it_last = stats[10];
        if (enforce_entropy) {
            dh_sum_last = stats[6];
            dh_max_last = stats[7];
            if (stats[6] > dh_sum_worst) dh_sum_worst = stats[6];
            if (stats[7] > dh_max_worst) dh_max_worst = stats[7];
        }
        skip_frac_last = stats[8];

        if (step % log_every == 0 || step == 1) {
            double mpop = 1e300, mu = 0.0, mrho = 0.0;
            for (int z = 0; z < nz; ++z) {
                for (int y = 1; y < ny - 1; ++y) {
                    const size_t row = ((size_t)z * ny + y) * nx;
                    for (int q = 0; q < Q; ++q) {
                        const float *b = f + (size_t)q * N + row;
                        for (int x = 0; x < nx; ++x) {
                            double val = b[x];
                            if (val < mpop) mpop = val;
                            if (!isfinite(val)) finite = 0;
                        }
                    }
                }
            }
            for (int z = 0; z < nz; ++z) {
                for (int y = 1; y < ny - 1; ++y) {
                    const size_t row = ((size_t)z * ny + y) * nx;
                    for (int x = 0; x < nx; ++x) {
                        double rho = 0.0, jx = 0.0;
                        for (int q = 0; q < Q; ++q) {
                            double v = (double)f[(size_t)q * N + row + x];
                            rho += v;
                            jx += CX[q] * v;
                        }
                        if (rho > 1e-12) {
                            double ux = jx / rho + 0.5 * force;
                            if (fabs(ux) > mu) mu = fabs(ux);
                        }
                        mrho += rho;
                    }
                }
            }
            mrho /= (double)((size_t)(ny - 2) * nz * nx);
            if (mpop < min_pop) min_pop = mpop;
            if (mu > max_u) max_u = mu;
            mean_rho = mrho;
            if (hist < 20000) {
                double *h = history + (size_t)hist * 12;
                h[0] = (double)step; h[1] = mu; h[2] = mrho; h[3] = mpop;
                h[4] = a_mean_last; h[5] = a_min_last; h[6] = a_max_last;
                h[7] = g_mean_last; h[8] = lim_frac_last; h[9] = ghost_last;
                h[10] = dh_sum_last; h[11] = dh_max_last;
                hist++;
            }
            if (!finite) break;
        }

        if (step > warmup && ((step - warmup) % sample_every) == 0) {
            for (int y = 0; y < ny; ++y) { sum_u[y] = 0.0; sum_v[y] = 0.0; sum_w[y] = 0.0; sum_p[y] = 0.0; }
            for (int z = 0; z < nz; ++z) {
                for (int y = 1; y < ny - 1; ++y) {
                    const size_t row = ((size_t)z * ny + y) * nx;
                    for (int x = 0; x < nx; ++x) {
                        double rho = 0.0, jx = 0.0, jy = 0.0, jz = 0.0, pi_xy = 0.0;
                        for (int q = 0; q < Q; ++q) {
                            double v = (double)f[(size_t)q * N + row + x];
                            rho += v;
                            jx += CX[q] * v;
                            jy += CY[q] * v;
                            jz += CZ[q] * v;
                            pi_xy += CX[q] * CY[q] * v;
                        }
                        sum_u[y] += jx / rho + 0.5 * force;
                        sum_v[y] += jy / rho;
                        sum_w[y] += jz / rho;
                        sum_p[y] += pi_xy;
                    }
                }
            }
            const double inv = 1.0 / (double)((size_t)nz * nx);
            for (int y = 1; y < ny - 1; ++y) {
                sum_u[y] *= inv; sum_v[y] *= inv; sum_w[y] *= inv; sum_p[y] *= inv;
                double auu = 0.0, avv = 0.0, aww = 0.0, auv = 0.0, au = 0.0, sr = 0.0, sp = 0.0;
                for (int z = 0; z < nz; ++z) {
                    const size_t row = ((size_t)z * ny + y) * nx;
                    for (int x = 0; x < nx; ++x) {
                        double rho = 0.0, jx = 0.0, jy = 0.0, jz = 0.0, pi_xy = 0.0;
                        for (int q = 0; q < Q; ++q) {
                            double v = (double)f[(size_t)q * N + row + x];
                            rho += v;
                            jx += CX[q] * v;
                            jy += CY[q] * v;
                            jz += CZ[q] * v;
                            pi_xy += CX[q] * CY[q] * v;
                        }
                        double ux = jx / rho + 0.5 * force;
                        double uy = jy / rho;
                        double uz = jz / rho;
                        double up = ux - sum_u[y], vp = uy - sum_v[y], wp = uz - sum_w[y];
                        auu += up * up; avv += vp * vp; aww += wp * wp; auv += up * vp;
                        au += ux; sr += rho; sp += pi_xy;
                    }
                }
                acc[0 * ny + y] += au * inv;
                acc[1 * ny + y] += auu * inv;
                acc[2 * ny + y] += avv * inv;
                acc[3 * ny + y] += aww * inv;
                acc[4 * ny + y] += auv * inv;
                acc[5 * ny + y] += sr * inv;
                acc[6 * ny + y] += sp * inv;
                acc[7 * ny + y] += (sp * inv - (sum_u[y] * sum_u[y])) ; /* total stress - rho u^2 */
            }
            /* wall momentum check: force input per wall area = force * half */
            wall_stress_sum += force * half;
            samples++;
        }
    }

    const double inv_s = samples > 0 ? 1.0 / (double)samples : 0.0;
    for (int y = 0; y < ny; ++y)
        for (int k = 0; k < 8; ++k) profiles[(size_t)k * ny + y] = acc[(size_t)k * ny + y] * inv_s;

    scalars[0] = (double)samples;
    scalars[1] = min_pop;
    scalars[2] = max_u;
    scalars[3] = a_mean_last;
    scalars[4] = a_min_last;
    scalars[5] = a_max_last;
    scalars[6] = dh_sum_worst;      /* worst (largest) global per-step dH */
    scalars[7] = (double)finite;
    scalars[8] = nu;
    scalars[9] = mean_rho;
    scalars[10] = g_mean_last;
    scalars[11] = g_min_last;
    scalars[12] = lim_frac_last;
    scalars[13] = skip_frac_last;
    scalars[14] = ghost_last;
    scalars[15] = dh_max_worst;
    scalars[16] = it_last;
    scalars[17] = wall_stress_sum * inv_s;
    *hist_count = (int)hist;
    free(f); free(fs); free(acc); free(sum_u); free(sum_v); free(sum_w); free(sum_p);
    return 0;
}

/* ELBM (entropic lattice Boltzmann) D3Q27 plane-channel kernel.
 *
 * OpenMP kernel behind tensorlbm.elbm_channel_native.  The Python module
 * tensorlbm.elbm_channel is the *reference implementation*: this file is a
 * straight port of it (same entropic equilibrium, same entropy-condition
 * Newton solve, same Guo forcing, same half-way bounce-back, same statistics).
 * tests/test_elbm.py pins the two to <=1e-4 relative per population for one
 * step and to <=1e-6 on the per-cell relaxation factor.
 *
 * Why a kernel at all: the per-cell entropy solve needs ~(1+n_iter)*27
 * logarithms, which torch executes as dozens of bandwidth-bound elementwise
 * passes (measured 0.1 Mcell/s on a 128-core node).  Fused in C with the
 * per-cell state in registers and the cell loop in *memory order* it runs at
 * O(10) Mcell/s per node, which is what makes a wall-resolved ELBM channel
 * affordable.
 *
 * Population arrays are C-contiguous with numpy/torch shape (Q, nz, ny, nx),
 * i.e. index = q*(nz*ny*nx) + z*(ny*nx) + y*nx + x.  Every loop below is
 * ordered z -> y -> x so the accesses are contiguous; getting this wrong
 * silently transposes y/z and the strided access costs ~10x (the Python
 * reference check in tests/test_elbm.py catches it).
 *
 * Build: gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC
 *        (see elbm_channel_native.py)
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define Q 27
#define FLOOR 1e-30f

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

/* Standard normal deviate (Box-Muller) from a deterministic xorshift64 RNG. */
static double rng_unit(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return ((double)(x >> 11) + 0.5) / 9007199254740992.0;
}

static double rng_normal(uint64_t *s) {
    double u1 = rng_unit(s), u2 = rng_unit(s);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* Gauss-entropy D3Q27 equilibrium, closed form (Ansumali & Karlin 2002):
 * per axis  S = sqrt(1+3u^2), r = (S-2u)/(1+u), P = 6/(4+r+1/r),
 * g(-1) = P*r, g(0) = P, g(+1) = P/r ;  f_i = w_i rho prod_axis g(c_i).
 * Mirrors _axis_factors()/entropic_equilibrium_gauss_d3q27() in elbm.py. */
static inline void entropic_equilibrium(double rho, double ux, double uy, double uz, float *feq) {
    double g[3][3];
    const double u[3] = {ux, uy, uz};
    for (int a = 0; a < 3; ++a) {
        double ua = u[a];
        if (ua > 0.9) ua = 0.9;
        if (ua < -0.9) ua = -0.9;
        double S = sqrt(1.0 + 3.0 * ua * ua);
        double r = (S - 2.0 * ua) / (1.0 + ua);
        double P = 6.0 / (4.0 + r + 1.0 / r);
        g[a][0] = P * r; /* c = -1 */
        g[a][1] = P;     /* c =  0 */
        g[a][2] = P / r; /* c = +1 */
    }
    for (int q = 0; q < Q; ++q) {
        feq[q] = (float)(W[q] * rho * g[0][CX[q] + 1] * g[1][CY[q] + 1] * g[2][CZ[q] + 1]);
    }
}

/* Entropy-conditioned relaxation factor for one cell.
 *
 * Two-stage solve, mirroring solve_entropy_condition() in elbm.py:
 *
 * 1. Evaluate H at the admissibility bound alpha_hi = min(positivity bound, cap).
 *    If H did not increase there, the entropy condition is satisfied by the
 *    bound itself -- the usual near-equilibrium case, and alpha_hi = cap = 1
 *    means plain BGK.  This costs one extra entropy evaluation and, crucially,
 *    it makes the solve immune to the 1e-7 float noise in H: a degenerate
 *    state (f == f^eq) lands here instead of taking a noise-driven Newton step.
 * 2. Otherwise H increases somewhere inside, so a root exists: safeguarded
 *    Newton/bisection on [0, alpha_hi].  The returned value is `lo`, the last
 *    iterate with a *verified* non-increasing entropy, so the H-theorem holds
 *    by construction even if the bracket has not fully collapsed. */
static inline double entropy_alpha(const float *f, const float *feq, double tau, int n_iter,
                                   double alpha_cap, double alpha_warm) {
    double delta[Q];
    double H0 = 0.0;
    double t_pos = 1e300;
    int any_neg = 0;
    for (int q = 0; q < Q; ++q) {
        double fq = f[q] > FLOOR ? (double)f[q] : (double)FLOOR;
        double d = (double)feq[q] - fq;
        delta[q] = d;
        H0 += fq * log(fq / W[q]);
        if (d < 0.0) {
            double b = -fq / d;
            if (b < t_pos) t_pos = b;
            any_neg = 1;
        }
    }
    double alpha_hi = any_neg ? tau * t_pos * (1.0 - 1e-6) : 1e300;
    if (alpha_hi > alpha_cap) alpha_hi = alpha_cap;
    if (!(alpha_hi > 1e-9)) alpha_hi = 1e-9;
    const double tol = 1e-6 * (fabs(H0) + 1.0);

    double Hc = 0.0;
    for (int q = 0; q < Q; ++q) {
        double ft = (double)f[q] + (alpha_hi / tau) * delta[q];
        if (ft < FLOOR) ft = (double)FLOOR;
        Hc += ft * log(ft / W[q]);
    }
    if (Hc - H0 <= tol) return alpha_hi;

    double lo = 0.0, hi = alpha_hi;
    double alpha = (alpha_warm > 1e-9 && alpha_warm < alpha_hi) ? alpha_warm : 0.5 * alpha_hi;
    for (int it = 0; it < n_iter; ++it) {
        double Ht = 0.0, dHt = 0.0;
        for (int q = 0; q < Q; ++q) {
            double ft = (double)f[q] + (alpha / tau) * delta[q];
            if (ft < FLOOR) ft = (double)FLOOR;
            double lg = log(ft / W[q]);
            Ht += ft * lg;
            dHt += delta[q] * (lg + 1.0);
        }
        dHt /= tau;
        double err = Ht - H0;
        if (err > 0.0) hi = alpha;
        else lo = alpha;
        double denom = fabs(dHt) < 1e-30 ? 1e-30 : dHt;
        double step = alpha - err / denom;
        if (!(step > lo && step < hi)) step = 0.5 * (lo + hi);
        alpha = step;
        if (alpha > alpha_hi) alpha = alpha_hi;
        if (alpha < 1e-9) alpha = 1e-9;
    }
    return lo;
}

/* One full step: macro -> ELBM collision -> Guo force -> streaming + bounce back.
 * `f` is in/out (pre-collision on entry, post-streaming on exit); `fs` is
 * scratch and holds the post-collision state on exit.  `alpha` carries the
 * warm start in and the used values out.
 * stats[0..3] = {alpha_mean, alpha_min, alpha_max, entropy_drop_max};
 * stats[4] = number of cells whose alpha sits at the admissibility cap.
 */
int elbm_channel_step(float *f, float *fs, float *alpha, int nx, int ny, int nz, double tau,
                      double force, int n_iter, double alpha_cap, double *stats) {
    const size_t N = (size_t)nx * ny * nz;
    const double cs2 = 1.0 / 3.0;
    double a_sum = 0.0, a_min = 1e300, a_max = -1e300, h_drop = -1e300;
    long n_cap = 0;

#pragma omp parallel for schedule(static) collapse(2) reduction(+ : a_sum, n_cap) \
    reduction(max : a_max) reduction(min : a_min) reduction(max : h_drop)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            const int wall = (y == 0 || y == ny - 1);
            const size_t row = ((size_t)z * ny + y) * nx;
            for (int x = 0; x < nx; ++x) {
                const size_t i = row + x;
                double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
                for (int q = 0; q < Q; ++q) {
                    double v = f[(size_t)q * N + i];
                    rho += v;
                    ux += CX[q] * v;
                    uy += CY[q] * v;
                    uz += CZ[q] * v;
                }
                double rs = rho > 1e-12 ? rho : 1e-12;
                ux /= rs; uy /= rs; uz /= rs;
                if (rho < 1e-9) rho = 1e-9;
                if (wall) { ux = 0.0; uy = 0.0; uz = 0.0; }
                /* Guo scheme: the equilibrium is built at the half-force-shifted
                 * velocity u + F/(2 rho), so that collision + source deliver
                 * exactly rho*F of momentum per step (see _guo_force docstring). */
                ux += 0.5 * force;
                float feq[Q];
                entropic_equilibrium(rho, ux, uy, uz, feq);
                float fv[Q];
                double H0 = 0.0;
                for (int q = 0; q < Q; ++q) {
                    fv[q] = f[(size_t)q * N + i];
                    double s0 = fv[q] > FLOOR ? (double)fv[q] : (double)FLOOR;
                    H0 += s0 * log(s0 / W[q]);
                }
                double a = entropy_alpha(fv, feq, tau, n_iter, alpha_cap, alpha[i]);
                double rate = a / tau;
                double post[Q];
                for (int q = 0; q < Q; ++q) {
                    post[q] = (double)fv[q] + rate * ((double)feq[q] - (double)fv[q]);
                }
                double H1 = 0.0;
                for (int q = 0; q < Q; ++q) {
                    double s1 = post[q] > FLOOR ? post[q] : (double)FLOOR;
                    H1 += s1 * log(s1 / W[q]);
                }
                const double gx = rho * force;
                for (int q = 0; q < Q; ++q) {
                    /* Guo (2002) body force along +x */
                    double cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
                    double v = post[q] + W[q] * (1.0 - 0.5 / tau) *
                                              ((CX[q] - ux) * gx / cs2 + cu * (CX[q] * gx) / (cs2 * cs2));
                    fs[(size_t)q * N + i] = (float)v;
                }
                alpha[i] = (float)a;
                a_sum += a;
                if (a < a_min) a_min = a;
                if (a > a_max) a_max = a;
                if (a >= alpha_cap * (1.0 - 1e-6)) n_cap++;
                double d = H1 - H0;
                if (d > h_drop) h_drop = d;
            }
        }
    }

    /* streaming with half-way bounce-back in y (x, z periodic), in memory order */
#pragma omp parallel for schedule(static)
    for (int q = 0; q < Q; ++q) {
        const int cx = CX[q], cy = CY[q], cz = CZ[q];
        const float *src = fs + (size_t)q * N;
        const float *bsrc = fs + (size_t)OPP[q] * N;
        float *dst = f + (size_t)q * N;
        for (int z = 0; z < nz; ++z) {
            int zs = z - cz;
            if (zs < 0) zs += nz;
            if (zs >= nz) zs -= nz;
            for (int y = 0; y < ny; ++y) {
                const size_t row_d = ((size_t)z * ny + y) * nx;
                if (y == 0 || y == ny - 1) { /* wall rows carry no population */
                    for (int x = 0; x < nx; ++x) dst[row_d + x] = 0.0f;
                    continue;
                }
                int ys = y - cy;
                int bounce = 0;
                if (ys < 1) { ys = 1; bounce = 1; }
                if (ys > ny - 2) { ys = ny - 2; bounce = 1; }
                if (!bounce) {
                    const size_t row_s = ((size_t)zs * ny + ys) * nx;
                    for (int x = 0; x < nx; ++x) {
                        int xs = x - cx;
                        if (xs < 0) xs += nx;
                        if (xs >= nx) xs -= nx;
                        dst[row_d + x] = src[row_s + xs];
                    }
                } else {
                    for (int x = 0; x < nx; ++x) dst[row_d + x] = bsrc[row_d + x];
                }
            }
        }
    }

    stats[0] = a_sum / (double)N;
    stats[1] = a_min;
    stats[2] = a_max;
    stats[3] = h_drop;
    stats[4] = (double)n_cap;
    return 0;
}

/* Debug hook: expose one cell's collision algebra for cross-validation against
 * the Python reference.
 * out[0..8] = {rho, ux, uy, uz, H0, alpha, alpha_warm, alpha, H1}
 * out[9 .. 9+3Q) = {feq, f, feq-f} */
int elbm_debug_cell(const float *f, int nx, int ny, int nz, int zi, int yi, int xi, double tau,
                    int n_iter, double alpha_cap, double alpha_warm, double *out) {
    const size_t N = (size_t)nx * ny * nz;
    const size_t i = ((size_t)zi * ny + yi) * nx + xi;
    double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
    for (int q = 0; q < Q; ++q) {
        double v = f[(size_t)q * N + i];
        rho += v; ux += CX[q] * v; uy += CY[q] * v; uz += CZ[q] * v;
    }
    ux /= rho; uy /= rho; uz /= rho;
    float feq[Q];
    entropic_equilibrium(rho, ux, uy, uz, feq);
    float fv[Q];
    double H0 = 0.0;
    for (int q = 0; q < Q; ++q) {
        fv[q] = f[(size_t)q * N + i];
        double s0 = fv[q] > FLOOR ? (double)fv[q] : (double)FLOOR;
        H0 += s0 * log(s0 / W[q]);
    }
    double a = entropy_alpha(fv, feq, tau, n_iter, alpha_cap, alpha_warm);
    double H1 = 0.0;
    for (int q = 0; q < Q; ++q) {
        double v = (double)fv[q] + (a / tau) * ((double)feq[q] - (double)fv[q]);
        if (v < FLOOR) v = (double)FLOOR;
        H1 += v * log(v / W[q]);
    }
    out[0] = rho; out[1] = ux; out[2] = uy; out[3] = uz;
    out[4] = H0; out[5] = a; out[6] = alpha_warm; out[7] = a; out[8] = H1;
    for (int q = 0; q < Q; ++q) out[9 + q] = (double)feq[q];
    for (int q = 0; q < Q; ++q) out[9 + Q + q] = (double)fv[q];
    for (int q = 0; q < Q; ++q) out[9 + 2 * Q + q] = (double)feq[q] - (double)fv[q];
    return 0;
}

/* Full channel driver: initialise (law of the wall + noise), spin up, sample.
 *
 * profiles: 6 * ny doubles -> u, uu, vv, ww, uv, rho   (per wall-normal row)
 * scalars : 10 doubles -> samples, min_pop, max_u, alpha_mean, alpha_min,
 *           alpha_max, entropy_drop_max, finite, nu_nominal, mean_rho
 * history : hist_count * 8 doubles -> step, max_u, mean_rho, min_pop,
 *           alpha_mean, alpha_min, alpha_max, entropy_drop_max
 */
int elbm_channel_run(int nx, int ny, int nz, int steps, int warmup, int sample_every, int n_iter,
                     double re_tau, double u_tau, double alpha_cap, double noise_amp, int seed,
                     int log_every, double *profiles, double *scalars, double *history,
                     int *hist_count) {
    const size_t N = (size_t)nx * ny * nz;
    float *f = (float *)calloc((size_t)Q * N, sizeof(float));
    float *fs = (float *)calloc((size_t)Q * N, sizeof(float));
    float *alpha = (float *)malloc(N * sizeof(float));
    double *acc = (double *)calloc((size_t)6 * ny, sizeof(double));
    double *sum_u = (double *)calloc((size_t)ny, sizeof(double));
    double *sum_v = (double *)calloc((size_t)ny, sizeof(double));
    if (!f || !fs || !alpha || !acc || !sum_u || !sum_v) {
        free(f); free(fs); free(alpha); free(acc); free(sum_u); free(sum_v);
        return -1;
    }
    /* half-way bounce-back puts the no-slip planes at y = 0.5 and y = ny-1.5 */
    const double H = (double)(ny - 2);
    const double half = 0.5 * H;
    const double force = u_tau * u_tau / half;
    const double nu = u_tau * half / re_tau;
    const double tau = 0.5 + 3.0 * nu;

    for (size_t i = 0; i < N; ++i) alpha[i] = 1.0f;

    /* law-of-the-wall start, mirrored about the centreline */
#pragma omp parallel for schedule(static)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            const size_t row = ((size_t)z * ny + y) * nx;
            double yw = (double)y - 0.5;
            if (yw > half) yw = H - yw;
            double yp = yw * re_tau / half;
            if (yp < 1e-6) yp = 1e-6;
            double up = yp < 11.4 ? yp : log(yp) / 0.41 + 5.2;
            double ux = (y > 0 && y < ny - 1) ? u_tau * up : 0.0;
            float feq[Q];
            entropic_equilibrium(1.0, ux, 0.0, 0.0, feq);
            for (int x = 0; x < nx; ++x) {
                for (int q = 0; q < Q; ++q) f[(size_t)q * N + row + x] = feq[q];
            }
        }
    }
    /* deterministic noise (sequential: the RNG state must not race) */
    uint64_t rng = (uint64_t)(seed ? seed : 1) * 2654435761u + 1u;
    for (int z = 0; z < nz; ++z) {
        for (int y = 1; y < ny - 1; ++y) {
            const size_t row = ((size_t)z * ny + y) * nx;
            for (int x = 0; x < nx; ++x) {
                const double amp = noise_amp * u_tau;
                const double n1 = amp * rng_normal(&rng);
                const double n2 = amp * rng_normal(&rng);
                const double n3 = amp * rng_normal(&rng);
                double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
                for (int q = 0; q < Q; ++q) {
                    double v = f[(size_t)q * N + row + x];
                    rho += v; ux += CX[q] * v; uy += CY[q] * v; uz += CZ[q] * v;
                }
                ux = ux / rho + n1; uy = uy / rho + n2; uz = uz / rho + n3;
                float feq[Q];
                entropic_equilibrium(rho, ux, uy, uz, feq);
                for (int q = 0; q < Q; ++q) f[(size_t)q * N + row + x] = feq[q];
            }
        }
    }

    long samples = 0, hist = 0;
    double min_pop = 1e300, max_u = 0.0, mean_rho = 0.0;
    double a_mean_last = 0.0, a_min_last = 0.0, a_max_last = 0.0, h_drop_max = -1e300;
    int finite = 1;

    for (int step = 1; step <= steps; ++step) {
        double stats[5];
        if (elbm_channel_step(f, fs, alpha, nx, ny, nz, tau, force, n_iter, alpha_cap, stats) != 0) {
            finite = 0;
            break;
        }
        a_mean_last = stats[0]; a_min_last = stats[1]; a_max_last = stats[2];
        if (stats[3] > h_drop_max) h_drop_max = stats[3];

        if (step % log_every == 0 || step == 1) {
            double mpop = 1e300, mu = 0.0, mrho = 0.0;
            /* positivity is a statement about the fluid cells: the wall rows
             * are structural zeros, so scan y = 1 .. ny-2 only */
            for (int z = 0; z < nz; ++z) {
                for (int y = 1; y < ny - 1; ++y) {
                    const size_t row = ((size_t)z * ny + y) * nx;
                    for (int q = 0; q < Q; ++q) {
                        const float *base = f + (size_t)q * N + row;
                        for (int x = 0; x < nx; ++x) {
                            double val = base[x];
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
                        double rho = 0.0, ux = 0.0;
                        for (int q = 0; q < Q; ++q) {
                            double v = f[(size_t)q * N + row + x];
                            rho += v; ux += CX[q] * v;
                        }
                        if (rho > 1e-12) {
                            ux = ux / rho + 0.5 * force;
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
            if (hist < 100000) {
                double *h = history + (size_t)hist * 8;
                h[0] = (double)step; h[1] = mu; h[2] = mrho; h[3] = mpop;
                h[4] = a_mean_last; h[5] = a_min_last; h[6] = a_max_last; h[7] = h_drop_max;
                hist++;
            }
            if (!finite) break;
        }

        if (step > warmup && ((step - warmup) % sample_every) == 0) {
            for (int y = 0; y < ny; ++y) { sum_u[y] = 0.0; sum_v[y] = 0.0; }
            for (int z = 0; z < nz; ++z) {
                for (int y = 1; y < ny - 1; ++y) {
                    const size_t row = ((size_t)z * ny + y) * nx;
                    for (int x = 0; x < nx; ++x) {
                        double rho = 0.0, ux = 0.0, uy = 0.0;
                        for (int q = 0; q < Q; ++q) {
                            double v = f[(size_t)q * N + row + x];
                            rho += v; ux += CX[q] * v; uy += CY[q] * v;
                        }
                        sum_u[y] += ux / rho + 0.5 * force; sum_v[y] += uy / rho;
                    }
                }
            }
            const double inv = 1.0 / (double)((size_t)nz * nx);
            for (int y = 1; y < ny - 1; ++y) {
                sum_u[y] *= inv; sum_v[y] *= inv;
                double auu = 0.0, avv = 0.0, aww = 0.0, auv = 0.0, au = 0.0, sr = 0.0;
                for (int z = 0; z < nz; ++z) {
                    const size_t row = ((size_t)z * ny + y) * nx;
                    for (int x = 0; x < nx; ++x) {
                        double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
                        for (int q = 0; q < Q; ++q) {
                            double v = f[(size_t)q * N + row + x];
                            rho += v; ux += CX[q] * v; uy += CY[q] * v; uz += CZ[q] * v;
                        }
                        ux = ux / rho + 0.5 * force; uy /= rho; uz /= rho;
                        double up = ux - sum_u[y], vp = uy - sum_v[y];
                        auu += up * up; avv += vp * vp; aww += uz * uz; auv += up * vp;
                        au += ux; sr += rho;
                    }
                }
                acc[0 * ny + y] += au * inv;
                acc[1 * ny + y] += auu * inv;
                acc[2 * ny + y] += avv * inv;
                acc[3 * ny + y] += aww * inv;
                acc[4 * ny + y] += auv * inv;
                acc[5 * ny + y] += sr * inv;
            }
            samples++;
        }
    }

    const double inv_s = samples > 0 ? 1.0 / (double)samples : 0.0;
    for (int y = 0; y < ny; ++y) {
        for (int k = 0; k < 6; ++k) profiles[(size_t)k * ny + y] = acc[(size_t)k * ny + y] * inv_s;
    }
    scalars[0] = (double)samples;
    scalars[1] = min_pop;
    scalars[2] = max_u;
    scalars[3] = a_mean_last;
    scalars[4] = a_min_last;
    scalars[5] = a_max_last;
    scalars[6] = h_drop_max;
    scalars[7] = (double)finite;
    scalars[8] = nu;
    scalars[9] = mean_rho;
    *hist_count = (int)hist;
    free(f); free(fs); free(alpha); free(acc); free(sum_u); free(sum_v);
    return 0;
}
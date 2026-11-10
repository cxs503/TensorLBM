/* mrt_channel3d.c -- D3Q19 MRT turbulent-channel kernel with Smagorinsky LES
 * and an optional Reichardt wall-stress model.
 *
 * Physics
 * -------
 *   * D3Q19 MRT collision, moment matrix read from disk (single source of
 *     truth: tensorlbm.turbulence_closure.mrt.build_moment_matrix).
 *   * Streamwise / spanwise periodicity, no-slip halfway bounce-back walls at
 *     y = 0.5 and y = ny - 1.5, constant body force g (Guo forcing) so that
 *         u_tau = sqrt(g h),  h = (ny - 2)/2,  Re_tau = u_tau h / nu.
 *   * Smagorinsky SGS: tau_eff = 1/2 + 3 (nu + (Cs Dx)^2 |S|) with |S| obtained
 *     from the non-equilibrium stress (two fixed-point iterations).
 *   * Wall mode "reichardt": the wall-adjacent cell gets the effective
 *     viscosity nu_w = u_tau y1 with u_tau inverted from the Reichardt (1951)
 *     composite law u+ = ln(1+0.4 y+)/kappa
 *                       + 7.8 (1 - exp(-y+/11) - (y+/11) exp(-y+/3)).
 *     This is a wall-stress (not wall-function-forcing) model: the momentum
 *     flux is imposed at the wall, the interior is unconstrained.
 *
 * Output
 * ------
 *   <prefix>_profile.txt : y   u_mean   u_rms   uy_rms   count
 *   <prefix>_diag.txt    : step ke maxu u_tau_wall mass
 *   <prefix>_meta.txt    : run record
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NQ 19
static const int CX[NQ] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
static const int CY[NQ] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
static const int CZ[NQ] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
static const double W[NQ] = {1.0 / 3,  1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18,
                             1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36,
                             1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36};
static const int OPP[NQ] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
static const double CS2 = 1.0 / 3.0;
static const int ST_IDX[5] = {5, 6, 7, 8, 9}; /* traceless 2nd-order rows */

static double M[NQ][NQ];
static int nx, ny, nz, ncell, nxy;
static double *f, *fpost;
static double *sum_ux, *sum_ux2, *sum_uy2;
static double *cnt;

static void die(const char *m) {
    fprintf(stderr, "mrt_channel3d: %s\n", m);
    exit(2);
}

static inline int idx(int x, int y, int z) { return (y * nz + z) * nx + x; }

static void load_matrix(const char *p) {
    FILE *fp = fopen(p, "r");
    if (!fp) die("cannot open MRT matrix");
    for (int i = 0; i < NQ; i++)
        for (int j = 0; j < NQ; j++)
            if (fscanf(fp, "%lf", &M[i][j]) != 1) die("matrix truncated");
    fclose(fp);
}

static inline void feq_q(int q, double rho, double ux, double uy, double uz, double *o) {
    double cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
    double usq = ux * ux + uy * uy + uz * uz;
    *o = rho * W[q] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
}

/* Reichardt (1951) composite law; returns u+ for a given y+ */
static inline double reichardt_uplus(double yp, double kappa) {
    return log(1.0 + 0.4 * yp) / kappa +
           7.8 * (1.0 - exp(-yp / 11.0) - (yp / 11.0) * exp(-yp / 3.0));
}

/* invert U = u_tau * reichardt(y1 u_tau / nu) for u_tau (bisection, 60 iters) */
static double wall_friction_velocity(double U, double y1, double nu, double kappa) {
    double target = fabs(U);
    if (target < 1e-12) return 0.0;
    double lo = 1e-8, hi = 1.0;
    for (int i = 0; i < 80; i++) {
        double mid = 0.5 * (lo + hi);
        double yp = y1 * mid / nu;
        double val = mid * reichardt_uplus(fmax(yp, 1e-12), kappa);
        if (val < target) lo = mid; else hi = mid;
    }
    double ut = 0.5 * (lo + hi);
    return U < 0.0 ? -ut : ut;
}

int main(int argc, char **argv) {
    int steps = 20000, warmup = 10000, sample_every = 10, diag_every = 500;
    double tau0 = 0.55, cs = 0.1, g = 1e-6, kappa = 0.41, nu_wall_override = 0.0;
    int wall_mode = 0; /* 0 = resolved, 1 = reichardt */
    const char *prefix = "ch", *matpath = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--nx")) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ny")) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nz")) nz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps")) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sample-every")) sample_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tau")) tau0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--cs")) cs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--force")) g = atof(argv[++i]);
        else if (!strcmp(argv[i], "--kappa")) kappa = atof(argv[++i]);
        else if (!strcmp(argv[i], "--wall")) wall_mode = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--matrix")) matpath = argv[++i];
        else if (!strcmp(argv[i], "--prefix")) prefix = argv[++i];
        else die("unknown argument");
    }
    if (!matpath || nx <= 0 || ny <= 3 || nz <= 0) die("bad arguments");
    nu_wall_override = (tau0 - 0.5) / 3.0;

    ncell = nx * ny * nz;
    nxy = nx * nz;
    f = (double *)calloc((size_t)NQ * ncell, sizeof(double));
    fpost = (double *)calloc((size_t)NQ * ncell, sizeof(double));
    sum_ux = (double *)calloc(ny, sizeof(double));
    sum_ux2 = (double *)calloc(ny, sizeof(double));
    sum_uy2 = (double *)calloc(ny, sizeof(double));
    cnt = (double *)calloc(ny, sizeof(double));
    if (!f || !fpost || !sum_ux || !sum_ux2 || !sum_uy2 || !cnt) die("oom");
    load_matrix(matpath);

    const double h = (ny - 2) / 2.0;
    const double nu = nu_wall_override;
    const double u_tau_target = sqrt(g * h);

    /* Initial state: the Reichardt mean profile at the target u_tau plus 5%
     * random fluctuations in all three components.  Starting from rest would be
     * hopeless: the viscous spin-up time of the mean flow is h^2/nu, two orders
     * of magnitude more steps than the statistics need. */
    srand(12345);
    for (int y = 1; y < ny - 1; y++) {
        double y1 = (y - 0.5);
        double y_plus = y1 * u_tau_target / nu;
        double u_mean = u_tau_target * reichardt_uplus(fmax(y_plus, 1e-6), kappa);
        for (int z = 0; z < nz; z++)
            for (int x = 0; x < nx; x++) {
                int i = idx(x, y, z);
                double amp = 0.05 * u_mean;
                double ux = u_mean + amp * ((double)rand() / RAND_MAX - 0.5);
                double uy = amp * ((double)rand() / RAND_MAX - 0.5);
                double uz = amp * ((double)rand() / RAND_MAX - 0.5);
                for (int q = 0; q < NQ; q++) {
                    double v;
                    feq_q(q, 1.0, ux, uy, uz, &v);
                    f[(size_t)q * ncell + i] = v;
                }
            }
    }

    char path[4096];
    snprintf(path, sizeof path, "%s_diag.txt", prefix);
    FILE *fdiag = fopen(path, "w");
    if (!fdiag) die("cannot open diag");
    fprintf(fdiag, "# step ke maxu utau_wall mass\n");

    for (int step = 1; step <= steps; step++) {
        /* ---- collision ---- */
#pragma omp parallel for schedule(static)
        for (int y = 1; y < ny - 1; y++)
            for (int z = 0; z < nz; z++)
                for (int x = 0; x < nx; x++) {
                    int i = idx(x, y, z);
                    double rho = 0, ux = 0, uy = 0, uz = 0;
                    for (int q = 0; q < NQ; q++) {
                        double fq = f[(size_t)q * ncell + i];
                        rho += fq;
                        ux += CX[q] * fq;
                        uy += CY[q] * fq;
                        uz += CZ[q] * fq;
                    }
                    ux /= rho;
                    uy /= rho;
                    uz /= rho;

                    /* --- stress relaxation time (molecular + SGS + wall model) --- */
                    double tau;
                    if (wall_mode == 1 && (y == 1 || y == ny - 2)) {
                        double y1 = 0.5; /* distance from wall to first cell centre */
                        double ut = wall_friction_velocity(ux, y1, nu, kappa);
                        double nu_w = fabs(ut) * y1;
                        if (nu_w < 0.5 * nu) nu_w = 0.5 * nu; /* safety floor */
                        tau = 0.5 + 3.0 * (nu_w + (tau0 - 0.5) / 3.0 * 0.0);
                        if (tau < 0.505) tau = 0.505;
                    } else {
                        double feq[NQ], fneq[NQ];
                        double pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
                        for (int q = 0; q < NQ; q++) {
                            feq_q(q, rho, ux, uy, uz, &feq[q]);
                            fneq[q] = f[(size_t)q * ncell + i] - feq[q];
                            pxx += CX[q] * CX[q] * fneq[q];
                            pyy += CY[q] * CY[q] * fneq[q];
                            pzz += CZ[q] * CZ[q] * fneq[q];
                            pxy += CX[q] * CY[q] * fneq[q];
                            pxz += CX[q] * CZ[q] * fneq[q];
                            pyz += CY[q] * CZ[q] * fneq[q];
                        }
                        double p2 = pxx * pxx + pyy * pyy + pzz * pzz +
                                    2.0 * (pxy * pxy + pxz * pxz + pyz * pyz);
                        tau = tau0;
                        for (int it = 0; it < 2; it++) {
                            double smag = sqrt(fmax(p2 * 0.5, 0.0)) / fmax(CS2 * tau * rho, 1e-30);
                            double nut = cs * cs * smag;
                            tau = 0.5 + 3.0 * (nu + nut);
                        }
                    }

                    /* ---- MRT in rank-1 form (see the comment block above) ---- */
                    double fneq[NQ], feq_c[NQ], coef[6];
                    double uxf = ux + 0.5 * g / rho;
                    for (int q = 0; q < NQ; q++) {
                        feq_q(q, rho, uxf, uy, uz, &feq_c[q]);
                        fneq[q] = f[(size_t)q * ncell + i] - feq_c[q];
                    }
                    /* Rank-1 form of the same MRT update:
                     *   f* = f_eq + sum_{k<=3} A_k M_k + sum_{k in R} (1-s_k) A_k M_k
                     * with A_k = M_k . (f - f_eq) and R = {energy 4, stress 5..9}.
                     * The rows left at equilibrium (k >= 10) drop out analytically.
                     * The conserved term is kept because the Guo velocity shift
                     * makes the conserved moments of f and f_eq differ. */
                    double A[10];
                    double s_rate = 1.0 / tau;
                    for (int k = 0; k < 10; k++) {
                        int row = (k < 4) ? k : ((k == 4) ? 4 : k - 4 + 4);
                        double dot = 0;
                        for (int q = 0; q < NQ; q++) dot += M[row][q] * fneq[q];
                        if (k < 4) {
                            A[k] = dot;
                        } else {
                            double s_k = (k == 4) ? 1.2 : s_rate;
                            A[k] = (1.0 - s_k) * dot;
                        }
                    }
                    for (int q = 0; q < NQ; q++) {
                        double acc = feq_c[q];
                        for (int k = 0; k < 4; k++) acc += A[k] * M[k][q];
                        acc += A[4] * M[4][q];
                        for (int k = 5; k < 10; k++) acc += A[k] * M[k][q];
                        acc += 3.0 * W[q] * CX[q] * g;
                        fpost[(size_t)q * ncell + i] = acc;
                    }
                }

        /* ---- streaming (periodic x,z; bounce-back walls in y) ---- */
#pragma omp parallel for schedule(static)
        for (int y = 1; y < ny - 1; y++)
            for (int z = 0; z < nz; z++)
                for (int x = 0; x < nx; x++) {
                    int i = idx(x, y, z);
                    for (int q = 0; q < NQ; q++) {
                        int xs = x - CX[q], ys = y - CY[q], zs = z - CZ[q];
                        if (xs < 0) xs += nx; else if (xs >= nx) xs -= nx;
                        if (zs < 0) zs += nz; else if (zs >= nz) zs -= nz;
                        if (ys <= 0 || ys >= ny - 1) {
                            f[(size_t)q * ncell + i] = fpost[(size_t)OPP[q] * ncell + i];
                        } else {
                            f[(size_t)q * ncell + i] = fpost[(size_t)q * ncell + idx(xs, ys, zs)];
                        }
                    }
                }

        /* ---- statistics ---- */
        if (step > warmup && (step - warmup) % sample_every == 0) {
#pragma omp parallel for schedule(static)
            for (int y = 1; y < ny - 1; y++)
                for (int z = 0; z < nz; z++)
                    for (int x = 0; x < nx; x++) {
                        int i = idx(x, y, z);
                        double rho = 0, ux = 0, uy = 0, uz = 0;
                        for (int q = 0; q < NQ; q++) {
                            double fq = f[(size_t)q * ncell + i];
                            rho += fq;
                            ux += CX[q] * fq;
                            uy += CY[q] * fq;
                            uz += CZ[q] * fq;
                        }
                        ux /= rho;
                        uy /= rho;
                        sum_ux[y] += ux;
                        sum_ux2[y] += ux * ux;
                        sum_uy2[y] += uy * uy;
                        cnt[y] += 1.0;
                    }
        }

        if (step % diag_every == 0) {
            double ke = 0, umax = 0, mass = 0;
            for (int y = 1; y < ny - 1; y++)
                for (int z = 0; z < nz; z++)
                    for (int x = 0; x < nx; x++) {
                        int i = idx(x, y, z);
                        double rho = 0, ux = 0, uy = 0, uz = 0;
                        for (int q = 0; q < NQ; q++) {
                            double fq = f[(size_t)q * ncell + i];
                            rho += fq;
                            ux += CX[q] * fq;
                            uy += CY[q] * fq;
                            uz += CZ[q] * fq;
                        }
                        ux /= rho;
                        uy /= rho;
                        uz /= rho;
                        ke += 0.5 * rho * (ux * ux + uy * uy + uz * uz);
                        mass += rho;
                        double un = sqrt(ux * ux + uy * uy + uz * uz);
                        if (un > umax) umax = un;
                    }
            /* wall-model friction velocity at the lower wall */
            double ut = 0.0;
            if (wall_mode == 1) {
                double acc = 0;
                for (int z = 0; z < nz; z++)
                    for (int x = 0; x < nx; x++) {
                        int i = idx(x, 1, z);
                        double rho = 0, ux = 0;
                        for (int q = 0; q < NQ; q++) {
                            double fq = f[(size_t)q * ncell + i];
                            rho += fq;
                            ux += CX[q] * fq;
                        }
                        acc += ux / rho;
                    }
                ut = wall_friction_velocity(acc / (nx * nz), 0.5, nu, kappa);
            }
            fprintf(fdiag, "%d %.12e %.12e %.12e %.12e\n", step, ke, umax, ut, mass);
            fflush(fdiag);
        }
        if (step % 2000 == 0) {
            fprintf(stderr, "step %d/%d\n", step, steps);
            fflush(stderr);
        }
    }
    fclose(fdiag);

    snprintf(path, sizeof path, "%s_profile.txt", prefix);
    FILE *fp = fopen(path, "w");
    if (!fp) die("cannot open profile");
    fprintf(fp, "# y u_mean u_rms uy_rms count\n");
    for (int y = 1; y < ny - 1; y++) {
        double c = cnt[y] > 0 ? cnt[y] : 1.0;
        double um = sum_ux[y] / c;
        double vr = sum_ux2[y] / c - um * um;
        if (vr < 0) vr = 0;
        double vv = sum_uy2[y] / c;
        fprintf(fp, "%d %.10e %.10e %.10e %.0f\n", y, um, sqrt(vr), sqrt(vv), cnt[y]);
    }
    fclose(fp);

    snprintf(path, sizeof path, "%s_meta.txt", prefix);
    FILE *fm = fopen(path, "w");
    fprintf(fm, "nx=%d\nny=%d\nnz=%d\nsteps=%d\nwarmup=%d\nsample_every=%d\ntau=%.12g\n"
                "nu=%.12g\ncs=%.12g\nforce=%.12e\nh=%.12g\nu_tau_target=%.12e\n"
                "wall_mode=%d\nkappa=%.12g\nmatrix=%s\n",
            nx, ny, nz, steps, warmup, sample_every, tau0, nu, cs, g, h, u_tau_target,
            wall_mode, kappa, matpath);
#ifdef _OPENMP
    fprintf(fm, "omp_threads=%d\n", omp_get_max_threads());
#endif
    fclose(fm);
    return 0;
}
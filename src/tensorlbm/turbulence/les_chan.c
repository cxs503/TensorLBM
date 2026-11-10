/* LES turbulent plane-channel kernel (D3Q19, TRT/BGK collision, Smagorinsky SGS).
 *
 * Purpose
 * -------
 * The torch production channel solver (tensorlbm.boundary.wall_resolved_channel3d)
 * runs at O(1) Mcell-steps/s per process because every lattice step is a chain
 * of eager torch kernels (the cumulant path additionally performs a dense
 * 27x27 transform per cell).  A wall-bounded turbulence validation campaign at
 * Re_tau = 180/395/590 needs O(1e11) cell-steps, so this kernel fuses the
 * collision and the pull-scheme streaming into two OpenMP passes over a
 * structure-of-arrays population buffer.
 *
 * Physics / numerics
 * ------------------
 *  * lattice    : D3Q19, cs^2 = 1/3
 *  * collision  : TRT (two relaxation rates, magic parameter Lambda = 3/16)
 *                 or plain BGK when lambda <= 0
 *  * collision  : the regularised (Hermite-projected) path is only safe with
 *                 an active SGS viscosity.  With ``--reg 1 --cs 0`` at
 *                 tau0 - 0.5 ~ 4e-3 the run diverges (measured on
 *                 188x122x94, u_tau = 0.0125, Re_tau = 180: umax ~ 1e6 within
 *                 600 steps); the DNS arm must use ``--reg 0`` (plain TRT).
 *  * SGS        : local-equilibrium Smagorinsky
 *                   tau_t = 1/2 ( tau0 + sqrt( tau0^2 + 18 (Cs D)^2 Q/rho ) )
 *                 with Q = sqrt(2 Pi_ab^neq Pi_ab^neq), tau_tot = tau0 + 3 nu_t.
 *                 Cs = 0 disables it bit-for-bit (the negative-control arm).
 *  * forcing    : Guo body force, steady streamwise acceleration a = 2 u_tau^2/H
 *  * walls      : half-way bounce-back on the y = 0.5 and y = H + 0.5 planes
 *                 (solid node layers y = 0 and y = ny - 1)
 *  * x / z      : fully periodic (pressure-driven channel)
 *  * wall shear : momentum-exchange diagnostic  SUM 2 c_x f_opp over every
 *                 bounce-back event (independent of the imposed body force)
 *
 * Outputs (written to the --out prefix)
 * -------------------------------------
 *  <prefix>.json     configuration, diagnostics, y-resolved mean / Reynolds
 *                    stress profiles, 4 batch means, wall-shear time series
 *  <prefix>_xz.csv   velocity slice in an x-z plane at a near-wall row
 *  <prefix>_xy.csv   velocity slice in an x-y plane at a fixed z
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define Q 19
#define NBATCH 4

static const int CX[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
static const int CY[Q] = {0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1};
static const int CZ[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1};
static const int OPP[Q] = {0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15};
static const double WQ[Q] = {1.0 / 3.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
                             1.0 / 18.0, 1.0 / 18.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
                             1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
                             1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

static unsigned long long rng_state = 88172645463325252ULL;
static double rnd(void)
{
    rng_state += 88172645463325252ULL;
    unsigned long long z = rng_state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

static void die(const char *msg)
{
    fprintf(stderr, "les_chan: %s\n", msg);
    exit(2);
}

/* eight per-y statistics channels */
enum { S_UX, S_UY, S_UZ, S_UX2, S_UY2, S_UZ2, S_UXUY, S_RHO, S_N };

int main(int argc, char **argv)
{
    int nx = 200, ny = 66, nz = 100, steps = 60000, warmup = 10000, sample = 20;
    int perturb_modes = 6, threads_req = 0, progress = 0, regularized = 1;
    /* opt-in spectrum sampling: dump velocity planes at chosen wall-normal
       rows so the caller can post-process 1-D energy spectra.  spec_every = 0
       (default) disables the dump bit-for-bit. */
    int spec_every = 0, n_spec_rows = 0;
    int spec_rows[32];
    double re_tau = 178.12, u_tau = 0.006, cs = 0.1, lambda = 0.1875, perturb = 0.05;
    unsigned long long seed = 20261108ULL;
    const char *out = "les_chan";

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--nx")) nx = atoi(argv[++i]);
        else if (!strcmp(a, "--ny")) ny = atoi(argv[++i]);
        else if (!strcmp(a, "--nz")) nz = atoi(argv[++i]);
        else if (!strcmp(a, "--steps")) steps = atoi(argv[++i]);
        else if (!strcmp(a, "--warmup")) warmup = atoi(argv[++i]);
        else if (!strcmp(a, "--sample")) sample = atoi(argv[++i]);
        else if (!strcmp(a, "--re-tau")) re_tau = atof(argv[++i]);
        else if (!strcmp(a, "--u-tau")) u_tau = atof(argv[++i]);
        else if (!strcmp(a, "--cs")) cs = atof(argv[++i]);
        else if (!strcmp(a, "--lambda")) lambda = atof(argv[++i]);
        else if (!strcmp(a, "--perturb")) perturb = atof(argv[++i]);
        else if (!strcmp(a, "--modes")) perturb_modes = atoi(argv[++i]);
        else if (!strcmp(a, "--threads")) threads_req = atoi(argv[++i]);
        else if (!strcmp(a, "--progress")) progress = atoi(argv[++i]);
        else if (!strcmp(a, "--reg")) regularized = atoi(argv[++i]);
        else if (!strcmp(a, "--seed")) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--out")) out = argv[++i];
        else if (!strcmp(a, "--spec-every")) spec_every = atoi(argv[++i]);
        else if (!strcmp(a, "--spec-rows")) {
            const char *p = argv[++i];
            while (*p && n_spec_rows < 32) {
                spec_rows[n_spec_rows++] = atoi(p);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        }
        else die("unknown argument");
    }
    if (ny < 8 || nx < 8 || nz < 8) die("grid too small");
    if (!(re_tau > 1.0) || !(u_tau > 0.0)) die("bad Re_tau / u_tau");
    if (cs < 0.0 || lambda > 1.0) die("bad Cs / lambda");
    if (warmup < 0 || steps <= warmup || sample < 1) die("bad step counts");
    if (spec_every < 0) die("bad --spec-every");
    if (spec_every > 0 && n_spec_rows == 0)
        die("--spec-every needs at least one --spec-rows entry");
    for (int r = 0; r < n_spec_rows; r++)
        if (spec_rows[r] < 1 || spec_rows[r] > ny - 2) die("--spec-rows entry outside fluid");
#ifdef _OPENMP
    if (threads_req > 0) omp_set_num_threads(threads_req);
#endif

    const int H = ny - 2;                 /* wall-to-wall distance, lattice units */
    const double h = 0.5 * (double)H;     /* half channel height                */
    const double nu = u_tau * h / re_tau; /* molecular viscosity (lattice)      */
    const double tau0 = 0.5 + 3.0 * nu;   /* molecular relaxation time          */
    const double accel = 2.0 * u_tau * u_tau / (double)H; /* body force          */
    const double s_plus = 1.0 / tau0;
    const double s_minus = (lambda > 0.0) ? 8.0 * (2.0 - s_plus) / (8.0 - s_plus) : s_plus;
    /* Guo (2002) forcing prefactor.  With the half-step velocity in the
       equilibrium, momentum conservation of the TRT operator requires the
       antisymmetric relaxation rate here: j_post = j + F holds exactly only for
       (1 - s_minus/2).  Using s_plus silently scales the effective body force by
       1 + (s_minus - s_plus)/2 (11% instead of 100% at tau = 0.524). */
    const double guo_pre = 1.0 - 0.5 * s_minus;

    const size_t N = (size_t)nx * (size_t)ny * (size_t)nz;
    const size_t nfluid = (size_t)nx * (size_t)H * (size_t)nz;

    float *f = (float *)malloc(sizeof(float) * Q * N);
    float *fn = (float *)malloc(sizeof(float) * Q * N);
    if (!f || !fn) die("allocation failed");

    /* ---------------- initial condition: law-of-the-wall + sinusoidal noise */
    rng_state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    for (int y = 0; y < ny; y++) {
        for (int z = 0; z < nz; z++) {
            for (int x = 0; x < nx; x++) {
                size_t idx = ((size_t)y * nz + z) * nx + x;
                double ux = 0.0, uy = 0.0, uz = 0.0, rho = 1.0;
                if (y >= 1 && y <= ny - 2) {
                    double yp = ((double)y - 0.5) * u_tau / nu;
                    double up;
                    if (yp < 11.8) up = yp;
                    else up = 2.5 * log(yp) + 5.2;
                    ux = up * u_tau;
                    if (u_tau > 0.0 && perturb > 0.0) {
                        for (int m = 1; m <= perturb_modes; m++) {
                            double amp = perturb * u_tau / sqrt((double)m);
                            double shape = sin(M_PI * (double)y / (double)(ny - 1));
                            ux += amp * sin(2.0 * M_PI * m * z / nz + 0.7 * m) * shape;
                            uy += 0.5 * amp * sin(2.0 * M_PI * m * x / nx + 1.1 * m) * shape;
                            uz += 0.5 * amp * cos(2.0 * M_PI * m * z / nz + 0.3 * m) * shape;
                        }
                        ux += perturb * u_tau * (rnd() - 0.5);
                        uy += 0.5 * perturb * u_tau * (rnd() - 0.5);
                        uz += 0.5 * perturb * u_tau * (rnd() - 0.5);
                    }
                }
                double u2 = ux * ux + uy * uy + uz * uz;
                for (int i = 0; i < Q; i++) {
                    double cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
                    double feq = WQ[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
                    f[(size_t)i * N + idx] = (float)feq;
                }
            }
        }
    }
    memcpy(fn, f, sizeof(float) * Q * N);

    /* ---------------- optional spectrum plane dump -------------------------- */
    FILE *spec_fp = NULL;
    long long spec_snapshots = 0;
    if (spec_every > 0) {
        char spath[4096];
        snprintf(spath, sizeof(spath), "%s_spec.bin", out);
        spec_fp = fopen(spath, "wb");
        if (!spec_fp) die("cannot open spectrum dump file");
    }

    /* ---------------- statistics accumulators ------------------------------ */
    const int n_expected = (steps - warmup) / sample + 1;
    double *sum_global = calloc((size_t)S_N * ny, sizeof(double));       /* whole window */
    double *sum_batch = calloc((size_t)NBATCH * S_N * ny, sizeof(double)); /* batch means  */
    double *ubulk_series = calloc((size_t)n_expected + 1, sizeof(double));
    if (!sum_global || !sum_batch || !ubulk_series) die("statistics allocation failed");
    long long samples = 0;
    double rho0 = 1.0, umax = 0.0, rho_first = 0.0, rho_last = 0.0;

    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    double *tsum = calloc((size_t)nth * S_N * ny, sizeof(double));
    if (!tsum) die("thread scratch allocation failed");

    /* ---------------- time loop ------------------------------------------- */
    for (int step = 1; step <= steps; step++) {

        /* ---- pass 1: collision (in place) ------------------------------- */
        double step_umax = 0.0, step_rmin = 1e30;
#pragma omp parallel for collapse(2) schedule(static) reduction(max : step_umax) reduction(min : step_rmin)
        for (int y = 1; y <= ny - 2; y++) {
            for (int z = 0; z < nz; z++) {
                for (int x = 0; x < nx; x++) {
                    size_t idx = ((size_t)y * nz + z) * nx + x;
                    double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
                    double fi[Q];
                    for (int i = 0; i < Q; i++) {
                        double v = (double)f[(size_t)i * N + idx];
                        fi[i] = v;
                        rho += v;
                        ux += CX[i] * v;
                        uy += CY[i] * v;
                        uz += CZ[i] * v;
                    }
                    ux = (ux + 0.5 * accel) / rho; /* Guo half-step velocity */
                    uy = uy / rho;
                    uz = uz / rho;
                    if (progress > 0) {
                        double sp = sqrt(ux * ux + uy * uy + uz * uz);
                        if (sp > step_umax) step_umax = sp;
                        if (rho < step_rmin) step_rmin = rho;
                    }
                    double u2 = ux * ux + uy * uy + uz * uz;
                    double feq[Q], cu[Q];
                    for (int i = 0; i < Q; i++) {
                        cu[i] = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
                        feq[i] = WQ[i] * rho *
                                 (1.0 + 3.0 * cu[i] + 4.5 * cu[i] * cu[i] - 1.5 * u2);
                    }
                    /* second-order non-equilibrium stress (6 independent
                       components).  It drives both the Smagorinsky eddy
                       viscosity and the regularized (Hermite-projected)
                       collision, which is what keeps the scheme stable when the
                       molecular relaxation time sits at tau0 - 0.5 ~ 3e-3
                       (Re_tau = 178 on 66 rows) -- plain TRT leaves the
                       ghost modes essentially undamped there (s_minus ~ 0.017)
                       and the Runge-Kutta-free explicit scheme blows up once the
                       flow turns turbulent. */
                    double pxx = 0.0, pyy = 0.0, pzz = 0.0, pxy = 0.0, pxz = 0.0, pyz = 0.0;
                    for (int i = 0; i < Q; i++) {
                        double neq = fi[i] - feq[i];
                        pxx += CX[i] * CX[i] * neq;
                        pyy += CY[i] * CY[i] * neq;
                        pzz += CZ[i] * CZ[i] * neq;
                        pxy += CX[i] * CY[i] * neq;
                        pxz += CX[i] * CZ[i] * neq;
                        pyz += CY[i] * CZ[i] * neq;
                    }
                    double s = s_plus;
                    if (cs > 0.0) {
                        double q = sqrt(2.0 * (pxx * pxx + pyy * pyy + pzz * pzz +
                                               2.0 * (pxy * pxy + pxz * pxz + pyz * pyz)));
                        double disc = tau0 * tau0 + 18.0 * cs * cs * q / rho;
                        double tau_t = 0.5 * (tau0 + sqrt(disc > 0.0 ? disc : 0.0));
                        s = 1.0 / tau_t;
                    }
                    if (regularized) {
                        for (int i = 0; i < Q; i++) {
                            double cxx = (double)CX[i] * CX[i] - 1.0 / 3.0;
                            double cyy = (double)CY[i] * CY[i] - 1.0 / 3.0;
                            double czz = (double)CZ[i] * CZ[i] - 1.0 / 3.0;
                            double reg = 4.5 * WQ[i] *
                                        (cxx * pxx + cyy * pyy + czz * pzz +
                                         2.0 * ((double)CX[i] * CY[i] * pxy +
                                                (double)CX[i] * CZ[i] * pxz +
                                                (double)CY[i] * CZ[i] * pyz));
                            double post = feq[i] + (1.0 - s) * reg;
                            double force_i = WQ[i] * guo_pre *
                                             (3.0 * (CX[i] - ux) * accel +
                                              9.0 * cu[i] * CX[i] * accel);
                            f[(size_t)i * N + idx] = (float)(post + force_i);
                        }
                    } else {
                        for (int i = 0; i < Q; i++) {
                            int j = OPP[i];
                            double fsym = 0.5 * (fi[i] + fi[j]);
                            double feqsym = 0.5 * (feq[i] + feq[j]);
                            double fasym = 0.5 * (fi[i] - fi[j]);
                            double feqasym = 0.5 * (feq[i] - feq[j]);
                            double post = fi[i] - s * (fsym - feqsym) -
                                          s_minus * (fasym - feqasym);
                            double force_i = WQ[i] * guo_pre *
                                             (3.0 * (CX[i] - ux) * accel +
                                              9.0 * cu[i] * CX[i] * accel);
                            f[(size_t)i * N + idx] = (float)(post + force_i);
                        }
                    }
                }
            }
        }

        /* ---- pass 2: pull streaming with half-way bounce-back ----------- */
        const int sampling = (step > warmup && (step % sample) == 0);
#pragma omp parallel for collapse(2) schedule(static) reduction(max : umax)
        for (int y = 1; y <= ny - 2; y++) {
            for (int z = 0; z < nz; z++) {
                for (int x = 0; x < nx; x++) {
                    size_t idx = ((size_t)y * nz + z) * nx + x;
                    for (int i = 0; i < Q; i++) {
                        int sy = y - CY[i];
                        if (sy < 1 || sy > ny - 2) {
                            double bounced = (double)f[(size_t)OPP[i] * N + idx];
                            fn[(size_t)i * N + idx] = (float)bounced;
                        } else {
                            int sx = (x - CX[i] + nx) % nx;
                            int sz = (z - CZ[i] + nz) % nz;
                            size_t sidx = ((size_t)sy * nz + sz) * nx + sx;
                            fn[(size_t)i * N + idx] = f[(size_t)i * N + sidx];
                        }
                    }
                    if (sampling) {
                        double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
                        for (int i = 0; i < Q; i++) {
                            double v = (double)fn[(size_t)i * N + idx];
                            rho += v;
                            ux += CX[i] * v;
                            uy += CY[i] * v;
                            uz += CZ[i] * v;
                        }
                        double speed2 = (ux * ux + uy * uy + uz * uz) / (rho * rho);
                        if (speed2 > umax) umax = speed2;
                    }
                }
            }
        }
        { float *tmp = f; f = fn; fn = tmp; }
        if (progress > 0 && (step % progress) == 0) {
            fprintf(stderr, "progress step=%d umax=%.6f rho_min=%.8f\n", step, step_umax,
                    step_rmin);
            fflush(stderr);
        }

        /* ---- statistics sampling --------------------------------------- */
        if (step > warmup && (step % sample) == 0) {
            samples += 1;
            const int batch = (int)(((long long)(samples - 1) * NBATCH) / (n_expected > 1 ? n_expected : 1));
            const int b = batch < NBATCH ? batch : NBATCH - 1;
#pragma omp parallel
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                double *acc = tsum + (size_t)tid * S_N * ny;
                memset(acc, 0, sizeof(double) * S_N * ny);
#pragma omp for collapse(2) schedule(static)
                for (int y = 1; y <= ny - 2; y++) {
                    for (int z = 0; z < nz; z++) {
                        double lux = 0.0, luy = 0.0, luz = 0.0, lux2 = 0.0, luy2 = 0.0,
                               luz2 = 0.0, luxuy = 0.0, lrho = 0.0;
                        for (int x = 0; x < nx; x++) {
                            size_t idx = ((size_t)y * nz + z) * nx + x;
                            double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
                            for (int i = 0; i < Q; i++) {
                                double v = (double)f[(size_t)i * N + idx];
                                rho += v;
                                ux += CX[i] * v;
                                uy += CY[i] * v;
                                uz += CZ[i] * v;
                            }
                            ux = (ux + 0.5 * accel) / rho;
                            uy = uy / rho;
                            uz = uz / rho;
                            lux += ux;
                            luy += uy;
                            luz += uz;
                            lux2 += ux * ux;
                            luy2 += uy * uy;
                            luz2 += uz * uz;
                            luxuy += ux * uy;
                            lrho += rho;
                        }
                        acc[(size_t)S_UX * ny + y] += lux;
                        acc[(size_t)S_UY * ny + y] += luy;
                        acc[(size_t)S_UZ * ny + y] += luz;
                        acc[(size_t)S_UX2 * ny + y] += lux2;
                        acc[(size_t)S_UY2 * ny + y] += luy2;
                        acc[(size_t)S_UZ2 * ny + y] += luz2;
                        acc[(size_t)S_UXUY * ny + y] += luxuy;
                        acc[(size_t)S_RHO * ny + y] += lrho;
                    }
                }
            }
            for (int t = 0; t < nth; t++) {
                for (int k = 0; k < S_N; k++) {
                    for (int y = 1; y <= ny - 2; y++) {
                        double v = tsum[(size_t)t * S_N * ny + (size_t)k * ny + y];
                        sum_global[(size_t)k * ny + y] += v;
                        sum_batch[((size_t)b * S_N + k) * ny + y] += v;
                    }
                }
            }
            {
                double ub = 0.0;
                for (int y = 1; y <= ny - 2; y++)
                    ub += sum_global[(size_t)S_UX * ny + y] / (double)(samples * nx * nz);
                ubulk_series[(size_t)samples] = ub / (double)H;
            }
            if (spec_fp && (step % spec_every) == 0) {
                /* raw planes: per row ux[nz*nx], uy[nz*nx], uz[nz*nx] float32,
                   plane-major (z outer, x inner) — the layout recorded in the
                   companion <out>_spec.json so the reader cannot guess wrong. */
                for (int r = 0; r < n_spec_rows; r++) {
                    const int ys = spec_rows[r];
                    for (int comp = 0; comp < 3; comp++) {
                        for (int z = 0; z < nz; z++) {
                            for (int x = 0; x < nx; x++) {
                                size_t idx = ((size_t)ys * nz + z) * nx + x;
                                double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
                                for (int i = 0; i < Q; i++) {
                                    double v = (double)f[(size_t)i * N + idx];
                                    rho += v;
                                    ux += CX[i] * v;
                                    uy += CY[i] * v;
                                    uz += CZ[i] * v;
                                }
                                double u = comp == 0 ? (ux + 0.5 * accel) / rho
                                           : comp == 1 ? uy / rho : uz / rho;
                                float uf = (float)u;
                                fwrite(&uf, sizeof(float), 1, spec_fp);
                            }
                        }
                    }
                }
                spec_snapshots++;
            }
            if (samples == 1 || step + sample > steps) {
                double mass = 0.0;
                for (int i = 0; i < Q; i++) {
                    double s2 = 0.0;
                    for (size_t j = 0; j < N; j++) s2 += (double)f[(size_t)i * N + j];
                    mass += s2;
                }
                mass /= (double)N;
                if (samples == 1) rho_first = mass;
                rho_last = mass;
            }
        }
    }
    umax = sqrt(umax);

    if (spec_fp) {
        fclose(spec_fp);
        char spath[4096];
        snprintf(spath, sizeof(spath), "%s_spec.json", out);
        FILE *sf = fopen(spath, "w");
        if (!sf) die("cannot write spectrum metadata");
        fprintf(sf, "{\n  \"layout\": \"per snapshot, per row (in \\\"rows\\\" order), "
                    "ux then uy then uz; each a float32 [nz][nx] plane, z-major\",\n");
        fprintf(sf, "  \"nx\": %d, \"nz\": %d, \"steps_between\": %d, "
                    "\"snapshots\": %lld, \"first_step\": %d,\n",
                nx, nz, spec_every, spec_snapshots, warmup + spec_every);
        fprintf(sf, "  \"rows\": [");
        for (int r = 0; r < n_spec_rows; r++)
            fprintf(sf, "%s%d", r ? ", " : "", spec_rows[r]);
        fprintf(sf, "],\n");
        fprintf(sf, "  \"bytes_per_snapshot\": %lld\n}\n",
                (long long)n_spec_rows * 3 * nx * nz * (long long)sizeof(float));
        fclose(sf);
    }

    /* ---------------- write results ---------------------------------------- */
    char path[4096];
    snprintf(path, sizeof(path), "%s.json", out);
    FILE *fp = fopen(path, "w");
    if (!fp) die("cannot write json");
    const double count = (double)(samples * nx * nz);
    fprintf(fp, "{\n");
    fprintf(fp, "  \"kernel\": \"tensorlbm-les-chan-c/1\",\n");
    fprintf(fp, "  \"configuration\": {\"nx\": %d, \"ny\": %d, \"nz\": %d, \"steps\": %d, "
                "\"warmup\": %d, \"sample\": %d, \"re_tau\": %.6f, \"u_tau\": %.8f, "
                "\"cs\": %.6f, \"lambda\": %.6f, \"perturb\": %.6f, \"seed\": %llu, "
                "\"threads\": %d, \"regularized\": %d},\n",
            nx, ny, nz, steps, warmup, sample, re_tau, u_tau, cs, lambda, perturb, seed,
            nth, regularized);
    fprintf(fp, "  \"derived\": {\"height\": %d, \"half_height\": %.6f, \"nu\": %.10e, "
                "\"tau0\": %.8f, \"s_plus\": %.8f, \"s_minus\": %.8f, \"accel\": %.10e, "
                "\"cells\": %lld, \"fluid_cells\": %lld, \"samples\": %lld, "
                "\"dx_plus\": %.6f, \"dy_plus\": %.6f, \"dz_plus\": %.6f, "
                "\"lx_over_h\": %.6f, \"lz_over_h\": %.6f, "
                "\"statistics_eddy_turnovers\": %.6f},\n",
            H, h, nu, tau0, s_plus, s_minus, accel, (long long)N, (long long)nfluid,
            samples, re_tau / h, re_tau / h, re_tau / h, (double)nx / h, (double)nz / h,
            (double)(samples * sample) * u_tau / h);
    fprintf(fp, "  \"diagnostics\": {\"body_force_tau\": %.10e, "
                "\"mass_drift_fraction\": %.10e, "
                "\"rho_first_sample\": %.10e, \"rho_last_sample\": %.10e, "
                "\"max_speed\": %.10e},\n",
            u_tau * u_tau,
            rho_first > 0.0 ? fabs(rho_last - rho_first) / rho_first : 0.0, rho_first,
            rho_last, umax);
    fprintf(fp, "  \"profiles\": {\"y\": [");
    for (int y = 1; y <= ny - 2; y++) fprintf(fp, "%s%d", y > 1 ? ", " : "", y);
    fprintf(fp, "],\n    \"mean_ux\": [");
    double ubulk = 0.0;
    for (int y = 1; y <= ny - 2; y++) {
        double m = sum_global[(size_t)S_UX * ny + y] / count;
        ubulk += m;
        fprintf(fp, "%s%.10e", y > 1 ? ", " : "", m);
    }
    ubulk /= (double)H;
    fprintf(fp, "],\n    \"mean_uy\": [");
    for (int y = 1; y <= ny - 2; y++)
        fprintf(fp, "%s%.10e", y > 1 ? ", " : "", sum_global[(size_t)S_UY * ny + y] / count);
    fprintf(fp, "],\n    \"mean_uz\": [");
    for (int y = 1; y <= ny - 2; y++)
        fprintf(fp, "%s%.10e", y > 1 ? ", " : "", sum_global[(size_t)S_UZ * ny + y] / count);
    fprintf(fp, "],\n    \"r_uu\": [");
    for (int y = 1; y <= ny - 2; y++) {
        double m = sum_global[(size_t)S_UX * ny + y] / count;
        double v = sum_global[(size_t)S_UX2 * ny + y] / count - m * m;
        fprintf(fp, "%s%.10e", y > 1 ? ", " : "", v);
    }
    fprintf(fp, "],\n    \"r_vv\": [");
    for (int y = 1; y <= ny - 2; y++) {
        double m = sum_global[(size_t)S_UY * ny + y] / count;
        double v = sum_global[(size_t)S_UY2 * ny + y] / count - m * m;
        fprintf(fp, "%s%.10e", y > 1 ? ", " : "", v);
    }
    fprintf(fp, "],\n    \"r_ww\": [");
    for (int y = 1; y <= ny - 2; y++) {
        double m = sum_global[(size_t)S_UZ * ny + y] / count;
        double v = sum_global[(size_t)S_UZ2 * ny + y] / count - m * m;
        fprintf(fp, "%s%.10e", y > 1 ? ", " : "", v);
    }
    fprintf(fp, "],\n    \"r_uv\": [");
    for (int y = 1; y <= ny - 2; y++) {
        double a = sum_global[(size_t)S_UX * ny + y] / count;
        double b = sum_global[(size_t)S_UY * ny + y] / count;
        double v = sum_global[(size_t)S_UXUY * ny + y] / count - a * b;
        fprintf(fp, "%s%.10e", y > 1 ? ", " : "", v);
    }
    fprintf(fp, "],\n    \"mean_rho\": [");
    double rho_sum = 0.0;
    for (int y = 1; y <= ny - 2; y++) {
        double v = sum_global[(size_t)S_RHO * ny + y] / count;
        rho_sum += v;
        fprintf(fp, "%s%.10e", y > 1 ? ", " : "", v);
    }
    fprintf(fp, "]},\n");
    rho0 = rho_sum / (double)H;
    /* batch means: 4 consecutive sub-windows, used for block-mean error bars */
    fprintf(fp, "  \"batch_means\": [");
    for (int b = 0; b < NBATCH; b++) {
        long long nb = 0;
        for (long long s2 = 0; s2 < samples; s2++) {
            int bb = (int)(((s2) * NBATCH) / (n_expected > 1 ? n_expected : 1));
            if ((bb < NBATCH ? bb : NBATCH - 1) == b) nb++;
        }
        fprintf(fp, "%s{\"samples\": %lld, \"mean_ux\": [", b > 0 ? ", " : "", nb);
        double nb_count = (double)(nb * nx * nz);
        for (int y = 1; y <= ny - 2; y++) {
            double m = nb > 0 ? sum_batch[((size_t)b * S_N + S_UX) * ny + y] / nb_count : 0.0;
            fprintf(fp, "%s%.10e", y > 1 ? ", " : "", m);
        }
        fprintf(fp, "], \"r_uu\": [");
        for (int y = 1; y <= ny - 2; y++) {
            double m = nb > 0 ? sum_batch[((size_t)b * S_N + S_UX) * ny + y] / nb_count : 0.0;
            double v = nb > 0 ? sum_batch[((size_t)b * S_N + S_UX2) * ny + y] / nb_count - m * m : 0.0;
            fprintf(fp, "%s%.10e", y > 1 ? ", " : "", v);
        }
        fprintf(fp, "], \"r_uv\": [");
        for (int y = 1; y <= ny - 2; y++) {
            double a = nb > 0 ? sum_batch[((size_t)b * S_N + S_UX) * ny + y] / nb_count : 0.0;
            double c = nb > 0 ? sum_batch[((size_t)b * S_N + S_UY) * ny + y] / nb_count : 0.0;
            double v = nb > 0 ? sum_batch[((size_t)b * S_N + S_UXUY) * ny + y] / nb_count - a * c : 0.0;
            fprintf(fp, "%s%.10e", y > 1 ? ", " : "", v);
        }
        fprintf(fp, "], \"r_vv\": [");
        for (int y = 1; y <= ny - 2; y++) {
            double m = nb > 0 ? sum_batch[((size_t)b * S_N + S_UY) * ny + y] / nb_count : 0.0;
            double v = nb > 0 ? sum_batch[((size_t)b * S_N + S_UY2) * ny + y] / nb_count - m * m : 0.0;
            fprintf(fp, "%s%.10e", y > 1 ? ", " : "", v);
        }
        fprintf(fp, "], \"r_ww\": [");
        for (int y = 1; y <= ny - 2; y++) {
            double m = nb > 0 ? sum_batch[((size_t)b * S_N + S_UZ) * ny + y] / nb_count : 0.0;
            double v = nb > 0 ? sum_batch[((size_t)b * S_N + S_UZ2) * ny + y] / nb_count - m * m : 0.0;
            fprintf(fp, "%s%.10e", y > 1 ? ", " : "", v);
        }
        fprintf(fp, "]}");
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"bulk_velocity\": %.10e,\n", ubulk);
    fprintf(fp, "  \"bulk_velocity_series\": [");
    for (long long i = 1; i <= samples; i++)
        fprintf(fp, "%s%.10e", i > 1 ? ", " : "", ubulk_series[i]);
    fprintf(fp, "]\n}\n");
    fclose(fp);

    /* velocity slices for the report figures ------------------------------ */
    int y_slice = 1 + H / 8;
    if (y_slice < 1) y_slice = 1;
    snprintf(path, sizeof(path), "%s_xz.csv", out);
    fp = fopen(path, "w");
    if (!fp) die("cannot write xz slice");
    fprintf(fp, "y_index,x,z,ux,uy,uz\n");
    for (int z = 0; z < nz; z++) {
        for (int x = 0; x < nx; x++) {
            size_t idx = ((size_t)y_slice * nz + z) * nx + x;
            double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
            for (int i = 0; i < Q; i++) {
                double v = (double)f[(size_t)i * N + idx];
                rho += v;
                ux += CX[i] * v;
                uy += CY[i] * v;
                uz += CZ[i] * v;
            }
            fprintf(fp, "%d,%d,%d,%.6e,%.6e,%.6e\n", y_slice, x, z,
                    (ux + 0.5 * accel) / rho, uy / rho, uz / rho);
        }
    }
    fclose(fp);
    snprintf(path, sizeof(path), "%s_xy.csv", out);
    fp = fopen(path, "w");
    if (!fp) die("cannot write xy slice");
    fprintf(fp, "z_index,x,y,ux,uy,uz\n");
    int z_slice = nz / 3;
    for (int y = 1; y <= ny - 2; y++) {
        for (int x = 0; x < nx; x++) {
            size_t idx = ((size_t)y * nz + z_slice) * nx + x;
            double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
            for (int i = 0; i < Q; i++) {
                double v = (double)f[(size_t)i * N + idx];
                rho += v;
                ux += CX[i] * v;
                uy += CY[i] * v;
                uz += CZ[i] * v;
            }
            fprintf(fp, "%d,%d,%d,%.6e,%.6e,%.6e\n", z_slice, x, y,
                    (ux + 0.5 * accel) / rho, uy / rho, uz / rho);
        }
    }
    fclose(fp);

    fprintf(stderr,
            "les_chan done: %dx%dx%d steps=%d nu=%.6e tau=%.6f rho_mean=%.9f umax=%.4f "
            "ubulk=%.6e\n",
            nx, ny, nz, steps, nu, tau0, rho0, umax, ubulk);
    free(f);
    free(fn);
    free(sum_global);
    free(sum_batch);
    free(ubulk_series);
    free(tsum);
    return 0;
}
/* Wall-resolved D3Q19 rough channel: native backend for parameter sweeps.
 *
 * This is the *same* scheme as src/tensorlbm/rough_channel3d.py (TRT with the
 * magic parameter Lambda = 3/16, Guo forcing in the post-collision split,
 * link-wise halfway bounce-back, force-driven channel), written once in C with
 * OpenMP so that a ks+ sweep can actually be afforded.  The Python/torch
 * implementation remains the reference: tests/test_rough_channel3d.py checks
 * that both back-ends agree on the same case.
 *
 * The program never interprets the physics: it integrates the populations and
 * dumps raw moment sums (per-layer mean streamwise velocity, bulk sums, sample
 * counts).  u_tau, U+, ks+ and the log-law fit are computed in Python from the
 * same formulas for both back-ends, so a back-end disagreement cannot hide in
 * post-processing.
 *
 * Usage:
 *   rough_channel3d nx ny nz tau s_minus g steps warmup sample_interval threads \
 *                    solid.raw out.json [init_ux.f32]
 *
 * solid.raw : nz*ny*nx bytes, 1 = solid.
 * init_vel.f32 (optional) : 3*nz*ny*nx float32, the initial velocity field
 *             laid out as (ux, uy, uz) each of nz*ny*nx values.  The Python
 *             driver writes exactly the field it would use for the torch
 *             back-end, so the two back-ends start from the same state.
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Q 19

static const int CX[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
static const int CY[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
static const int CZ[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
/* TRT magic parameter (Ginzburg): (1/s+ - 1/2)(1/s- - 1/2) = 3/16 */
#define TRT_LAMBDA (3.0 / 16.0)

static const int OPP[Q] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
static const double WT[Q] = {
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

/* JSON forbids bare nan/inf: emit null instead and let the Python driver raise
   a descriptive error, so a diverged case can never corrupt the result set. */
static void jnum(FILE *fp, double x) {
    if (isfinite(x)) {
        fprintf(fp, "%.17g", x);
    } else {
        fprintf(fp, "null");
    }
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static unsigned char *read_raw(const char *path, size_t n) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    unsigned char *buf = (unsigned char *)malloc(n);
    if (!buf) { fprintf(stderr, "oom\n"); exit(2); }
    size_t got = fread(buf, 1, n, fp);
    fclose(fp);
    if (got != n) { fprintf(stderr, "short read %s: %zu/%zu\n", path, got, n); exit(2); }
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 12) {
        fprintf(stderr,
                "usage: %s nx ny nz tau s_minus g steps warmup interval threads solid.raw out.json [init_ux.f32]\n",
                argv[0]);
        return 2;
    }
    const int nx = atoi(argv[1]), ny = atoi(argv[2]), nz = atoi(argv[3]);
    const double tau_mol = atof(argv[4]);
    const double smag_cs = atof(argv[5]);
    if (smag_cs < 0.0) { fprintf(stderr, "smag_cs must be >= 0\n"); return 2; }
    const double smag2 = smag_cs * smag_cs; /* Delta = 1 lattice unit */
    const double g = atof(argv[6]);
    const int steps = atoi(argv[7]);
    const int warmup = atoi(argv[8]);
    const int interval = atoi(argv[9]);
    const int threads = atoi(argv[10]);
    const char *mask_path = argv[11];
    const char *out_path = argv[12];
    const char *init_path = (argc > 13) ? argv[13] : NULL;

    const size_t N = (size_t)nz * ny * nx;
    const int nxy = ny * nx;
    unsigned char *solid = read_raw(mask_path, N);
    double *f = (double *)calloc((size_t)Q * N, sizeof(double));
    double *fs = (double *)calloc((size_t)Q * N, sizeof(double));
    double *fn = (double *)calloc((size_t)Q * N, sizeof(double));
    double *init = NULL;
    if (init_path) {
        FILE *fp = fopen(init_path, "rb");
        if (!fp) { fprintf(stderr, "cannot open %s\n", init_path); return 2; }
        float *tmp = (float *)malloc(3 * N * sizeof(float));
        if (!tmp) { fprintf(stderr, "oom\n"); return 2; }
        if (fread(tmp, sizeof(float), 3 * N, fp) != 3 * N) { fprintf(stderr, "short init read\n"); return 2; }
        fclose(fp);
        init = (double *)malloc(3 * N * sizeof(double));
        if (!init) { fprintf(stderr, "oom\n"); return 2; }
        for (size_t i = 0; i < 3 * N; i++) init[i] = (double)tmp[i];
        free(tmp);
    }
    if (!f || !fs || !fn) { fprintf(stderr, "oom\n"); return 2; }

    /* initial state: local equilibrium of (rho=1, u=(ux,0,0)); solid nodes stay
       empty (bounce-back never reads them). */
    for (size_t idx = 0; idx < N; idx++) {
        if (solid[idx]) continue;
        const double ux = init ? init[idx] : 0.0;
        const double uy = init ? init[N + idx] : 0.0;
        const double uz = init ? init[2 * N + idx] : 0.0;
        const double usq = ux * ux + uy * uy + uz * uz;
        for (int d = 0; d < Q; d++) {
            const double cu = CX[d] * ux + CY[d] * uy + CZ[d] * uz;
            f[(size_t)d * N + idx] = WT[d] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
        }
    }
    free(init);

    double *prof_sum = (double *)calloc((size_t)nz, sizeof(double));
    double *k_count = (double *)calloc((size_t)nz, sizeof(double));
    for (int k = 0; k < nz; k++) {
        size_t c = 0;
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                c += (solid[(size_t)k * nxy + j * nx + i] == 0);
        k_count[k] = (double)c;
    }
    double n_fluid = 0.0;
    for (int k = 0; k < nz; k++) n_fluid += k_count[k];

    const double s_plus_mol = 1.0 / tau_mol;
    double sum_ux = 0.0, sum_ux2 = 0.0, sum_rho = 0.0, max_ux = 0.0;
    long n_samples = 0;
    const double t_start = now_seconds();

    double *pr = (double *)calloc((size_t)threads * nz, sizeof(double));
    double *acc = (double *)calloc((size_t)threads * 5, sizeof(double));
    if (!pr || !acc) { fprintf(stderr, "oom\n"); return 2; }

    for (int t = 1; t < steps; t++) {
        const int sampling = (t >= warmup) && ((t - warmup) % interval == 0);
        memset(pr, 0, (size_t)threads * nz * sizeof(double));
        memset(acc, 0, (size_t)threads * 5 * sizeof(double));
        double local_max[512];
        for (int q = 0; q < threads && q < 512; q++) local_max[q] = 0.0;

#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            double *myp = pr + (size_t)tid * nz;
            double *mya = acc + (size_t)tid * 5;
            double mymax = 0.0;
            double feq[Q];
            double fstar[Q];
#pragma omp for schedule(static)
            for (int k = 1; k < nz - 1; k++) {
                for (int j = 0; j < ny; j++) {
                    const size_t row = (size_t)k * nxy + (size_t)j * nx;
                    for (int i = 0; i < nx; i++) {
                        const size_t idx = row + i;
                        if (solid[idx]) continue;
                        double rho = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
                        for (int d = 0; d < Q; d++) {
                            const double fv = f[(size_t)d * N + idx];
                            rho += fv;
                            mx += CX[d] * fv;
                            my += CY[d] * fv;
                            mz += CZ[d] * fv;
                        }
                        const double ux = (mx + 0.5 * g * rho) / rho;
                        const double uy = my / rho;
                        const double uz = mz / rho;
                        const double usq = ux * ux + uy * uy + uz * uz;
                        const double ug = ux * g;
                        /* equilibrium of every direction first: the TRT update
                           needs feq of the opposite direction. */
                        double pxx = 0.0, pyy = 0.0, pzz = 0.0, pxy = 0.0, pxz = 0.0, pyz = 0.0;
                        for (int d = 0; d < Q; d++) {
                            const double cu = CX[d] * ux + CY[d] * uy + CZ[d] * uz;
                            feq[d] = WT[d] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
                            const double fneq = f[(size_t)d * N + idx] - feq[d];
                            pxx += CX[d] * CX[d] * fneq;
                            pyy += CY[d] * CY[d] * fneq;
                            pzz += CZ[d] * CZ[d] * fneq;
                            pxy += CX[d] * CY[d] * fneq;
                            pxz += CX[d] * CZ[d] * fneq;
                            pyz += CY[d] * CZ[d] * fneq;
                        }
                        /* Smagorinsky: tau_eff^2 - tau_mol tau_eff - 1.5 (Cs D)^2 Q = 0,
                           Q = 3 sqrt(2 : PiPi) / rho  */
                        double s_plus = s_plus_mol;
                        double s_minus = 1.0 / (TRT_LAMBDA / (tau_mol - 0.5) + 0.5);
                        if (smag2 > 0.0) {
                            const double pipi = pxx * pxx + pyy * pyy + pzz * pzz +
                                                2.0 * (pxy * pxy + pxz * pxz + pyz * pyz);
                            const double Qv = 3.0 * sqrt(2.0 * pipi) / rho;
                            const double disc = tau_mol * tau_mol + 6.0 * smag2 * Qv;
                            const double tau_eff = 0.5 * (tau_mol + sqrt(disc > 0.0 ? disc : 0.0));
                            s_plus = 1.0 / tau_eff;
                            s_minus = 1.0 / (TRT_LAMBDA / (tau_eff - 0.5) + 0.5);
                        }
                        for (int d = 0; d < Q; d++) {
                            const double cu = CX[d] * ux + CY[d] * uy + CZ[d] * uz;
                            /* Guo (2002): w rho [3 (c-u).g + 9 (c.u)(c.g)] */
                            const double force = WT[d] * rho * (3.0 * (CX[d] * g - ug) + 9.0 * cu * (CX[d] * g));
                            const double fv = f[(size_t)d * N + idx];
                            const double fov = f[(size_t)OPP[d] * N + idx];
                            fstar[d] = fv - 0.5 * (s_plus + s_minus) * (fv - feq[d]) -
                                       0.5 * (s_plus - s_minus) * (fov - feq[OPP[d]]) + force * (1.0 - 0.5 * s_plus);
                            fs[(size_t)d * N + idx] = fstar[d];
                        }
                        if (sampling) {
                            myp[k] += ux;
                            mya[0] += ux;
                            mya[1] += ux * ux;
                            mya[2] += rho;
                            const double aux = fabs(ux);
                            if (aux > mymax) mymax = aux;
                        }
                    }
                }
            }
#pragma omp critical
            {
                if (mymax > max_ux) max_ux = mymax;
            }
        }

        /* streaming + link-wise halfway bounce-back (gather form) */
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int k = 1; k < nz - 1; k++) {
            for (int j = 0; j < ny; j++) {
                const size_t row = (size_t)k * nxy + (size_t)j * nx;
                for (int i = 0; i < nx; i++) {
                    const size_t idx = row + i;
                    if (solid[idx]) continue;
                    for (int d = 0; d < Q; d++) {
                        const int ks = k - CZ[d];
                        const int js = (j - CY[d] + ny) % ny;
                        const int is = (i - CX[d] + nx) % nx;
                        double val;
                        if (ks < 1 || ks > nz - 2) {
                            val = fs[(size_t)OPP[d] * N + idx]; /* wall link */
                        } else {
                            const size_t src = (size_t)ks * nxy + (size_t)js * nx + is;
                            val = solid[src] ? fs[(size_t)OPP[d] * N + idx] : fs[(size_t)d * N + src];
                        }
                        fn[(size_t)d * N + idx] = val;
                    }
                }
            }
        }

        double *tmp = f; f = fn; fn = tmp;

        if (sampling) {
            for (int t2 = 0; t2 < threads; t2++) {
                sum_ux += acc[(size_t)t2 * 5 + 0];
                sum_ux2 += acc[(size_t)t2 * 5 + 1];
                sum_rho += acc[(size_t)t2 * 5 + 2];
                for (int k = 0; k < nz; k++) prof_sum[k] += pr[(size_t)t2 * nz + k];
            }
            n_samples++;
        }
    }

    const double wall = now_seconds() - t_start;
    FILE *fp = fopen(out_path, "w");
    if (!fp) { fprintf(stderr, "cannot write %s\n", out_path); return 2; }
    fprintf(fp, "{\"nx\":%d,\"ny\":%d,\"nz\":%d,\"tau\":%.17g,\"smag_cs\":%.17g,\"g\":%.17g,", nx, ny, nz, tau_mol,
            smag_cs, g);
    fprintf(fp, "\"steps\":%d,\"warmup\":%d,\"interval\":%d,\"n_samples\":%ld,", steps, warmup, interval, n_samples);
    fprintf(fp, "\"n_fluid\":%.17g,\"sum_ux\":", n_fluid);
    jnum(fp, sum_ux);
    fprintf(fp, ",\"sum_ux2\":");
    jnum(fp, sum_ux2);
    fprintf(fp, ",\"sum_rho\":");
    jnum(fp, sum_rho);
    fprintf(fp, ",\"max_ux\":");
    jnum(fp, max_ux);
    fprintf(fp, ",");
    fprintf(fp, "\"wall_seconds\":%.6f,\n", wall);
    fprintf(fp, "\"k_count\":[");
    for (int k = 0; k < nz; k++) {
        if (k) fprintf(fp, ",");
        jnum(fp, k_count[k]);
    }
    fprintf(fp, "],\n\"prof_sum\":[");
    for (int k = 0; k < nz; k++) {
        if (k) fprintf(fp, ",");
        jnum(fp, prof_sum[k]);
    }
    fprintf(fp, "]}\n");
    fclose(fp);
    fprintf(stderr, "native: steps=%d wall=%.1fs samples=%ld\n", steps, wall, n_samples);
    return 0;
}
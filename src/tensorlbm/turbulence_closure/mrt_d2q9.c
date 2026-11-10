/* mrt_d2q9.c -- D2Q9 MRT solver with non-equilibrium-extrapolation (NEEM)
 * boundary conditions for the planar turbulence-validation cases
 * (backward-facing step, circular cylinder).
 *
 * The 9x9 moment matrix is NOT hard-coded: it is read from a text file written
 * by tensorlbm.turbulence_closure.mrt.write_matrix(), so the Python reference
 * implementation and this kernel provably use the same operator.
 *
 * Boundary handling (all of it non-periodic):
 *   - solid cells   : halfway bounce-back (flag array, 1 = solid)
 *   - inlet  x = 0  : NEEM velocity Dirichlet, f = feq(rho_nb, u_in)
 *                                          + [f_nb - feq_nb(rho_nb, u_nb)]
 *   - outlet x = nx-1: NEEM pressure outlet, f = feq(1, u_nb)
 *                                          + [f_nb - feq_nb(rho_nb, u_nb)]
 *   - the flag array must contain a solid wall on y = 0 and y = ny-1.
 *
 * Outputs (plain text, no hand-copying anywhere downstream):
 *   <prefix>_mean.txt   : x y rho ux uy (time-averaged, after warm-up)
 *   <prefix>_force.txt  : step Fx Fy (instantaneous momentum-exchange force)
 *   <prefix>_diag.txt   : step ke maxu   (stability / convergence trace)
 *   <prefix>_meta.txt   : key=value run record (args + provenance)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NQ 9
static const int CX[NQ] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static const int CY[NQ] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static const double W[NQ] = {4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9,
                             1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36};
static const int OPP[NQ] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
static const double CS2 = 1.0 / 3.0;

static double M[NQ][NQ];
static int S_IDX[NQ];      /* relaxation rate per moment */
static int ST_IDX[2] = {4, 5}; /* traceless 2nd-order (stress) rows */

static int nx, ny, ncell;
static int *solid;
static int *body; /* subset of solid: momentum-exchange force is accumulated here */
static double *f, *fpost;
static double *rm, *sm; /* mean accumulators */

static void die(const char *msg) {
    fprintf(stderr, "mrt_d2q9: %s\n", msg);
    exit(2);
}

static void load_matrix(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) die("cannot open MRT matrix file");
    for (int i = 0; i < NQ; i++)
        for (int j = 0; j < NQ; j++)
            if (fscanf(fp, "%lf", &M[i][j]) != 1) die("matrix file truncated");
    fclose(fp);
}

static void load_mask(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) die("cannot open solid mask file");
    for (int i = 0; i < ncell; i++)
        if (fscanf(fp, "%d", &solid[i]) != 1) die("mask file truncated");
    fclose(fp);
}

static inline void feq_q(int q, double rho, double ux, double uy, double *out) {
    double cu = CX[q] * ux + CY[q] * uy;
    double usq = ux * ux + uy * uy;
    *out = rho * W[q] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
}

/* Smagorinsky-consistent stress relaxation rate; falls back to molecular tau. */
static inline double sgs_tau(double fneq[NQ], double rho, double tau0, double cs) {
    if (cs <= 0.0) return tau0;
    double pxx = 0, pyy = 0, pxy = 0;
    for (int q = 0; q < NQ; q++) {
        double cxcx = (double)CX[q] * CX[q], cycy = (double)CY[q] * CY[q];
        double cxcy = (double)CX[q] * CY[q];
        pxx += cxcx * fneq[q];
        pyy += cycy * fneq[q];
        pxy += cxcy * fneq[q];
    }
    double p2 = pxx * pxx + pyy * pyy + 2.0 * pxy * pxy;
    double tau = tau0;
    for (int it = 0; it < 2; it++) {
        double smag = sqrt(fmax(p2 * 0.5, 0.0)) / fmax(CS2 * tau * rho, 1e-30);
        double nut = cs * cs * smag;
        tau = 0.5 + 3.0 * ((tau0 - 0.5) / 3.0 + nut);
    }
    return tau;
}

int main(int argc, char **argv) {
    int steps = 10000, warmup = 5000, sample_every = 10, force_every = 1, diag_every = 500;
    double u_in = 0.1, tau0 = 0.6, cs = 0.0, body_force = 0.0;
    double inlet_y0 = 0.5, inlet_y1 = 0.0; /* 0 => full domain height */
    int parabolic = 0, periodic_x = 0;
    const char *prefix = "run", *matpath = NULL, *maskpath = NULL, *bodypath = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--nx")) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ny")) ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps")) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sample-every")) sample_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--force-every")) force_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--u-in")) u_in = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tau")) tau0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--cs")) cs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--parabolic")) parabolic = 1;
        else if (!strcmp(argv[i], "--matrix")) matpath = argv[++i];
        else if (!strcmp(argv[i], "--mask")) maskpath = argv[++i];
        else if (!strcmp(argv[i], "--body")) bodypath = argv[++i];
        else if (!strcmp(argv[i], "--body-force")) body_force = atof(argv[++i]);
        else if (!strcmp(argv[i], "--inlet-y0")) inlet_y0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--inlet-y1")) inlet_y1 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--periodic-x")) periodic_x = 1;
        else if (!strcmp(argv[i], "--prefix")) prefix = argv[++i];
        else die("unknown argument");
    }
    if (!matpath || !maskpath || nx <= 0 || ny <= 0) die("need --matrix --mask --nx --ny");

    ncell = nx * ny;
    solid = (int *)calloc(ncell, sizeof(int));
    f = (double *)calloc((size_t)NQ * ncell, sizeof(double));
    fpost = (double *)calloc((size_t)NQ * ncell, sizeof(double));
    rm = (double *)calloc(ncell, sizeof(double));
    sm = (double *)calloc(ncell * 3, sizeof(double)); /* ux, uy, rho */
    if (!solid || !f || !fpost || !rm || !sm) die("out of memory");

    body = (int *)calloc(ncell, sizeof(int));
    if (!body) die("out of memory");
    load_matrix(matpath);
    load_mask(maskpath);
    if (bodypath) {
        FILE *fb = fopen(bodypath, "r");
        if (!fb) die("cannot open body mask file");
        for (int i = 0; i < ncell; i++)
            if (fscanf(fb, "%d", &body[i]) != 1) die("body mask truncated");
        fclose(fb);
    } else {
        for (int i = 0; i < ncell; i++) body[i] = solid[i];
    }

    for (int i = 0; i < NQ; i++) S_IDX[i] = 1; /* higher-order modes -> equilibrium */
    S_IDX[0] = 0;
    S_IDX[1] = 0;
    S_IDX[2] = 0;
    S_IDX[3] = 1; /* energy (bulk) */

    /* initialise at rest, then a short ramp of the inlet velocity */
    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++) {
            int i = y * nx + x;
            if (solid[i]) continue;
            for (int q = 0; q < NQ; q++) f[q * ncell + i] = W[q];
        }

    char path[4096];
    FILE *fdiag = NULL, *fforce = NULL;
    snprintf(path, sizeof path, "%s_diag.txt", prefix);
    fdiag = fopen(path, "w");
    snprintf(path, sizeof path, "%s_force.txt", prefix);
    fforce = fopen(path, "w");
    if (!fdiag || !fforce) die("cannot open output files");
    fprintf(fdiag, "# step ke maxu\n");
    fprintf(fforce, "# step Fx Fy\n");

    double ramp_steps = fmax(1.0, 0.05 * steps);
    for (int step = 1; step <= steps; step++) {
        double u_now = u_in * fmin(1.0, step / ramp_steps);

        /* ---------- 1. moments + collision (in place into fpost) ---------- */
#pragma omp parallel for schedule(static)
        for (int y = 1; y < ny - 1; y++)
            for (int x = 0; x < nx; x++) {
                int i = y * nx + x;
                if (solid[i]) continue;
                double rho = 0, ux = 0, uy = 0;
                for (int q = 0; q < NQ; q++) {
                    double fq = f[q * ncell + i];
                    rho += fq;
                    ux += CX[q] * fq;
                    uy += CY[q] * fq;
                }
                ux /= rho;
                uy /= rho;
                /* Guo forcing: relax towards equilibrium at u + F/(2 rho) */
                double ux_f = ux + 0.5 * body_force / rho;
                double feq[NQ], fneq[NQ], m[NQ], meq[NQ], mnew[NQ];
                for (int q = 0; q < NQ; q++) {
                    feq_q(q, rho, ux_f, uy, &feq[q]);
                    fneq[q] = f[q * ncell + i] - feq[q];
                }
                double tau = sgs_tau(fneq, rho, tau0, cs);
                for (int k = 0; k < NQ; k++) {
                    double a = 0, b = 0;
                    for (int q = 0; q < NQ; q++) {
                        a += M[k][q] * f[q * ncell + i];
                        b += M[k][q] * feq[q];
                    }
                    m[k] = a;
                    meq[k] = b;
                }
                double s_rate = 1.0 / tau;
                for (int k = 0; k < NQ; k++) {
                    double s = S_IDX[k] == 0 ? 0.0 : (S_IDX[k] == 1 ? 1.0 : s_rate);
                    mnew[k] = m[k] + s * (meq[k] - m[k]);
                }
                /* stress rows use the (possibly SGS-enhanced) rate */
                for (int k = 0; k < 2; k++)
                    mnew[ST_IDX[k]] = m[ST_IDX[k]] + s_rate * (meq[ST_IDX[k]] - m[ST_IDX[k]]);
                for (int q = 0; q < NQ; q++) {
                    double acc = 0;
                    for (int k = 0; k < NQ; k++) acc += M[k][q] * mnew[k];
                    /* second half of the Guo forcing term */
                    acc += 3.0 * W[q] * (double)CX[q] * body_force;
                    fpost[q * ncell + i] = acc;
                }
            }

        /* ---------- 2. streaming with halfway bounce-back + force ---------- */
        double Fx = 0, Fy = 0;
#pragma omp parallel for schedule(static) reduction(+ : Fx, Fy)
        for (int y = 1; y < ny - 1; y++)
            for (int x = 0; x < nx; x++) {
                int i = y * nx + x;
                if (solid[i]) continue;
                for (int q = 0; q < NQ; q++) {
                    int xs = x - CX[q], ys = y - CY[q];
                    if (periodic_x) {
                        xs = (xs + nx) % nx;
                    }
                    int inside = (xs >= 0 && xs < nx && ys >= 0 && ys < ny);
                    if (periodic_x && ys >= 0 && ys < ny) inside = 1;
                    int j = inside ? ys * nx + xs : -1;
                    if (!inside || solid[j]) {
                        f[q * ncell + i] = fpost[OPP[q] * ncell + i];
                        if (inside && body[j]) {
                            Fx += 2.0 * (double)CX[q] * fpost[q * ncell + i];
                            Fy += 2.0 * (double)CY[q] * fpost[q * ncell + i];
                        }
                    } else {
                        f[q * ncell + i] = fpost[q * ncell + j];
                    }
                }
            }

        /* ---------- 3. NEEM inlet / outlet ---------- */
        /* inlet: velocity Dirichlet */
#pragma omp parallel for schedule(static)
        for (int y = 1; y < (periodic_x ? 1 : ny - 1); y++) {
            int i = y * nx + 0, nb = y * nx + 1;
            if (solid[i] || solid[nb]) continue;
            double rnb = 0, uxnb = 0, uynb = 0;
            for (int q = 0; q < NQ; q++) {
                double fq = f[q * ncell + nb];
                rnb += fq;
                uxnb += CX[q] * fq;
                uynb += CY[q] * fq;
            }
            uxnb /= rnb;
            uynb /= rnb;
            double uy_in = 0.0;
            double ux_in = u_now;
            if (parabolic) {
                /* parabolic profile across the inlet opening [inlet_y0, inlet_y1] */
                double hi = inlet_y1 > 0.0 ? inlet_y1 : (ny - 1.5);
                double yc = (y - inlet_y0) / (hi - inlet_y0); /* 0..1 */
                if (yc < 0.0) yc = 0.0;
                if (yc > 1.0) yc = 1.0;
                ux_in = u_now * 6.0 * yc * (1.0 - yc);
            }
            for (int q = 0; q < NQ; q++) {
                double feq_t, feq_nb;
                feq_q(q, rnb, ux_in, uy_in, &feq_t);
                feq_q(q, rnb, uxnb, uynb, &feq_nb);
                f[q * ncell + i] = feq_t + (f[q * ncell + nb] - feq_nb);
            }
        }
        /* outlet: pressure Dirichlet (rho = 1) via NEEM */
#pragma omp parallel for schedule(static)
        for (int y = 1; y < (periodic_x ? 1 : ny - 1); y++) {
            int i = y * nx + (nx - 1), nb = y * nx + (nx - 2);
            if (solid[i] || solid[nb]) continue;
            double rnb = 0, uxnb = 0, uynb = 0;
            for (int q = 0; q < NQ; q++) {
                double fq = f[q * ncell + nb];
                rnb += fq;
                uxnb += CX[q] * fq;
                uynb += CY[q] * fq;
            }
            uxnb /= rnb;
            uynb /= rnb;
            for (int q = 0; q < NQ; q++) {
                double feq_t, feq_nb;
                feq_q(q, 1.0, uxnb, uynb, &feq_t);
                feq_q(q, rnb, uxnb, uynb, &feq_nb);
                f[q * ncell + i] = feq_t + (f[q * ncell + nb] - feq_nb);
            }
        }

        /* ---------- 4. diagnostics + statistics ---------- */
        if (step % force_every == 0) fprintf(fforce, "%d %.12e %.12e\n", step, Fx, Fy);

        if (step > warmup && (step - warmup) % sample_every == 0) {
#pragma omp parallel for schedule(static)
            for (int y = 1; y < ny - 1; y++)
                for (int x = 0; x < nx; x++) {
                    int i = y * nx + x;
                    if (solid[i]) continue;
                    double rho = 0, ux = 0, uy = 0;
                    for (int q = 0; q < NQ; q++) {
                        double fq = f[q * ncell + i];
                        rho += fq;
                        ux += CX[q] * fq;
                        uy += CY[q] * fq;
                    }
                    rm[i] += 1.0;
                    sm[3 * i + 0] += ux / rho;
                    sm[3 * i + 1] += uy / rho;
                    sm[3 * i + 2] += rho;
                }
        }

        if (step % diag_every == 0) {
            double ke = 0, umax = 0;
            for (int i = 0; i < ncell; i++) {
                if (solid[i]) continue;
                double rho = 0, ux = 0, uy = 0;
                for (int q = 0; q < NQ; q++) {
                    double fq = f[q * ncell + i];
                    rho += fq;
                    ux += CX[q] * fq;
                    uy += CY[q] * fq;
                }
                ux /= rho;
                uy /= rho;
                ke += 0.5 * rho * (ux * ux + uy * uy);
                double un = sqrt(ux * ux + uy * uy);
                if (un > umax) umax = un;
            }
            fprintf(fdiag, "%d %.12e %.12e\n", step, ke, umax);
            fflush(fdiag);
        }
        if (step % 5000 == 0) {
            fprintf(stderr, "step %d/%d Fx=%.6e Fy=%.6e\n", step, steps, Fx, Fy);
            fflush(stderr);
        }
    }
    fclose(fdiag);
    fclose(fforce);

    snprintf(path, sizeof path, "%s_mean.txt", prefix);
    FILE *fmean = fopen(path, "w");
    if (!fmean) die("cannot open mean output");
    fprintf(fmean, "# x y rho ux uy\n");
    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++) {
            int i = y * nx + x;
            if (solid[i] || rm[i] == 0.0) {
                fprintf(fmean, "%d %d nan nan nan\n", x, y);
                continue;
            }
            fprintf(fmean, "%d %d %.10e %.10e %.10e\n", x, y, sm[3 * i + 2] / rm[i],
                    sm[3 * i + 0] / rm[i], sm[3 * i + 1] / rm[i]);
        }
    fclose(fmean);

    snprintf(path, sizeof path, "%s_meta.txt", prefix);
    FILE *fmeta = fopen(path, "w");
    fprintf(fmeta,
            "nx=%d\nny=%d\nsteps=%d\nwarmup=%d\nsample_every=%d\nu_in=%.12g\n"
            "tau=%.12g\ncs=%.12g\nparabolic=%d\nbody_force=%.12g\nperiodic_x=%d\n"
            "inlet_y0=%.12g\ninlet_y1=%.12g\nmatrix=%s\nmask=%s\nbody=%s\n",
            nx, ny, steps, warmup, sample_every, u_in, tau0, cs, parabolic, body_force,
            periodic_x, inlet_y0, inlet_y1, matpath, maskpath,
            bodypath ? bodypath : "(all-solid)");
#ifdef _OPENMP
    fprintf(fmeta, "omp_threads=%d\n", omp_get_max_threads());
#else
    fprintf(fmeta, "omp_threads=1\n");
#endif
    fclose(fmeta);
    free(solid);
    free(body);
    free(f);
    free(fpost);
    free(rm);
    free(sm);
    return 0;
}
/* Under-ice keel channel: native D3Q19 kernel with wall-force diagnostics.
 *
 * Same lattice scheme as src/tensorlbm/underice_drag.py (TRT, magic parameter
 * Lambda = 3/16, link-wise halfway bounce-back, momentum-exact body force
 * 3 w_i c_ix rho g added post-collision).  The Python/torch implementation is
 * the reference: tests/test_underice_drag.py checks that both back-ends agree.
 *
 * What this program adds over the torch path is speed: a 2-parameter x 5-ratio
 * x 2-Re sweep of wall-resolved channels is 10^10 cell updates, which the torch
 * path cannot afford.  It dumps only *raw sums*; every physical quantity
 * (u_tau, Cf, C_d, skin/form split, closures) is computed in Python from the
 * same formulas for both back-ends, so a back-end disagreement cannot hide in
 * post-processing.
 *
 * Usage:
 *   underice_channel nx ny nz tau s_minus g nu steps warmup sample_interval \
 *                    threads solid.raw out.json [init.f32]
 *
 * solid.raw : nz*ny*nx bytes, 1 = solid.
 * init.f32  : 3*nz*ny*nx float32 (ux, uy, uz blocks) optional initial velocity.
 *
 * Diagnostic sums (per sample, over the whole domain, lattice units):
 *   sum_me      sum over wall-crossing links of c_ix (f*_i + f*_opp)
 *   sum_form    sum over fluid cells of p (+1 if solid at x+1, -1 at x-1),
 *               p = rho/3  (first-order wall pressure on x-normal faces)
 *   sum_skin    sum over fluid cells touching a solid in z of 2 nu u_x
 *               (viscous shear on the horizontal wetted faces)
 *   sum_skin_y  sum over fluid cells touching a solid in y of nu du_x/dy
 *   sum_ux, sum_ux2, sum_rho, max_ux, prof[nz]
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
static const int OPP[Q] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
static const double WT[Q] = {
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

/* JSON-safe scalar: json.loads accepts nan/inf spellings, but not "-nan". */
static const char *jnum_s(double x) {
    static char buf[8][40];
    static int slot = 0;
    char *b = buf[slot++ & 7];
    if (!isfinite(x)) {
        snprintf(b, 40, "%s", (x != x) ? "NaN" : (x > 0 ? "Infinity" : "-Infinity"));
        return b;
    }
    snprintf(b, 40, "%.17g", x);
    return b;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 13) {
        fprintf(stderr,
                "usage: %s nx ny nz tau s_minus g nu steps warmup sample_interval "
                "threads solid.raw out.json [init.f32]\n",
                argv[0]);
        return 2;
    }
    const int nx = atoi(argv[1]), ny = atoi(argv[2]), nz = atoi(argv[3]);
    const double tau = atof(argv[4]);
    const double s_minus = atof(argv[5]);
    const double g = atof(argv[6]);
    const double nu = atof(argv[7]);
    const int steps = atoi(argv[8]);
    const int warmup = atoi(argv[9]);
    const int sample_interval = atoi(argv[10]);
    omp_set_num_threads(atoi(argv[11]));
    const char *solid_path = argv[12];
    const char *out_path = argv[13];
    const char *init_path = (argc > 14) ? argv[14] : NULL;
    const char *dump_path = (argc > 15) ? argv[15] : NULL;

    const long N = (long)nx * ny * nz;
    unsigned char *solid = (unsigned char *)malloc((size_t)N);
    if (!solid) { fprintf(stderr, "oom solid\n"); return 1; }
    FILE *fs = fopen(solid_path, "rb");
    if (!fs || fread(solid, 1, (size_t)N, fs) != (size_t)N) {
        fprintf(stderr, "cannot read %s\n", solid_path);
        return 1;
    }
    fclose(fs);

    /* links[i*N+n] = fluid[n] and solid[n - c_i]  (the incoming population
       would come from a solid cell -> this link bounces back) */
    unsigned char *links = (unsigned char *)calloc((size_t)Q * N, 1);
    float *f = (float *)calloc((size_t)Q * N, sizeof(float));
    float *fstar = (float *)calloc((size_t)Q * N, sizeof(float));
    float *rho = (float *)calloc((size_t)N, sizeof(float));
    float *ux = (float *)calloc((size_t)N, sizeof(float));
    float *uy = (float *)calloc((size_t)N, sizeof(float));
    float *uz = (float *)calloc((size_t)N, sizeof(float));
    double *prof = (double *)calloc((size_t)nz, sizeof(double));
    long *kc = (long *)calloc((size_t)nz, sizeof(long));
    if (!links || !f || !fstar || !rho || !ux || !uy || !uz || !prof || !kc) {
        fprintf(stderr, "oom\n");
        return 1;
    }

    long n_fluid = 0;
    for (long z = 0; z < nz; z++)
        for (long y = 0; y < ny; y++)
            for (long x = 0; x < nx; x++) {
                long n = (z * ny + y) * nx + x;
                if (!solid[n]) { n_fluid++; kc[z]++; }
            }
    for (long z = 0; z < nz; z++)
        for (long y = 0; y < ny; y++)
            for (long x = 0; x < nx; x++) {
                long n = (z * ny + y) * nx + x;
                if (solid[n]) continue;
                for (int i = 1; i < Q; i++) {
                    long zs = z - CZ[i], ys = y - CY[i], xs = x - CX[i];
                    if (zs < 0 || zs >= nz) continue;
                    ys = ((ys % ny) + ny) % ny;
                    xs = ((xs % nx) + nx) % nx;
                    long s = (zs * ny + ys) * nx + xs;
                    if (solid[s]) links[(long)i * N + n] = 1;
                }
            }

    /* initial state: equilibrium of the supplied velocity field (or rest) */
    float *init = (float *)calloc((size_t)3 * N, sizeof(float));
    if (init_path) {
        FILE *fi = fopen(init_path, "rb");
        if (!fi || fread(init, sizeof(float), (size_t)3 * N, fi) != (size_t)(3 * N)) {
            fprintf(stderr, "cannot read %s\n", init_path);
            return 1;
        }
        fclose(fi);
    }
    for (long z = 0; z < nz; z++)
        for (long y = 0; y < ny; y++)
            for (long x = 0; x < nx; x++) {
                long n = (z * ny + y) * nx + x;
                rho[n] = 1.0f;
                ux[n] = init[n];
                uy[n] = init[N + n];
                uz[n] = init[2 * N + n];
                double usq = (double)ux[n] * ux[n] + (double)uy[n] * uy[n] + (double)uz[n] * uz[n];
                for (int i = 0; i < Q; i++) {
                    double cu = CX[i] * (double)ux[n] + CY[i] * (double)uy[n] + CZ[i] * (double)uz[n];
                    f[(long)i * N + n] = (float)(WT[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq));
                }
            }

    const double s_plus = 1.0 / tau;
    double sum_me = 0.0, sum_form = 0.0, sum_skin = 0.0, sum_skin_y = 0.0;
    double sum_form_face = 0.0, sum_skin_grad = 0.0;
    double sum_flux = 0.0; long n_wall = 0;
    double sum_ux = 0.0, sum_ux2 = 0.0, sum_rho = 0.0;
    double mom_first = 0.0, mom_last = 0.0;
    double sux_first = 0.0, sux_last = 0.0;
    float max_ux = 0.0f;
    int n_samples = 0;
    const double t_start = now_seconds();

    for (int t = 1; t <= steps; t++) {
        /* ---- macroscopic moments ---- */
        #pragma omp parallel for schedule(static)
        for (long n = 0; n < N; n++) {
            double r = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
            for (int i = 0; i < Q; i++) {
                double v = f[(long)i * N + n];
                r += v;
                mx += CX[i] * v;
                my += CY[i] * v;
                mz += CZ[i] * v;
            }
            if (r <= 1e-12) r = 1.0; /* solid/garbage cells: neutral */
            rho[n] = (float)r;
            ux[n] = (float)(mx / r);
            uy[n] = (float)(my / r);
            uz[n] = (float)(mz / r);
        }
        /* ---- TRT collision + momentum-exact body force ---- */
        #pragma omp parallel for schedule(static)
        for (long n = 0; n < N; n++) {
            double r = solid[n] ? 1.0 : rho[n];
            double u = solid[n] ? 0.0 : ux[n];
            double v = solid[n] ? 0.0 : uy[n];
            double w = solid[n] ? 0.0 : uz[n];
            const double usq = u * u + v * v + w * w;
            double feq[Q];
            for (int i = 0; i < Q; i++) {
                double cu = CX[i] * u + CY[i] * v + CZ[i] * w;
                feq[i] = WT[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
            }
            for (int i = 0; i < Q; i++) {
                int j = OPP[i];
                double fi = f[(long)i * N + n], fj = f[(long)j * N + n];
                double fplus = 0.5 * (fi + fj), fminus = 0.5 * (fi - fj);
                double feqplus = 0.5 * (feq[i] + feq[j]), feqminus = 0.5 * (feq[i] - feq[j]);
                double out = fi - s_plus * (fplus - feqplus) - s_minus * (fminus - feqminus);
                /* solid nodes receive no force, so their state cannot drift */
                if (!solid[n]) out += 3.0 * g * WT[i] * CX[i] * r;
                fstar[(long)i * N + n] = (float)out;
            }
        }
        /* ---- streaming + link-wise halfway bounce-back ---- */
        #pragma omp parallel for schedule(static)
        for (long z = 0; z < nz; z++)
            for (long y = 0; y < ny; y++)
                for (long x = 0; x < nx; x++) {
                    long n = (z * ny + y) * nx + x;
                    f[(long)0 * N + n] = fstar[(long)0 * N + n];
                    for (int i = 1; i < Q; i++) {
                        if (links[(long)i * N + n]) {
                            f[(long)i * N + n] = fstar[(long)OPP[i] * N + n];
                            continue;
                        }
                        long zs = z - CZ[i], ys = y - CY[i], xs = x - CX[i];
                        if (zs < 0 || zs >= nz) { f[(long)i * N + n] = fstar[(long)i * N + n]; continue; }
                        ys = ((ys % ny) + ny) % ny;
                        xs = ((xs % nx) + nx) % nx;
                        f[(long)i * N + n] = fstar[(long)i * N + ((zs * ny + ys) * nx + xs)];
                    }
                }

        if (t >= warmup && ((t - warmup) % sample_interval) == 0) {
            double me = 0.0, form = 0.0, skin = 0.0, skin_y = 0.0;
            double form_face = 0.0, skin_grad = 0.0;
            double flux = 0.0; long nwall = 0;
            double mom_now = 0.0;
            double sux = 0.0, sux2 = 0.0, srho = 0.0;
            float mx = 0.0f;
            #pragma omp parallel
            {
                double lme = 0.0, lform = 0.0, lskin = 0.0, lskin_y = 0.0, lmom = 0.0;
                double lform_face = 0.0, lskin_grad = 0.0;
                double lflux = 0.0; long lnwall = 0;
                double lsux = 0.0, lsux2 = 0.0, lsrho = 0.0;
                float lmax = 0.0f;
                double lprof[512];
                for (int k = 0; k < nz && k < 512; k++) lprof[k] = 0.0;
                #pragma omp for schedule(static)
                for (long z = 0; z < nz; z++)
                    for (long y = 0; y < ny; y++)
                        for (long x = 0; x < nx; x++) {
                            long n = (z * ny + y) * nx + x;
                            if (solid[n]) continue;
                            /* (1) wall force: exact link-wise momentum
                               exchange.  links[i](n)=1 means the solid sits at
                               X - c_i, so the incoming population of this fluid
                               node in direction i is the reflected one,
                               f*_opp(i).  The fluid gives up 2 c_i f*_opp(i) of
                               x-momentum per wall link -- the exact wall force,
                               with no equilibrium split and no relaxation
                               calibration.  The 2026-11 form
                               2 c_x (f*_i - feq_i)/(1 - s+) is a stress estimate
                               and over-predicts by ~1e2; the pre-2026 form
                               c_x (f*_i - f*_opp) measures ~U_wall/3.
                               Projected on the local staircase normal
                               (normalised wetted-axis sum, 45 deg on a corner)
                               it splits into pressure (form) and friction. */
                            for (int i = 1; i < Q; i++) {
                                if (CX[i] == 0) continue;
                                if (!links[(long)i * N + n]) continue;
                                double iwx = 0.0, iwy = 0.0, iwz = 0.0;
                                if (CX[i] != 0) {
                                    int xs = ((x - CX[i]) % nx + nx) % nx;
                                    if (solid[(long)z * ny * nx + (long)y * nx + xs])
                                        iwx = CX[i] > 0 ? 1.0 : -1.0;
                                }
                                if (CY[i] != 0) {
                                    int ys = ((y - CY[i]) % ny + ny) % ny;
                                    if (solid[(long)z * ny * nx + (long)ys * nx + x])
                                        iwy = CY[i] > 0 ? 1.0 : -1.0;
                                }
                                if (CZ[i] != 0) {
                                    int zs = z - CZ[i];
                                    if (zs >= 0 && zs < nz &&
                                        solid[(long)zs * ny * nx + (long)y * nx + x])
                                        iwz = CZ[i] > 0 ? 1.0 : -1.0;
                                }
                                if (iwx == 0.0 && iwy == 0.0 && iwz == 0.0) {
                                    iwx = (double)CX[i]; iwy = (double)CY[i]; iwz = (double)CZ[i];
                                }
                                double inorm = sqrt(iwx * iwx + iwy * iwy + iwz * iwz);
                                iwx /= inorm; iwy /= inorm; iwz /= inorm;
                                double cdn = CX[i] * iwx + CY[i] * iwy + CZ[i] * iwz;
                                double wi = 2.0 * (double)fstar[(long)OPP[i] * N + n] *
                                            (double)links[(long)i * N + n];
                                lme += wi * (double)CX[i];
                                lform += wi * cdn * iwx;
                            }
                            /* legacy: x-normal faces, p = rho/3 (diagnostic only) */
                            {
                                long xp = (((x + 1) % nx) * 1) + (y * nx) + (z * ny * nx);
                                long xm = ((((x - 1) + nx) % nx) + y * nx + z * ny * nx);
                                double p = (double)rho[n] / 3.0;
                                if (solid[xp]) lform_face += p;
                                if (solid[xm]) lform_face -= p;
                            }
                            /* horizontal faces: 2 nu u_x (one-sided, halfway wall) */
                            {
                                int touch = 0;
                                if (z + 1 < nz && solid[n + (long)nx * ny]) touch = 1;
                                if (z - 1 >= 0 && solid[n - (long)nx * ny]) touch = 1;
                                if (touch) lskin_grad += 2.0 * nu * (double)ux[n];
                                int touchy = 0;
                                if (solid[(z * ny + ((y + 1) % ny)) * nx + x]) touchy = 1;
                                if (solid[(z * ny + (((y - 1) + ny) % ny)) * nx + x]) touchy = 1;
                                if (touchy) {
                                    long yp = (z * ny + ((y + 1) % ny)) * nx + x;
                                    long ym = (z * ny + (((y - 1) + ny) % ny)) * nx + x;
                                    lskin_y += nu * 0.5 * ((double)ux[yp] - (double)ux[ym]);
                                }
                            }
                            /* wall stress from the x-z momentum flux at the
                               wall-adjacent row: tau_xz = -sum_i c_x c_z f_i^neq
                               (the equilibrium part sum c_x c_z feq = rho u_x u_z
                               and is subtracted).  One entry per wall-adjacent
                               node, i.e. per unit of wall-plan area. */
                            {
                                int zwall = 0;
                                if (z + 1 < nz && solid[n + (long)nx * ny]) zwall = 1;
                                if (z - 1 >= 0 && solid[n - (long)nx * ny]) zwall = 1;
                                if (zwall) {
                                    double rr = (double)rho[n];
                                    double uu = rr > 1e-12 ? (double)ux[n] : 0.0;
                                    double vv = rr > 1e-12 ? (double)uy[n] : 0.0;
                                    double ww = rr > 1e-12 ? (double)uz[n] : 0.0;
                                    double usq = uu * uu + vv * vv + ww * ww;
                                    for (int i = 1; i < Q; i++) {
                                        if (CX[i] == 0 || CZ[i] == 0) continue;
                                        double cu = CX[i] * uu + CY[i] * vv + CZ[i] * ww;
                                        double feqi = WT[i] * rr *
                                            (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
                                        lflux += (double)CX[i] * (double)CZ[i] *
                                                 ((double)fstar[(long)i * N + n] - feqi);
                                    }
                                    lnwall += 1;
                                }
                            }
                            {
                                double m = 0.0;
                                for (int i = 0; i < Q; i++) m += CX[i] * (double)f[(long)i * N + n];
                                lmom += m;
                            }
                            lsux += ux[n];
                            lsux2 += (double)ux[n] * ux[n];
                            lsrho += (double)rho[n];
                            lprof[z] += ux[n];
                            if (fabsf(ux[n]) > lmax) lmax = fabsf(ux[n]);
                        }
                #pragma omp critical
                {
                    me += lme; form += lform; skin_y += lskin_y;
                    form_face += lform_face; skin_grad += lskin_grad;
                    flux += lflux; nwall += lnwall;
                    sux += lsux; sux2 += lsux2; srho += lsrho;
                    mom_now += lmom;
                    if (lmax > mx) mx = lmax;
                    for (int k = 0; k < nz && k < 512; k++) prof[k] += lprof[k];
                }
            }
            if (n_samples == 0) { mom_first = mom_now; sux_first = sux; }
            sux_last = sux;
            mom_last = mom_now;
            /* f* - feq = (1 - s+)(f - feq) here (s- = s+ = 1/tau), so undo the
               relaxation attenuation; the 2 c_ix factor is the bounce-back
               momentum delivery. */
            /* (1) exact exchange, (2) its pressure part, (3) friction =
               total - pressure (exact by construction per link) */
            /* sign: the exchange 2 c_ix f*_opp is negative along the flow, so
               the drag magnitude is its negative (same convention as the
               torch back-end, which reports positive drag). */
            sum_me += -me;
            sum_form += -form;
            sum_skin += -(me - form);
            sum_skin_y += skin_y;
            sum_form_face += form_face;
            sum_skin_grad += skin_grad;
            sum_flux += 2.0 * flux / (1.0 - s_minus);
            n_wall += nwall;
            sum_ux += sux; sum_ux2 += sux2; sum_rho += srho;
            if (mx > max_ux) max_ux = mx;
            n_samples++;
        }
    }
    const double wall = now_seconds() - t_start;

    if (dump_path) { /* optional raw final state, for cross-back-end tests */
        FILE *fd = fopen(dump_path, "wb");
        if (!fd) { fprintf(stderr, "cannot write %s\n", dump_path); return 1; }
        fwrite(f, sizeof(float), (size_t)Q * N, fd);
        fclose(fd);
    }
    FILE *fo = fopen(out_path, "w");
    if (!fo) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    fprintf(fo, "{\n");
    fprintf(fo, "\"nx\": %d, \"ny\": %d, \"nz\": %d,\n", nx, ny, nz);
    fprintf(fo, "\"n_fluid\": %ld, \"n_samples\": %d,\n", n_fluid, n_samples);
    fprintf(fo, "\"sum_me\": %s, \"sum_form\": %s, \"sum_skin\": %s,",
            jnum_s(sum_me), jnum_s(sum_form), jnum_s(sum_skin));
    fprintf(fo, " \"sum_flux\": %s, \"n_wall\": %d,\n", jnum_s(sum_flux), n_wall);
    fprintf(fo, " \"sum_form_face\": %s, \"sum_skin_grad\": %s,\n",
            jnum_s(sum_form_face), jnum_s(sum_skin_grad));
    fprintf(fo, " \"sum_skin_y\": %s,\n", jnum_s(sum_skin_y));
    fprintf(fo, "\"sum_ux\": %s, \"sum_ux2\": %s, \"sum_rho\": %s, \"max_ux\": %s,\n",
            jnum_s(sum_ux), jnum_s(sum_ux2), jnum_s(sum_rho), jnum_s((double)max_ux));
    fprintf(fo, "\"steps\": %d, \"warmup\": %d, \"sample_interval\": %d,\n", steps, warmup,
            sample_interval);
    fprintf(fo, "\"tau\": %.17g, \"s_minus\": %.17g, \"g\": %.17g, \"nu\": %.17g,\n", tau,
            s_minus, g, nu);
    fprintf(fo, "\"wall_seconds\": %.17g,\n", wall);
    fprintf(fo, "\"mom_first\": %s, \"mom_last\": %s,\n", jnum_s(mom_first), jnum_s(mom_last));
    fprintf(fo, "\"sux_first\": %s, \"sux_last\": %s,\n", jnum_s(sux_first), jnum_s(sux_last));
    fprintf(fo, "\"k_count\": [");
    for (long z = 0; z < nz; z++) fprintf(fo, "%s%ld", z ? ", " : "", kc[z]);
    fprintf(fo, "],\n\"prof_sum\": [");
    for (long z = 0; z < nz; z++) fprintf(fo, "%s%s", z ? ", " : "", jnum_s(prof[z]));
    fprintf(fo, "]\n}\n");
    fclose(fo);
    return 0;
}
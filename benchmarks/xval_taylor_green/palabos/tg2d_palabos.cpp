/* XVAL-LBM-TAYLORGREEN -- third-party reference tool #1 (palabos).

Standalone driver that runs the *same* case specification as the in-house
TensorLBM runner (see ``tensorlbm/xval/case_spec.py``) on the palabos
D2Q9/BGK lattice, and dumps raw measurements so the Python side can build the
cross-validation table without any hand-copied number.

Case (identical in all tools)
-----------------------------
    domain      : 2D periodic box [0,L) x [0,L), L = N lattice units, N = 128
    lattice     : D2Q9, BGK (single relaxation time)
    IC          : u = -U0 cos(kx) sin(ky),  v = +U0 sin(kx) cos(ky),  rho = 1
                  cell centre coordinate x_i = i + 0.5 (units of dx = 1)
    k           : 2 pi / L
    Re          : U0 L / nu = 100  ->  nu = 0.064  ->  tau = 0.5 + 3 nu = 0.692
    U0          : 0.05 (Ma = 0.087)
    steps       : 4900, sample every 100 steps
    metrics     : K(t) = 0.5 <u^2+v^2>,  |u|_max(t), velocity field at the
                  field-dump steps (for the field-error / phase metrics)

Outputs (written to the directory given as argv[1])
---------------------------------------------------
    k_history_palabos.csv   step,kinetic_energy,umax
    field_palabos_<step>.csv  i,j,ux,uy     (row-major, i fastest)
    run_info_palabos.json   machine-readable run metadata (best effort)

Usage
-----
    ./tg2d_palabos <out_dir> [N] [Re] [U0] [steps] [every] [field_steps_csv]
*/

#include "palabos2D.h"
#include "palabos2D.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace plb;
using plb::descriptors::D2Q9Descriptor;
typedef double T;

static const T PI = 3.14159265358979323846;

struct Config {
    plint N;
    T Re;
    T U0;
    T boost;
    plint steps;
    plint every;
    std::vector<plint> fieldSteps;
};

static void initTaylorGreen(MultiBlockLattice2D<T, D2Q9Descriptor>& lat, const Config& c, T k)
{
    for (plint iY = 0; iY < c.N; ++iY) {
        for (plint iX = 0; iX < c.N; ++iX) {
            T x = (T)iX + 0.5;
            T y = (T)iY + 0.5;
            Array<T, 2> u(c.boost - c.U0 * std::cos(k * x) * std::sin(k * y),
                          c.U0 * std::sin(k * x) * std::cos(k * y));
            Cell<T, D2Q9Descriptor>& cell = lat.get(iX, iY);
            iniCellAtEquilibrium(cell, (T)1, u);
        }
    }
}

struct Sample {
    T energy;
    T umax;
    T phaseDeg;   // phase of the (k,k) Fourier mode of ux
    T peakX;      // position of max|u| as a fraction of the box length
    T peakY;
};

/* Globally identical metric definitions (see tensorlbm/xval/metrics.py):
   K = 0.5*mean(u^2+v^2),  umax = max|u|,
   c = sum_ij ux_ij exp(-i k (x_i + y_j)),  x_i = i+0.5,  phase = arg(c),
   peak = argmax|u| position / N. */
static void measure(MultiBlockLattice2D<T, D2Q9Descriptor>& lat, const Config& c,
                    T k, Sample& s)
{
    T sumE = 0.0, maxU2 = -1.0;
    T cre = 0.0, cim = 0.0;
    plint iPeak = 0, jPeak = 0;
    Array<T, 2> u;
    for (plint iY = 0; iY < c.N; ++iY) {
        for (plint iX = 0; iX < c.N; ++iX) {
            lat.get(iX, iY).computeVelocity(u);
            T u2 = u[0] * u[0] + u[1] * u[1];
            sumE += (T)0.5 * u2;
            if (u2 > maxU2) { maxU2 = u2; iPeak = iX; jPeak = iY; }
            T ang = k * (((T)iX + (T)0.5) + ((T)iY + (T)0.5));
            cre += u[0] * std::cos(ang);
            cim -= u[0] * std::sin(ang);
        }
    }
    s.energy = sumE / (T)(c.N * c.N);
    s.umax = std::sqrt(maxU2);
    s.phaseDeg = std::atan2(cim, cre) * (T)180.0 / PI;
    s.peakX = ((T)iPeak + (T)0.5) / (T)c.N;
    s.peakY = ((T)jPeak + (T)0.5) / (T)c.N;
}

static void dumpField(MultiBlockLattice2D<T, D2Q9Descriptor>& lat, const Config& c,
                      const std::string& path)
{
    std::ofstream out(path.c_str());
    out.precision(17);
    out << "i,j,ux,uy\n";
    Array<T, 2> u;
    for (plint iY = 0; iY < c.N; ++iY) {
        for (plint iX = 0; iX < c.N; ++iX) {
            lat.get(iX, iY).computeVelocity(u);
            out << iX << "," << iY << "," << u[0] << "," << u[1] << "\n";
        }
    }
}

int main(int argc, char* argv[])
{
    plbInit(&argc, &argv);

    if (argc < 2) {
        pcout << "usage: " << argv[0]
              << " <out_dir> [N] [Re] [U0] [steps] [every] [field_steps_csv]" << std::endl;
        return 1;
    }

    const std::string outDir = argv[1];
    Config cfg;
    cfg.N = (argc > 2) ? atoi(argv[2]) : 128;
    cfg.Re = (argc > 3) ? (T)atof(argv[3]) : (T)100.0;
    cfg.U0 = (argc > 4) ? (T)atof(argv[4]) : (T)0.05;
    cfg.steps = (argc > 5) ? atoi(argv[5]) : 4900;
    cfg.every = (argc > 6) ? atoi(argv[6]) : 100;
    cfg.fieldSteps.clear();
    if (argc > 7) {
        std::stringstream ss(argv[7]);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) cfg.fieldSteps.push_back(atoi(tok.c_str()));
        }
    }
    cfg.boost = (argc > 8) ? (T)atof(argv[8]) : (T)0.0;

    const T L = (T)cfg.N;
    const T k = 2.0 * PI / L;
    const T nu = cfg.U0 * L / cfg.Re;
    const T tau = 0.5 + 3.0 * nu;
    const T omega = 1.0 / tau;

    MultiBlockLattice2D<T, D2Q9Descriptor> lattice(
        cfg.N, cfg.N, new BGKdynamics<T, D2Q9Descriptor>(omega));
    lattice.periodicity().toggleAll(true);

    initTaylorGreen(lattice, cfg, k);

    std::ostringstream info;
    info.precision(17);
    info << "{\n"
         << "  \"tool\": \"palabos\",\n"
         << "  \"N\": " << cfg.N << ",\n"
         << "  \"Re\": " << cfg.Re << ",\n"
         << "  \"U0\": " << cfg.U0 << ",\n"
         << "  \"boost_x\": " << cfg.boost << ",\n"
         << "  \"nu\": " << nu << ",\n"
         << "  \"tau\": " << tau << ",\n"
         << "  \"k\": " << k << ",\n"
         << "  \"steps\": " << cfg.steps << ",\n"
         << "  \"every\": " << cfg.every << ",\n"
         << "  \"lattice\": \"D2Q9\",\n"
         << "  \"collision\": \"BGKdynamics (single relaxation time)\",\n"
         << "  \"cell_centre_convention\": \"x_i = i + 0.5\"\n"
         << "}\n";
    {
        std::ofstream f((outDir + "/run_info_palabos.json").c_str());
        f << info.str();
    }

    std::ofstream hist((outDir + "/k_history_palabos.csv").c_str());
    hist.precision(17);
    hist << "step,kinetic_energy,umax,phase_deg,peak_x_frac,peak_y_frac\n";

    // initial sample (t = 0)
    {
        Sample s0;
        measure(lattice, cfg, k, s0);
        hist << 0 << "," << s0.energy << "," << s0.umax << "," << s0.phaseDeg << ","
             << s0.peakX << "," << s0.peakY << "\n";
        for (std::size_t s = 0; s < cfg.fieldSteps.size(); ++s) {
            if (cfg.fieldSteps[s] == 0) {
                std::ostringstream p;
                p << outDir << "/field_palabos_0.csv";
                dumpField(lattice, cfg, p.str());
            }
        }
    }

    for (plint step = 1; step <= cfg.steps; ++step) {
        lattice.collideAndStream();
        if (step % cfg.every == 0) {
            Sample sm;
            measure(lattice, cfg, k, sm);
            hist << step << "," << sm.energy << "," << sm.umax << "," << sm.phaseDeg << ","
                 << sm.peakX << "," << sm.peakY << "\n";
            hist.flush();
        }
        for (std::size_t s = 0; s < cfg.fieldSteps.size(); ++s) {
            if (cfg.fieldSteps[s] == step) {
                std::ostringstream p;
                p << outDir << "/field_palabos_" << step << ".csv";
                dumpField(lattice, cfg, p.str());
            }
        }
    }

    pcout << "palabos TG2D done: N=" << cfg.N << " steps=" << cfg.steps
          << " tau=" << tau << " -> " << outDir << std::endl;
    return 0;
}
/**
 * \file    lbt_ref_dump.cpp
 * \brief   Executable OSS reference for the 2b-t/LB-t "layout pattern" (A-A access
 *          pattern, NP-stacked populations).  Emits, as JSON on stdout, the exact
 *          linear indices produced by the *unmodified* upstream headers:
 *
 *            - lbt::Population<D2Q9<double>, NP>::spatialToLinear / linearToSpatial
 *            - indexRead<Even/Odd>  and  indexWrite<Even/Odd>  (parity semantics)
 *            - lattice constants (ND / OFF / DX / DY / W / MASK)
 *
 *          Nothing is re-derived by hand: every number in the JSON comes out of
 *          the upstream template instantiation at run time (zero hand-copy).
 *
 *          Build (upstream checkout required):
 *            g++ -O2 -std=c++17 -I/tmp/LB-t/include lbt_ref_dump.cpp -o lbt_ref_dump
 *          Run:
 *            ./lbt_ref_dump > lbt_ref_layout.json
 */
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "lbt/lattices/D2Q9.hpp"
#include "lbt/populations/population.hpp"

namespace {

template <class Lattice, std::int32_t NP>
void dumpLayout(std::ostream& os, std::int32_t const NX, std::int32_t const NY, std::int32_t const NZ,
                char const* const name, bool const last) {
  lbt::Population<Lattice, NP> pop {NX, NY, NZ};

  os << "    \"" << name << "\": {\n";
  os << "      \"NX\": " << NX << ", \"NY\": " << NY << ", \"NZ\": " << NZ << ",\n";
  os << "      \"NP\": " << NP << ",\n";
  os << "      \"ND\": " << Lattice::ND << ", \"OFF\": " << Lattice::OFF
     << ", \"SPEEDS\": " << Lattice::SPEEDS << ",\n";

  // lattice constants as instantiated upstream (ND includes the padding slot)
  os << "      \"DX\": [";
  for (std::int32_t i = 0; i < Lattice::ND; ++i) { os << (i ? ", " : "") << static_cast<double>(Lattice::DX[i]); }
  os << "],\n      \"DY\": [";
  for (std::int32_t i = 0; i < Lattice::ND; ++i) { os << (i ? ", " : "") << static_cast<double>(Lattice::DY[i]); }
  os << "],\n      \"W\": [";
  for (std::int32_t i = 0; i < Lattice::ND; ++i) { os << (i ? ", " : "") << static_cast<double>(Lattice::W[i]); }
  os << "],\n      \"MASK\": [";
  for (std::int32_t i = 0; i < Lattice::ND; ++i) { os << (i ? ", " : "") << static_cast<std::int32_t>(Lattice::MASK[i]); }
  os << "],\n";

  // spatialToLinear: exact memory layout formula ---------------------------------
  os << "      \"spatial_to_linear\": [";
  bool first {true};
  for (std::int32_t p = 0; p < NP; ++p) {
    for (std::int32_t n = 0; n < 2; ++n) {
      for (std::int32_t d = 0; d < Lattice::OFF; ++d) {
        for (std::int32_t x = 0; x < NX; ++x) {
          for (std::int32_t y = 0; y < NY; ++y) {
            for (std::int32_t z = 0; z < NZ; ++z) {
              auto const idx {pop.spatialToLinear(x, y, z, n, d, p)};
              os << (first ? "" : ", ") << "[" << x << "," << y << "," << z << "," << p << "," << n << "," << d << "," << idx << "]";
              first = false;
            }
          }
        }
      }
    }
  }
  os << "],\n";

  // read / write parity semantics ------------------------------------------------
  os << "      \"parity\": [";
  first = true;
  for (std::int32_t p = 0; p < NP; ++p) {
    for (std::int32_t n = 0; n < 2; ++n) {
      for (std::int32_t d = 0; d < Lattice::OFF; ++d) {
        for (std::int32_t x = 0; x < NX; ++x) {
          for (std::int32_t y = 0; y < NY; ++y) {
            auto const rEven {pop.template indexRead<lbt::Timestep::Even>(x, y, 0, n, d, p)};
            auto const rOdd  {pop.template indexRead<lbt::Timestep::Odd >(x, y, 0, n, d, p)};
            auto const wEven {pop.template indexWrite<lbt::Timestep::Even>(x, y, 0, n, d, p)};
            auto const wOdd  {pop.template indexWrite<lbt::Timestep::Odd >(x, y, 0, n, d, p)};
            os << (first ? "" : ", ") << "[" << x << "," << y << "," << p << "," << n << "," << d << ","
               << rEven << "," << rOdd << "," << wEven << "," << wOdd << "]";
            first = false;
          }
        }
      }
    }
  }
  os << "],\n";

  // linearToSpatial must round-trip: full inverse table ---------------------------
  std::int64_t const total {static_cast<std::int64_t>(NZ) * NY * NX * NP * Lattice::ND};
  bool ok {true};
  for (std::int64_t i = 0; i < total; ++i) {
    auto const [x, y, z, n, d, p] {pop.linearToSpatial(i)};
    if (pop.spatialToLinear(x, y, z, n, d, p) != i) { ok = false; break; }
  }
  os << "      \"roundtrip_ok\": " << (ok ? "true" : "false") << ",\n";
  os << "      \"total_slots\": " << total << "\n";
  os << "    }" << (last ? "\n" : ",\n");
}

} // namespace

int main(int argc, char** argv) {
  using Lattice = lbt::lattice::D2Q9<double>;
  // Upstream Indexing ctor prints NZ to std::cout (see indexing.hpp:47); route our
  // JSON to a file so the upstream chatter cannot corrupt the payload.
  std::ofstream file;
  if (argc > 1) { file.open(argv[1]); }
  std::ostream& out = (argc > 1) ? static_cast<std::ostream&>(file) : std::cout;
  constexpr std::int32_t NX {5};
  constexpr std::int32_t NY {4};
  constexpr std::int32_t NZ {1};

  out << "{\n";
  out << "  \"repo\": \"2b-t/LB-t\",\n";
  out << "  \"lattice\": \"D2Q9<double>\",\n";
  dumpLayout<Lattice, 1>(out, NX, NY, NZ, "npop1", false);
  dumpLayout<Lattice, 2>(out, NX, NY, NZ, "npop2", true);
  out << "}\n";
  return 0;
}
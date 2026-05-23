# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added
- Modern Python packaging via `pyproject.toml` with editable-install and dev extras support.
- Public package version metadata via `tensorlbm.__version__`.
- Structured logging helpers for long-running simulations.
- Checkpoint save/load utilities for resumable runs.
- Scientific VTK and HDF5 output helpers.
- Post-processing utilities for velocity profiles, pressure coefficient, and Q-criterion.
- Protocol-based extension interfaces for custom collision operators and boundary conditions.
- YAML/TOML configuration loading with environment-variable overrides.
- Reproducibility metadata capture including git commit, Python version, and package versions.
- Mass-correction helpers for 2-D and 3-D solvers.
- Bouzidi interpolated bounce-back support for curved 2-D boundaries.
- Full D3Q27 lattice implementation with BGK collision and streaming helpers.
- Verification tests for Poiseuille flow and Taylor-Green vortex decay.
- Property-based tests for collision invariants and D3Q27 lattice properties.
- MLUPS benchmark scripts for 2-D and 3-D kernels.
- Progress bars for selected long-running example runners.

### Changed
- Example runners now emit structured log messages instead of raw `print()` output.
- Runner metadata now includes reproducibility information.
- CI installs from the package metadata, runs mypy, and reports test coverage.
- Runtime requirements now include `tqdm` for optional progress reporting.

## [0.1.0] - Initial release

### Added
- D2Q9 and D3Q19 lattice definitions with equilibrium and macroscopic reconstruction.
- 2-D and 3-D BGK/MRT collision operators and periodic streaming kernels.
- 2-D and 3-D boundary-condition helpers including Zou/He inlet/outlet variants.
- Example runners for cylinder, sphere, ship-hull, and water-entry simulations.
- Marine engineering extensions including Wigley hull geometry, obstacle diagnostics, and wave boundary conditions.
- Smagorinsky LES turbulence models and shared simulation utilities.
- Unit and smoke tests covering core solvers, runners, and marine features.

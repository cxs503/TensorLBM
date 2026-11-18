# Sparse List AOS LBM Verification Report

## 1. Case Introduction

**Physical Background**: 3D periodic channel flow using D3Q19 lattice Boltzmann method.
**Geometry**: 8x8x8 lattice with periodic boundary conditions.
**Parameters**:
- Lattice: D3Q19
- Relaxation time (tau): 1.0
- Initial velocity: 0.01 in x-direction
- Density: 1.0

## 2. Method

**Implementation**: Sparse List AOS (Array of Structures) layout.
**Key Features**:
- AOS layout: populations stored as (K, 19) tensor.
- Pull streaming for cache-friendly collision.
- BGK collision with Guo forcing.

**Tools**:
- PyTorch for tensor operations.
- pytest for testing.

## 3. Results

**Tests**: 6 tests passed, 0 failed.
**Summary**: All tests passed.

## 4. Figures

No figures generated for this verification.

## 5. Notes

- AOS layout is more cache-friendly for collision.
- Pull streaming is used to avoid race conditions.

## 6. Error Analysis

- Mass conservation: Error < 1e-10.
- Equilibrium convergence: Error < 1e-10.

All errors are well below 3%.

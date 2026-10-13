# Changelog

## 2.0.0 (2026)
First release of the revived libnxc. The history of 0.x is in the git log.

### Added
- `exporter/`: JAX/Equinox → TorchScript exporter with a numerical gate (exc, vrho,
  vsigma and vtau must agree to a relative error of 1e-10). Recipes: `sara_unpol_log10`,
  `xcdiff_pol`, `mgga_x3` and `demo_mgga_alpha`.
- Self-contained model directories. Each holds a manifest (sources with sha256, networks and
  versions) and `source/` with the original checkpoints. Archives carry no local file paths.
- New models: spin-aware and unpolarized neural PBE clones, and neural r2SCAN clones (with and
  without a one-electron gate).
- `qe/`: Quantum ESPRESSO 7.4.1 interface (XClib patch, `input_dft='NXC'`), including
  meta-GGA stress. It comes with validation against analytic twin functionals, and EOS and
  band-gap scripts.
- CMake build for current libtorch (C++20), the Fortran module `nxc` (ISO_C_BINDING, error
  codes), and `test/revival` (C++ and Fortran APIs against pylibnxc, 49 cases).
- `devtools/hpc`: Linux cluster build recipe (conda-forge compilers, Slurm).
- `validation/`: PySCF end-to-end and finite-difference checks.

### Changed
- pylibnxc ported to PySCF ≥ 2.
- Libxc-style density, σ and τ floors (with the Fermi-hole-curvature cap σ ≤ 8ρτ) for
  models that do not handle thresholds internally. Exported models declare
  `"thresholds": "internal"` and skip them.
- Shipped TorchScript models no longer give NaN potentials at ζ = ±1 or σ = 0.

### Unchanged
- The C/C++ API and the legacy `nxc_f90_*` routines. The legacy Makefile build and the Libxc
  4.3.4 hook (`src/libxc`, `utils/libxc`) are kept but no longer maintained.

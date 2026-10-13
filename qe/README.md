# Quantum ESPRESSO with libnxc (machine-learned LDA/GGA/meta-GGA)

The interface is a patch to Quantum ESPRESSO 7.4.1 (`patches/`, GPL-2.0-or-later like QE
itself). Apply it to a QE 7.4.1 source tree:
```bash
git clone --branch qe-7.4.1 https://gitlab.com/QEF/q-e.git q-e-7.4.1
cd q-e-7.4.1 && git am /path/to/libnxc/qe/patches/0001-*.patch
```
The patch is small:
- a new module `XClib/xc_nxc.f90`;
- intercepts in XClib's `xc`, `xc_gcx` and `xc_metagcx` wrappers;
- the name `NXC` in `xclib_set_dft_from_name`;
- the option `-DQE_ENABLE_NXC=ON` in the top-level CMake.

Because the intercepts sit at the XClib wrapper level, SCF, forces and stress (including
meta-GGA stress) all go through libnxc. Nothing changes unless `input_dft='NXC'`.

## Build
First build libnxc with CMake (see the top-level README). Example (macOS arm64, serial,
Homebrew Libxc/OpenBLAS/FFTW; on Linux drop `SDKROOT` and use your own library paths):
```bash
export SDKROOT=$(xcrun --show-sdk-path) PKG_CONFIG_PATH=/opt/homebrew/opt/libxc/lib/pkgconfig
unset LDFLAGS
X=/opt/homebrew/opt/libxc
cmake -S q-e-7.4.1 -B build-nxc \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran \
  -DQE_ENABLE_MPI=OFF -DQE_ENABLE_OPENMP=OFF -DQE_ENABLE_LIBXC=ON -DQE_ENABLE_SCALAPACK=OFF \
  -DQE_FFTW_VENDOR=FFTW3 -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/openblas;/opt/homebrew/opt/fftw" \
  -DLIBXC_LIBRARIES=$X/lib/libxc.dylib -DLIBXC_LIBRARIES_F03=$X/lib/libxcf03.dylib \
  -DLIBXC_INCLUDE_DIR=$X/include -DLIBXC_INCLUDE_DIR_F03=$X/include \
  -DQE_ENABLE_NXC=ON -DNXC_ROOT=/path/to/libnxc/build-cmake
make -C build-nxc -j10 pw
```
Notes on this recipe:
- QE's Libxc finder only looks for static libraries, hence the explicit `LIBXC_*` paths for
  Homebrew's dylibs.
- `NXC_ROOT` points at a libnxc CMake build or install (`libnxc`, `nxc_fortran`, `nxc.mod`).
  The Fortran compiler must be the same one that built libnxc's `nxc.mod`.

## Use
```
&SYSTEM
  input_dft = 'NXC'
/
```
```bash
export NXC_MODEL=/path/to/libnxc/models/GGA_XC_PBECLONE_TL   # directory containing 'xc'
export OMP_NUM_THREADS=4                                              # libtorch threads per process
pw.x < pw.in > pw.out
```
- The model directory name must start with `LDA_`, `GGA_` or `MGGA_`; that sets the family.
  Exported models and libnxc's shipped models (e.g. `.../libnxc/models/MGGA_XC_SCAN`) both
  work.
- `nspin = 1` and `2` are supported (noncollinear through QE's local up/down frame).
- Units and conventions are those of the other XClib wrappers: Hartree; τ in Libxc units
  (PW passes `kin_r/e2`).
- For LSDA, the LDA branch converts (ρ, m) with QE's convention: ζ = m/|ρ| clipped to [−1, 1].

## Limitations
- **Placeholders.** For NXC, XClib sets the PBE (GGA) or TPSS (meta-GGA) indices only as
  placeholders, so that PW computes gradients and τ. Second derivatives (DFPT: `ph.x`, `hp.x`)
  stop with an error for `input_dft='NXC'`.
- **Legacy Makefile builds.** `xc_nxc.o` is listed in `XClib/Makefile`, but `make.inc` would
  also need `-D__NXC`, the `nxc.mod` include path and the libnxc libraries. Use CMake.
- **Meta-GGAs with nonlinear core corrections.** The core density has no matching τ, so
  τ < τ_W there. Libxc caps σ_σσ ≤ 8ρ_σ τ_σ, and libnxc now does the same for non-exporter
  models (`func_param.tau_w_cap`). Exporter models must handle α < 0 themselves.

- **QE 7.4.1 issue with Libxc meta-GGAs (not libnxc).** XClib's Libxc meta-GGA wrapper sets
  ε_xc and the potentials to zero at grid points where ∇ρ = 0 exactly (σ ≤ 1e-24). In
  high-symmetry crystals (e.g. the bond centres of diamond Si when the FFT dimension is
  divisible by 8), this gives grid-dependent total energies at the mRy level for Libxc
  meta-GGAs. Compare with Libxc meta-GGAs through stress-based equations of state, or avoid
  such FFT grids. libnxc evaluates every point and is not affected.

## Validation
The scripts take the `pw.x` binary from `$PW_X` and the pseudopotentials from
`$QE_PSEUDO_DIR` (QE's test-suite `pseudo/`) and `$NXC_SG15_DIR` (SG15 ONCV 1.2).
- `twins/make_twins.py`, `run_twins.py`: analytic twins of Slater, PBE, PKZB and LTA, written as
  libnxc models, against the QE-native and QE+Libxc functionals. Energies, forces and stresses
  agree to 1e-8 Ry wherever QE's own threshold rules do not apply
  (`twins/make_pkzb_qe_rule_twin.py` reproduces those rules).
- `fd_force_check.py`: forces against finite differences of the energy.
- `run_eos.py`: equations of state of the neural functionals against PBE, SCAN and r2SCAN.
  - The PBE clones reproduce PBE to 0.0001–0.005 Å and 1–3 GPa (Si, c-BN, bcc Mo).
  - The r2SCAN clone reproduces Libxc r2SCAN for Si (SG15) to 0.001 Å and 0.1 GPa.
- `run_gaps.py`: Si band gaps (generalized-KS eigenvalues). The r2SCAN clones reproduce
  r2SCAN to 0.01 eV.

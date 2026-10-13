# Lib**n**xc

libnxc is a library for running **machine-learned exchange-correlation functionals**
(LDA, GGA, meta-GGA, and NeuralXC-type) inside density functional theory codes. It is
written in C++ with Fortran bindings and mirrors the API of [Libxc](https://libxc.gitlab.io),
so that codes that call Libxc can call libnxc in the same way. A pure-Python implementation,
`pylibnxc`, is included and plugs into [PySCF](https://pyscf.org). Models are TorchScript
files, and PyTorch's automatic differentiation provides all the potential terms.

**libnxc 2.0 (2026)** is a revival and substantial rewrite of the original libnxc
(S. Dick and M. Fernández-Serra, 2020–2021; see [Credits and history](#credits-and-history)).
It adds:

- **An exporter from JAX/Equinox to libnxc** (`exporter/`). Models trained with
  differentiable-DFT codes in JAX are converted into libnxc TorchScript models. An export
  fails unless exc, vrho, vsigma and vtau agree with the JAX model to a relative error of
  1e-10. The check uses random and molecular densities, spin-polarized and unpolarized, and goes
  through the same code path used in SCF. Every exported model carries a manifest (sources
  with sha256, architecture, versions) and the source checkpoints themselves.
- **Full meta-GGA (τ) support** for exported models, with Libxc-compatible density, σ and
  τ thresholds.
- **A Quantum ESPRESSO 7.4.1 interface** (`qe/`). A small patch to XClib (`input_dft='NXC'`)
  routes the SCF, forces and stress (including meta-GGA stress) through libnxc, for
  plane-wave calculations of solids.
- **A modern build.** CMake against current libtorch (C++20; tested with torch 2.14) on Linux
  and macOS (Apple silicon). There is a new Fortran module `nxc` (ISO_C_BINDING, with error
  codes), and the legacy `nxc_f90_*` routines are kept.
- **pylibnxc for PySCF ≥ 2** (RKS/UKS, mixtures with exact exchange).
- **Tests.** The C++ and Fortran APIs are checked against pylibnxc on 49 reference cases. The
  exporter and PySCF test suites are included, and QE is validated against analytic "twin"
  functionals.
- **New models** (`models/`): spin-aware neural PBE clones and neural r2SCAN clones (see
  [Models](#models)).

## Installation

### pylibnxc (Python, PySCF)
```bash
pip install -e .                 # torch, numpy
pip install -e .[pyscf]          # + PySCF interface
pip install -e .[export]         # + JAX/Equinox exporter
```

### libnxc (C++/Fortran)
```bash
TORCH_PREFIX=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build-cmake -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DNXC_TEST_PYTHON=$(which python)
cmake --build build-cmake -j
ctest --test-dir build-cmake --output-on-failure
```
- libtorch ≥ 2.10 needs a C++20 compiler (GCC ≥ 11 or a recent clang). The libtorch of a
  pip/conda torch works. Use the torch version that exported the models, or newer: the
  models in `models/` were written with torch 2.14.
- On macOS with Homebrew gfortran, `export SDKROOT=$(xcrun --show-sdk-path)` first.
- Use pthreads OpenBLAS in the same environment. The OpenMP build loads a second OpenMP
  runtime next to torch's, and the process aborts.
- Cluster recipe (conda-forge compilers, Slurm): [devtools/hpc](devtools/hpc/README.md).

The C++ API is in `src/nxc.h`, and the Fortran module is in `src/nxc_mod.f90`. Arrays are
spin-blocked with the grid index fastest (`rho(np, nspin)`); see `docs/cpp.rst`. Any directory
`<LDA|GGA|MGGA>_<...>/xc` can be passed by path as a model.

## Quickstart

### PySCF
```python
from pyscf import gto
from pylibnxc.pyscf import RKS, UKS

mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='def2-tzvp', spin=1)
mf = UKS(mol, nxc='models/MGGA_XC_R2SCAN_CLONE_SC')      # a model directory, or a shipped name
mf.kernel()
```
Mixtures use the PySCF syntax, e.g. `nxc='0.25*HF + 0.75*GGA_X_PBE, GGA_C_PBE'`.

### Quantum ESPRESSO
Build `pw.x` with the patch in [qe/](qe/README.md), then
```
&SYSTEM
  input_dft = 'NXC'
/
```
```bash
export NXC_MODEL=$PWD/models/GGA_XC_PBECLONE_TL     # directory containing 'xc'
pw.x < pw.in > pw.out
```

### Export a JAX/Equinox model
```bash
python -m exporter.export --recipe xcdiff_pol \
    --fx <dir>/GGA_Fx_xcdiff_d3_n16_s42_v_20000 --fc <dir>/GGA_Fc_xcdiff_d3_n16_s42_v_20000 \
    --name GGA_XC_MYCLONE --outdir models
python -m exporter.export --recipe mgga_x3 --xc <dir>/xc.eqx --name MGGA_XC_MYMODEL --outdir models
```
This writes `models/<name>/`, which contains:
- `xc`: the TorchScript model;
- `manifest.json`;
- `weights.npz`: raw weights, for evaluators that do not use libtorch;
- `check.json`: the numerical gate;
- `source/`: the Equinox checkpoints.

| recipe | network classes | spin |
|---|---|---|
| `sara_unpol_log10` | `GGA_F{x,c}_G_transf[_lin]` (log10 features) | unpolarized only; the model refuses \|ζ\| > 1e-6 |
| `xcdiff_pol` | `GGA_F{x,c}_xcdiff[_unct]` (spin-aware, ln features) | exchange by spin scaling; F_c(x0, ln ζ′, x2)·PW92(r_s, ζ) |
| `mgga_x3` | meta-GGA `MXC` (r2SCAN-type ᾱ) | as `xcdiff_pol`, plus x3 = (1−ᾱ)/(1+ᾱ); optional one-electron gate on F_c |
| `demo_mgga_alpha` | `MGGA_F{x,c}_demo` (untrained) | for testing vtau |

To add a new model family, add three things:
- its Equinox class in `exporter/eqx_classes.py`;
- a JAX recipe in `exporter/recipes_jax.py`;
- the same maths in `exporter/nxc_torch.py`.

The numerical gate checks that the two implementations agree.

## Models

New in 2.0 (exported with the exporter; the source checkpoints are in `models/<name>/source`):

| name | description |
|---|---|
| `GGA_XC_PBECLONE_TL` | spin-aware neural PBE clone with exact constraints [3] |
| `GGA_XC_PBECLONE_UT` | the same, unconstrained [3] |
| `GGA_XC_PBECLONE_UNPOL_S05R1` | unpolarized neural PBE clone [3] (refuses spin-polarized input) |
| `MGGA_XC_R2SCAN_CLONE` | neural meta-GGA clone of r2SCAN |
| `MGGA_XC_R2SCAN_CLONE_SC` | the same, with a one-electron (self-correlation-free) gate on correlation |
| `MGGA_XC_DEMO_RANDOM` | untrained meta-GGA, for testing only |

The shipped models of libnxc 0.x are listed below ([Shipped functionals](#shipped-functionals)).
They remain available by name, e.g. `GGA_XC_PBE`, `MGGA_XC_SCAN` and `MGGA_XC_XCDIFF` [2].

## Validation (summary)

- **PySCF.** Exported models reproduce the SCF energies of the JAX code that trained them to
  0.000 µEh (H2O, OH; def2-TZVP). The r2SCAN clones are within 0.04–0.3 mEh of Libxc r2SCAN,
  and the PBE clones within 0.06–0.5 mEh of Libxc PBE.
- **Quantum ESPRESSO 7.4.1.**
  - Analytic twins of Slater, PBE, PKZB and LTA reproduce QE-native and QE+Libxc energies,
    forces and stresses to 1e-8 Ry.
  - The neural PBE clones reproduce PBE lattice constants to 0.0001–0.005 Å and bulk moduli to
    1–3 GPa (Si, c-BN, bcc Mo).
  - The neural r2SCAN clone reproduces Libxc r2SCAN for Si to 0.001 Å, 0.1 GPa, and band gaps
    to 0.01 eV.

  The scripts are in `qe/`, and the limitations and caveats are in [qe/README.md](qe/README.md).
- **Scripts.** `validation/` holds the PySCF end-to-end scripts, the finite-difference V_xc
  check and the regression against libnxc 0.x numbers.

## Credits and history

libnxc was created by **Sebastian Dick** in the group of **Marivi Fernández-Serra** at Stony
Brook University (2020–2021, [github.com/semodi/libnxc](https://github.com/semodi/libnxc)).
It was used to deploy NeuralXC [1] and the differentiable-programming functional
xc-diff [2]. This repository continues that project. The full development history of the
original libnxc is preserved in this repository's git history, with its original authorship.

libnxc 2.0 (2026) was developed by M. Fernández-Serra's group at Stony Brook University. Its
main pieces are:
- the exporter, the meta-GGA work and the C++/Fortran modernization;
- the PySCF ≥ 2 port;
- the Quantum ESPRESSO interface.

The neural PBE clones are from S. Navarro-Rodríguez *et al.* [3]. The shipped NN-LDA/GGA/meta-GGA
"HM" models are from Nagai, Akashi and Sugino (see below).

See [AUTHORS.md](AUTHORS.md) and [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use libnxc, please cite this repository ([CITATION.cff](CITATION.cff)) and the papers
of the functionals you use:

1. S. Dick and M. Fernández-Serra, *Machine learning accurate exchange and correlation
   functionals of the electronic density*, Nat. Commun. **11**, 3509 (2020).
2. S. Dick and M. Fernández-Serra, *Highly accurate and constrained density functional
   obtained with differentiable programming*, Phys. Rev. B **104**, L161109 (2021).
3. S. Navarro-Rodríguez, A. Wills, K. J. Daas, M. Camarasa-Gómez and M. Fernández-Serra,
   *Constraint-aware functional cloning for stable and transferable machine-learned density
   functional theory*, arXiv:2605.10331 (2026).

## Acknowledgments

The development of libnxc 2.0 was supported by the U.S. National Science Foundation under
award 2427902 (Collaborative Research: Optimally designed exchange-correlation functionals
for solids using machine learning; Stony Brook University, with award 2427903 at UC Irvine).
Any opinions, findings, and conclusions or recommendations expressed in this material are those
of the authors and do not necessarily reflect the views of the National Science Foundation.

## License

libnxc is distributed under the Mozilla Public License 2.0 ([LICENSE](LICENSE)), the license
of the original project. The Quantum ESPRESSO patch in `qe/patches/` modifies Quantum ESPRESSO
and is distributed under QE's license (GPL-2.0-or-later).

---

## Original libnxc documentation

The sections below are the user documentation of libnxc 0.x. They still describe the C++,
Fortran and Python APIs and the shipped functionals. Online docs:
[libnxc.readthedocs.io](https://libnxc.readthedocs.io) (0.x). Libnxc can use MPI, and model
inference on GPUs through CUDA. The serialized model is treated as a black box, so any
TorchScript model with the documented [input/output form](https://libnxc.readthedocs.io/en/latest/functionals.html)
can be used, even a simple polynomial.

### Dependencies

Libnxc depends on PyTorch. The C++/Fortran implementation requires that libtorch is made available at compile time, pylibnxc depends on pytorch.
Both libraries are available for download on the [pytorch website](https://pytorch.org/get-started/locally/).

pylibnxc further depends on numpy, which is a requirement for pytorch and should therefore be installed automatically.

**Optional dependencies:**

For unit testing, pylibnxc currently requires [pyscf](https://sunqm.github.io/pyscf/install.html) which can be obtained through

`pip install pyscf`

To unit test the C++/Fortran implementation [GoogleTest](https://github.com/google/googletest) is required.

### Installation (legacy Makefile build)

To compile Libnxc:

1. Create a work directory and copy `utils/Makefile` into it (this can be done quickly with `sh config.sh`)
1. Adjust arch.make according to your system
1. Change into the work directory and run `make`. The default target is `make all` which will build the library `libnxc.so`
  and if `LIBXCDIR` is set in `arch.make` will create the necessary symbolic links in your Libxc installation

To install pylibnxc:

pylibnxc operates independent of Libnxc and can be installed from within the root directory with

`pip install -e .`


### Quickstart

The functionality of Libnxc may be best explained by an example

```c++
#include "nxc.h"

int main()
{
  nxc_func_type p;
  func_param fp;
  int nspin = NXC_UNPOLARIZED;

  double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
  double sigma[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
  double exc[5];

  double vrho[5];
  double vsigma[5];

  nxc_func_init(&p, "GGA_XC_PBE", fp, nspin);

  nxc_gga_exc_vxc(&p, 5, rho, sigma, exc, vrho, vsigma);

}
```

- `nxc_func_type p` stores the functional
- `func_param fp` stores parameters that determine how the functional is evaluated and how Libnxc communicates with the external code. For now, it suffices to know that these parameters are set to sensible default values. We will return to it later.
- `nxc_func_init(&p, "GGA_XC_PBE", fp, nspin)` loads the functional "GGA_PBE", the second argument can either be the name of a shipped functional (see below for a list thereof) or a path to a serialized TorchScript model (at the moment custom models are only available in pylibnxc).
- `nxc_gga_exc_vxc(&p, 5, rho, sigma, exc, vrho, vsigma)` evaluates the functional and stores the energy per unit particle, and the potential terms $\delta E/ \delta \rho(\mathbf r)$ and $\delta E/ \delta \sigma(\mathbf r)$ in `exc`, `vrho` and `vsigma`, respectively.



Running the program above should produce the following output:
```
0.1     -0.459810
0.2     -0.507294
0.3     -0.562470
0.4     -0.611119
0.5     -0.653536
```

The same program can be run from Fortran:

```fortran
PROGRAM test

      implicit none
      integer               :: ierr, i
      character(len=100)    :: nxc_path
      real(8)               :: rho(5), sigma(5), exc(5), vrho(5), vsigma(5)
      ! Initialize grid, basis, etc.
      nxc_path='GGA_PBE'

      rho = (/0.1,0.2,0.3,0.4,0.5/)
      sigma = (/0.2,0.3,0.4,0.5,0.6/)
      call nxc_f90_set_code(0)
      call nxc_f90_func_init(nxc_path, LEN_TRIM(nxc_path), ierr)

      call nxc_f90_gga_exc_vxc(5, rho, sigma, exc, vrho, vsigma)

      do i=1,5
        write(*,"(T1,F8.6,T12,F8.6)") rho(i) , exc(i)
      end do

END PROGRAM test
```

... and Python

```python
from pylibnxc import LibNXCFunctional
import numpy as np


if __name__ == '__main__':

    func = LibNXCFunctional("GGA_XC_PBE")
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sigma = np.array([0.2, 0.3, 0.4, 0.5, 0.6])

    inp = {'rho': rho,
           'sigma': sigma}

    results = func.compute(inp)

    print(np.stack([inp['rho'], np.round(results['zk'],6)],axis=-1))
```

### Using pylibnxc with PySCF

Using Pylibnxc in PySCF is as simple as changing two lines of code:

Starting from
```python
from pyscf import gto
from pyscf.dft import RKS

mol = gto.M(atom='H 0 0 0; H 0 0 0.7', basis='6-311G')
mf = RKS(mol)
mf.xc = 'PBE'
mf.kernel()
```
one can use pylibnxc
```python
from pyscf import gto
from pylibnxc.pyscf import RKS

mol = gto.M(atom='H 0 0 0; H 0 0 0.7', basis='6-311G')
mf = RKS(mol, nxc='GGA_XC_PBE', nxc_kind='grid')
mf.kernel()
```
The second version would run a SCF calculation using our machine-learned version of
PBE (see [Shipped functionals](#shipped-functionals).

For unrestricted Kohn-Sham calculations `pylibnxc.pyscf.UKS` is available as well.
The `nxc` keyword supports mixing of functionals similar to pyscf, e.g.
`nxc ='0.25*HF + 0.75*GGA_X_PBE, GGA_C_PBE'` would correspond to a neural network version
of PBE0.
Currently, mixing of libxc functionals with libnxc functionals is not supported.

### Using libnxc with Libxc (legacy)

The simplest way to use machine learned functionals in existing electronic
structure codes is by linking Libxc with Libnxc. This way, any code that supports
Libxc functionals (Quantum Espresso, CP2K, Vasp, Psi4, ...) has access to Libnxc routines.
This comes with two caveats:
1) Only grid based functionals are supported through this solution. There is currently
no way to pass the additional information needed by Libnxc (atomic positions, unit cell size etc.)
through Libxc.
2) Without adjusting the electronic structure code, using Libnxc with Neural Network functionals
may be significantly slower than calling traditional functionals. This is because some codes pass
the density grid point by grid point. This is fine for traditional functionals, but to efficiently
evaluate NN functionals, grid points should be processed in a vectorized fashion. Codes like
Quantum espresso already do so, but others, like CP2K, will need to be modified to ensure efficient
evaluation.

Using Libnxc from within Libxc is straightforward:

- Set the `LIBXCDIR` variable in arch.make to the
location of the libxc `src` directory.
- In order to be able to run unit tests using Libxc `LIBXC_INCLUDE`
and `LIBXC_LD` need to be set accordingly.
- From within the build directory `make libxc` will create symbolic links of the required
Libnxc source files within the Libxc `src` directory and copy a modified `Makefile.am` that includes
instructions to link to Libnxc. In some cases automake needs to be run in the Libxc directory.
- After successfully compiling Libnxc, Libxc should be re-compiled to include the new functionals.

### Shipped functionals

The following functionals were introduced in

[1] *Nagai, Ryo, Ryosuke Akashi, and Osamu Sugino. "Completing density functional theory by machine learning hidden messages from molecules." npj Computational Materials 6.1 (2020): 1-8.*

Please consider citing the paper when using them.

- **LDA_HM**: NN-LSDA introduced in [1]
- **GGA_HM**: NN-GGA introduced in [1]
- **MGGA_HM**: NN-meta-GGA introduced in [1]


The following functionals are only included for testing purposes. For small molecules an accuracy of about 1 mHartree can be expected.

- **GGA_XC_PBE**: Neural Network fitted to reproduce the popular PBE functional
- **GGA_X_PBE**: Exchange part of GGA_PBE
- **GGA_C_PBE**: Correlation part of GGA_PBE

- **MGGA_XC_SCAN**: Neural Network fitted to reproduce SCAN
- **MGGA_X_SCAN**: Exchange part of GGA_SCAN
- **MGGA_C_SCAN**: Correlation part of GGA_SCAN

### Custom functionals

Entries for the functionals listed above are hard-coded into Libnxc. To include custom functionals without the need to change any code and re-compile
Libnxc we have added the entries **GGA_XC_CUSTOM** and **MGGA_XC_MCUSTOM**. Using these functionals will instruct Libnxc to look for the corresponding model in the current working directory (e.g. the location from where the DFT calculation is run). This way, any model that follows the form defined in the [documentation](https://libnxc.readthedocs.io/en/latest/functionals.html) can be deployed by copying (or creating a symbolic link of) the custom model into the working directory.
